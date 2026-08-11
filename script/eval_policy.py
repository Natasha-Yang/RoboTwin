"""Policy evaluation driver (expert feasibility check, seed loop, per-chunk rollout).

If the policy exposes a ``model.online_critic`` (pi05 builds one only when
``guidance_scale != 0``), the rollout additionally collects a chunk-level transition after
every control step and runs TD updates on that critic, which steers the frozen pi0.5 flow
sampler; the critic persists and keeps learning across episodes for the whole eval run, and
progress is logged to W&B. Setting ``train_critic_online: false`` keeps the guidance but leaves
the critic frozen at its checkpoint -- no transitions are collected and no TD update runs. With
no critic this is the plain baseline rollout. Configure via
``policy/<policy_name>/deploy_policy.yml``.
"""

import sys
import os
import json
import re
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from envs.utils.debug_vis import QValueRecorder, RolloutFrameLog, TCPWrenchRecorder

import numpy as np
from pathlib import Path
from collections import deque
import traceback
import pandas as pd

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = Path(__file__).resolve()
parent_directory = current_file_path.parent


# ===== Online critic logging =====
# Only used when the policy exposes a trained-online critic (pi05 with guidance_scale != 0);
# the plain baseline never reaches any of this, so wandb stays an optional dependency.


def init_wandb(usr_args, save_dir, current_time, resume_id=None):
    """Open the run's W&B session, rejoining `resume_id` when this is a resumed run.

    `resume="allow"` rather than `"must"`: a run whose W&B side was never created (offline, or
    the id predates a project change) should still start logging, not abort the eval over it.
    """
    import wandb

    if not usr_args.get("wandb_enabled", True):
        wandb.init(mode="disabled")
        return None

    run_name = usr_args.get("wandb_run_name")
    if run_name is None:
        run_name = (
            f"{usr_args['task_name']}-{usr_args['policy_name']}-"
            f"{usr_args.get('model_name', 'model')}-seed{usr_args['seed']}-{current_time}"
        )

    init_kwargs = {
        "project": usr_args.get("wandb_project", "qmfm"),
        "name": run_name,
        "config": {**usr_args, "eval_save_dir": str(save_dir)},
        "dir": str(save_dir),
        "tags": ["pi05", "online-critic", "eval"],
        "entity": "natashayang04-university-of-toronto",
    }
    wandb_mode = usr_args.get("wandb_mode", None)
    if wandb_mode is not None:
        init_kwargs["mode"] = wandb_mode
    if resume_id:
        init_kwargs["id"] = resume_id
        init_kwargs["resume"] = "allow"

    run = wandb.init(**init_kwargs)
    wandb.define_metric("critic/update")
    wandb.define_metric("critic/*", step_metric="critic/update")
    wandb.define_metric("rollout/*", step_metric="critic/update")
    wandb.define_metric("eval/episode")
    wandb.define_metric("eval/*", step_metric="eval/episode")
    return run


def _uses_online_critic(model):
    """Whether this policy intends to run a trained-online critic.

    Distinct from `model.online_critic`, which stays None until the critic is actually built
    -- pi05 defers that to the first observation, since the critic's action width comes from
    the embodiment's joint vector.
    """
    return bool(getattr(model, "uses_online_critic", getattr(model, "online_critic", None) is not None))


def _trains_online_critic(model):
    """Whether that critic is also *learning* here, or is frozen at its checkpoint.

    pi05's `train_critic_online: false` keeps the guidance but skips the replay collection and
    the TD updates below, which is how an offline-trained critic is evaluated as-is. Policies
    that do not know the knob keep the historical behavior (train).
    """
    return _uses_online_critic(model) and bool(getattr(model, "train_critic_online", True))


def as_bool(value, default):
    """A config flag that may arrive as a CLI override string.

    The yml gives real booleans, but `--overrides use_step_reward false` reaches here as the
    *string* `"false"` (`parse_args_and_config`'s `eval()` raises NameError on it and the parser
    keeps the text), and `bool("false")` is True -- exactly backwards for a switch. Anything
    unrecognised raises rather than defaulting, so a typo cannot silently turn a flag on.

    The pi05 policy has its own copy of this for the knobs it reads itself
    (`policy/pi05/deploy_policy.py::_as_bool`); it runs on the other side of
    `eval_function_decorator`, which loads the policy package rather than importing from here.
    """
    if value is None:
        return default
    if not isinstance(value, str):
        return bool(value)
    text = value.strip().lower()
    if text in ("true", "yes", "on", "1"):
        return True
    if text in ("false", "no", "off", "0", ""):
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def control_step_reward(TASK_ENV, success_now, prev_success, use_step_reward=True):
    """Reward for the control step (action chunk) that just executed.

    Success pays 1.0 on the transition that first reaches it; every other step is worth
    whatever the task's own `step_reward` says -- a *delta* (progress since the last call, see
    e.g. `envs/lift_pot.py`), so it must be called exactly once per control step, and only by
    the driver running the rollout. Tasks that define no shaping fall back to 0.0, which leaves
    a sparse terminal reward.

    `use_step_reward=False` drops the shaping for every task, leaving that same sparse terminal
    reward (`use_step_reward` in deploy_policy.yml / eval.sh's 10th arg). The task's
    `step_reward` is then never called at all, so its internal "since the last call" state never
    advances -- which is what keeps a disabled run from paying a huge accumulated delta if the
    flag were flipped mid-rollout.

    Shared with `script/collect_dataset.py` so an offline dataset's `reward` column and the
    reward the online critic is fed by `commit()` are the same quantity -- nothing downstream
    can tell them apart if they drift (see `multisensory_steering/cfgs/qmfm.yaml`).
    """
    if success_now and not prev_success:
        return 1.0
    if not use_step_reward:
        return 0.0
    return float(getattr(TASK_ENV, "step_reward", lambda: 0.0)())


def _scalar(value):
    return float(np.asarray(value))


def _window_mean(values):
    return float(np.mean(values)) if values else 0.0


class BestOfNRecorder:
    """Per-episode record of what best-of-N selection is actually buying.

    The policy (pi05: `PI0.get_action`, only when `best_of_n > 1`) leaves the ensemble-mean Q of
    every candidate chunk it chose between in `model.last_best_scores`. Two numbers summarize a
    control step's selection:

    * `q_gain` -- the winner's Q minus the mean over the candidates, i.e. how much value the
      selection added over executing an arbitrary one of them. This is the whole point of
      best-of-N: a gain that sits near zero means the critic cannot separate the chunks the
      sampler draws, so the extra N-fold denoising is buying nothing.
    * `q_spread` -- max minus min, the range it was choosing over. Puts the gain in context: a
      small gain over a wide spread is a different failure from both being small.

    Values are in the critic's own output space (normalized return units for a checkpoint
    trained offline), so they are comparable within a run but not across critics.

    Consumed like `QValueRecorder.record` -- the scores are cleared as they are read, so a
    control step that sampled nothing cannot re-log the previous step's selection.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self.gain = []
        self.spread = []

    def record(self, model):
        scores = getattr(model, "last_best_scores", None)
        if scores is None:
            return
        model.last_best_scores = None
        scores = np.asarray(scores, dtype=np.float64).ravel()
        self.gain.append(float(scores.max() - scores.mean()))
        self.spread.append(float(scores.max() - scores.min()))

    def metrics(self):
        """This episode's averages, then start a fresh episode. Empty when nothing was logged."""
        if not self.gain:
            return {}
        out = {
            "bestofn/q_gain": _window_mean(self.gain),
            "bestofn/q_spread": _window_mean(self.spread),
            "bestofn/chunks": len(self.gain),
        }
        self._reset()
        return out


def log_critic_update(wandb_run, online_critic, model, info, chunk_count, episode_idx, action_count):
    if wandb_run is None or info is None:
        return
    buf = online_critic.buffer.size if online_critic.buffer is not None else 0
    wandb_run.log({
        "critic/update": int(online_critic.num_updates),
        "critic/loss": _scalar(info["critic_loss"]),
        "critic/q_mean": _scalar(info["q_mean"]),
        "critic/q_max": _scalar(info["q_max"]),
        "critic/q_min": _scalar(info["q_min"]),
        "critic/target_q_mean": _scalar(info["target_q_mean"]),
        "critic/reward_mean": _scalar(info["reward_mean"]),
        # The rate this update was actually taken at: the critic's Adam warms up over
        # `lr_warmup_steps` and cosine-decays over `lr_decay_steps` (cfgs/qmfm.yaml), both
        # counted in updates of this run, so the curve doubles as a check that the schedule
        # covers the update budget the run really has.
        "critic/lr": _scalar(info["lr"]),
        "critic/buffer_size": int(buf),
        "critic/guidance_scale": float(model.scheduled_guidance_scale()),
        "critic/guidance_scale_target": float(model.guidance_scale_target),
        "rollout/chunk_count": int(chunk_count),
        "rollout/episode": int(episode_idx),
        "rollout/action_count": int(action_count),
    })


def log_episode(
    wandb_run,
    online_critic,
    model,
    episode_idx,
    success,
    num_steps,
    episode_reward,
    success_rate,
    success_rate_ma,
    reward_ma,
    ma_window,
    best_of_n_log=None,
):
    if wandb_run is None:
        return
    metrics = {
        "eval/episode": int(episode_idx),
        "eval/success": float(success),
        "eval/num_steps": int(num_steps),
        "eval/reward": float(episode_reward),
        "eval/success_rate": float(success_rate),
        "eval/success_rate_ma": float(success_rate_ma),
        "eval/reward_ma": float(reward_ma),
        "eval/ma_window": int(ma_window),
    }
    if online_critic is not None:
        buf = online_critic.buffer.size if online_critic.buffer is not None else 0
        metrics.update({
            "critic/update": int(online_critic.num_updates),
            "critic/buffer_size": int(buf),
            "critic/guidance_scale": float(model.scheduled_guidance_scale()),
            "critic/guidance_scale_target": float(model.guidance_scale_target),
        })
    if best_of_n_log is not None:
        metrics.update(best_of_n_log.metrics())
    wandb_run.log(metrics)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def visualize_debug_obs(observation, step_idx=0, save_dir=None, show=True, task_env=None, wrench_recorder=None):
    """Visualize per-camera images (rgb / depth / segmentation) and the point cloud.

    When a ``wrench_recorder`` is passed it also samples the end-effector contact wrench
    from ``task_env`` at this step (see ``envs/utils/debug_vis.py::TCPWrenchRecorder``); that
    part needs neither depth nor point clouds, so it works under any task config that sets
    `debug: true`.

    Enabled by `debug: true` in the task config. The observation layout follows
    ``_base_task.get_obs`` (see envs/_base_task.py); each entry is present only when
    the matching ``data_type`` flag is enabled:
      - observation["observation"][camera_name]["rgb"]:                H x W x 3 uint8
      - observation["observation"][camera_name]["depth"]:              H x W float64 depth in mm
      - observation["observation"][camera_name]["mesh_segmentation"]:  H x W x 3 uint8 (palette-colored)
      - observation["observation"][camera_name]["actor_segmentation"]: H x W x 3 uint8 (palette-colored)
      - observation["pointcloud"]:                                     N x 6 (xyz + rgb in [0, 1])

    Whichever image modalities are present get tiled into one figure (rows = modality,
    columns = camera). When ``show`` is True the figure/point-cloud open interactively
    (blocking, so close each window to step forward). PNG/PLY copies are always written
    to ``save_dir`` when it is provided, which keeps debug output usable headlessly.
    """
    if wrench_recorder is not None and task_env is not None:
        wrench_recorder.record(task_env, observation, step_idx)

    import matplotlib
    if not show:
        matplotlib.use("Agg")  # no display: render to file only
    import matplotlib.pyplot as plt

    obs = observation.get("observation", {})
    # Per-camera image modalities to tile, in display order (rendered when present).
    image_modalities = ["rgb", "depth", "mesh_segmentation", "actor_segmentation"]
    cam_names = [name for name, data in obs.items()
                 if isinstance(data, dict) and any(m in data for m in image_modalities)]
    # Rows = only the modalities that at least one camera actually carries.
    rows = [m for m in image_modalities if any(m in obs[name] for name in cam_names)]

    if save_dir is not None and (rows or observation.get("pointcloud", [])):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ---- Image modalities (rgb / depth / mesh & actor segmentation) ----
    if cam_names and rows:
        n, r = len(cam_names), len(rows)
        fig, axes = plt.subplots(r, n, figsize=(4 * n, 3.5 * r), squeeze=False)
        for ci, name in enumerate(cam_names):
            cam = obs[name]
            for ri, mod in enumerate(rows):
                ax = axes[ri, ci]
                ax.axis("off")
                if mod not in cam:
                    continue
                if mod == "depth":
                    # Depth stored in mm; convert to meters and mask invalid (0) pixels.
                    depth_m = np.asarray(cam["depth"], dtype=np.float32) / 1000.0
                    masked = np.ma.masked_where(depth_m <= 0, depth_m)
                    im = ax.imshow(masked, cmap="turbo")
                    ax.set_title(f"{name} depth [m]")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    # rgb / mesh_segmentation / actor_segmentation are H x W x 3 uint8.
                    ax.imshow(cam[mod])
                    ax.set_title(f"{name} {mod}")
        fig.suptitle(f"step {step_idx}")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(save_dir / f"obs_step{step_idx:04d}.png", dpi=100)
        if show:
            plt.show()
        plt.close(fig)

    # ---- Point cloud ----
    pcd = np.asarray(observation.get("pointcloud", []))
    if pcd.ndim == 2 and pcd.shape[0] > 0:
        try:
            import open3d as o3d

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
            if pcd.shape[1] >= 6:
                cloud.colors = o3d.utility.Vector3dVector(np.clip(pcd[:, 3:6], 0.0, 1.0))
            if save_dir is not None:
                # open3d's bindings take a str filename, not a PathLike.
                o3d.io.write_point_cloud(str(Path(save_dir) / f"pcd_step{step_idx:04d}.ply"), cloud)
            if show:
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                o3d.visualization.draw_geometries([cloud, frame], window_name=f"pointcloud step {step_idx}")
        except Exception as e:
            print(f"[debug] point cloud visualization failed: {e}")

def get_camera_config(camera_type):
    camera_config_path = parent_directory.parent / "task_config" / "_camera_config.yml"

    assert camera_config_path.is_file(), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = Path(robot_file) / "config.yml"
    with robot_config_file.open("r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def snapshot_config(src_path, values, dst_dir):
    """Copy a yml config into `dst_dir` with the values actually used written in.

    Comments and layout are preserved; a line is rewritten only when `values` resolves that
    top-level key differently (a CLI override, or -- in the critic config -- a key the deploy
    config shadows). Keys whose value opens a block on the next line are skipped: nothing
    overrides those.
    """
    src_path = Path(src_path)
    with src_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    original = yaml.safe_load("".join(lines)) or {}

    for i, line in enumerate(lines):
        match = re.match(r"^([A-Za-z_][\w.-]*):[ \t]+(\S[^\n]*?)[ \t]*$", line)
        if match is None:  # comment, blank, indented, or a key with a block value
            continue
        key, raw_value = match.group(1), match.group(2)
        if key not in values or values[key] == original.get(key):
            continue
        # Keep any trailing `# ...` comment, unless the value itself is quoted (a `#` could
        # then be part of the string rather than the start of a comment).
        comment = ""
        if not raw_value.startswith(("'", '"')):
            hash_at = raw_value.find(" #")
            if hash_at != -1:
                comment = "  " + raw_value[hash_at:].lstrip()
        lines[i] = yaml.safe_dump({key: values[key]}, sort_keys=False).strip() + comment + "\n"

    header = f"# Snapshot of {src_path.resolve()} as used by this eval run.\n"
    with (Path(dst_dir) / src_path.name).open("w", encoding="utf-8") as f:
        f.writelines([header] + lines)


# ===== Incremental results, and resuming an interrupted run =====
# An eval run is hours long and used to write nothing at all until it finished, so a crash --
# or the renderer wedging, which it does -- discarded every episode it had already paid for.
# Each episode now commits to the run directory in dependency order:
#
#   1. `_episode_results.csv`     the episode's own row, appended
#   2. `online_value_critic.pkl`  the critic: params, target, and Adam state
#   3. `resume_state.json`        the loop counters -- written last, and atomically
#
# The json is the commit marker: it only names an episode once 1 and 2 are durably on disk, so
# an interruption anywhere in the sequence rewinds to the last episode that finished all three.
# The csv can legitimately be one row ahead of it, which `load_resume_state` trims.
#
# What is *not* checkpointed is the replay buffer -- gigabytes, mostly SigLIP features (see
# CLAUDE.md §5a) -- so a resumed critic keeps its weights and optimizer but refills its buffer
# from empty, and runs no TD update until `start_training` transitions are back in it.
#
# The guidance ramp carries across the break too: `critic_ramp_baseline` records the update
# count the ramp is measured from, and is handed back to the policy on resume. Without it the
# resumed run reloads its own critic as an ordinary `critic_ckpt`, re-bases the ramp at that
# checkpoint's counter, and comes back at guidance 0 to climb the whole ramp again.

RESUME_STATE = "resume_state.json"
EPISODE_CSV = "_episode_results.csv"
CRITIC_CKPT = "online_value_critic.pkl"
EPISODE_COLUMNS = ["episode", "seed", "num_steps", "success", "reward", "success_rate_ma", "reward_ma"]


def append_episode_row(csv_path, row):
    """Append one finished episode to the results csv, writing the header for the first."""
    pd.DataFrame([row], columns=EPISODE_COLUMNS).to_csv(
        csv_path, mode="a", header=not csv_path.exists(), index=False
    )


def save_critic_atomically(online_critic, path):
    """Checkpoint the critic via a temp file, so an interrupted save cannot shred the old one.

    `resume_state.json` is written after this returns and points at the result, so a torn
    pickle here would otherwise be a checkpoint the next run is told to trust.
    """
    tmp = path.with_name(path.name + ".tmp")
    online_critic.save(tmp)
    tmp.replace(path)


def numpy_random_state():
    """The global numpy RNG state, as json.

    Part of the seed bookkeeping and not a formality: the episode's language instruction is
    drawn with `np.random.choice`, so a resume that skipped this would replay the recorded
    seeds against different instructions.
    """
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return {"kind": kind, "keys": keys.tolist(), "pos": int(pos),
            "has_gauss": int(has_gauss), "cached_gaussian": float(cached)}


def set_numpy_random_state(blob):
    np.random.set_state((blob["kind"], np.array(blob["keys"], dtype=np.uint32),
                         int(blob["pos"]), int(blob["has_gauss"]), float(blob["cached_gaussian"])))


def write_resume_state(save_dir, state):
    """Record the loop state atomically -- the last write of an episode."""
    path = Path(save_dir) / RESUME_STATE
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(path)  # rename is atomic within a filesystem: a torn write never becomes state


def load_resume_state(save_dir):
    """The state left by the last fully-committed episode, or None if there is nothing to resume."""
    save_dir = Path(save_dir)
    path = save_dir / RESUME_STATE
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        state = json.load(f)

    # The csv is written before the state, so an episode interrupted between the two leaves a
    # row nothing accounts for. Drop it instead of double-counting it on the next append.
    csv_path = save_dir / EPISODE_CSV
    if csv_path.exists():
        rows = pd.read_csv(csv_path)
        if len(rows) > state["episodes"]:
            print(f"[resume] dropping {len(rows) - state['episodes']} uncommitted csv row(s)")
            rows.iloc[: state["episodes"]].to_csv(csv_path, index=False)
    return state


def find_resumable_run(run_root):
    """The most recent run directory under `run_root` carrying resume state, if any."""
    run_root = Path(run_root)
    if not run_root.is_dir():
        return None
    candidates = [d for d in run_root.iterdir() if d.is_dir() and (d / RESUME_STATE).exists()]
    return max(candidates, key=lambda d: (d / RESUME_STATE).stat().st_mtime, default=None)


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    # Shaped per-step progress reward on/off (see control_step_reward). Resolved here, before
    # the config snapshot and the W&B init below, so both record the boolean actually used
    # rather than the `"false"` string a CLI override arrives as.
    usr_args["use_step_reward"] = as_bool(usr_args.get("use_step_reward"), True)
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with (Path("./task_config") / f"{task_config}.yml").open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"

    with embodiment_config_path.open("r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with (Path(CONFIGS_PATH) / "_camera_config.yml").open("r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # `resume: true` continues the newest interrupted run for this task/policy/config/checkpoint
    # in place rather than opening a fresh timestamped directory -- the episodes, the critic and
    # the seed sequence all live in there and only mean anything together. With no such run (or
    # `resume: false`) this is an ordinary new run.
    run_root = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}")
    resume_dir = find_resumable_run(run_root) if usr_args.get("resume", False) else None
    save_dir = resume_dir if resume_dir is not None else run_root / current_time
    save_dir.mkdir(parents=True, exist_ok=True)
    args["eval_save_dir"] = str(save_dir)

    resume_state = load_resume_state(save_dir) if resume_dir is not None else None
    if resume_state is not None:
        print(f"\033[93m[resume] continuing {save_dir}\033[0m")
        print(f"\033[93m[resume] {resume_state['episodes']} episode(s) done, "
              f"{resume_state['successes']} successful, next seed {resume_state['now_seed']}\033[0m")
        # Point the policy at this run's own critic and ask for the optimizer back with it.
        # Without `restore_optimizer` the critic would reload its weights but restart Adam and
        # the LR schedule from the top of warmup (see OnlineValueCritic._adopt_pending_optimizer).
        critic_ckpt = save_dir / CRITIC_CKPT
        if critic_ckpt.exists():
            # The guidance ramp is measured from the update count the run started at, and a
            # warm start normally re-bases it at the checkpoint's counter. Here the checkpoint
            # IS this run's own critic, so re-basing would zero the ramp and make a run that had
            # already ramped to full guidance crawl back up from 0. Hand the original baseline
            # back instead. (Only meaningful with the critic actually reloaded, hence in here:
            # applying it to a critic starting at 0 updates would pin guidance at 0 instead.)
            baseline = resume_state.get("critic_ramp_baseline")
            if baseline is None and not usr_args.get("critic_ckpt"):
                # State files written before 2026-08-08 have no such key. A run that trained
                # its critic from scratch ramped from 0 updates, so that is exactly its
                # baseline; one that warm-started from an offline critic cannot be
                # reconstructed and keeps the old (restart-the-ramp) behavior.
                baseline = 0
            if baseline is not None:
                usr_args["critic_ramp_baseline"] = int(baseline)
            usr_args["critic_ckpt"] = str(critic_ckpt)
            usr_args["restore_optimizer"] = True
            print(f"\033[93m[resume] critic + optimizer from {critic_ckpt}"
                  + (f", guidance ramp from update {baseline}" if baseline is not None
                     else ", guidance ramp restarts (pre-2026-08-08 state file)") + "\033[0m")
            print("\033[93m[resume] the replay buffer is not checkpointed -- it refills from "
                  "empty before TD updates restart\033[0m")

    # Snapshot the deploy config (and the critic config it includes) next to the results, with
    # the CLI overrides written in, so a run's settings stay readable -- and accurate -- after
    # the ymls are edited.
    for config_path in (usr_args.get("_config_path"), usr_args.get("critic_config_path")):
        if config_path and Path(config_path).is_file():
            snapshot_config(config_path, usr_args, save_dir)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    # Online-critic knobs (see deploy_policy.yml) forwarded into the eval loop; inert unless
    # the policy actually built a critic.
    args["train_freq"] = usr_args.get("train_freq", 1)
    args["wandb_ma_window"] = usr_args.get("wandb_ma_window", 20)
    # Checkpointed after every episode, not just at the end, so a resume has a recent critic
    # to restore. Same flag that governs the final save below.
    args["save_critic"] = usr_args.get("save_critic", False)
    # Off leaves the sparse terminal reward, which is what the critic is then trained on.
    args["use_step_reward"] = usr_args["use_step_reward"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = usr_args.get("test_num", 100)
    topk = 1

    model = get_model(usr_args)
    # The policy decides whether guidance is on (pi05: guidance_scale != 0). The critic object
    # itself may not exist until the first observation (its shape depends on the embodiment),
    # so W&B keys off the policy's declared intent rather than off `model.online_critic`.
    # A resumed run rejoins its own W&B run, so the critic curves stay one continuous series
    # across the interruption instead of restarting at update 0 in a new one.
    wandb_run = (init_wandb(usr_args, save_dir, current_time,
                            resume_id=(resume_state or {}).get("wandb_run_id"))
                 if _uses_online_critic(model) else None)

    st_seed, suc_num, episode_results = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   wandb_run=wandb_run,
                                   resume_state=resume_state)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = save_dir / "_result.txt"
    with file_path.open("w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    # `_episode_results.csv` needs nothing here: the rollout appended each episode's row as it
    # finished, which is what makes an interrupted run salvageable.
    episode_file_path = save_dir / EPISODE_CSV

    # Persist the online-trained critic alongside the eval results. A frozen one has nothing to
    # persist -- it is a byte-for-byte copy of `critic_ckpt`, which the snapshotted config
    # already names -- so the file is skipped rather than written misleadingly.
    online_critic = getattr(model, "online_critic", None)
    if online_critic is not None and usr_args.get("save_critic", False):
        if _trains_online_critic(model):
            critic_path = save_dir / CRITIC_CKPT
            online_critic.save(critic_path)
            print(f"saved online critic to {critic_path}")
        else:
            print("train_critic_online is off -- not saving the critic (unchanged from "
                  f"{usr_args.get('critic_ckpt')})")

    print(f"Data has been saved to {file_path}")
    if wandb_run is not None:
        wandb_run.finish()
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                wandb_run=None,
                resume_state=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    episode_results = {col: [] for col in EPISODE_COLUMNS}

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    # The contact wrench is one of the sensor modalities a guided critic can condition on, and
    # the only one the env has to be told to log -- every other one rides along in the
    # observation. Switched by the task config's `data_type.wrench` like the rest of them; a
    # critic configured for `wrench.*` against a config that has it off fails at startup,
    # saying which modalities the run does provide.
    args["record_step_wrench"] = bool(args["data_type"].get("wrench", False))

    # ===== Online QMFM Value critic (trained across the whole eval run) =====
    # The critic object is read fresh via `getattr(model, "online_critic", None)` at each use
    # site rather than captured here: pi05 builds it lazily on the first observation, because
    # its action width comes from the embodiment. It stays None for the plain baseline, which
    # makes every critic block below inert.
    #
    # `train_critic` is the second switch: with `train_critic_online: false` the critic still
    # steers the sampler, but it is frozen at `critic_ckpt`, so no transition is collected and
    # no TD update runs. Unlike the critic object this is known up front -- it is a
    # construction-time property of the policy, not something built on the first observation.
    train_critic = _trains_online_critic(model)
    train_freq = int(args.get("train_freq", 1))
    # The reward every consumer below sees -- the critic's `commit()`, the Q recorder and the
    # per-episode totals in the csv. With the shaping off it is sparse: 1.0 on success, 0.0
    # everywhere else, regardless of whether the task defines a `step_reward`.
    use_step_reward = bool(args.get("use_step_reward", True))
    ma_window = max(1, int(args.get("wandb_ma_window", 20)))
    success_window = deque(maxlen=ma_window)
    reward_window = deque(maxlen=ma_window)
    chunk_count = 0
    last_info = None
    print(f"\033[95mStep reward (shaped progress):\033[0m "
          + ("ON" if use_step_reward else "OFF (sparse success reward only)"))

    # Debug visualization of depth maps / point clouds (see visualize_debug_obs).
    debug = args.get("debug", False)
    debug_show = bool(os.environ.get("DISPLAY"))  # only pop up windows when a display exists
    save_dir = Path(args.get("eval_save_dir", "eval_result"))
    debug_save_dir = save_dir / "debug_vis" if debug else None
    # All of these share `debug_vis/episode<N>/`: each recorder appends the episode dir itself,
    # since it only learns the episode index at flush time. The frame log holds the head-camera
    # frames both GIFs are drawn on, so the rollout is downscaled and kept once.
    frame_log = RolloutFrameLog() if debug else None
    wrench_recorder = TCPWrenchRecorder(debug_save_dir, frame_log) if debug else None
    # The Q recorder needs a critic to read a value off; without guidance there is none, and
    # the policy's own per-step scoring stays off (it is an extra critic forward per chunk).
    q_recorder = QValueRecorder(debug_save_dir, frame_log) if debug and _uses_online_critic(model) else None
    if q_recorder is not None:
        model.record_q_values = True
    # Best-of-N selection diagnostics. Unlike the Q recorder this costs nothing (the scores are
    # a by-product of the selection the sampler already made) and needs no `debug`, so it is on
    # whenever the policy is actually choosing between candidates.
    best_of_n = int(getattr(model, "best_of_n", 1) or 1)
    best_of_n_recorder = BestOfNRecorder() if best_of_n > 1 else None
    if best_of_n_recorder is not None:
        print(f"\033[96m[critic]\033[0m best-of-{best_of_n} sampling ON: {best_of_n} candidate "
              f"chunks per control step, highest ensemble-mean Q executed")
    if debug:
        print(f"\033[93m[debug] depth/point-cloud visualization ON "
              f"(interactive={debug_show}, saving to {debug_save_dir})\033[0m")
        print(f"\033[93m[debug] TCP wrench logging ON (saving to {debug_save_dir}/episode<N>/)\033[0m")
        print(f"\033[93m[debug] critic Q logging "
              + (f"ON (saving to {debug_save_dir}/episode<N>/)" if q_recorder is not None
                 else "OFF (no critic: guidance_scale is 0 and best_of_n is 1)") + "\033[0m")

    # Where each finished episode is committed (see the notes above `append_episode_row`).
    csv_path = save_dir / EPISODE_CSV
    critic_path = save_dir / CRITIC_CKPT
    save_critic = bool(args.get("save_critic", True))

    if resume_state is not None:
        # Pick the loop back up exactly where it stopped. `now_seed` is the load-bearing one:
        # the seed sequence is not a function of the episode index, since the expert check
        # rejects an unpredictable subset, so it cannot be recomputed -- only restored.
        now_id = resume_state["now_id"]
        now_seed = resume_state["now_seed"]
        succ_seed = resume_state["episodes"]
        suc_test_seed_list = list(resume_state["suc_test_seed_list"])
        TASK_ENV.suc = resume_state["successes"]
        TASK_ENV.test_num = resume_state["test_num"]
        chunk_count = resume_state["chunk_count"]
        set_numpy_random_state(resume_state["numpy_random_state"])

        # The csv is the record of the episodes themselves; reload it so the moving averages
        # continue over the interruption rather than restarting from an empty window.
        past = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=EPISODE_COLUMNS)
        for col in EPISODE_COLUMNS:
            episode_results[col] = past[col].tolist()
        success_window.extend(float(v) for v in episode_results["success"][-ma_window:])
        reward_window.extend(float(v) for v in episode_results["reward"][-ma_window:])
        print(f"\033[93m[resume] {len(past)} episode(s) reloaded, resuming at seed {now_seed} "
              f"({succ_seed}/{test_num} done)\033[0m")

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        expert_success = False
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                expert_success = TASK_ENV.plan_success and TASK_ENV.check_success()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", e)
                print(stack_trace)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or expert_success:
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        prev_success = False
        episode_reward = 0.0
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            # The step this observation was taken at, held because `take_action_cnt` advances
            # during the chunk -- the Q recorder logs against the observation's own index.
            step = TASK_ENV.take_action_cnt
            observation = TASK_ENV.get_obs()
            if debug:
                visualize_debug_obs(
                    observation,
                    step_idx=TASK_ENV.take_action_cnt,
                    save_dir = (
                        debug_save_dir / f"episode{TASK_ENV.test_num}"
                        if debug_save_dir else None
                    ),
                    show=debug_show,
                    task_env=TASK_ENV,
                    wrench_recorder=wrench_recorder,
                )
            eval_func(TASK_ENV, model, observation)
            success_now = bool(TASK_ENV.eval_success)
            reward = control_step_reward(TASK_ENV, success_now, prev_success, use_step_reward)
            episode_reward += reward

            # The chunk's Q only exists once the policy has sampled it, so unlike the wrench
            # this is logged after the control step -- but against `step`, the count the
            # observation it was drawn from was taken at, so it lines up with that frame.
            if q_recorder is not None:
                q_recorder.record(model, observation, step, reward)
            if best_of_n_recorder is not None:
                best_of_n_recorder.record(model)

            # Online critic: close the chunk transition (SARSA), then run a TD update. Skipped
            # for a frozen critic -- it guides, but its parameters and buffer stay untouched.
            online_critic = getattr(model, "online_critic", None) if train_critic else None
            if online_critic is not None:
                done = success_now or (TASK_ENV.take_action_cnt >= TASK_ENV.step_lim)
                online_critic.commit(reward, done)
                chunk_count += 1
                if chunk_count % train_freq == 0:
                    info = online_critic.train_step()
                    if info is not None:
                        last_info = info
                        log_critic_update(
                            wandb_run,
                            online_critic,
                            model,
                            info,
                            chunk_count,
                            TASK_ENV.test_num,
                            TASK_ENV.take_action_cnt,
                        )

            prev_success = success_now
            if success_now:
                succ = True
                break
        episode_steps = TASK_ENV.take_action_cnt
        episode_seed = now_seed
        episode_results["episode"].append(TASK_ENV.test_num)
        episode_results["seed"].append(episode_seed)
        episode_results["num_steps"].append(episode_steps)
        episode_results["success"].append(succ)
        episode_results["reward"].append(episode_reward)
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()
        if wrench_recorder is not None:
            wrench_recorder.flush(TASK_ENV.test_num)
        if q_recorder is not None:
            q_recorder.flush(TASK_ENV.test_num)
        if frame_log is not None:
            frame_log.flush()  # last: both GIFs above are drawn from it

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        # Online-critic diagnostics.
        online_critic = getattr(model, "online_critic", None)
        if online_critic is not None:
            buf = online_critic.buffer.size if online_critic.buffer is not None else 0
            guidance = model.scheduled_guidance_scale()
            if not train_critic:
                print(f"\033[96m[critic]\033[0m frozen at {online_critic.num_updates} updates "
                      f"guidance={guidance:.4g}/{model.guidance_scale_target:.4g} "
                      f"(train_critic_online is off: no replay collection, no TD updates)")
            elif last_info is not None:
                print(f"\033[96m[critic]\033[0m buffer={buf} updates={online_critic.num_updates} "
                      f"guidance={guidance:.4g}/{model.guidance_scale_target:.4g} "
                      f"loss={float(last_info['critic_loss']):.4f} "
                      f"q_mean={float(last_info['q_mean']):.3f} "
                      f"target_q={float(last_info['target_q_mean']):.3f} "
                      f"reward_mean={float(last_info['reward_mean']):.3f}")
            else:
                print(f"\033[96m[critic]\033[0m buffer={buf} "
                      f"guidance={guidance:.4g}/{model.guidance_scale_target:.4g} "
                      f"(warming up, no update yet)")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1
        success_window.append(float(succ))
        reward_window.append(float(episode_reward))
        success_rate = TASK_ENV.suc / TASK_ENV.test_num
        success_rate_ma = _window_mean(success_window)
        reward_ma = _window_mean(reward_window)
        episode_results["success_rate_ma"].append(success_rate_ma)
        episode_results["reward_ma"].append(reward_ma)
        log_episode(
            wandb_run,
            online_critic,
            model,
            TASK_ENV.test_num,
            succ,
            TASK_ENV.take_action_cnt,
            episode_reward,
            success_rate,
            success_rate_ma,
            reward_ma,
            ma_window,
            best_of_n_log=best_of_n_recorder,
        )

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(success_rate*100, 1)}%\033[0m "
            f"(MA{ma_window}: \033[95m{round(success_rate_ma*100, 1)}%\033[0m, reward={reward_ma:.3f}), "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

        # Commit the episode. Strict order: the row, then the critic, then the state that
        # asserts both are on disk (see the notes above `append_episode_row`). Everything
        # recorded here is state the next run cannot recompute -- above all `now_seed`, which
        # depends on which seeds the expert check happened to reject.
        append_episode_row(csv_path, {
            "episode": TASK_ENV.test_num,
            "seed": episode_seed,
            "num_steps": episode_steps,
            "success": succ,
            "reward": episode_reward,
            "success_rate_ma": success_rate_ma,
            "reward_ma": reward_ma,
        })
        if save_critic and online_critic is not None and train_critic:
            save_critic_atomically(online_critic, critic_path)
        write_resume_state(save_dir, {
            "episodes": succ_seed,
            "successes": TASK_ENV.suc,
            "test_num": TASK_ENV.test_num,
            "now_id": now_id,
            "now_seed": now_seed,
            "chunk_count": chunk_count,
            "suc_test_seed_list": suc_test_seed_list,
            "numpy_random_state": numpy_random_state(),
            "critic_updates": int(online_critic.num_updates) if online_critic is not None else 0,
            # The update count this run's guidance ramp is measured from -- 0 for a critic
            # trained from scratch here, the checkpoint's lifetime count for one warm-started
            # from `critic_ckpt`. Restored on resume so the ramp continues rather than
            # restarting against the run's own checkpoint (see the resume block in `main`).
            "critic_ramp_baseline": int(getattr(model, "critic_ramp_baseline", 0)),
            "wandb_run_id": wandb_run.id if wandb_run is not None else None,
        })

    return now_seed, TASK_ENV.suc, episode_results


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Policy-specific hyperparameters may be factored out into a file that ships with the
    # implementation they configure (e.g. the critic config in the multisensory_steering
    # repo, so it stays in sync with the critic that reads it). Merge it in *underneath* the
    # deploy config: precedence is CLI overrides > deploy config > included file.
    include_path = config.get("critic_config_path")
    if include_path:
        with Path(include_path).open("r", encoding="utf-8") as f:
            included = yaml.safe_load(f) or {}
        config = {**included, **config}

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    # Kept so `main` can copy the deploy config into the eval_result dir.
    config["_config_path"] = args.config

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
