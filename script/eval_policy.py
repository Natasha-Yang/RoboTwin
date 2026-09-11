"""Policy evaluation driver (expert feasibility check, seed loop, per-chunk rollout).

If the policy exposes a ``model.online_critic`` (pi05 builds one only when
``guidance_scale != 0``), the rollout additionally collects a chunk-level transition after
every control step and runs TD updates on that critic, which steers the frozen pi0.5 flow
sampler; the critic persists and keeps learning across episodes for the whole eval run, and
progress is logged to W&B. Under ``save_critic`` it is written to the result directory after
every episode, so an interrupted run leaves a checkpoint the next one can resume from via
``critic_ckpt``; a second copy of the best episode's critic is kept alongside it. Which episode
counts as best is decided by ``eval_interval``: with it set, every that many episodes the run
re-plays a fixed set of ``eval_seed`` episodes with the policy frozen and scores it
(``run_holdout_eval``), and that held-out success rate is the criterion; with it 0 the criterion
falls back to ``success_rate_ma`` over the training episodes. Setting ``train_critic_online:
false`` keeps the guidance but leaves
the critic frozen at its checkpoint -- no transitions are collected and no TD update runs. With
no critic this is the plain baseline rollout. Configure via
``policy/<policy_name>/deploy_policy.yml``.
"""

import sys
import os
import contextlib
import json
import random
import re
import subprocess
import shutil

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs._base_task import resolve_background_texture_pool
from envs.utils.create_actor import UnStableError
from envs.utils.debug_vis import (
    DemoRetrievalRecorder,
    QValueRecorder,
    RolloutFrameLog,
    ScriptedInterventionRecorder,
    TCPWrenchRecorder,
)
from envs.utils.obs_modalities import obs_modalities
from envs.utils import eval_profiler

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
    wandb.define_metric("intervention/episode")
    wandb.define_metric("intervention/*", step_metric="intervention/episode")
    # The periodic held-out evaluation (`eval_interval`), logged against the training episode it
    # ran after -- so its curve lines up with `eval/success_rate_ma`, the criterion it replaces.
    wandb.define_metric("holdout/*", step_metric="eval/episode")
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


def as_bool_or_none(value):
    """`as_bool` without the raise -- None when the value is not a boolean at all.

    For a key that is a switch *or* something else, `resume` being the only one: `true` continues
    the newest run, a path continues that one (see `find_resumable_run`).
    """
    try:
        return as_bool(value, False)
    except ValueError:
        return None


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
    # Whatever the update reported, under its own name -- the two critic families share the TD
    # diagnostics (`critic_loss`, `q_mean`, `target_q_mean`, `reward_mean`, and the `lr` the
    # Adam schedule reached) and each adds its own on top, DSRL's actor and temperature among
    # them. `critic/loss` is kept as an alias so a run's curves stay comparable with older ones.
    metrics = {f"critic/{key}": _scalar(value) for key, value in info.items()}
    metrics.update({
        "critic/update": int(online_critic.num_updates),
        "critic/loss": _scalar(info["critic_loss"]),
        "critic/buffer_size": int(buf),
        "critic/guidance_scale": float(model.scheduled_guidance_scale()),
        "critic/guidance_scale_target": float(model.guidance_scale_target),
        "rollout/chunk_count": int(chunk_count),
        "rollout/episode": int(episode_idx),
        "rollout/action_count": int(action_count),
    })
    wandb_run.log(metrics)


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

    When a ``wrench_recorder`` is passed it also drains the end-effector contact wrench the
    previous chunk logged -- one sample per physics step, not one per call (see
    ``envs/utils/debug_vis.py::TCPWrenchRecorder``). That part needs neither depth nor point
    clouds, so it works under any task config that sets `debug: true`; with `data_type.wrench`
    off there is no log to drain and it falls back to a single sample taken here.

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


SNAPSHOT_HEADER_PREFIX = "# Snapshot of "


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
    # A resumed run snapshots the snapshot it was configured from (src and dst are then the same
    # file), so drop any header this function wrote before rather than stacking a new one on it.
    while lines and lines[0].startswith(SNAPSHOT_HEADER_PREFIX):
        lines.pop(0)
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

    header = f"{SNAPSHOT_HEADER_PREFIX}{src_path.resolve()} as used by this eval run.\n"
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
# The critic is the one file that can also be *ahead* of the marker, because it is additionally
# written every `critic_save_every_updates` TD updates -- an episode runs for up to `step_lim`
# control steps, so waiting for the boundary can put hundreds of updates at risk. That is
# deliberate and harmless: a resume replays the interrupted episode's seed against a critic that
# already saw part of it, which duplicates a little training data but loses none of it. Only the
# `critic_updates` field of the state file goes stale as a result, and nothing reads it back --
# the ramp is measured from `critic_ramp_baseline` against the checkpoint's own counter, which
# is correct precisely because those updates really did happen.
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
# The best critic seen so far, alongside the latest one. `CRITIC_CKPT` is the run's *state* --
# what a resume must pick up, and by construction whatever the last episode happened to leave --
# while online TD on a few thousand correlated transitions is not monotone, so the end of a run
# is not reliably its best point. `CRITIC_CKPT_BEST` is the checkpoint at the best score any
# episode of this run reached, which is what you want to evaluate or ship. Never resumed from:
# resuming from it would rewind the seed sequence and the optimizer to an episode the csv says
# is already done.
#
# *Which* score is `eval_interval`'s business. With the periodic held-out evaluation on (the
# default in policy/pi05/deploy_policy.yml) it is the success rate over a fixed set of
# `eval_seed` episodes, re-run with the critic frozen every `eval_interval` episodes; with it
# off it falls back to `success_rate_ma`, the moving average over the training episodes
# themselves. The held-out score is the better criterion for the same reason a held-out set
# always is: the training episodes are the ones the critic just learned from, they are a
# different set of seeds every time, and their moving average moves with the seeds as much as
# with the critic. See `run_holdout_eval`.
CRITIC_CKPT_BEST = "online_value_critic_best.pkl"
EPISODE_COLUMNS = ["episode", "seed", "num_steps", "success", "reward", "success_rate_ma", "reward_ma"]
# One row per held-out evaluation: which training episode it ran after, the critic's lifetime
# update count at the time (what identifies the checkpoint being scored) and the score itself.
HOLDOUT_CSV = "_holdout_results.csv"
# How far above the run's own first seed the held-out search starts by default. The run consumes
# seeds upward from `st_seed` -- one per candidate, so more than one per episode, since the
# expert gate rejects some -- and the held-out set is only held out while it stays below this.
# 1000 is comfortable for the usual `test_num: 300`; `eval_policy` warns if a run ever reaches it.
HOLDOUT_SEED_OFFSET = 1000
HOLDOUT_COLUMNS = ["episode", "critic_updates", "guidance_scale",
                   "episodes", "successes", "success_rate", "reward_mean", "steps_mean"]


def append_holdout_row(csv_path, row):
    pd.DataFrame([row], columns=HOLDOUT_COLUMNS).to_csv(
        csv_path, mode="a", header=not csv_path.exists(), index=False
    )


def append_episode_row(csv_path, row):
    """Append one finished episode to the results csv, writing the header for the first."""
    pd.DataFrame([row], columns=EPISODE_COLUMNS).to_csv(
        csv_path, mode="a", header=not csv_path.exists(), index=False
    )


def save_critic_atomically(online_critic, path):
    """Checkpoint the critic via a temp file, so an interrupted save cannot shred the old one.

    `OnlineValueCritic.save` pickles straight into its destination, which is fine for a single
    write at the end of a run. This is called after *every* episode instead, so overwriting in
    place at that rate would mean a kill mid-write destroys the last good checkpoint along with
    the new one. Pickle into a sibling temp file and rename it into position instead: the rename
    is atomic on POSIX, so the destination is always either the previous complete checkpoint or
    this one. `resume_state.json` is written after this returns and points at the result, so a
    torn pickle here would otherwise be a checkpoint the next run is told to trust.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        online_critic.save(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)  # no-op after a successful replace; cleans up after a failed one


def promote_best_critic(latest_path, best_path):
    """Copy the just-written latest checkpoint over the best one, atomically.

    A copy rather than a second `online_critic.save`: the pickle for this episode was written a
    moment ago from the same object, so re-pickling would only burn the time again and risk the
    two files disagreeing if anything about the critic moved in between. Same temp-file-then-
    rename discipline as `save_critic_atomically`, for the same reason -- this runs after every
    improving episode, so a kill mid-write must not shred the best checkpoint the run has.
    """
    tmp = best_path.with_name(best_path.name + ".tmp")
    try:
        shutil.copyfile(latest_path, tmp)
        tmp.replace(best_path)
    finally:
        tmp.unlink(missing_ok=True)


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


# The run identity: the keys that decide *which* experiment a run directory holds. A resume
# reloads them from the run's own config snapshot (see `parse_args_and_config`), so they cannot
# be taken from the working-tree yml -- that file describes whatever is being set up next, not
# what this run was launched as. Anything passed on the command line is checked against the
# snapshot rather than applied, because an override here does not reconfigure the run, it
# mislabels it: the episodes, seeds and critic in the directory are still the old experiment's.
IDENTITY_KEYS = ("policy_name", "task_name", "task_config", "ckpt_setting",
                 "train_config_name", "model_name", "seed")


def find_resumable_run(run_root, resume=True):
    """The run directory to continue, or None to start a fresh one.

    `resume` is normally the run directory to continue -- what `eval_tasks.sh --resume` passes,
    and the only form that names one run unambiguously. `true` instead picks the most recent
    interrupted run under `run_root`, which is right for a single task evaluated one job at a
    time but ambiguous the moment two evals share a root: each rewrites its state file every
    episode, so whichever ran last is "most recent" however long ago yours stopped.

    A named directory must live under `run_root`, i.e. must belong to the task/policy/config/
    checkpoint this process was configured for. It is the check that makes a wrong `--resume`
    fail instead of appending one task's episodes to another task's results -- which is exactly
    what a run directory pinned in a shared config file used to do to every task submitted
    after it (see the `resume` note in policy/pi05/deploy_policy.yml).
    """
    if isinstance(resume, str) and as_bool_or_none(resume) is None:
        run_dir = Path(resume).expanduser()
        if not (run_dir / RESUME_STATE).exists():
            raise FileNotFoundError(f"resume: {run_dir} has no {RESUME_STATE}")
        run_root = Path(run_root)
        if run_dir.resolve().parent != run_root.resolve():
            raise ValueError(
                f"resume: {run_dir} does not belong to this eval.\n"
                f"  it lives under : {run_dir.resolve().parent}\n"
                f"  this run wants : {run_root.resolve()}\n"
                "Resuming it would append this task's episodes to another run's results. "
                "Check the directory, or the task/config/checkpoint this job was launched with."
            )
        return run_dir
    if not as_bool(resume, False):
        return None
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

    # One scene for the whole run (see Base_Task._init_task_env_): with `env_seed` set, the
    # background texture, light colors, table height and head-camera jitter come from it instead
    # of the episode's own seed, so every episode looks the same while object poses keep varying.
    #
    # It is a task-config key and deliberately has no override here. That is what makes it
    # shared: `script/collect_data.py` reads the same key out of the same file, so evaluating
    # under the task config a policy's demos were collected with puts it in the environment
    # those demos were recorded in.
    env_seed = args.get("env_seed")
    if isinstance(env_seed, str):
        env_seed = None if env_seed.strip().lower() in ("", "none", "null") else int(env_seed)
    args["env_seed"] = None if env_seed is None else int(env_seed)

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
    # the seed sequence all live in there and only mean anything together. `resume: <run dir>`
    # continues that one instead, for when "newest" is ambiguous. With neither (or
    # `resume: false`) this is an ordinary new run.
    run_root = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}")
    resume_dir = find_resumable_run(run_root, usr_args.get("resume", False))
    save_dir = resume_dir if resume_dir is not None else run_root / current_time
    save_dir.mkdir(parents=True, exist_ok=True)
    args["eval_save_dir"] = str(save_dir)

    # Profiling (`profile:` in deploy_policy.yml, or `--profile true` through eval.sh's
    # pass-through overrides). Off by default and off means *not installed* -- nothing is
    # wrapped and the run is byte-identical to one that never heard of the profiler. Installed
    # here because the hooks are class patches that must be in place before the env, the policy
    # and the critic are built, and because the report needs `save_dir` to exist.
    profile_mode = eval_profiler.parse_flag(usr_args.get("profile"))
    profiler = eval_profiler.install(profile_mode) if profile_mode != "off" else None

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

    # Snapshot the deploy config (and the adaptation config it includes) next to the results, with
    # the CLI overrides written in, so a run's settings stay readable -- and accurate -- after
    # the ymls are edited.
    for config_path in (usr_args.get("_config_path"), usr_args.get("adaptation_config_path")):
        expanded_path = Path(config_path).expanduser() if config_path else None
        if expanded_path is not None and expanded_path.is_file():
            snapshot_config(expanded_path, usr_args, save_dir)

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
    print("\033[95mEnv Seed:\033[0m " +
          (f'{args["env_seed"]} (scene fixed for the whole run)' if args["env_seed"] is not None
           else "None (scene redrawn per episode)"))
    # Which texture pool this eval draws from, resolved by the same function the env uses so the
    # banner cannot claim one thing and the scene do another. `env_seed` pins *which* texture is
    # drawn; the pool is `background_texture_pool`'s business alone.
    if args["domain_randomization"]["random_background"]:
        configured = args["domain_randomization"].get("background_texture_pool")
        pool = resolve_background_texture_pool(configured, eval_mode=True)
        print(f" - Texture Pool: {pool}/" + (" (explicit -- collection draws from the same one)"
                                             if configured else " (RoboTwin's held-out split)"))
        # The one combination that surprises: a pinned scene whose background still differs from
        # the one its demonstrations were collected on, because the split is still in force.
        if configured is None and args["env_seed"] is not None:
            print("   note: env_seed fixes the texture *within* a pool, and collection draws "
                  "from `seen/`, so this eval's background differs from its demos'. Set "
                  "domain_randomization.background_texture_pool: seen to match them.")

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    # `setup_demo` / `play_once` / `check_success` are defined by the task, not by `Base_Task`,
    # so the profiler can only wrap them once the concrete class exists. No-op when off.
    eval_profiler.install_env(TASK_ENV)
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
    # ... and again every N TD updates within an episode, so a long episode's training is not
    # all riding on reaching the episode boundary. 0 disables the mid-episode writes.
    args["critic_save_every_updates"] = usr_args.get("critic_save_every_updates", 200)
    # Off leaves the sparse terminal reward, which is what the critic is then trained on.
    args["use_step_reward"] = usr_args["use_step_reward"]
    # Post-failure adaptation is configured beside the critic, in the included
    # `adaptation_config_path`, while `eval_policy` receives the task config (`args`). Forward
    # just the lifecycle knobs it consumes; the critic's own keys already travel through
    # deploy_policy.get_model.
    for key in (
        "post_failure_intervention", "adaptation_adapter",
        "rewind_control_steps", "retry_rewind_control_steps", "intervention_start_episode",
        "intervention_chunk_offsets",
        "q_reduction", "q_source", "intervention_replay_fraction", "restore_atol",
        "inversion_num_steps", "inversion_fixed_point_iterations",
        "inversion_batch_size", "inversion_mse_threshold",
        "inversion_audit_fraction", "inversion_audit_seed",
    ):
        if key in usr_args:
            args[key] = usr_args[key]
    # Fixed length of the `wrench.*` trace a control step sees, and the cap on the env's log
    # (`_base_task._init_task_env_`). The env commits one row per primitive step, so a chunk
    # drains exactly the policy's steps-per-call and `pi0_step` is the right width -- the config
    # only has to say so when it wants a different one. It is an architecture key in all but
    # name: the critic's obs shape is fixed when it is built, so a checkpoint warm-started here
    # -- or pretrained on a rollout dataset -- has to have been made with the same value.
    args["wrench_trace_len"] = int(usr_args.get("wrench_trace_len") or usr_args.get("pi0_step", 10))
    # ===== Periodic held-out evaluation =====
    # `eval_interval` training episodes apart, re-run a fixed set of `eval_episodes` episodes
    # with the policy frozen and score it; that score picks the best critic checkpoint. 0 = off.
    args["eval_interval"] = max(0, int(usr_args.get("eval_interval") or 0))
    args["eval_episodes"] = max(1, int(usr_args.get("eval_episodes") or 10))
    # Where the held-out seed search starts -- an actual seed, not a `seed`-style block index.
    # Default: the run's own block offset by `HOLDOUT_SEED_OFFSET`, i.e. `100000 * (1 + seed) +
    # 1000`. The offset is the whole guard: the run walks its own seeds upward from `st_seed`
    # (one per candidate, accepted or rejected), so the held-out episodes are held out exactly
    # as long as it does not walk 1000 seeds. `eval_policy` checks that as it goes rather than
    # guessing here, since how many seeds a run consumes depends on the expert's reject rate.
    #
    # Staying inside the run's own block rather than jumping to the next one is a readability
    # choice, not a distributional one: nothing about a seed's magnitude biases the scene. With
    # `env_seed` set the scene is pinned for the whole run and every episode renders identically
    # whatever its seed is; with it unset the scene is drawn from the episode's own seed, so it
    # already varies episode to episode and one seed is as good as another.
    eval_seed = usr_args.get("eval_seed")
    args["eval_seed"] = (100000 * (1 + int(usr_args["seed"])) + HOLDOUT_SEED_OFFSET
                         if eval_seed is None else int(eval_seed))


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

    # The profiled region is the seed loop and nothing else: model loading and the JAX
    # compilation it triggers are one-off startup, and folding them into the per-control-step
    # numbers would make a long run look like a slow one.
    try:
        with eval_profiler.cprofile_to(save_dir / "_profile_cprofile",
                                       enabled=profile_mode in ("cprofile", "both")):
            st_seed, suc_num, episode_results = eval_policy(
                task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=test_num,
                video_size=video_size,
                instruction_type=instruction_type,
                wandb_run=wandb_run,
                resume_state=resume_state)
    finally:
        # Also on the way out of a crash or a Ctrl-C -- which is the run you most want the
        # numbers for.
        if profiler is not None:
            print(profiler.dump(save_dir, extra=[
                f"task {task_name}, config {task_config}, ckpt {ckpt_setting}",
                f"written to {save_dir / '_profile.txt'} (and _profile.json)",
            ]))
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

    # The online-trained critic is already persisted alongside the eval results: `eval_policy`
    # checkpoints it after every episode, and reaching here means the last episode completed, so
    # the file on disk is this run's final state. Re-pickling it would write identical bytes --
    # only report where it is. A frozen critic has nothing to persist in the first place; it is a
    # byte-for-byte copy of `critic_ckpt`, which the snapshotted config already names.
    online_critic = getattr(model, "online_critic", None)
    if online_critic is not None and usr_args.get("save_critic", False):
        if _trains_online_critic(model):
            print(f"saved online critic to {save_dir / CRITIC_CKPT} "
                  f"({online_critic.num_updates} updates)")
            # The other file: the critic as of the run's best episode, which is generally not
            # its last. Read the bar back off the state file rather than threading it out of
            # `eval_policy`, which is called once per checkpoint setting.
            state_path = save_dir / RESUME_STATE  # read directly: `load_resume_state` also trims the csv
            final_state = json.loads(state_path.read_text()) if state_path.exists() else {}
            best_score = final_state.get("best_score", final_state.get("best_success_rate_ma"))
            criterion = final_state.get("best_score_criterion") or "success rate"
            if (save_dir / CRITIC_CKPT_BEST).exists():
                print(f"saved best online critic to {save_dir / CRITIC_CKPT_BEST}"
                      + (f" (episode {final_state.get('best_score_episode')}, "
                         f"{criterion} {best_score:.3f})" if best_score is not None
                         else " (the best-checkpoint criterion never produced a score: same as "
                              "the latest)"))
        else:
            print("train_critic_online is off -- not saving the critic (unchanged from "
                  f"{usr_args.get('critic_ckpt')})")

    print(f"Data has been saved to {file_path}")
    if wandb_run is not None:
        wandb_run.finish()
    # return task_reward


# ===== Shared rollout pieces =====
# The main seed loop and the periodic held-out evaluation run the same two things -- the expert
# feasibility gate and a policy rollout -- and have to keep running the *same* two, or the
# held-out score would not be measuring the number the run reports for itself. They live here
# rather than inline so there is only one of each.


def frozen_for_eval(model):
    """Context in which rollouts measure the policy without changing it.

    pi05 implements this (`PI0.frozen_for_eval`): no transition is stashed, the DSRL warmup
    budget and its RNG stay put, the debug recorders' pending values are cleared on the way out,
    and -- deliberately -- the guidance scale is left exactly where the ramp has it, so what is
    scored is the policy as it behaves right now. A policy that does not know the idea has
    nothing to freeze: it has no critic, so its rollouts change nothing anyway.
    """
    freeze = getattr(model, "frozen_for_eval", None)
    return freeze() if callable(freeze) else contextlib.nullcontext()


def run_expert_check(TASK_ENV, args, seed, now_ep_num):
    """Run the scripted expert once on `seed`; its `info` if it solved the task, else None.

    The per-seed feasibility gate both loops apply, so an episode a policy is scored on is one
    the task is known to be solvable from -- and the same gate on both sides is what makes a
    held-out success rate comparable with the run's own. Rendering is off for the duration (the
    expert's trajectory is not what the video is of) and restored however this returns.

    An unstable spawn (`UnStableError`) is a rejected seed like any other and stays quiet; any
    other exception is a bug in the task or the planner, so it is printed before the seed is
    dropped.
    """
    render_freq = args["render_freq"]
    args["render_freq"] = 0
    try:
        with eval_profiler.PROFILER.phase("expert_check"):
            TASK_ENV.setup_demo(now_ep_num=now_ep_num, seed=seed, is_test=True, **args)
            episode_info = TASK_ENV.play_once()
            solved = TASK_ENV.plan_success and TASK_ENV.check_success()
            TASK_ENV.close_env()
            return episode_info if solved else None
    except UnStableError:
        TASK_ENV.close_env()
        return None
    except Exception as e:
        print(" -------------")
        print("Error: ", e)
        print(traceback.format_exc())
        print(" -------------")
        TASK_ENV.close_env()
        print("error occurs !")
        return None
    finally:
        args["render_freq"] = render_freq


def set_episode_instruction(TASK_ENV, task_name, episode_info, instruction_type, test_num):
    """Draw this episode's language instruction from the expert run's own `info` and set it."""
    results = generate_episode_descriptions(task_name, [episode_info["info"]], test_num)
    instruction = np.random.choice(results[0][instruction_type])
    TASK_ENV.set_instruction(instruction=instruction)
    return instruction


def rollout_episode(TASK_ENV, model, eval_func, reset_func, use_step_reward,
                    on_observation=None, on_step=None):
    """One policy rollout, to the first success or `step_lim`.

    Returns `(success, steps, total_reward)`. The two hooks are where everything a *training*
    episode does on top of the rollout lives -- the debug visualisation before the chunk is
    drawn, and the recorders plus the critic's `commit`/`train_step` after it -- so the held-out
    evaluation can run the identical loop with neither.

    `control_step_reward` is called exactly once per control step whether or not anyone wants
    the number: the task's `step_reward` is a delta against its own previous call, so skipping
    it in the held-out rollouts would leave the next training episode paying an accumulated one.
    """
    reset_func(model)
    succ = False
    prev_success = False
    episode_reward = 0.0
    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        # The step this observation was taken at, held because `take_action_cnt` advances
        # during the chunk -- the Q recorder logs against the observation's own index.
        step = TASK_ENV.take_action_cnt
        observation = TASK_ENV.get_obs()
        if on_observation is not None:
            on_observation(step, observation)
        # One chunk sampled and `pi0_step` of it executed -- the unit the profile's per-step
        # budget is over.
        eval_profiler.PROFILER.count("control_steps")
        eval_func(TASK_ENV, model, observation)
        success_now = bool(TASK_ENV.eval_success)
        reward = control_step_reward(TASK_ENV, success_now, prev_success, use_step_reward)
        episode_reward += reward
        if on_step is not None:
            on_step(step, observation, reward, success_now)
        prev_success = success_now
        if success_now:
            succ = True
            break
    return succ, TASK_ENV.take_action_cnt, episode_reward


# ===== Periodic held-out evaluation =====
# Every `eval_interval` training episodes, re-run a *fixed* set of episodes with the critic
# frozen and score it. That score -- not the training episodes' moving average -- is what
# decides which critic `online_value_critic_best.pkl` holds.
#
# Fixed is the whole point, and it is fixed twice over. The seeds are the first `eval_episodes`
# the expert can solve counting up from `eval_seed`, which is deterministic in `eval_seed` and
# therefore identical at every evaluation and across runs; and the `random` module is reseeded
# from `eval_seed` for the duration, so each evaluation draws the same language instructions in
# the same order (RoboTwin draws instructions from `random`, which nothing else seeds). Two
# evaluations of the same critic therefore differ only by the sampler's own noise, and two
# evaluations of different critics differ by the critic.
#
# The seed *search* is not cheap -- every rejected seed is a full expert rollout -- so it runs
# once and the answer is carried in `resume_state.json`.


def find_holdout_seeds(TASK_ENV, args, start_seed, num_episodes, info_cache):
    """The held-out seed set: the first `num_episodes` seeds >= `start_seed` the expert solves.

    Fills `info_cache` with each accepted seed's expert `info` on the way past. That is not an
    optimisation detail: the episode's language instruction is built from those placeholders, so
    without the cache every evaluation would have to re-run the expert pass in every held-out
    scene just to name the objects. The scene is a function of the seed, so one pass is enough
    for the whole run -- and a resumed run, whose cache starts empty, refills it the same way.
    """
    seeds = []
    seed = start_seed
    print(f"\033[96m[holdout]\033[0m searching for {num_episodes} expert-feasible seed(s) "
          f"from {start_seed} (once per run; the result is carried in {RESUME_STATE})")
    while len(seeds) < num_episodes:
        info = run_expert_check(TASK_ENV, args, seed, now_ep_num=len(seeds))
        if info is not None:
            info_cache[seed] = info
            seeds.append(seed)
        seed += 1
    print(f"\033[96m[holdout]\033[0m seed set: {seeds}")
    return seeds


def run_holdout_eval(TASK_ENV, args, model, eval_func, reset_func, seeds, info_cache,
                     instruction_type, use_step_reward, test_num, eval_seed):
    """Score the current policy+critic on the fixed held-out seeds, changing nothing.

    Returns `(success_rate, successes, reward_mean, steps_mean)`.

    Three things are saved and restored around it, because an evaluation must not be visible to
    the run that ran it: the policy's own collection/RNG state (`frozen_for_eval`), the global
    numpy RNG (`setup_demo` reseeds it per episode anyway, but the instruction draw reads it) and
    the `random` module's state, which is reseeded from `eval_seed` so the instructions are the
    same at every evaluation. `TASK_ENV.suc` / `test_num` are the run's own counters and are
    deliberately not touched -- a held-out episode is not one of the `test_num` episodes the run
    was asked for.

    Video and the debug recorders are off: `eval_video_save_dir` is dropped from the env args
    (the env writes head-camera frames into the run's ffmpeg pipe whenever that path is set, and
    there is no pipe here), and `rollout_episode` is called with no hooks.
    """
    holdout_args = {k: v for k, v in args.items() if k != "eval_video_save_dir"}
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    random.seed(eval_seed)
    successes = 0
    rewards, steps = [], []
    try:
        with eval_profiler.PROFILER.phase("holdout"), frozen_for_eval(model):
            for idx, seed in enumerate(seeds):
                # The instruction is built from the expert run's `info` placeholders, which is
                # why the gate's `info` was cached: the scene is a function of the seed, so the
                # expert pass that accepted this seed answers for every later evaluation of it.
                # A resumed run has no cache and pays for one pass per seed, once.
                if seed not in info_cache:
                    info_cache[seed] = run_expert_check(TASK_ENV, args, seed, now_ep_num=idx)
                    if info_cache[seed] is None:
                        # The gate accepted this seed once and it is deterministic in the seed,
                        # so this means the task or the config changed under a resumed run --
                        # in which case the held-out set is not the one the earlier scores were
                        # over and the comparison is meaningless. Say so rather than scoring on.
                        raise RuntimeError(
                            f"held-out seed {seed} is no longer expert-feasible: the task or the "
                            f"task config has changed since this run's seed set was chosen, so "
                            f"its held-out scores are not comparable. Start a fresh run "
                            f"(resume: false), or change eval_seed."
                        )
                TASK_ENV.setup_demo(now_ep_num=idx, seed=seed, is_test=True, **holdout_args)
                set_episode_instruction(TASK_ENV, args["task_name"], info_cache[seed],
                                        instruction_type, test_num)
                succ, episode_steps, episode_reward = rollout_episode(
                    TASK_ENV, model, eval_func, reset_func, use_step_reward)
                successes += int(succ)
                rewards.append(episode_reward)
                steps.append(episode_steps)
                TASK_ENV.close_env()
                print(f"\033[96m[holdout]\033[0m seed {seed}: "
                      + ("\033[92msuccess\033[0m" if succ else "\033[91mfail\033[0m")
                      + f" ({episode_steps} steps, reward {episode_reward:.3f}) "
                      f"-- {successes}/{idx + 1}")
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
    return (successes / len(seeds), successes, _window_mean(rewards), _window_mean(steps))


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
    # The debug recorder plots the same per-physics-step trace, but it reads *before* the
    # control step and so cannot drain the critic's log; the env keeps it a second copy of the
    # same samples instead (`_base_task._log_step_wrench`). Only worth the memory when there is
    # a recorder to consume it.
    args["record_debug_wrench"] = bool(args["record_step_wrench"] and args.get("debug", False))

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
    # Checkpoint the critic after every episode rather than once at the end, so a run killed
    # part-way (SLURM time limit, node failure) still leaves the latest critic on disk and the
    # next run can pick it up via `critic_ckpt`. Only a *learning* critic is worth writing: a
    # frozen one is a byte-for-byte copy of the checkpoint it was loaded from, which the
    # snapshotted config already names.
    save_critic = train_critic and bool(args.get("save_critic", False))
    critic_ckpt_announced = False
    # The best critic of the run, tracked alongside the latest (see CRITIC_CKPT_BEST).
    # `best_score` is None until the criterion in force has produced its first number; while it
    # is, the best file simply tracks the latest, so a run killed early still leaves both files
    # valid. Which criterion that is, is `eval_interval`'s business -- see below.
    best_score = None
    best_score_episode = None
    best_criterion = None
    # Second, finer checkpoint cadence: write the latest critic every N TD updates as well as
    # after every episode (0 turns it off). Only the *latest* file moves -- the best-checkpoint
    # criterion is an episode-level number, so there is no bar to rank a mid-episode critic
    # against.
    save_every_updates = int(args.get("critic_save_every_updates", 200) or 0) if save_critic else 0
    periodic_ckpt_announced = False
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

    # ===== Periodic held-out evaluation (see run_holdout_eval) =====
    # Every `eval_interval` training episodes, re-run a fixed set of `eval_episodes` episodes
    # drawn from `eval_seed` with the policy frozen -- no transitions collected, no TD updates --
    # and make that success rate the criterion for `online_value_critic_best.pkl`. 0 turns it off
    # and falls back to `success_rate_ma` over the training episodes, which is what this used
    # before: cheaper, but it moves with the seeds as much as with the critic, since the training
    # episodes are a different (and unrepeatable) set every time.
    eval_interval = max(0, int(args.get("eval_interval") or 0))
    eval_episodes = max(1, int(args.get("eval_episodes") or 10))
    # Where the held-out seed search starts: an actual seed, defaulting to `st_seed +
    # HOLDOUT_SEED_OFFSET` (see `main`). It is above the run's own first seed and the run walks
    # upward, so the two sets are disjoint only while the run stays below it -- the top of the
    # seed loop checks that as it goes, since the reject rate decides how fast the run gets there.
    eval_seed = args.get("eval_seed")
    eval_seed = st_seed + HOLDOUT_SEED_OFFSET if eval_seed is None else int(eval_seed)
    holdout_overlap_warned = False
    # The seed set itself, and the expert `info` each one's instruction is built from. Found once
    # (every rejected seed costs a full expert rollout) and carried in `resume_state.json`.
    holdout_seeds = None
    holdout_info_cache = {}
    print(f"\033[95mStep reward (shaped progress):\033[0m "
          + ("ON" if use_step_reward else "OFF (sparse success reward only)"))
    print(f"\033[95mHeld-out evaluation:\033[0m "
          + (f"every {eval_interval} episode(s), {eval_episodes} episode(s) from seed "
             f"{eval_seed} ({eval_seed - st_seed:+d} from this run's first seed), policy frozen"
             + (f" -- and the criterion for {CRITIC_CKPT_BEST}" if save_critic
                else " (no critic to checkpoint: scored and logged only)")
             if eval_interval else
             "OFF (eval_interval is 0"
             + (f"; {CRITIC_CKPT_BEST} falls back to the MA{ma_window} success rate over the "
                f"training episodes)" if save_critic else ")")))

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
    # Demo retrieval diagnostics: what the top-K similarity actually matched, per control step.
    # Needs the policy to be retrieving at all (the two proposal `data_type` flags), and the flag
    # has to be set before the first bank is built -- the thumbnails it draws are the only part
    # of a demo episode that would otherwise be dropped after encoding.
    retriever = getattr(model, "demo_retriever", None)
    retrieval_recorder = (DemoRetrievalRecorder(debug_save_dir, frame_log)
                          if debug and retriever is not None else None)
    if retrieval_recorder is not None:
        model.record_demo_retrieval = True
    # Best-of-N selection diagnostics. Unlike the Q recorder this costs nothing (the scores are
    # a by-product of the selection the sampler already made) and needs no `debug`, so it is on
    # whenever the policy is actually choosing between candidates.
    best_of_n = int(getattr(model, "best_of_n", 1) or 1)
    best_of_n_recorder = BestOfNRecorder() if best_of_n > 1 else None

    # The included adaptation config chooses the external lifecycle adapter dynamically. Most
    # implementation remains in multisensory-steering; this driver supplies only environment
    # callbacks and the existing debug output location.
    intervention = None
    intervention_recorder = None
    intervention_enabled = as_bool(args.get("post_failure_intervention"), False)
    if intervention_enabled and not _uses_online_critic(model):
        # Same class of mistake as `train_online: false` below, and the more expensive one to
        # find: the run does not fail, it just quietly becomes an unguided baseline, and the
        # only sign is a banner three lines into a log that then runs for hours. The rewind
        # point is chosen from the critic's Q trace, so no critic means no interventions.
        # Read off the model, which is what actually decided, rather than off `args`, which
        # only carries the lifecycle keys forwarded above.
        raise ValueError(
            "post_failure_intervention is on but no critic was built, so no episode can be "
            "recovered. deploy_policy.get_model builds one only for `critic_type: dsrl`, "
            f"`guidance_scale != 0` or `best_of_n > 1`; this run has "
            f"critic_type={getattr(model, 'critic_type', 'qmfm')!r}, "
            f"guidance_scale={getattr(model, 'guidance_scale_target', 0.0)!r}, "
            f"best_of_n={best_of_n}. "
            "Set one of them in deploy_policy.yml, or turn off post_failure_intervention.")
    if intervention_enabled and _uses_online_critic(model):
        if not train_critic:
            raise ValueError("post_failure_intervention requires train_online: true")
        adapter_spec = str(args.get(
            "adaptation_adapter",
            "multisensory_steering.interventions.robotwin:build_adapter",
        ))
        module_name, separator, factory_name = adapter_spec.partition(":")
        if not separator:
            raise ValueError("adaptation_adapter must have the form 'module:function'")
        factory = getattr(importlib.import_module(module_name), factory_name)
        intervention_recorder = (
            ScriptedInterventionRecorder(debug_save_dir) if debug else None
        )
        intervention = factory(
            args,
            model,
            output_dir=save_dir,
            reward_fn=control_step_reward,
            obs_modalities_fn=obs_modalities,
            debug_recorder=intervention_recorder,
        )
        start_episode = int(args.get("intervention_start_episode", 0) or 0)
        print("\033[95mPost-failure expert intervention:\033[0m ON "
              f"(adapter {adapter_spec}, debug={'ON' if debug else 'OFF'}, "
              + (f"from episode {start_episode} -- the first {start_episode} are autonomous "
                 "warm-up for the critic)" if start_episode > 0 else "from episode 0)"))
    elif intervention_enabled:
        # The shared adaptation config is also included by the plain pi0.5 baseline.  With
        # guidance_scale=0 and best_of_n=1 there is deliberately no critic, hence no Q-drop
        # schedule or replay to intervene into; keep that established baseline runnable.
        print("\033[95mPost-failure expert intervention:\033[0m OFF "
              "(plain pi0.5 baseline has no critic)")
    if best_of_n_recorder is not None:
        print(f"\033[96m[critic]\033[0m best-of-{best_of_n} sampling ON: {best_of_n} candidate "
              f"chunks per control step, highest ensemble-mean Q executed")
    if debug:
        print(f"\033[93m[debug] depth/point-cloud visualization ON "
              f"(interactive={debug_show}, saving to {debug_save_dir})\033[0m")
        print(f"\033[93m[debug] end-effector wrench logging ON, per gripper link, "
              + ("every physics step" if args["record_debug_wrench"] else
                 "one sample per policy call (data_type.wrench is off)")
              + f" (saving to {debug_save_dir}/episode<N>/)\033[0m")
        print(f"\033[93m[debug] critic Q logging "
              + (f"ON (saving to {debug_save_dir}/episode<N>/)" if q_recorder is not None
                 else "OFF (no critic: guidance_scale is 0, best_of_n is 1 and critic_type is "
                      "not dsrl)") + "\033[0m")
        print(f"\033[93m[debug] demo retrieval logging "
              + (f"ON, top-{max(retriever.top_k, retriever.debug_top_k)} "
                 f"(saving to {debug_save_dir}/episode<N>/)"
                 if retrieval_recorder is not None
                 else "OFF (the critic's encoder_modalities names no proposals)") + "\033[0m")

    # Where each finished episode is committed (see the notes above `append_episode_row`).
    csv_path = save_dir / EPISODE_CSV
    critic_path = save_dir / CRITIC_CKPT
    best_critic_path = save_dir / CRITIC_CKPT_BEST

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
        # The best-so-far bar, so a resumed run does not overwrite a better checkpoint from
        # before the break with a worse one after it. `best_score` is the criterion-agnostic
        # name; `best_success_rate_ma` is what state files written before the held-out
        # evaluation existed called the same field (a moving-average success rate, which is
        # still what the bar means when `eval_interval` is 0). Absent in state files written
        # before 2026-08-29 (and in one reconstructed by `resume_state_from_log.py`, which has
        # no critic to speak of), in which case the bar starts over -- the moving averages in
        # the csv would give the right number, but the checkpoint that earned it is already gone.
        best_score = resume_state.get("best_score", resume_state.get("best_success_rate_ma"))
        best_score_episode = resume_state.get("best_score_episode",
                                              resume_state.get("best_success_rate_ma_episode"))
        best_criterion = resume_state.get("best_score_criterion")
        if best_score is not None:
            print(f"\033[93m[resume] best critic so far: {best_score:.3f} "
                  f"({best_criterion or f'MA{ma_window} success rate'}) "
                  f"at episode {best_score_episode}\033[0m")
        # The held-out seed set, so a resumed run scores on the same episodes rather than paying
        # for the search again -- and, more to the point, rather than scoring on a different set
        # than the numbers already in `_holdout_results.csv`.
        if resume_state.get("holdout_seeds"):
            holdout_seeds = [int(v) for v in resume_state["holdout_seeds"]]
            print(f"\033[93m[resume] held-out seed set: {holdout_seeds}\033[0m")

        # The csv is the record of the episodes themselves; reload it so the moving averages
        # continue over the interruption rather than restarting from an empty window.
        past = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=EPISODE_COLUMNS)
        for col in EPISODE_COLUMNS:
            episode_results[col] = past[col].tolist()
        success_window.extend(float(v) for v in episode_results["success"][-ma_window:])
        reward_window.extend(float(v) for v in episode_results["reward"][-ma_window:])
        print(f"\033[93m[resume] {len(past)} episode(s) reloaded, resuming at seed {now_seed} "
              f"({succ_seed}/{test_num} done)\033[0m")

    # What a *training* episode does on top of the bare rollout, as the two hooks
    # `rollout_episode` calls. The held-out evaluation runs the same rollout with neither: it
    # writes no debug output, feeds nothing to the recorders, and above all collects no
    # transition and runs no TD update.
    def on_observation(step, observation):
        if intervention is not None:
            intervention.pre_action_snapshot(TASK_ENV)
        if debug:
            visualize_debug_obs(
                observation,
                step_idx=step,
                save_dir=(debug_save_dir / f"episode{TASK_ENV.test_num}"
                          if debug_save_dir else None),
                show=debug_show,
                task_env=TASK_ENV,
                wrench_recorder=wrench_recorder,
            )

    def train_for_collected_transition(episode_step):
        nonlocal chunk_count, last_info, periodic_ckpt_announced
        online_critic = getattr(model, "online_critic", None) if train_critic else None
        if online_critic is None:
            return
        chunk_count += 1
        if chunk_count % train_freq != 0:
            return
        info = online_critic.train_step()
        if info is None:
            return
        last_info = info
        log_critic_update(wandb_run, online_critic, model, info, chunk_count,
                          TASK_ENV.test_num, episode_step)
        updates = int(online_critic.num_updates)
        if save_every_updates and updates % save_every_updates == 0:
            save_critic_atomically(online_critic, critic_path)
            if not periodic_ckpt_announced:
                print(f"\033[96m[critic]\033[0m also checkpointing every "
                      f"{save_every_updates} updates to {critic_path}")
                periodic_ckpt_announced = True

    def on_step(step, observation, reward, success_now):
        # The chunk's Q only exists once the policy has sampled it, so unlike the wrench this is
        # logged after the control step -- but against `step`, the count the observation it was
        # drawn from was taken at, so it lines up with that frame.
        if q_recorder is not None:
            q_recorder.record(model, observation, step, reward)
        # Retrieval happens *before* the chunk is drawn, so it belongs to `observation`; it is
        # read here only because the policy runs it inside `get_action`.
        if retrieval_recorder is not None:
            retrieval_recorder.record(model, observation, step)
        if best_of_n_recorder is not None:
            best_of_n_recorder.record(model)

        # Read the stashed action before terminal commit clears it. The adapter's parameter tree
        # was retained at episode start, so mid-episode Polyak/online updates do not move this
        # diagnostic Q sequence.
        if intervention is not None:
            intervention.post_sampling_q(reward)

        # Online critic: close the chunk transition (SARSA), then run a TD update. Skipped for a
        # frozen critic -- it guides, but its parameters and buffer stay untouched.
        online_critic = getattr(model, "online_critic", None) if train_critic else None
        if online_critic is None:
            return
        done = success_now or (TASK_ENV.take_action_cnt >= TASK_ENV.step_lim)
        online_critic.commit(reward, done)
        train_for_collected_transition(TASK_ENV.take_action_cnt)

    # Everything before this -- loading the checkpoint, building the critic, the sampler's first
    # JAX compilation -- is one-off startup, and folding it into the per-control-step budget
    # would make a long run read as a slow one.
    eval_profiler.PROFILER.mark_startup_done()

    while succ_seed < test_num:
        # The run walks its seeds upward and the held-out set sits `HOLDOUT_SEED_OFFSET` above
        # the start, so a long run (or an unusually high expert reject rate) can eventually
        # reach it -- at which point the "held-out" episodes are also episodes the critic
        # trained on, and the scores after this point are no longer a clean measurement. Said
        # once, and not fatal: the run is still valid, its best-checkpoint criterion is just no
        # longer held out. Raise `eval_seed` (or lower `test_num`) next time.
        if eval_interval and not holdout_overlap_warned and now_seed >= eval_seed:
            holdout_overlap_warned = True
            print(f"\033[93m[holdout] the run has reached seed {now_seed}, at or past the "
                  f"held-out set's start ({eval_seed}) -- from here the held-out episodes are "
                  f"no longer held out. Raise eval_seed next time.\033[0m")

        # The per-seed feasibility gate (`run_expert_check` -- rendering off, the seed dropped
        # on failure). `episode_info` is the expert's own placeholder dict, which the episode's
        # language instruction is built from.
        episode_info = run_expert_check(TASK_ENV, args, now_seed, now_ep_num=now_id) if expert_check else None
        if expert_check and episode_info is None:
            now_seed += 1
            continue
        succ_seed += 1
        suc_test_seed_list.append(now_seed)

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        set_episode_instruction(TASK_ENV, args["task_name"], episode_info, instruction_type, test_num)

        if intervention is not None:
            intervention.begin_episode(TASK_ENV, TASK_ENV.test_num)

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

        with eval_profiler.PROFILER.phase("rollout"):
            succ, episode_steps, episode_reward = rollout_episode(
                TASK_ENV, model, eval_func, reset_func, use_step_reward,
                on_observation=on_observation, on_step=on_step)
        eval_profiler.PROFILER.count("episodes")
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
        if retrieval_recorder is not None:
            retrieval_recorder.flush(TASK_ENV.test_num)
        if frame_log is not None:
            frame_log.flush()  # last: both GIFs above are drawn from it

        recovery = None
        if not succ and intervention is not None:
            with eval_profiler.PROFILER.phase("expert_intervention"):
                recovery = intervention.recover_failed_episode(
                    TASK_ENV, TASK_ENV.test_num, use_step_reward=use_step_reward
                )
            # Each accepted expert transition gets one ordinary train-frequency opportunity;
            # add_intervention_episode already stored it, so there is no second commit here.
            for _ in range(recovery.inserted_transitions):
                train_for_collected_transition(episode_steps)
            if wandb_run is not None:
                wandb_run.log({
                    "intervention/episode": int(TASK_ENV.test_num),
                    "intervention/expert_success": float(recovery.expert_success),
                    "intervention/accepted_chunks": int(recovery.accepted_chunks),
                    "intervention/rejected_chunks": int(recovery.rejected_chunks),
                    "intervention/inserted_transitions": int(recovery.inserted_transitions),
                    "intervention/corrections": int(recovery.interventions_inserted),
                    "intervention/attempts": len(recovery.attempts),
                    "intervention/restore_exact": float(bool(recovery.restore_exact)),
                    "intervention/restore_max_abs_error": float(
                        recovery.restore_max_abs_error or 0.0
                    ),
                    "intervention/buffer_size": int(recovery.intervention_buffer_size),
                })
            attempted = "/".join(
                f"{r.snapshot_index}:{r.outcome.split(':')[0]}" for r in recovery.attempts
            )
            print(f"\033[95m[intervention]\033[0m {recovery.reason}: "
                  f"rewind={recovery.requested_snapshot}, "
                  f"attempts=[{attempted}], "
                  f"corrections={recovery.interventions_inserted}, "
                  f"accepted/rejected={recovery.accepted_chunks}/{recovery.rejected_chunks}, "
                  f"inserted={recovery.inserted_transitions}")
        if intervention_recorder is not None and recovery is not None:
            intervention_recorder.flush(TASK_ENV.test_num, recovery)

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
            episode_steps,
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
        # `save_critic` already folds in `train_critic`: a frozen critic is a byte-for-byte copy
        # of the checkpoint it was loaded from, so there is nothing to write. Announced only the
        # first time -- it happens every episode, and the `[critic]` line above already reports
        # the update count that identifies which state was written. This is the *latest* file
        # and it is written before the held-out evaluation below, which is what `promote_best_
        # critic` then copies from -- the evaluation changes nothing about the critic, so the
        # bytes are the same either way, but the copy needs the file to exist.
        if save_critic and online_critic is not None:
            with eval_profiler.PROFILER.span("critic.save_checkpoint"):
                save_critic_atomically(online_critic, critic_path)
            if not critic_ckpt_announced:
                print(f"\033[96m[critic]\033[0m checkpointing after every episode to "
                      f"{critic_path} (set `critic_ckpt` to it to resume this run); the best "
                      f"episode's critic is kept alongside it as {best_critic_path.name}")
                critic_ckpt_announced = True

        # ===== Periodic held-out evaluation =====
        # Every `eval_interval` episodes, score the current policy + critic on the fixed
        # `eval_seed` episode set, frozen. This is the only place `best_score` gets a number
        # when the evaluation is on, so between evaluations the best checkpoint simply does not
        # move.
        holdout_score = None
        if eval_interval and TASK_ENV.test_num % eval_interval == 0:
            if holdout_seeds is None:
                holdout_seeds = find_holdout_seeds(
                    TASK_ENV, args, eval_seed, eval_episodes, holdout_info_cache)
            guidance = float(model.scheduled_guidance_scale()) if online_critic is not None else 0.0
            updates = int(online_critic.num_updates) if online_critic is not None else 0
            print(f"\033[96m[holdout]\033[0m episode {TASK_ENV.test_num}: evaluating "
                  f"{len(holdout_seeds)} held-out episode(s) frozen "
                  f"(critic updates={updates}, guidance={guidance:.4g})")
            holdout_score, holdout_successes, holdout_reward, holdout_steps = run_holdout_eval(
                TASK_ENV, args, model, eval_func, reset_func, holdout_seeds, holdout_info_cache,
                instruction_type, use_step_reward, test_num, eval_seed)
            print(f"\033[96m[holdout]\033[0m episode {TASK_ENV.test_num}: "
                  f"\033[95m{holdout_successes}/{len(holdout_seeds)}\033[0m "
                  f"=> \033[95m{round(holdout_score * 100, 1)}%\033[0m "
                  f"(reward {holdout_reward:.3f}, {holdout_steps:.0f} steps mean)")
            append_holdout_row(save_dir / HOLDOUT_CSV, {
                "episode": TASK_ENV.test_num,
                "critic_updates": updates,
                "guidance_scale": guidance,
                "episodes": len(holdout_seeds),
                "successes": holdout_successes,
                "success_rate": holdout_score,
                "reward_mean": holdout_reward,
                "steps_mean": holdout_steps,
            })
            if wandb_run is not None:
                wandb_run.log({
                    "eval/episode": int(TASK_ENV.test_num),
                    "holdout/success_rate": float(holdout_score),
                    "holdout/successes": int(holdout_successes),
                    "holdout/episodes": int(len(holdout_seeds)),
                    "holdout/reward_mean": float(holdout_reward),
                    "holdout/steps_mean": float(holdout_steps),
                    "holdout/critic_updates": updates,
                })

        # ... and keep a second copy of the best episode's critic, on whichever criterion is in
        # force. With `eval_interval` on that is the held-out success rate and it exists only on
        # the episodes an evaluation just ran; with it off it is the training episodes' moving
        # average, which is only meaningful once the window has filled (a mean over one or two
        # episodes makes a single early success read as 1.0, which no honest window can beat, and
        # would freeze "best" at episode 1). Until the criterion has produced anything the best
        # file just follows the latest, so a run killed early leaves both files valid.
        if save_critic and online_critic is not None:
            if eval_interval:
                score, criterion = holdout_score, f"held-out success rate ({eval_episodes} ep)"
            elif len(success_window) >= ma_window:
                score, criterion = success_rate_ma, f"MA{ma_window} success rate"
            else:
                score, criterion = None, None
            if score is None:
                if best_score is None:
                    promote_best_critic(critic_path, best_critic_path)
            elif best_score is None or score >= best_score:
                promote_best_critic(critic_path, best_critic_path)
                best_score, best_score_episode, best_criterion = score, TASK_ENV.test_num, criterion
                print(f"\033[96m[critic]\033[0m new best: {criterion} "
                      f"{best_score:.3f} -> {best_critic_path.name}")
        # Refreshed every episode rather than only at the end: an eval slow enough to be worth
        # profiling is one you are likely to kill, and the report is a few kB.
        if eval_profiler.PROFILER.enabled:
            eval_profiler.PROFILER.dump(save_dir)
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
            # The bar `online_value_critic_best.pkl` currently holds, and what it is a bar on,
            # so a resumed run keeps comparing against it rather than replacing it with its own
            # first episode. `best_success_rate_ma` is the name the same field had before the
            # held-out evaluation existed; it is still written so an older reader keeps working.
            "best_score": best_score,
            "best_score_episode": best_score_episode,
            "best_score_criterion": best_criterion,
            "best_success_rate_ma": best_score,
            "best_success_rate_ma_episode": best_score_episode,
            # The held-out episode set, so a resume scores on the same episodes the numbers
            # already in `_holdout_results.csv` were measured over.
            "holdout_seeds": holdout_seeds,
            "wandb_run_id": wandb_run.id if wandb_run is not None else None,
        })

    return now_seed, TASK_ENV.suc, episode_results


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

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

    overrides = parse_override_pairs(args.overrides) if args.overrides else {}

    # ===== Resuming: the run's own config, not the working tree's =====
    # Every run snapshots its deploy config (and the critic config it includes) into its result
    # directory with the values actually used written in (`snapshot_config`). Resuming reads
    # those back instead of the working-tree ymls, because the ymls have moved on -- they
    # describe the run being set up next, and a resumed run must continue with the settings its
    # episodes, its seed sequence and its critic were produced under. It also means a resume
    # needs no knobs on the command line: `--resume <dir>` is the whole instruction, and a
    # `--critic-ckpt` (or guidance, or reward) flag meant for some *other* submission cannot
    # reach it.
    resume_dir = overrides.get("resume")
    if resume_dir is not None and as_bool_or_none(resume_dir) is None:
        resume_dir = Path(str(resume_dir)).expanduser()
        snapshot = resume_dir / config_path.name
        if not snapshot.is_file():
            raise FileNotFoundError(
                f"resume: {resume_dir} has no {config_path.name} snapshot to continue from. "
                "Runs from before config snapshotting have to be resumed by pointing --config "
                "at a config that matches what they were launched with."
            )
        config_path = snapshot
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # The identity keys are the run's, full stop. eval.sh passes them positionally on every
        # invocation, so they arrive here whether or not the caller meant to set them -- compare
        # rather than apply, and say which one disagrees.
        mismatched = {k: (config.get(k), overrides[k]) for k in IDENTITY_KEYS
                      if k in overrides and overrides[k] != config.get(k)}
        if mismatched:
            detail = "\n".join(f"  {k}: run is {was!r}, command says {now!r}"
                               for k, (was, now) in sorted(mismatched.items()))
            raise ValueError(
                f"resume: {resume_dir} was not launched with these settings.\n{detail}\n"
                "A resumed run continues the experiment in the directory; these keys cannot be "
                "changed. Drop them, or start a new run instead of resuming."
            )
        for key in IDENTITY_KEYS:
            overrides.pop(key, None)

    # Policy-specific hyperparameters may be factored out into a file that ships with the
    # implementation they configure (the adaptation config in multisensory-steering includes
    # both the critic and intervention settings, so it stays in sync with the code that reads
    # them). Merge it in *underneath* the
    # deploy config: precedence is CLI overrides > deploy config > included file.
    #
    # Which file that is has to be resolved against the overrides first, or `--overrides
    # adaptation_config_path .../dsrl.yaml` would swap the recorded path while still merging in the
    # defaults of the file it replaced -- silently mixing two critic families' configs.
    include_path = overrides.get("adaptation_config_path", config.get("adaptation_config_path"))
    if include_path:
        # Same reasoning as the deploy config above: a resumed run takes the critic config it
        # was launched under, which is the copy in its own directory. The recorded path may name
        # a checkout that has since been edited, or is not on this machine at all.
        if config_path.parent != Path(args.config).parent:
            snapshot = config_path.parent / Path(include_path).name
            if snapshot.is_file():
                include_path = str(snapshot)
        with Path(include_path).expanduser().open("r", encoding="utf-8") as f:
            included = yaml.safe_load(f) or {}
        config = {**included, **config, "adaptation_config_path": include_path}

    config.update(overrides)

    # Kept so `main` can copy the deploy config into the eval_result dir. On a resume this is
    # the snapshot itself, so re-snapshotting rewrites the file with the same content.
    config["_config_path"] = str(config_path)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
