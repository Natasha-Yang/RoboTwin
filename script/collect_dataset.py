"""Roll out a trained policy in RoboTwin and save the rollouts as a HuggingFace dataset.

Given a checkpoint (via the policy's ``deploy_policy`` interface), a task config and a
task name, this collects rollout episodes and stores, for every policy inference step,
the observations from the three cameras (head / left wrist / right wrist), the action
chunk executed, the end-effector contact wrench at every primitive step of that chunk, the
reward that chunk earned, and whether the episode ultimately succeeded. Whatever else the
task config's ``data_type`` block enables -- depth, segmentation, point cloud, third-person view, end-effector poses --
is recorded alongside them (see extra_obs_columns), so ``demo_clean_privileged`` yields a
much wider dataset than ``demo_clean``.

Model loading stays policy-specific: we reuse the ``get_model`` / ``eval`` / ``encode_obs``
/ ``reset_model`` interface defined in ``policy/<policy_name>/deploy_policy.py`` (the same
interface ``eval_policy.py`` uses). Only the rollout loop and dataset construction live here.

Each episode is written to its own shard under ``<output_dir>/.../<ckpt>_shards`` as soon as it
finishes and the shards are concatenated at the end, so a run's rows never all sit in RAM and an
interrupted run resumes from ``progress.json`` (``resume: false`` starts over instead).

Usage (mirrors eval_policy.py):
    python script/collect_dataset.py --config policy/pi05/collect_dataset.yml \
        --overrides --task_name <task> --task_config <config> \
        --train_config_name <cfg> --model_name <model> --ckpt_setting <model> \
        --seed <seed> --policy_name pi05
"""
import json
import os
import shutil
import sys

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # for `import eval_policy`

import numpy as np
from pathlib import Path
import traceback

from PIL import Image
import datasets

from envs.utils.create_actor import UnStableError
from envs.utils.obs_modalities import camera_suffix
from envs.utils.wrench import stack_step_wrench
from generate_episode_instructions import *

# Reuse env-setup helpers from the evaluation entrypoint so the two stay in sync.
from eval_policy import (
    class_decorator,
    control_step_reward,
    eval_function_decorator,
    get_embodiment_config,
    parse_args_and_config,
)
from envs import CONFIGS_PATH

import yaml


def wrench_columns(step_wrench, num_steps):
    """Per-arm end-effector contact wrench, one sample per primitive step since the last row.

    `step_wrench` is what `_base_task.pop_step_wrench` logged while the *previous* chunk ran: a
    `{arm: (6,)}` sample per `take_action`, `[Fx, Fy, Fz, Tx, Ty, Tz]` in the world frame. It
    comes from the same `envs/utils/wrench.py` helper the eval driver's debug plots use -- the
    only difference is the rate: `eval_policy.py` samples once per policy call, here every step
    in between is kept. Stacking and NaN padding to `(num_steps, 6)` (i.e. `(pi0_step, 6)`, one
    fixed shape across the dataset) is `stack_step_wrench`, shared with the critic's online view
    of the same modality (`envs/utils/obs_modalities.py`).
    """
    return {f"observation.wrench.{arm}": samples
            for arm, samples in stack_step_wrench(step_wrench, num_steps).items()}


def extra_obs_columns(observation, step_wrench=(), num_steps=0, fixed_pcd=True):
    """Columns for whatever extra data types the task config enabled, plus the contact wrench.

    The `data_type` block of `task_config/*.yml` decides what `_base_task.get_obs` puts in the
    observation (see `envs/_base_task.py::get_obs`): `depth`, `mesh_segmentation` and
    `actor_segmentation` land per camera, `third_view` / `pointcloud` / `endpose` at the top
    level. So `demo_clean` yields nothing here while `demo_clean_privileged` yields all of them.
    `rgb` and `qpos` are already recorded as the image and state columns, and camera intrinsics
    / extrinsics come along whenever depth or a point cloud does, since depth is not unprojectable
    without them and the wrist cameras move every step.

    Driven off what the observation actually contains rather than off the flags, so a data type
    added upstream is picked up without a change here. The wrench is the exception: contacts are
    a scene query rather than part of the observation, so it is sampled during the rollout and
    handed in as `step_wrench` (see wrench_columns).
    """
    cols = wrench_columns(step_wrench, num_steps)
    cameras = observation.get("observation", {})
    wants_geometry = len(observation.get("pointcloud", [])) > 0 or any(
        "depth" in cam_obs for cam_obs in cameras.values())

    for cam_name, cam_obs in cameras.items():
        suffix = camera_suffix(cam_name)
        if "depth" in cam_obs:
            # Millimetres, float64 out of the renderer; float32 keeps sub-micron precision at
            # half the size, and depth is already the heaviest column here (a 320x240 map per
            # camera per row).
            cols[f"observation.depth.{suffix}"] = np.asarray(cam_obs["depth"], dtype=np.float32)
        for level in ("mesh", "actor"):
            key = f"{level}_segmentation"
            if key in cam_obs:
                # Already palette-colored (H, W, 3) uint8, so it stores as a PNG image column
                # like the rgb frames -- flat label regions compress well.
                cols[f"observation.{key}.{suffix}"] = Image.fromarray(
                    np.asarray(cam_obs[key], dtype=np.uint8))
        if wants_geometry:
            for key in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl"):
                if key in cam_obs:
                    cols[f"observation.camera.{suffix}.{key}"] = np.asarray(cam_obs[key], dtype=np.float32)

    if "third_view_rgb" in observation:
        cols["observation.images.third_view"] = Image.fromarray(
            np.asarray(observation["third_view_rgb"], dtype=np.uint8))

    pointcloud = observation.get("pointcloud", [])
    if len(pointcloud) > 0:
        # (N, 6): world-frame xyz + rgb, downsampled to `pcd_down_sample_num` points. That fixed
        # N is what makes it an Array2D column; with downsampling off, N varies per row and it
        # has to be a ragged nested list instead.
        pointcloud = np.asarray(pointcloud, dtype=np.float32)
        cols["observation.pointcloud"] = pointcloud if fixed_pcd else pointcloud.tolist()

    # left/right end-effector pose (xyz + quat) and normalized gripper width.
    for key, value in observation.get("endpose", {}).items():
        value = np.asarray(value, dtype=np.float32)
        cols[f"observation.endpose.{key}"] = value.tolist() if value.ndim else float(value)

    return cols


def build_env_args(usr_args):
    """Replicate eval_policy.main() env/arg construction (without running eval)."""
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["policy_name"] = policy_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
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
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    # No video logging during dataset collection.
    args["eval_video_log"] = False

    TASK_ENV = class_decorator(args["task_name"])
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    return args, TASK_ENV


def collect_rollouts(usr_args, start=None):
    """Roll out episodes, yielding ``(records, state)`` once per collected episode.

    ``records`` are that episode's per-inference-step rows and ``state`` is the loop position
    *after* it (``collected`` / ``now_id`` / ``now_seed``), which the caller persists so an
    interrupted run can resume at exactly the seed it stopped on. Yielding per episode rather
    than returning the whole run is what lets the caller flush each episode to disk: a run's
    rows are far too large to all sit in RAM (see main).

    ``start`` resumes from such a state; None starts from the configured seed.
    """
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    num_episodes = usr_args["num_episodes"]
    expert_check = usr_args.get("expert_check", True)

    get_model = eval_function_decorator(policy_name, "get_model")
    encode_obs = eval_function_decorator(policy_name, "encode_obs")
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    args, TASK_ENV = build_env_args(usr_args)
    args["eval_mode"] = True
    # `data_type.wrench` has the env log the end-effector contact wrench after every primitive
    # step, so each row carries the whole (pi0_step, 6) trace of the chunk it executed (see
    # wrench_columns). It is a task-config switch like the other data types, but it has to be
    # passed explicitly because contacts are a scene query rather than part of the observation.
    args["record_step_wrench"] = bool(args["data_type"].get("wrench", False))
    clear_cache_freq = args["clear_cache_freq"]
    # Point clouds are only a fixed-shape column when the sim downsamples them to a set number
    # of points (see extra_obs_columns).
    fixed_pcd = int(args.get("pcd_down_sample_num", 0)) > 0

    model = get_model(usr_args)
    # Actions actually executed per inference call, i.e. how many wrench samples a full chunk
    # produces. The model's own value wins: it is what deploy_policy slices the chunk with.
    pi0_step = int(getattr(model, "pi0_step", usr_args["pi0_step"]))

    st_seed = 100000 * (1 + usr_args["seed"])
    now_seed = st_seed
    now_id = 0
    collected = 0
    if start is not None:
        now_seed, now_id, collected = start["now_seed"], start["now_id"], start["collected"]

    while collected < num_episodes:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        # Expert feasibility check: only roll out the policy on solvable task instances.
        episode_info = None
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                expert_success = TASK_ENV.plan_success and TASK_ENV.check_success()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                print(" -------------")
                print("Error: ", e)
                print(traceback.format_exc())
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

            if not expert_success:
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        args["render_freq"] = render_freq

        # Set up the actual policy rollout on this (validated) seed.
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        if expert_check and episode_info is not None:
            results = generate_episode_descriptions(args["task_name"], [episode_info["info"]], num_episodes)
            instruction = np.random.choice(results[0][instruction_type])
            TASK_ENV.set_instruction(instruction=instruction)
        instruction = TASK_ENV.get_instruction()

        succ = False
        prev_success = False
        frame_records = []
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            # Actions taken so far, NOT inference calls: `_base_task.take_action` bumps
            # `take_action_cnt` once per call and `deploy_policy.eval` calls it once per action,
            # `pi0_step` times per chunk -- so the indices run 0, 10, 20, ... at pi0_step=10 (as
            # the collected datasets show). Consumers read elapsed time off this column
            # (multisensory_steering's offline trainer discounts by the gap between consecutive
            # rows), so a row counter would understate it by exactly that factor -- and scaling
            # this one by pi0_step would overstate it by the same factor.
            frame_index = TASK_ENV.take_action_cnt
            # Record the first observation seen before this inference call.
            observation = TASK_ENV.get_obs()
            # Drained *before* the chunk runs, so what the row carries is the wrench the
            # PREVIOUS chunk produced -- the steps between the last observation and this one.
            # That is the only wrench that exists at the moment the row's action is chosen,
            # which makes the column a genuine observation rather than an outcome, and makes it
            # identical to what the online critic reads (deploy_policy.py::critic_obs_modalities
            # drains it at the same point). An episode's first row has no previous chunk and
            # falls back to a single sample of the current contact state (pop_step_wrench).
            step_wrench = TASK_ENV.pop_step_wrench()

            actions, initial_obs = eval_func(TASK_ENV, model, observation)

            # Reward for the chunk that just ran, i.e. for this row's own (state, action) --
            # unlike the wrench above, which is an observation of what came before it. Computed
            # by the same helper `eval_policy.py` feeds the online critic's `commit()` with, at
            # the same point in the loop, so a critic pretrained on this column and one trained
            # online during eval see the same reward. It has to be read here rather than
            # reconstructed later: the task's `step_reward` is a delta against its own last
            # call, so it exists only while the episode is running.
            success_now = bool(TASK_ENV.eval_success)
            reward = control_step_reward(TASK_ENV, success_now, prev_success)
            prev_success = success_now

            input_rgb_arr, input_state = initial_obs
            head_rgb, right_rgb, left_rgb = input_rgb_arr

            record = {
                "frame_index": frame_index,
                "reward": reward,
                "observation.images.head": Image.fromarray(np.asarray(head_rgb, dtype=np.uint8)),
                "observation.images.left_wrist": Image.fromarray(np.asarray(left_rgb, dtype=np.uint8)),
                "observation.images.right_wrist": Image.fromarray(np.asarray(right_rgb, dtype=np.uint8)),
                "observation.state": np.asarray(input_state, dtype=np.float32).tolist(),
                "action": np.asarray(actions, dtype=np.float32).tolist(),
                "task": instruction,
            }

            # The per-step contact wrench, plus whatever else the task config's data_type block
            # turned on -- depth, segmentation, point cloud, third-person view, end-effector
            # poses (those are empty for the plain configs).
            record.update(extra_obs_columns(observation, step_wrench=step_wrench,
                                            num_steps=pi0_step, fixed_pcd=fixed_pcd))

            # Model-space copies of the same step, for critics that score the policy's own
            # normalized action chunk (as `Pi0.sample_actions` does when steering). The columns
            # above are raw robot space -- env qpos, and the chunk after the output transform
            # unnormalized it -- so a critic trained on them cannot be applied inside the
            # sampler, which steers before that transform runs.
            critic_obs = getattr(model, "last_critic_obs", None)
            if critic_obs is not None:
                record["observation.state.model"] = critic_obs["state"].tolist()
                record["action.model"] = critic_obs["action"].tolist()
                # SigLIP patch features for each camera view the policy sees, straight from the
                # sampler's own image tower -- what the critic conditions on, keyed by the same
                # `siglip.<view>` names it uses online. Kept as numpy arrays (Array2D columns,
                # see build_features) rather than nested lists: at 256x1152 per view per row,
                # `.tolist()` would cost ~24 MB of Python floats a view a step.
                record.update(critic_obs.get("siglip", {}))

            frame_records.append(record)

            if success_now:
                succ = True
                break

        TASK_ENV.close_env(clear_cache=((collected + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        # Stamp episode-level metadata onto every frame of this episode.
        for rec in frame_records:
            rec["episode_index"] = collected
            rec["success"] = succ

        status = "\033[92mSuccess!\033[0m" if succ else "\033[91mFail!\033[0m"
        print(f"Episode {collected} (seed {now_seed}): {status} "
              f"| {len(frame_records)} steps | total collected: {collected + 1}/{num_episodes}")

        collected += 1
        now_id += 1
        now_seed += 1
        yield frame_records, {"collected": collected, "now_id": now_id, "now_seed": now_seed}


def infer_feature(value):
    """The datasets feature that stores `value` as-is.

    Used for the columns whose presence and shape are decided at runtime by the task config
    (see extra_obs_columns), so they do not have to be enumerated here. ndarrays become
    fixed-shape ArrayND columns -- far cheaper than the equivalent nested Sequence -- which is
    also why anything genuinely ragged (a non-downsampled point cloud) arrives as a list.
    """
    if isinstance(value, Image.Image):
        return datasets.Image()
    if isinstance(value, np.ndarray):
        dtype = str(value.dtype)
        if value.ndim == 0:
            return datasets.Value(dtype)
        if value.ndim == 1:
            return datasets.Sequence(datasets.Value(dtype))
        array_feature = {2: datasets.Array2D, 3: datasets.Array3D, 4: datasets.Array4D}[value.ndim]
        return array_feature(shape=value.shape, dtype=dtype)
    if isinstance(value, list):
        inner = infer_feature(value[0]) if value else datasets.Value("float32")
        return datasets.Sequence(inner)
    if isinstance(value, bool):
        return datasets.Value("bool")
    if isinstance(value, int):
        return datasets.Value("int32")
    if isinstance(value, float):
        return datasets.Value("float32")
    return datasets.Value("string")


def build_features(record):
    """The dataset schema implied by one collected row.

    Derived from a row rather than fixed because the optional columns depend on what the policy
    was asked to expose (collect_critic_obs / collect_siglip) and on which data types the task
    config enabled (extra_obs_columns). Every shard of a run is written with the same schema --
    `concatenate_datasets` rejects a mismatch.
    """
    features = datasets.Features({
        "episode_index": datasets.Value("int32"),
        # Primitive sim step, so consecutive rows are `pi0_step` apart -- not a row counter.
        "frame_index": datasets.Value("int32"),
        "observation.images.head": datasets.Image(),
        "observation.images.left_wrist": datasets.Image(),
        "observation.images.right_wrist": datasets.Image(),
        "observation.state": datasets.Sequence(datasets.Value("float32")),
        # Action chunk executed at this inference step: shape (chunk_len, action_dim).
        "action": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
        # Reward earned by that chunk -- task shaping, or 1.0 on the step that first succeeds.
        # Point `multisensory_steering`'s `dataset.reward_col` at it to train on the shaped
        # reward instead of deriving the sparse `terminal_reward` from `success`.
        "reward": datasets.Value("float32"),
        "success": datasets.Value("bool"),
        "task": datasets.Value("string"),
    })
    # Present only when the policy exposes them (pi05 with collect_critic_obs). Both are
    # *normalized* and in embodiment dims, with the model's zero padding stripped back off:
    # state is (critic_action_dim,) and the chunk is the full-horizon sample
    # (action_horizon, critic_action_dim) -- e.g. (14,) and (50, 14) for aloha agilex.
    if "action.model" in record:
        features["observation.state.model"] = datasets.Sequence(datasets.Value("float32"))
        features["action.model"] = datasets.Sequence(datasets.Sequence(datasets.Value("float32")))
    # Per-camera SigLIP patch features (collect_siglip): the same (256, 1152) columns
    # `multisensory_steering.create_dataset siglip` used to add in a second pass, produced here
    # by the sampler's own image tower. Array2D keeps them compact fixed-shape ndarray columns
    # instead of nested-list Sequences, which is what makes storing them inline affordable.
    for key, value in record.items():
        if key.startswith("siglip."):
            value = np.asarray(value)
            features[key] = datasets.Array2D(shape=value.shape, dtype=str(value.dtype))
    # Everything the task config's data_type block added (depth / segmentation / pointcloud /
    # third view / endpose), typed from the value itself.
    for key, value in record.items():
        if key not in features:
            features[key] = infer_feature(value)
    return features


def shard_paths(shard_dir):
    """Per-episode shard dirs, in collection order."""
    return sorted(p for p in shard_dir.glob("episode_*") if p.is_dir())


def read_progress(shard_dir):
    """The loop state left by the last completed episode, or None if there is nothing to resume."""
    path = shard_dir / "progress.json"
    if not path.exists() or not shard_paths(shard_dir):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(usr_args):
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    output_dir = Path(usr_args["output_dir"]) / task_name / task_config / str(ckpt_setting)

    # Each episode is flushed to its own on-disk shard the moment it finishes, so RAM holds one
    # episode (tens of MB) rather than the whole run: a row costs ~1 MB resident (three
    # uncompressed camera frames, plus ~576 KB per SigLIP view collected, and ~1 MB more
    # under a privileged task config -- depth alone is a float32 map per camera), and a long
    # task at 100 episodes runs to five figures of rows. It also means an interrupted run keeps the
    # episodes it already paid for -- `progress.json` records the seed to resume from, since the
    # seed sequence depends on which seeds the expert check rejected along the way.
    # Kept beside output_dir, not inside it: the final dataset is memory-mapped from these
    # shards while it is written and pushed, so they must survive save_to_disk.
    shard_dir = output_dir.parent / f"{output_dir.name}_shards"
    start = read_progress(shard_dir) if usr_args.get("resume", True) else None
    if start is None and shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    features = None
    if start is not None:
        # Reuse the schema the existing shards carry rather than re-deriving it: a config
        # changed between runs (collect_siglip, say) would otherwise produce shards that
        # cannot be concatenated.
        features = datasets.load_from_disk(str(shard_paths(shard_dir)[0])).features
        print(f"Resuming: {start['collected']} episode(s) already in {shard_dir}, "
              f"next seed {start['now_seed']}")

    for records, state in collect_rollouts(usr_args, start):
        if features is None:
            features = build_features(records[0])
            print("Dataset columns: " + ", ".join(features))
        elif set(records[0]) != set(features):
            # Resuming with a schema the existing shards do not have -- a different task config
            # (data_type flags) or collect_critic_obs / collect_siglip setting. from_list would
            # fail with a bare KeyError, so say what actually changed.
            added = sorted(set(records[0]) - set(features))
            dropped = sorted(set(features) - set(records[0]))
            raise SystemExit(
                f"Resumed run collects different columns than the shards in {shard_dir}:\n"
                f"  new: {added or 'none'}\n  missing: {dropped or 'none'}\n"
                f"Restore the config the shards were collected with, or set resume: false to "
                f"discard them and start over.")
        shard = shard_dir / f"episode_{state['collected'] - 1:05d}"
        datasets.Dataset.from_list(records, features=features).save_to_disk(str(shard))
        with open(shard_dir / "progress.json", "w", encoding="utf-8") as f:
            json.dump(state, f)

    shards = shard_paths(shard_dir)
    if not shards:
        raise SystemExit("No rollout frames were collected.")

    # Memory-mapped, so the whole run is never resident: save_to_disk and push_to_hub stream
    # from the shard files.
    dataset = datasets.concatenate_datasets([datasets.load_from_disk(str(p)) for p in shards])
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    success = np.asarray(dataset.with_format("numpy")["success"])
    episodes = np.asarray(dataset.with_format("numpy")["episode_index"])
    num_success = len(np.unique(episodes[success]))
    print(f"\nSaved {len(dataset)} frames from {len(shards)} episodes "
          f"({num_success} successful) to {output_dir}")

    if usr_args.get("push_to_hub"):
        repo_id = usr_args.get("hub_repo_id")
        if not repo_id:
            raise SystemExit("push_to_hub is true but hub_repo_id is not set.")
        # Private unless the config opts out: this only takes effect when the repo is created,
        # so a repo that already exists keeps whatever visibility it has.
        private = usr_args.get("hub_private", True)
        dataset.push_to_hub(repo_id, private=private)
        print(f"Pushed {'private' if private else 'PUBLIC'} dataset to "
              f"https://huggingface.co/datasets/{repo_id}")

    # Only now that everything is written (and pushed) are the shards redundant; a failure
    # above leaves them in place so the run can be resumed or salvaged.
    del dataset
    shutil.rmtree(shard_dir)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()
    main(usr_args)
