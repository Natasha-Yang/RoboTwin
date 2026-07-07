"""Roll out a trained policy in RoboTwin and save the rollouts as a HuggingFace dataset.

Given a checkpoint (via the policy's ``deploy_policy`` interface), a task config and a
task name, this collects rollout episodes and stores, for every policy inference step,
the observations from the three cameras (head / left wrist / right wrist), the action
chunk executed, and whether the episode ultimately succeeded.

Model loading stays policy-specific: we reuse the ``get_model`` / ``eval`` / ``encode_obs``
/ ``reset_model`` interface defined in ``policy/<policy_name>/deploy_policy.py`` (the same
interface ``eval_policy.py`` uses). Only the rollout loop and dataset construction live here.

Usage (mirrors eval_policy.py):
    python script/collect_dataset.py --config policy/pi05/collect_dataset.yml \
        --overrides --task_name <task> --task_config <config> \
        --train_config_name <cfg> --model_name <model> --ckpt_setting <model> \
        --seed <seed> --policy_name pi05
"""
import os
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
from generate_episode_instructions import *

# Reuse env-setup helpers from the evaluation entrypoint so the two stay in sync.
from eval_policy import (
    class_decorator,
    eval_function_decorator,
    get_embodiment_config,
    parse_args_and_config,
)
from envs import CONFIGS_PATH

import yaml


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


def collect_rollouts(usr_args):
    """Run rollouts and return a list of per-inference-step records."""
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
    clear_cache_freq = args["clear_cache_freq"]

    model = get_model(usr_args)

    st_seed = 100000 * (1 + usr_args["seed"])
    now_seed = st_seed
    now_id = 0
    collected = 0
    records = []

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
        frame_records = []
        frame_index = 0
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            # Record the first observation seen before this inference call.
            observation = TASK_ENV.get_obs()

            actions, initial_obs = eval_func(TASK_ENV, model, observation)

            input_rgb_arr, input_state = initial_obs
            head_rgb, right_rgb, left_rgb = input_rgb_arr

            frame_records.append({
                "frame_index": frame_index,
                "observation.images.head": Image.fromarray(np.asarray(head_rgb, dtype=np.uint8)),
                "observation.images.left_wrist": Image.fromarray(np.asarray(left_rgb, dtype=np.uint8)),
                "observation.images.right_wrist": Image.fromarray(np.asarray(right_rgb, dtype=np.uint8)),
                "observation.state": np.asarray(input_state, dtype=np.float32).tolist(),
                "action": np.asarray(actions, dtype=np.float32).tolist(),
                "task": instruction,
            })
            frame_index += 1

            if TASK_ENV.eval_success:
                succ = True
                break

        TASK_ENV.close_env(clear_cache=((collected + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        # Stamp episode-level metadata onto every frame of this episode.
        for rec in frame_records:
            rec["episode_index"] = collected
            rec["success"] = succ
            records.append(rec)

        status = "\033[92mSuccess!\033[0m" if succ else "\033[91mFail!\033[0m"
        print(f"Episode {collected} (seed {now_seed}): {status} "
              f"| {len(frame_records)} steps | total collected: {collected + 1}/{num_episodes}")

        collected += 1
        now_id += 1
        now_seed += 1

    return records


def build_dataset(records):
    features = datasets.Features({
        "episode_index": datasets.Value("int32"),
        "frame_index": datasets.Value("int32"),
        "observation.images.head": datasets.Image(),
        "observation.images.left_wrist": datasets.Image(),
        "observation.images.right_wrist": datasets.Image(),
        "observation.state": datasets.Sequence(datasets.Value("float32")),
        # Action chunk executed at this inference step: shape (chunk_len, action_dim).
        "action": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
        "success": datasets.Value("bool"),
        "task": datasets.Value("string"),
    })
    return datasets.Dataset.from_list(records, features=features)


def main(usr_args):
    records = collect_rollouts(usr_args)
    if not records:
        raise SystemExit("No rollout frames were collected.")

    dataset = build_dataset(records)

    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    output_dir = Path(usr_args["output_dir"]) / task_name / task_config / str(ckpt_setting)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(output_dir))
    num_episodes = len(set(r["episode_index"] for r in records))
    num_success = len(set(r["episode_index"] for r in records if r["success"]))
    print(f"\nSaved {len(dataset)} frames from {num_episodes} episodes "
          f"({num_success} successful) to {output_dir}")

    if usr_args.get("push_to_hub"):
        repo_id = usr_args.get("hub_repo_id")
        if not repo_id:
            raise SystemExit("push_to_hub is true but hub_repo_id is not set.")
        dataset.push_to_hub(repo_id)
        print(f"Pushed dataset to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()
    main(usr_args)
