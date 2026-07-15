"""Online critic-guided pi0.5 evaluation driver.

Same as ``script/eval_policy.py`` (expert feasibility check, seed loop, per-chunk rollout),
but during the policy rollout it (a) collects a chunk-level transition after every control
step and (b) runs QMFM TD updates on the online ``Value`` critic that steers the frozen
pi0.5 (``policy/pi05/qmfm_critic.py`` + ``Pi0.sample_actions``). The critic persists and keeps
learning across episodes for the whole eval run. Configure via ``deploy_policy_online.yml``.
"""

import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

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
import wandb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def init_wandb(usr_args, save_dir, current_time):
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

    run = wandb.init(**init_kwargs)
    wandb.define_metric("critic/update")
    wandb.define_metric("critic/*", step_metric="critic/update")
    wandb.define_metric("rollout/*", step_metric="critic/update")
    wandb.define_metric("eval/episode")
    wandb.define_metric("eval/*", step_metric="eval/episode")
    return run


def _scalar(value):
    return float(np.asarray(value))


def _window_mean(values):
    return float(np.mean(values)) if values else 0.0


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


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
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
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(usr_args, save_dir, current_time)

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

    # Online-critic knobs (from deploy_policy_online.yml) forwarded into the eval loop.
    args["train_freq"] = usr_args.get("train_freq", 1)
    args["wandb_ma_window"] = usr_args.get("wandb_ma_window", 20)

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = usr_args.get("test_num", 100)
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num, episode_results = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   wandb_run=wandb_run)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    episode_file_path = os.path.join(save_dir, f"_episode_results.csv")
    # write num_steps and success to csv file
    episode_results_df = pd.DataFrame(episode_results)
    episode_results_df.to_csv(episode_file_path)

    # Persist the online-trained critic alongside the eval results.
    if getattr(model, "online_critic", None) is not None and usr_args.get("save_critic", False):
        critic_path = os.path.join(save_dir, "online_value_critic.pkl")
        model.online_critic.save(critic_path)
        print(f"saved online critic to {critic_path}")

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
                wandb_run=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    episode_results = {"num_steps": [], "success": [], "reward": [], "success_rate_ma": [], "reward_ma": []}

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    # ===== Online QMFM Value critic (trained across the whole eval run) =====
    online_critic = getattr(model, "online_critic", None)
    train_freq = int(args.get("train_freq", 1))
    ma_window = max(1, int(args.get("wandb_ma_window", 20)))
    success_window = deque(maxlen=ma_window)
    reward_window = deque(maxlen=ma_window)
    chunk_count = 0
    last_info = None

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
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            success_now = bool(TASK_ENV.eval_success)
            reward = 1.0 if (success_now and not prev_success) else 0.0
            episode_reward += reward

            # Online critic: close the chunk transition (SARSA), then run a TD update.
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
        episode_results["num_steps"].append(TASK_ENV.take_action_cnt)
        episode_results["success"].append(succ)
        episode_results["reward"].append(episode_reward)
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        # Online-critic diagnostics.
        if online_critic is not None:
            buf = online_critic.buffer.size if online_critic.buffer is not None else 0
            guidance = model.scheduled_guidance_scale()
            if last_info is not None:
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
        )

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(success_rate*100, 1)}%\033[0m "
            f"(MA{ma_window}: \033[95m{round(success_rate_ma*100, 1)}%\033[0m, reward={reward_ma:.3f}), "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, episode_results


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

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

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
