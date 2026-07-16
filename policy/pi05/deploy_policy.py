import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
sys.path.insert(0, os.path.join(parent_directory, "src"))

from pi_model import *


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    critic_ckpt = usr_args.get("critic_ckpt", None)
    guidance_scale = usr_args.get("guidance_scale", 0.0)
    guidance_ramp_updates = usr_args.get("guidance_ramp_updates", 0)
    online_critic = usr_args.get("online_critic", False)
    # CriticCluster gradient guidance toggle. Robust bool parse: CLI `--overrides cluster false`
    # eval()s to the truthy string "false" otherwise (yaml `cluster: false` is already a bool).
    cluster = usr_args.get("cluster", False)
    cluster = cluster.strip().lower() in ("true", "1", "yes") if isinstance(cluster, str) else bool(cluster)
    # Online QMFM Value-critic hyperparameters (forwarded from deploy_policy_online.yml).
    critic_config = {
        k: usr_args[k]
        for k in (
            "value_hidden_dims", "value_layer_norm", "num_qs", "rho", "discount", "tau", "lr",
            "clip_grad", "cnn_features", "cnn_out_dim", "batch_size", "buffer_size",
            "start_training", "utd_ratio",
        )
        if k in usr_args
    }
    critic_seed = usr_args.get("critic_seed", usr_args.get("seed", 0) or 0)
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step,
               critic_ckpt=critic_ckpt, guidance_scale=guidance_scale,
               guidance_ramp_updates=guidance_ramp_updates,
               online_critic=online_critic, critic_config=critic_config, critic_seed=critic_seed,
               cluster=cluster)


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    initial_obs = (input_rgb_arr, input_state)

    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================
    return actions, initial_obs


def reset_model(model):
    model.reset_obsrvationwindows()
    if getattr(model, "online_critic", None) is not None:
        model.online_critic.reset_episode()
