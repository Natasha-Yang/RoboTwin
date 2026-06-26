import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
sys.path.insert(0, os.path.join(parent_directory, "src"))

from molmoact_model import *


def encode_obs(observation):  # Post-Process Observation
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    checkpoint_path, norm_tag, molmoact_step = (
        usr_args["ckpt_path"],
        usr_args["ckpt_setting"],
        usr_args["molmoact_step"],
    )
    return MolmoAct(checkpoint_path, norm_tag, molmoact_step)


def eval(TASK_ENV, model, observation):
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.molmoact_step]

    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos") # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obsrvationwindows()
