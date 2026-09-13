# -- coding: UTF-8
"""RoboTwin policy-adapter trio for the released pi05-DROID checkpoint.

`script/eval_policy.py` imports `get_model` / `eval` / `reset_model` by name out of the package
named by `policy_name` (`eval_function_decorator`), so this file is the whole contract.

The checkpoint drives ONE Franka arm, which constrains the setup in two ways worth stating up
front:

* the embodiment must be **`[franka-panda, franka-panda, <dis>]`** -- 7 arm joints per arm is
  what the DROID state space is, and the one-entry form does not work for a `dual_arm: False`
  embodiment anyway (it looks for a `curobo_left.yml` that does not exist);
* the task must be solvable one-handed. `open_microwave` is, and it drives whichever arm's
  base is nearer the microwave -- the right arm under franka-panda-droid, matching `arm:`
  in deploy_policy.yml.

The other arm is held at its current commanded pose for the whole episode.
"""
import os
import sys

import numpy as np

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import PI05Droid


# RoboTwin's wrist camera for each arm. DROID's single wrist view is the driven arm's.
_WRIST_CAMERA = {"left": "left_camera", "right": "right_camera"}

# DROID's state is a Franka's: 7 joints plus a gripper. Anything else means the task config
# names an embodiment this checkpoint cannot be fed.
_DROID_ARM_DOF = 7


def encode_obs(observation, arm):
    """The sim observation -> what `DroidInputs` expects, plus the idle arm's hold pose.

    Returns `(exterior_rgb, wrist_rgb, joint_position, gripper_position, other_arm_state)`.
    The two state halves come from `joint_action`'s own per-arm keys rather than from slicing
    `vector`, so the split follows the embodiment instead of a hardcoded width.
    """
    ja = observation["joint_action"]
    arm_qpos = np.asarray(ja[f"{arm}_arm"], dtype=np.float64)
    arm_gripper = float(ja[f"{arm}_gripper"])

    other = "right" if arm == "left" else "left"
    other_arm_state = np.concatenate([np.asarray(ja[f"{other}_arm"], dtype=np.float64),
                                      [float(ja[f"{other}_gripper"])]])

    if arm_qpos.shape[0] != _DROID_ARM_DOF:
        raise ValueError(
            f"pi05_droid expects a {_DROID_ARM_DOF}-DoF arm (a Franka Panda), but the "
            f"{arm} arm of this embodiment has {arm_qpos.shape[0]}. Set "
            f"`embodiment: [franka-panda, franka-panda, 0.6]` in the task config."
        )

    exterior_rgb = observation["observation"]["exterior_camera"]["rgb"]
    wrist_rgb = observation["observation"][_WRIST_CAMERA[arm]]["rgb"]

    # DROID's gripper_position is 1 = closed; RoboTwin's normalized val is 1 = open. The state
    # fed to the model has to be in DROID's convention, the same way the action coming back is
    # converted on the way out (`PI05Droid.to_robot_action`).
    droid_gripper = np.array([1.0 - arm_gripper], dtype=np.float32)

    return exterior_rgb, wrist_rgb, arm_qpos, droid_gripper, other_arm_state


def get_model(usr_args):
    return PI05Droid(
        checkpoint_dir=usr_args["checkpoint_dir"],
        pi0_step=usr_args["pi0_step"],
        arm=usr_args.get("arm", "right"),
        velocity_dt=usr_args.get("velocity_dt", 1.0 / 15.0),
        binarize_gripper=usr_args.get("binarize_gripper", True),
        gripper_threshold=usr_args.get("gripper_threshold", 0.5),
        train_config_name=usr_args.get("train_config_name") or "pi05_droid",
    )


def eval(TASK_ENV, model, observation):
    if model.observation_window is None:
        model.set_language(TASK_ENV.get_instruction())

    exterior, wrist, joint_pos, gripper, other = encode_obs(observation, model.arm)
    model.update_observation_window(exterior, wrist, joint_pos, gripper)

    # (action_horizon, 8): 7 joint velocities + 1 absolute gripper position.
    actions = model.get_action()

    for action in actions[:model.pi0_step]:
        # Integrate from the arm's *current* commanded joints each step rather than from the
        # target we issued last step, so the chunk cannot accumulate drift away from where the
        # arm actually is.
        robot_action = model.to_robot_action(action, joint_pos, gripper, other)
        TASK_ENV.take_action(robot_action)
        observation = TASK_ENV.get_obs()
        exterior, wrist, joint_pos, gripper, other = encode_obs(observation, model.arm)
        model.update_observation_window(exterior, wrist, joint_pos, gripper)

    return actions[:model.pi0_step], None


def reset_model(model):
    model.reset_obsrvationwindows()
