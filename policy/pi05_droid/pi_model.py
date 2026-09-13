# -- coding: UTF-8
"""The released pi05-DROID checkpoint, wrapped for the RoboTwin eval loop.

This is a deliberately thin sibling of `policy/pi05/pi_model.py`. None of that file's
machinery applies here -- no critic, no demo retrieval, no rollout collection -- because
pi05-DROID is a *frozen, released* checkpoint being scored zero-shot, not a policy this
repo fine-tunes. What it does instead is bridge three mismatches between DROID and RoboTwin:

1. **Observation layout.** pi0.5-aloha takes three camera views and a 14-dim bimanual state;
   DROID takes one exterior view, one wrist view, 7 joint positions and a gripper scalar
   (`openpi.policies.droid_policy.DroidInputs`).
2. **Action semantics.** `DroidOutputs` returns 8 dims: **7 joint velocities** plus an
   **absolute** gripper position. RoboTwin's `take_action` wants absolute joint targets, so
   the velocities are integrated here (`VELOCITY_DT`).
3. **Arm count.** DROID drives one Franka. RoboTwin's franka-panda embodiment is two of them
   (`embodiment: [franka-panda, franka-panda, <dis>]`), so one arm is driven and the other is
   held at its current commanded pose.

The checkpoint lives in a bucket rather than under `policy/pi05/checkpoints/`, so
`checkpoint_dir` is taken verbatim from the config -- `create_trained_policy` runs it through
`openpi.shared.download.maybe_download`, which handles `gs://` and caches under
`~/.cache/openpi`.
"""
import os
import sys

import numpy as np

# openpi itself lives in the pi05 tree; this adapter only adds the DROID-side glue.
_PI05_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pi05")
if os.path.join(_PI05_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_PI05_ROOT, "src"))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


# DROID runs its policy at 15 Hz, so one chunk step is 1/15 s of joint *velocity*
# (examples/droid/main.py: `RobotEnv(action_space="joint_velocity", ...)`). RoboTwin has no
# such clock -- a `take_action` is a TOPP-retimed move to an absolute target, 34-196 physics
# steps depending on the size of the joint delta -- so this is the conversion factor between
# the two, and the first thing to tune if the arm moves too far or too little per step.
VELOCITY_DT = 1.0 / 15.0

# DROID's gripper action is an absolute position in [0, 1] where **1 is closed**; RoboTwin's
# normalized gripper val is the opposite (`robot.is_left_gripper_open` is `val > 0.8`). The
# DROID runtime also binarizes the command rather than tracking it continuously
# (examples/droid/main.py), which is reproduced here.
GRIPPER_BINARIZE_THRESHOLD = 0.5


class PI05Droid:

    def __init__(self, checkpoint_dir, pi0_step, arm="right",
                 velocity_dt=VELOCITY_DT, binarize_gripper=True,
                 gripper_threshold=GRIPPER_BINARIZE_THRESHOLD,
                 train_config_name="pi05_droid"):
        self.checkpoint_dir = checkpoint_dir
        self.pi0_step = int(pi0_step)
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        self.arm = arm
        self.velocity_dt = float(velocity_dt)
        self.binarize_gripper = bool(binarize_gripper)
        self.gripper_threshold = float(gripper_threshold)

        config = _config.get_config(train_config_name)
        self.model_config = config.model
        # 15 for pi05_droid. The chunk is this long regardless of how many steps of it the
        # rollout actually executes (`pi0_step`), exactly as on the pi05 side.
        self.action_horizon = int(config.model.action_horizon)
        if self.pi0_step > self.action_horizon:
            raise ValueError(
                f"pi0_step={self.pi0_step} exceeds the checkpoint's action_horizon="
                f"{self.action_horizon}; there are not that many steps in a chunk to execute."
            )

        print(f"[pi05_droid] loading {checkpoint_dir} (may download on first use)")
        self.policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        print("[pi05_droid] loading model success!")

        self.observation_window = None
        self.instruction = None
        # The parked arm's joint vector, latched once per episode (see `to_robot_action`).
        self._idle_hold = None

    # ---- language ----------------------------------------------------------------

    def set_language(self, instruction):
        self.instruction = instruction
        print(f"[pi05_droid] instruction: {instruction}")

    # ---- observation -------------------------------------------------------------

    def update_observation_window(self, exterior_rgb, wrist_rgb, joint_position, gripper_position):
        """Set the observation the next `get_action` runs on.

        Keys are `DroidInputs`' own (`openpi/policies/droid_policy.py`); the resize to 224x224
        and the normalization by the checkpoint's own DROID norm stats happen downstream in the
        transform chain, so the frames go in at whatever resolution the sim renders.
        """
        self.observation_window = {
            "observation/exterior_image_1_left": exterior_rgb,
            "observation/wrist_image_left": wrist_rgb,
            "observation/joint_position": np.asarray(joint_position, dtype=np.float32),
            "observation/gripper_position": np.asarray(gripper_position, dtype=np.float32),
            "prompt": self.instruction,
        }

    def reset_obsrvationwindows(self):
        self.observation_window = None
        self.instruction = None
        self._idle_hold = None

    # ---- action ------------------------------------------------------------------

    def get_action(self):
        if self.observation_window is None:
            raise RuntimeError("no observation set; call update_observation_window first")
        actions = self.policy.infer(self.observation_window)["actions"]
        return np.asarray(actions)   # (action_horizon, 8)

    def to_robot_action(self, droid_action, joint_position, gripper_position, other_arm_state):
        """One 8-dim DROID action -> RoboTwin's 16-dim absolute-qpos vector.

        `droid_action` is 7 joint velocities + 1 absolute gripper position. `joint_position` is
        the driven arm's current 7 joint targets, which the velocities integrate from.

        `other_arm_state` is the 8 dims (7 joints + gripper) of the arm this policy does not
        drive. It is **latched on the first call of an episode** and that same vector is
        re-commanded every step after, so the parked arm holds one constant position for the
        whole rollout rather than tracking whatever the observation last reported.

        Two things make that the right filler rather than zeros. `take_action` treats the
        vector as ABSOLUTE joint targets and TOPP-interpolates from the arm's current pose, so
        zero-padding would command joint angles [0]*7 -- a real, extended pose -- and swing the
        arm there every step. And latching rather than re-reading means the commanded value
        cannot drift with the observation, so the TOPP path stays exactly degenerate.

        The result is ordered the way `take_action` reads it (`envs/_base_task.py`):
        `[left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]`.
        """
        droid_action = np.asarray(droid_action, dtype=np.float64)
        joint_velocity, gripper = droid_action[:7], float(droid_action[7])

        arm_target = np.asarray(joint_position, dtype=np.float64) + joint_velocity * self.velocity_dt

        # DROID: 1 = closed. RoboTwin: 1 = open. The runtime binarizes rather than tracking the
        # continuous command, so by default so do we.
        if self.binarize_gripper:
            gripper = 1.0 if gripper > self.gripper_threshold else 0.0
        gripper_target = float(np.clip(1.0 - gripper, 0.0, 1.0))

        driven = np.concatenate([arm_target, [gripper_target]])
        if self._idle_hold is None:
            self._idle_hold = np.asarray(other_arm_state, dtype=np.float64).copy()
        other = self._idle_hold
        if self.arm == "left":
            return np.concatenate([driven, other])
        return np.concatenate([other, driven])
