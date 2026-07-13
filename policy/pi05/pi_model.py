#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import os

class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step, critic_ckpt=None, guidance_scale=0.0):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)
        checkpoint_dir = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}"

        # Optional critic gradient guidance for the flow-matching sampler (see
        # critic_guidance.py and Pi0.sample_actions). When a critic checkpoint is given,
        # its d(value)/d(action) steers each denoising step toward higher critic value.
        # The critic works in the raw 14-d state/action space, so we hand it this config's
        # norm stats to un-normalize the sampler's normalized, padded action chunk.
        sample_kwargs = None
        if critic_ckpt:
            from critic_guidance import load_critic
            from openpi.training import checkpoints as _checkpoints

            data_config = config.data.create(config.assets_dirs, config.model)
            norm_stats = _checkpoints.load_norm_stats(
                os.path.join(checkpoint_dir, "assets"), data_config.asset_id
            )
            critic = load_critic(critic_ckpt, norm_stats, data_config.use_quantile_norm)
            sample_kwargs = {"critic": critic, "guidance_scale": float(guidance_scale)}
            print(f"loaded critic for guidance: {critic_ckpt} (guidance_scale={guidance_scale})")

        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            sample_kwargs=sample_kwargs,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
