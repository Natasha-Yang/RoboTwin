#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import jax.numpy as jnp
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

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step,
                 critic_ckpt=None, guidance_scale=0.0,
                 guidance_ramp_updates=0,
                 online_critic=False, critic_config=None, critic_seed=0):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.guidance_scale_target = float(guidance_scale)
        self.guidance_ramp_updates = max(0, int(guidance_ramp_updates))
        self.current_guidance_scale = 0.0

        config = _config.get_config(self.train_config_name)
        checkpoint_dir = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}"

        # Online QMFM Value-critic gradient guidance for the flow-matching sampler (see
        # qmfm_critic.py and Pi0.sample_actions). The critic (an ensemble Q) is trained ONLINE
        # during eval rollouts; its d(value)/d(action), differentiated through pi0.5's velocity,
        # steers each denoising step toward higher Q (QMFM denoised-estimate steering).
        self.online_critic = None
        sample_kwargs = None
        if online_critic:
            from qmfm_critic import OnlineValueCritic

            cc = dict(critic_config or {})
            cc.setdefault("value_hidden_dims", [512, 512, 512, 512])
            cc.setdefault("value_layer_norm", True)
            cc.setdefault("num_qs", 10)
            cc.setdefault("rho", 0.5)
            cc.setdefault("discount", 0.99)
            cc.setdefault("tau", 0.005)
            cc.setdefault("lr", 3e-4)
            cc.setdefault("clip_grad", True)
            cc.setdefault("cnn_features", [128, 128])
            cc.setdefault("cnn_out_dim", 128)
            cc.setdefault("batch_size", 256)
            cc.setdefault("buffer_size", 5000)
            cc.setdefault("start_training", 256)
            cc.setdefault("utd_ratio", 1)
            # Dims derived from the model config (pi0.5: action_dim=32, action_horizon=50).
            cc["action_dim_flat"] = int(config.model.action_horizon * config.model.action_dim)
            cc["state_dim"] = int(config.model.action_dim)  # observation.state is padded to action_dim
            cc["siglip_channels"] = 1152
            cc["siglip_grid"] = 16
            cc["horizon"] = int(pi0_step)  # primitive sim steps executed per chunk (gamma^H in TD)
            self.online_critic = OnlineValueCritic(seed=int(critic_seed), config=cc)
            sample_kwargs = {
                "critic_apply": self.online_critic.critic_apply,
                "guidance_scale": jnp.asarray(0.0, dtype=jnp.float32),
            }
            print(f"[pi_model] online QMFM Value critic enabled "
                  f"(guidance_scale_target={self.guidance_scale_target}, "
                  f"guidance_ramp_updates={self.guidance_ramp_updates}, "
                  f"num_qs={cc['num_qs']}, "
                  f"action_dim_flat={cc['action_dim_flat']})")
        elif critic_ckpt:
            raise NotImplementedError(
                "The offline torch SigLIPCritic guidance path was replaced by the online QMFM "
                "Value critic. Set online_critic: true (see deploy_policy_online.yml) instead of "
                "critic_ckpt."
            )

        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            sample_kwargs=sample_kwargs,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    def scheduled_guidance_scale(self):
        if self.online_critic is None or self.online_critic.num_updates <= 0:
            return 0.0
        if self.guidance_ramp_updates <= 0:
            return self.guidance_scale_target
        progress = min(1.0, self.online_critic.num_updates / float(self.guidance_ramp_updates))
        return self.guidance_scale_target * progress

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
        if self.online_critic is not None:
            # Inject the current (traced) critic params so online updates take effect without an
            # XLA recompile. The scheduled guidance scale is also traced, so changing it per
            # call ramps guidance without compiling a sampler for each scalar value.
            self.current_guidance_scale = self.scheduled_guidance_scale()
            self.policy._sample_kwargs["critic_params"] = self.online_critic.params
            self.policy._sample_kwargs["guidance_scale"] = jnp.asarray(
                self.current_guidance_scale, dtype=jnp.float32
            )
            # Stash this control step's (obs, action) for the replay buffer.
            out = self.policy.infer(self.observation_window)
            self.online_critic.stash(out["critic_obs_img"], out["critic_obs_state"], out["critic_action"])
            return out["actions"]
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
