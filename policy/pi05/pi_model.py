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

from multisensory_steering import load_critic

import os

class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step,
                 critic_ckpt=None, guidance_scale=0.0,
                 guidance_ramp_updates=0,
                 online_critic=False, critic_config=None, critic_seed=0,
                 collect_critic_obs=False, collect_siglip=True):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.guidance_scale_target = float(guidance_scale)
        self.guidance_ramp_updates = max(0, int(guidance_ramp_updates))
        self.current_guidance_scale = 0.0
        # Rollout-dataset collection: record the critic's model-space view of each control
        # step (see get_action / last_critic_obs). Independent of guidance.
        self.collect_critic_obs = bool(collect_critic_obs)
        # Also keep the head-camera SigLIP patch features the critic conditions on. They are
        # by far the largest thing per row (256x1152 fp16 = 576 KB), so they can be dropped
        # while still recording the state/action columns.
        self.collect_siglip = self.collect_critic_obs and bool(collect_siglip)
        self.last_critic_obs = None

        config = _config.get_config(self.train_config_name)
        self.model_config = config.model
        checkpoint_dir = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}"

        # Online QMFM Value-critic gradient guidance for the flow-matching sampler (see
        # multisensory_steering and Pi0.sample_actions). Enabled by the caller iff the target
        # guidance_scale is nonzero. The critic (an ensemble Q) is trained ONLINE during eval
        # rollouts; its d(value)/d(action), differentiated through pi0.5's velocity, steers
        # each denoising step toward higher Q (QMFM denoised-estimate steering).
        #
        # The critic scores the action chunk in *embodiment* dims -- the width of
        # observation["joint_action"]["vector"] -- which nothing here knows until the sim hands
        # over the first observation. So the critic is built on the first
        # update_observation_window (see _init_critic), not here. `uses_online_critic` states
        # up front whether one is coming, because the eval driver has to decide about W&B
        # before any rollout starts.
        self.uses_online_critic = bool(online_critic)
        self.online_critic = None
        self.critic_action_dim = None
        self._critic_updates_at_start = 0
        self._critic_ckpt = critic_ckpt
        self._critic_config = dict(critic_config or {})
        self._critic_config["seed"] = critic_seed
        # primitive sim steps executed per chunk (gamma^H in TD)
        self._critic_config["horizon"] = int(pi0_step)
        if not online_critic and critic_ckpt:
            # Not an error: the baseline is selected by guidance_scale, and leaving a
            # critic_ckpt configured while running it is a normal A/B thing to do.
            print(f"[pi_model] guidance_scale is 0 -- ignoring critic_ckpt {critic_ckpt} "
                  f"and running the plain pi0.5 baseline")

        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    def _init_critic(self, state):
        """Set up the critic path now that the embodiment's action width is known.

        ``state`` is ``observation["joint_action"]["vector"]`` -- the sim's own joint vector
        (both arms plus grippers), so its width is exactly the number of dims the embodiment
        acts in. The model itself works in a padded ``action_dim`` (32); AlohaInputs zero-pads
        14 -> 32 (state as well as actions) and those trailing dims normalize to constant zero,
        so feeding them to the critic would only widen it with dead weights. Both the state and
        the action chunk it sees are therefore narrowed back to this width. Called once, from
        update_observation_window.
        """
        self.critic_action_dim = int(np.shape(state)[-1])
        horizon = int(self.model_config.action_horizon)
        chunk = f"{horizon}x{self.critic_action_dim}"

        if not self.uses_online_critic:
            if self.collect_critic_obs:
                self.policy._sample_kwargs.update({
                    "return_critic_obs": True,
                    "critic_action_dim": self.critic_action_dim,
                })
                siglip = "with head-camera SigLIP patch features" if self.collect_siglip else "no SigLIP"
                print(f"[pi_model] recording model-space critic observations for dataset "
                      f"collection (action chunk={chunk}, {siglip})")
            return

        cc = dict(self._critic_config)
        cc["action_dim_flat"] = horizon * self.critic_action_dim
        # State is the model-space state narrowed back to the embodiment's own dims, exactly as
        # the sampler emits it (Pi0.sample_actions::critic_observation) and as the collected
        # `observation.state.model` column stores it -- not the padded action_dim.
        cc["state_dim"] = self.critic_action_dim
        cc["siglip_channels"] = 1152
        cc["siglip_grid"] = 16
        self.online_critic = load_critic(cc, self._critic_ckpt)
        # A warm-started critic restores its lifetime update counter from the checkpoint (an
        # offline-trained one is in the hundreds/thousands), so the ramp has to be measured
        # against where *this* run started -- otherwise it reads as already finished and
        # guidance jumps to the target on the very first chunk.
        self._critic_updates_at_start = int(self.online_critic.num_updates)

        # A checkpoint's architecture keys override the caller's, so a critic whose shapes do not
        # match (wrong embodiment, wrong action horizon) loads "successfully" and then fails with
        # an opaque dot_general shape error inside sample_actions. Catch it here instead. Note
        # this cannot tell a *raw-space* critic apart from a model-space one: the raw
        # observation.state / action columns have the same widths as their .model counterparts,
        # differing only in normalization. Getting that right is on whoever trains the critic.
        for key in ("action_dim_flat", "state_dim"):
            got, want = int(self.online_critic.config[key]), int(cc[key])
            if got != want:
                raise ValueError(
                    f"critic checkpoint {self._critic_ckpt!r} was trained with {key}={got}, but "
                    f"pi0.5 guidance feeds {key}={want} (state is the "
                    f"{self.critic_action_dim}-dim model state; the action chunk is "
                    f"{chunk} normalized). Collect a rollout dataset with collect_critic_obs "
                    f"and train on the observation.state.model / action.model columns."
                )

        self.policy._sample_kwargs.update({
            "critic_apply": self.online_critic.critic_apply,
            "guidance_scale": jnp.asarray(0.0, dtype=jnp.float32),
            "critic_action_dim": self.critic_action_dim,
        })
        warm = (f"warm-started from {self._critic_ckpt} at "
                f"{self._critic_updates_at_start} updates" if self._critic_ckpt else "from scratch")
        print(f"[pi_model] online QMFM Value critic enabled ({warm}, "
              f"guidance_scale_target={self.guidance_scale_target}, "
              f"guidance_ramp_updates={self.guidance_ramp_updates} (from this run's first "
              f"TD update), "
              f"num_qs={cc['num_qs']}, action chunk={chunk} "
              f"-> action_dim_flat={cc['action_dim_flat']})")

    def scheduled_guidance_scale(self):
        """Guidance ramps 0 -> target over the first `guidance_ramp_updates` TD updates.

        Counted from the start of this run, so a critic warm-started from `critic_ckpt` ramps
        in exactly like one trained from scratch: its values are trained on a different
        (offline) state distribution, so easing the sampler into them is worth doing even
        though the network is not random.
        """
        if self.online_critic is None:
            return 0.0
        updates = self.online_critic.num_updates - self._critic_updates_at_start
        if updates <= 0:
            return 0.0
        if self.guidance_ramp_updates <= 0:
            return self.guidance_scale_target
        progress = min(1.0, updates / float(self.guidance_ramp_updates))
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
        if self.critic_action_dim is None:
            # First observation of the run: the embodiment's action width is now known.
            self._init_critic(state)
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
            self._stash_critic_obs(out)
            return out["actions"]
        out = self.policy.infer(self.observation_window)
        self._stash_critic_obs(out)
        return out["actions"]

    def _stash_critic_obs(self, out):
        """Keep the last control step's model-space critic view for dataset collection.

        ``critic_obs_state`` (critic_action_dim,) is the normalized model state and
        ``critic_action`` (action_horizon, critic_action_dim) the normalized chunk, both in
        embodiment dims -- the tensors the critic is scored on. ``out["actions"]`` is the same
        chunk after the output transform has unnormalized it.

        ``critic_obs_img`` is the head-camera SigLIP patch map the critic's CNN encoder reads,
        as produced by ``Pi0.sample_actions``' own image tower -- so recording it here makes the
        separate ``multisensory_steering.create_dataset siglip`` pass unnecessary. It is stored
        as the flat ``(256, 1152)`` patch sequence that pass wrote (the encoders reshape to the
        16x16 grid themselves) and in fp16, matching the online replay buffer's ``stash``.
        """
        if not self.collect_critic_obs or "critic_action" not in out:
            return
        self.last_critic_obs = {
            "state": np.asarray(out["critic_obs_state"], dtype=np.float32),
            "action": np.asarray(out["critic_action"], dtype=np.float32),
        }
        if self.collect_siglip and "critic_obs_img" in out:
            siglip = np.asarray(out["critic_obs_img"], dtype=np.float16)  # (16, 16, 1152)
            self.last_critic_obs["siglip"] = siglip.reshape(-1, siglip.shape[-1])

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
