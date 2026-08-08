import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
sys.path.insert(0, os.path.join(parent_directory, "src"))

from pi_model import *

from envs.utils.obs_modalities import obs_modalities


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def critic_obs_modalities(TASK_ENV, model, observation):
    """The sim's sensor modalities for this control step, or None when nothing consumes them.

    Only the guided path has a critic, so the baseline and dataset-collection runs skip this
    entirely rather than copying depth maps for nobody.

    The contact-wrench trace is the one thing here that is not part of the observation: it is
    logged per primitive step by the env and drained here, which means what a control step sees
    is the trace of the *previous* chunk -- the steps between the last observation and this one.
    That is the only wrench a policy could ever condition on (the current chunk has not been
    executed yet), and rollout collection drains it at the same point, so `observation.wrench.*`
    in a dataset is the same array against the same row's state and action.
    """
    if not getattr(model, "uses_online_critic", False):
        return None
    # Draining here is also what keeps the log bounded during eval. Rollout collection has its
    # own drain at its own observation, and never runs a critic, so the two never compete.
    return obs_modalities(observation, TASK_ENV.pop_step_wrench(), model.pi0_step)


def _as_bool(value, default):
    """A config flag that may arrive as a CLI override string.

    The yml gives real booleans, but `--overrides train_online false` reaches here as
    the *string* `"false"` (eval() raises NameError on it and the parser keeps the text), and
    `bool("false")` is True -- exactly backwards for a switch. Anything unrecognised raises
    rather than defaulting, so a typo cannot silently turn a flag on.
    """
    if value is None:
        return default
    if not isinstance(value, str):
        return bool(value)
    text = value.strip().lower()
    if text in ("true", "yes", "on", "1"):
        return True
    if text in ("false", "no", "off", "0", ""):
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    critic_ckpt = usr_args.get("critic_ckpt", None)
    # `guidance_scale` is the single switch for the online critic path: 0 (the default) is
    # plain pi0.5 sampling, anything else builds and TD-trains the online Value critic.
    # Robust parse: `--overrides guidance_scale 0.3` eval()s to a float, but a malformed
    # value stays a string, which must not silently read as "guidance on".
    guidance_scale = float(usr_args.get("guidance_scale", 0.0) or 0.0)
    guidance_ramp_updates = usr_args.get("guidance_ramp_updates", 0)
    online_critic = guidance_scale != 0.0
    # Whether that critic keeps learning during the rollouts. False freezes it at whatever
    # `critic_ckpt` holds: it still steers the sampler, but nothing is stashed into the replay
    # buffer and no TD update runs. Only meaningful when a critic exists at all.
    #
    # It is a critic-side knob, so it lives in the `critic_config_path` file as `train_online`
    # (next to `freeze_encoder`, which is the same idea one level down -- freeze the encoder
    # and keep training the value head) and reaches usr_args through that include. The attribute
    # keeps the longer name on this side, where "online" alone would not say online *what*.
    train_critic_online = _as_bool(usr_args.get("train_online"), True)
    # Online QMFM Value-critic hyperparameters. These reach usr_args via the deploy config's
    # `critic_config_path` include (see eval_policy.parse_args_and_config), so the critic's
    # own cfg file stays the source of truth for them.
    critic_config = {
        k: usr_args[k]
        for k in (
            "value_hidden_dims", "value_layer_norm", "num_qs", "rho", "discount", "tau", "lr",
            # Control steps of reward the TD target carries before it bootstraps (1 = QMFM's
            # one-step target).
            "n_steps",
            "lr_warmup_steps", "lr_decay_steps", "lr_final_frac",
            "clip_grad", "cnn_features", "cnn_out_dim", "batch_size", "buffer_size",
            "start_training", "utd_ratio",
            # Offline rollouts mixed into every TD batch (a `{enabled, frac, config, ...}`
            # block; see cfgs/qmfm.yaml). `train_online` rides along with it so the critic can
            # skip loading the dataset when no update is going to run -- it is otherwise read
            # as `train_critic_online` above.
            "offline_mix", "train_online",
            # Train the value head only, keeping the checkpoint's observation encoder.
            "freeze_encoder",
            # Which observation modalities the critic conditions on, and with what encoders.
            "encoder_modalities",
        )
        if k in usr_args
    }
    critic_seed = usr_args.get("critic_seed", usr_args.get("seed", 0) or 0)
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step,
               critic_ckpt=critic_ckpt, guidance_scale=guidance_scale,
               guidance_ramp_updates=guidance_ramp_updates,
               online_critic=online_critic, train_critic_online=train_critic_online,
               critic_config=critic_config, critic_seed=critic_seed,
               collect_critic_obs=usr_args.get("collect_critic_obs", False),
               collect_siglip=usr_args.get("collect_siglip", True))


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    initial_obs = (input_rgb_arr, input_state)

    model.update_observation_window(input_rgb_arr, input_state,
                                    critic_obs=critic_obs_modalities(TASK_ENV, model, observation))

    # ======== Get Action ========

    actions = model.get_action()

    for action in actions[:model.pi0_step]:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================
    # Trim the chunk to the embodiment's own action dims, i.e. the width of
    # observation["joint_action"]["vector"] (1-D: both arms plus grippers). That is the same
    # width the critic scores -- see PI0._init_critic.
    action_dim = np.shape(initial_obs[1])[-1]
    return actions[:, :action_dim], initial_obs


def reset_model(model):
    model.reset_obsrvationwindows()
    if getattr(model, "online_critic", None) is not None:
        model.online_critic.reset_episode()
