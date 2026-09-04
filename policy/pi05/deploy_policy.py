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

    Only a run with a critic has anything to condition on them, so the baseline and
    dataset-collection runs skip this entirely rather than copying depth maps for nobody.

    The contact-wrench trace is the one thing here that is not part of the observation: the env
    commits a row per primitive step -- the contact wrench averaged over that step's physics
    steps -- and this drains them, which means what a control step sees is the trace of the
    *previous* chunk, one row per action executed since the last observation, stacked to the
    fixed `wrench_trace_len` width the critic's obs shape was built with.
    That is the only wrench a policy could ever condition on (the current chunk has not been
    executed yet), and rollout collection drains it at the same point, so `observation.wrench.*`
    in a dataset is the same array against the same row's state and action.
    """
    if not getattr(model, "uses_online_critic", False):
        return None
    # Draining here is also what keeps the log bounded during eval. Rollout collection has its
    # own drain at its own observation, and never runs a critic, so the two never compete.
    return obs_modalities(observation, TASK_ENV.pop_step_wrench(), model.wrench_trace_len)


# Keys of the `critic_config_path` file that are the critic's own hyperparameters, forwarded
# into `load_critic`. They reach `usr_args` flattened alongside the deploy config
# (`eval_policy.parse_args_and_config` merges the included file underneath it), so this is what
# separates them back out again -- anything not listed stays a deploy-side knob.
CRITIC_CONFIG_KEYS = (
    # Which family to build, and what it observes.
    "critic_type", "encoder_modalities",
    # Value network / TD / replay, shared by both families.
    "value_hidden_dims", "value_layer_norm", "num_qs", "rho", "discount", "tau", "lr",
    # Control steps of reward the TD target carries before it bootstraps (1 = QMFM's one-step
    # target).
    "n_steps",
    "lr_warmup_steps", "lr_decay_steps", "lr_final_frac",
    "clip_grad", "cnn_features", "cnn_out_dim", "batch_size", "buffer_size",
    "start_training", "utd_ratio",
    # Offline rollouts mixed into every TD batch (a `{enabled, frac, config, ...}` block; see
    # cfgs/qmfm.yaml). `train_online` rides along with it so the critic can skip loading the
    # dataset when no update is going to run -- it is otherwise read as `train_critic_online`
    # below.
    "offline_mix", "train_online",
    # Train the value head only, keeping the checkpoint's observation encoder.
    "freeze_encoder",
    # Set by eval_policy when it resumes a run: take Adam's moments and the LR schedule
    # position from `critic_ckpt` too, rather than restarting the schedule at the bottom of
    # warmup. Off for an ordinary warm start.
    "restore_optimizer",
    # ---- critic_type: dsrl only (see cfgs/dsrl.yaml) ----
    # SAC's actor and temperature, the noise space it acts in, and jaxrl2's pixel augmentation.
    "actor_hidden_dims", "actor_lr", "temp_lr", "init_temperature", "target_entropy",
    "backup_entropy", "critic_reduction", "latent_dim", "use_bottleneck", "dropout_rate",
    "action_magnitude", "noise_horizon", "noise_param", "noise_rank",
    "color_jitter", "aug_next", "crop_padding",
    "num_proposals", "proposal_key", "proposal_hidden_dims", "proposal_out_dim",
)


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
    # `guidance_scale` and `best_of_n` are the two ways the critic can act on sampling, and
    # either one on its own turns the critic path on: 0 / 1 is plain pi0.5 sampling, anything
    # else builds and (by default) TD-trains the online Value critic.
    # Robust parse: `--overrides guidance_scale 0.3` eval()s to a float, but a malformed
    # value stays a string, which must not silently read as "guidance on".
    guidance_scale = float(usr_args.get("guidance_scale", 0.0) or 0.0)
    # Episodes over which guidance ramps 0 -> guidance_scale, counted from the critic's first
    # TD update of this run. The clock is the driver's own completed-episode count, pushed in
    # per episode (PI0.set_episodes_done).
    guidance_ramp_episodes = usr_args.get("guidance_ramp_episodes", 10)
    # Set only by eval_policy when it resumes an interrupted run: the episode that run's ramp
    # started at, so the resumed critic comes back at the guidance it had reached instead of
    # waiting for a "first" update it already did. Not a user-facing config key.
    guidance_ramp_start_episode = usr_args.get("guidance_ramp_start_episode")
    # Candidate chunks sampled per control step; the highest-Q one is executed. 1 = off.
    best_of_n = int(usr_args.get("best_of_n", 1) or 1)
    # The third way, and the only one that is not a knob: `critic_type: dsrl` steers by picking
    # the sampler's noise, so it needs no guidance scale and no candidate count -- naming the
    # family is what turns it on (pi_model.CRITIC_TYPES).
    critic_type = str(usr_args.get("critic_type") or "qmfm")
    online_critic = critic_type == "dsrl" or guidance_scale != 0.0 or best_of_n > 1
    # Whether that critic keeps learning during the rollouts. False freezes it at whatever
    # `critic_ckpt` holds: it still steers the sampler, but nothing is stashed into the replay
    # buffer and no TD update runs. Only meaningful when a critic exists at all.
    #
    # It is a critic-side knob, so it lives in the `critic_config_path` file as `train_online`
    # (next to `freeze_encoder`, which is the same idea one level down -- freeze the encoder
    # and keep training the value head) and reaches usr_args through that include. The attribute
    # keeps the longer name on this side, where "online" alone would not say online *what*.
    train_critic_online = _as_bool(usr_args.get("train_online"), True)
    # The critic's own hyperparameters. These reach usr_args via the deploy config's
    # `critic_config_path` include (see eval_policy.parse_args_and_config), so the critic's own
    # cfg file stays the source of truth for them.
    critic_config = {k: usr_args[k] for k in CRITIC_CONFIG_KEYS if k in usr_args}
    critic_seed = usr_args.get("critic_seed", usr_args.get("seed", 0) or 0)
    # DSRL only: control steps to spend on the base policy's own Gaussian latent before the
    # actor starts choosing. A rollout-loop knob rather than a critic hyperparameter, so it is
    # kept out of `critic_config` (and out of the critic checkpoint it would otherwise land in).
    noise_warmup_chunks = int(usr_args.get("noise_warmup_chunks", 0) or 0)
    # Demo retrieval (openpi/policies/demo_retrieval.py). Which of the two proposal modalities
    # is on comes from the *task config's* `data_type` block, like every other modality --
    demo_retrieval = dict(usr_args.get("demo_retrieval") or {})
    demo_retrieval.setdefault("seed", usr_args.get("seed", 0) or 0)

    return PI0(train_config_name, model_name, checkpoint_id, pi0_step,
               critic_ckpt=critic_ckpt, guidance_scale=guidance_scale,
               best_of_n=best_of_n,
               guidance_ramp_episodes=guidance_ramp_episodes,
               guidance_ramp_start_episode=guidance_ramp_start_episode,
               online_critic=online_critic, train_critic_online=train_critic_online,
               critic_config=critic_config, critic_seed=critic_seed,
               noise_warmup_chunks=noise_warmup_chunks,
               collect_critic_obs=usr_args.get("collect_critic_obs", False),
               collect_siglip=usr_args.get("collect_siglip", True),
               demo_retrieval=demo_retrieval,
               task_name=usr_args.get("task_name"),
               wrench_trace_len=usr_args.get("wrench_trace_len"))  # None -> pi0_step


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
