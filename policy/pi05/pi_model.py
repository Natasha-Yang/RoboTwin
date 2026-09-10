#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import contextlib
import json
import sys
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models.pi0 import SIGLIP_MODALITIES
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

from openpi.policies.demo_retrieval import DemoRetriever

from multisensory_steering import load_critic
from multisensory_steering.critics.dsrl import noise_latent_shape

import os

# The two demo-retrieval modalities, switched by the task config's `data_type` block like the
# rest of them (see envs/utils/obs_modalities.py for the sensor ones). Unlike those, these are
# produced by the policy rather than by the sim, so the flags reach the model through
# `deploy_policy.get_model` instead of through the observation.
PROPOSAL_MODALITIES = ("action_proposals", "noise_proposals")

# The critic families `multisensory_steering.load_critic` can build, and how each one gets to
# act on the frozen sampler:
#
#   qmfm  an ensemble Q over *action chunks*. It steers the denoising by value gradient
#         (`guidance_scale`) and/or ranks the chunks it produces (`best_of_n`); either one on
#         its own turns the critic path on.
#   dsrl  SAC over the sampler's *latent noise*. The denoising itself is untouched -- the actor
#         picks the noise chunk pi0.5 denoises from, inside `Pi0.sample_actions`
#         (`noise_apply`), and its Q is over that latent rather than over the chunk.
#
# They are mutually exclusive by construction: a DSRL Q(s, w) cannot score an action chunk, so
# it has nothing to guide or rank with, and `_init_critic` refuses the combination.
CRITIC_TYPES = ("qmfm", "qmfm_iql", "dsrl")


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step,
                 critic_ckpt=None, guidance_scale=0.0, best_of_n=1,
                 guidance_ramp_updates=0, critic_ramp_baseline=None,
                 online_critic=False, train_critic_online=True,
                 critic_config=None, critic_seed=0, noise_warmup_chunks=0,
                 collect_critic_obs=False, collect_siglip=True,
                 demo_retrieval=None, task_name=None,
                 wrench_trace_len=None):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.guidance_scale_target = float(guidance_scale)
        self.guidance_ramp_updates = max(0, int(guidance_ramp_updates))
        self.current_guidance_scale = 0.0
        # Best-of-N: draw this many candidate chunks per control step and execute the one the
        # critic scores highest (`Pi0.sample_actions`). Independent of the gradient guidance --
        # either, both or neither -- but it needs the same critic to rank with, so >1 also turns
        # the critic path on (see deploy_policy.get_model). The candidates are extra batch
        # elements in one sampler call, so the SigLIP tower and the prefix pass are still paid
        # once; the denoising loop and the KV cache are what scale with N.
        self.best_of_n = max(1, int(best_of_n or 1))
        # The index the last control step's selection landed on and the ensemble-mean Q of every
        # candidate it chose between, for the eval driver to log. None until a chunk is drawn,
        # and always None at best_of_n 1 (there is nothing to select).
        self.last_best_index = None
        self.last_best_scores = None
        # Rollout-dataset collection: record the critic's model-space view of each control
        # step (see get_action / last_critic_obs). Independent of guidance.
        self.collect_critic_obs = bool(collect_critic_obs)
        # Also keep the SigLIP patch features the critic conditions on, one column per camera
        # view. They are by far the largest thing per row (256x1152 fp16 = 576 KB *per view*),
        # so `collect_siglip` also takes a list of views to record a subset -- see
        # _siglip_views. Empty tuple = record none, and the state/action columns still go in.
        self.collect_siglip = self._siglip_views(collect_siglip) if self.collect_critic_obs else ()
        self.last_critic_obs = None
        # Debug diagnostics: score each sampled chunk with the critic that steered it and keep
        # the result in `last_q_values` (one entry per ensemble member) for the eval driver to
        # plot. Off by default -- it is an extra critic forward per control step -- and turned
        # on by `script/eval_policy.py` when the task config sets `debug: true` and a critic is
        # actually running. Never affects sampling: it reads the chunk after it was drawn.
        self.record_q_values = False
        self.last_q_values = None
        # The same idea for demo retrieval: keep a thumbnail of every bank row and remember which
        # rows each control step matched, so `envs/utils/debug_vis.py::DemoRetrievalRecorder` can
        # draw the query frame beside the demo frames it retrieved. Set by the eval driver when
        # the task config has `debug: true`; must be set before the first bank is built.
        self.record_demo_retrieval = False
        self.last_demo_retrieval = None
        # The bank the critic's offline half was ranked against, so it happens exactly once.
        self._offline_retrieval_bank = None
        # Which key modalities each cross-attending proposal set needs, filled in from the
        # critic's own config once it exists (`_init_critic`).
        self._proposal_key_modalities = {}

        # Demo retrieval (see openpi/policies/demo_retrieval.py). The two proposal entries --
        # `action_proposals` (the retrieved demo chunks) and `noise_proposals` (the seeds that
        # map to them under the current observation) -- are NOT observation modalities and are
        # not switched by the task config's `data_type` block. Nothing in `get_obs` produces
        # them and nothing but a critic consumes them: they are the mechanism by which
        # everything else the critic encodes *queries* a bank of demonstrations. So the critic's
        # own `encoder_modalities` is the only switch, exactly as it is for which cameras it
        # reads -- naming one turns retrieval on, and a run with no critic never builds the
        # retriever at all.
        self.proposal_modalities = self._configured_proposals(critic_config)
        self.demo_retriever = None
        self._demo_retrieval_config = dict(demo_retrieval or {})
        self.task_name = task_name

        config = _config.get_config(self.train_config_name)
        self.model_config = config.model
        checkpoint_dir = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}"

        # Online critic for the flow-matching sampler (see multisensory_steering and
        # Pi0.sample_actions), of whichever family CRITIC_TYPES names. The QMFM one is an
        # ensemble Q over action chunks whose d(value)/d(action), differentiated through pi0.5's
        # velocity, steers each denoising step toward higher Q (QMFM denoised-estimate
        # steering); the DSRL one is a SAC actor-critic over the sampler's latent noise, which
        # leaves the denoising alone and changes what it starts from. Either way the critic is
        # trained ONLINE during the eval rollouts, off the same transitions.
        #
        # The critic scores the action chunk in *embodiment* dims -- the width of
        # observation["joint_action"]["vector"] -- which nothing here knows until the sim hands
        # over the first observation. So the critic is built on the first
        # update_observation_window (see _init_critic), not here. `uses_online_critic` states
        # up front whether one is coming, because the eval driver has to decide about W&B
        # before any rollout starts.
        #
        # `train_critic_online` decides whether that critic keeps learning here. False freezes
        # it at the checkpoint's parameters: it is still built and still steers the sampler, but
        # nothing is stashed into the replay buffer and the eval driver runs no TD update -- an
        # offline-trained critic evaluated as-is, with no eval-time distribution shift in the
        # values. It only makes sense against a `critic_ckpt` (see below).
        self.uses_online_critic = bool(online_critic)
        self.train_critic_online = bool(train_critic_online)
        # The second half of that switch, and the one a *held-out evaluation* flips: whether the
        # control steps being run are collected at all. `train_critic_online` is a property of
        # the run and never moves; this goes False for the duration of a `frozen_for_eval` block
        # so the driver's periodic held-out episodes (eval_interval in deploy_policy.yml) score
        # the critic without also feeding it -- they are a measurement of the current critic, and
        # a critic that trained on its own test set could not be compared across evaluations.
        self._collect_transitions = True
        self.online_critic = None
        self.critic_action_dim = None
        # Which family that critic is, and therefore how it acts on the sampler (CRITIC_TYPES).
        # It is a critic-side choice, so it arrives in the critic config like every other one.
        self.critic_type = str((critic_config or {}).get("critic_type") or "qmfm")
        if self.critic_type not in CRITIC_TYPES:
            raise ValueError(
                f"unknown critic_type {self.critic_type!r}; expected one of {list(CRITIC_TYPES)}."
            )
        if self.critic_type == "dsrl" and (self.guidance_scale_target != 0.0 or self.best_of_n > 1):
            raise ValueError(
                f"critic_type: dsrl steers by choosing the sampler's noise, so it has no Q over "
                f"action chunks to guide or rank with, but guidance_scale="
                f"{self.guidance_scale_target} / best_of_n={self.best_of_n} ask it for one. Set "
                f"guidance_scale: 0 and best_of_n: 1 (they are what turn the *qmfm* critic on)."
            )
        # DSRL only: control steps to spend on the base policy's own Gaussian latent before the
        # actor takes over, so the buffer starts with transitions from the distribution pi0.5
        # was trained to denoise rather than from an untrained tanh actor (dsrl_pi0 collects its
        # whole first trajectory this way). Counted across the run, not per episode.
        self._noise_warmup_left = (max(0, int(noise_warmup_chunks or 0))
                                   if self.critic_type == "dsrl" else 0)
        self._noise_rng = np.random.default_rng(critic_seed)
        # Sensor modalities from the sim observation (depth / point cloud / contact wrench --
        # see envs/utils/obs_modalities.py), handed in by deploy_policy.eval. The model itself
        # only produces `state` and a `siglip.<view>` map per camera; everything else the critic
        # conditions on arrives this way. Populated only when a critic is actually running.
        self.critic_obs_extra = {}
        self._critic_extra_shapes = {}
        self._critic_updates_at_start = 0
        # Where the ramp counts from, when the caller knows better than "wherever the
        # checkpoint left off". Only `script/eval_policy.py` resuming an interrupted run does:
        # it reloads that run's *own* critic as `critic_ckpt`, so re-basing at the restored
        # counter would restart the ramp at 0 and re-ramp a critic the run had already ramped
        # in. It passes the interrupted run's own baseline back instead. None = derive it from
        # the checkpoint, which is right for every other warm start.
        self._critic_ramp_baseline = (None if critic_ramp_baseline is None
                                      else int(critic_ramp_baseline))
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
        if online_critic and not self.train_critic_online and not critic_ckpt:
            # A frozen critic never leaves its initialization, so this would steer the sampler
            # off a randomly initialized network for the whole run.
            raise ValueError(
                "train_critic_online is false and critic_ckpt is null: the critic would stay at "
                "its random initialization and steer the sampler on meaningless values. "
                "Point critic_ckpt at an offline-trained critic, or set train_critic_online "
                "true to train one during the rollouts."
            )

        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        # Rows of one `wrench.*` modality. The env commits one row per primitive step -- the
        # contact wrench averaged over the whole TOPP trajectory that step ran -- so a chunk
        # drains exactly `pi0_step` of them, which is the default. It fixes the critic's obs
        # shape, so it has to match the value a warm-start checkpoint (or an offline rollout
        # dataset) was made with.
        self.wrench_trace_len = int(wrench_trace_len or pi0_step)

        if self.proposal_modalities and not self.uses_online_critic:
            print(f"[pi_model] no critic this run -- ignoring encoder_modalities "
                  f"{list(self.proposal_modalities)} (nothing would consume the proposals)")
            self.proposal_modalities = ()
        if self.proposal_modalities:
            self._init_demo_retriever()

    def _init_demo_retriever(self):
        """Open the demo dataset and index its episodes by RoboTwin task.

        Up front rather than on the first observation: it reads a dataset off disk and matches
        every episode's instruction back to a task, so a bad `repo_id` or a task the dataset
        does not cover should stop the run before it spends an hour of rollouts. Only the
        *bank* is per-episode (`set_language`), and only the encoding in it is expensive.
        """
        cfg = dict(self._demo_retrieval_config)
        repo_id = cfg.pop("repo_id", None)
        if not repo_id:
            raise ValueError(
                f"the critic's encoder_modalities names {list(self.proposal_modalities)}, which "
                f"needs a demo dataset: set `demo_retrieval.repo_id` in deploy_policy.yml to the "
                f"LeRobot dataset the demonstrations come from."
            )
        if not self.task_name:
            raise ValueError("demo retrieval needs `task_name` to pick demonstrations of the right task.")
        cfg.pop("enabled", None)
        cfg = {k: v for k, v in cfg.items() if v is not None}
        self.demo_retriever = DemoRetriever(
            self.policy._model,
            self.policy._input_transform,
            repo_id=repo_id,
            invert="noise_proposals" in self.proposal_modalities,
            # A demo dataset that carries `observation.wrench.*` can serve a bank row's own
            # contact trace as a cross-attention key, and the width has to be the one the
            # critic's `wrench.*` modality was built with -- i.e. this run's, not a second
            # setting inside the retrieval block (which may still override it deliberately).
            wrench_trace_len=cfg.pop("wrench_trace_len", self.wrench_trace_len),
            # Only consulted when `sensor_modalities` is left unset: the recorded columns to
            # load are then whichever ones this critic asks for, so a depth camera does not have
            # to be named twice (once for the encoder, once to have it loaded).
            critic_modalities=self._critic_wanted_modalities(),
            **cfg,
        )
        episodes = self.demo_retriever.episodes_for_task(self.task_name)
        print(f"[pi_model] retrieval similarity: {self.demo_retriever.describe_similarity()}")
        print(f"[pi_model] proposal action space: {self.demo_retriever.describe_action_space()}")
        print(f"[pi_model] demo retrieval: {list(self.proposal_modalities)} from {repo_id} "
              f"({len(episodes)} episodes of {self.task_name}), "
              f"{self.demo_retriever.num_demos} demo(s) per episode, top_k="
              f"{self.demo_retriever.top_k}, views="
              f"{[v.split('.', 1)[1] for v in self.demo_retriever.views]}"
              + (f", inverted at num_steps={self.demo_retriever.num_steps} x "
                 f"{self.demo_retriever.num_inner_steps} fixed-point iterations"
                 if self.demo_retriever.invert else ", no inversion (action_proposals only)"))
        print(f"[pi_model] demo rows can be keys for: "
              f"{list(self.demo_retriever.cotrain_modalities)}")

    def _critic_wanted_modalities(self) -> tuple[str, ...]:
        """Every modality name this critic's config asks for, keys included.

        The union of `encoder_modalities` (what it encodes at all, which is what a co-training
        row must supply) and any explicit `key_modalities` on a cross-attending proposal spec
        (what a retrieved row must supply). It is a superset on purpose -- the retriever
        intersects it with the columns the demo dataset actually has.

        Read from the config, so it is known before the critic is built; a warm start can still
        override the architecture from its checkpoint, which `_init_critic` re-checks. A proposal
        spec written as a plain `true` carries no `key_modalities`, and the critic's own default
        for those is `siglip.*` + `state`, which every demo dataset serves anyway.
        """
        modalities = self._critic_config.get("encoder_modalities") or {}
        if isinstance(modalities, dict):
            names = [name for name, spec in modalities.items() if spec]
            specs = [spec for spec in modalities.values() if isinstance(spec, dict)]
        else:
            names, specs = list(modalities), []
        keys = [k for spec in specs for k in (spec.get("key_modalities") or ())]
        return tuple(dict.fromkeys([*names, *keys]))

    @staticmethod
    def _configured_proposals(critic_config):
        """Which of `PROPOSAL_MODALITIES` the critic's `encoder_modalities` asks for.

        Read from the critic config rather than from the run's observation modalities, because a
        proposal is not one: `envs/utils/obs_modalities.py` cannot produce it, so validating it
        against what the sim offers would reject every run. The list is accepted as a mapping
        (`{name: bool}`, how `cfgs/qmfm.yaml` writes it) or as a plain sequence of names.

        Read from the *config* and not from the built critic because the retriever is opened up
        front -- a bad `repo_id` should stop the run before an hour of rollouts, and the critic
        does not exist until the first observation. The two can disagree in one case: a warm
        start takes its architecture keys from the checkpoint, so `critic_ckpt` can override
        `encoder_modalities`. `_init_critic` re-checks against the critic that was actually
        built and turns retrieval back off if it does not read the proposals after all.
        """
        modalities = (critic_config or {}).get("encoder_modalities") or ()
        enabled = (
            [name for name, on in modalities.items() if on]
            if isinstance(modalities, dict)
            else list(modalities)
        )
        return tuple(name for name in PROPOSAL_MODALITIES if name in enabled)

    @staticmethod
    def _siglip_views(collect_siglip):
        """`collect_siglip` -> the SigLIP modalities to record, in the policy's camera order.

        `true` takes every view the policy sees (head plus both wrists for aloha agilex),
        `false` none. A list records a subset -- each 256x1152 fp16 map is ~576 KB/row, so the
        three of them are what dominates a rollout dataset's size -- named either the short way
        (`head`, `left_wrist`) or as the modality itself (`siglip.head`).
        """
        views = tuple(SIGLIP_MODALITIES.values())
        if collect_siglip is None or isinstance(collect_siglip, bool):
            return views if collect_siglip else ()
        if isinstance(collect_siglip, str):
            collect_siglip = [collect_siglip]
        wanted = {v if str(v).startswith("siglip.") else f"siglip.{v}" for v in collect_siglip}
        if unknown := sorted(wanted - set(views)):
            raise ValueError(
                f"collect_siglip names camera view(s) the policy does not have: {unknown}. "
                f"Available: {[v.split('.', 1)[1] for v in views]} (or true / false)."
            )
        return tuple(view for view in views if view in wanted)

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
                siglip = (f"with SigLIP patch features: {', '.join(self.collect_siglip)}"
                          if self.collect_siglip else "no SigLIP")
                print(f"[pi_model] recording model-space critic observations for dataset "
                      f"collection (action chunk={chunk}, sampler noise="
                      f"{horizon}x{self.model_config.action_dim}, {siglip})")
            return

        if self.demo_retriever is not None:
            # The proposals are model-produced, so unlike the sim's sensor modalities they are
            # not in `critic_obs_extra` yet -- the first one is retrieved on the first
            # get_action. Seeding them here puts them in `obs_shapes` (and so in
            # `_critic_extra_shapes`) alongside everything else, each at the width
            # `_proposal_width` keeps it at -- which is not the same for the two of them.
            for name in self.proposal_modalities:
                self.critic_obs_extra[name] = np.zeros(
                    (self.demo_retriever.top_k, horizon, self._proposal_width(name)),
                    dtype=np.float32,
                )

        cc = dict(self._critic_config)
        # State is the model-space state narrowed back to the embodiment's own dims, exactly as
        # the sampler emits it (Pi0.sample_actions::critic_observation) and as the collected
        # `observation.state.model` column stores it -- not the padded action_dim.
        cc["state_dim"] = self.critic_action_dim
        # What the critic's Q takes as its *action*, which is what distinguishes the two
        # families. For qmfm it is the chunk the sampler produces, in embodiment dims. For DSRL
        # it is the latent that chunk was denoised from, and that lives in the model's own
        # padded `action_dim` (32): the trailing dims are dead in an *action* but not in a
        # *noise*, since all 32 go through `action_in_proj` and shape the 14 that come out.
        # The full latent is `horizon x action_dim` (50 x 32 = 1600), far too wide for SAC to
        # act in, so the actor acts in a low-rank family inside it and `noise_param` says which
        # one: `noise_horizon` rows held out to the horizon (dsrl_pi0's scheme), or the factors
        # of a rank-`noise_rank` product. `dsrl.noise_latent_shape` is the one place that turns
        # that choice into a width.
        if self.critic_type == "dsrl":
            cc["noise_action_dim"] = int(self.model_config.action_dim)
            cc["sampler_horizon"] = horizon
            cc["action_dim_flat"] = int(np.prod(noise_latent_shape(cc, horizon)))
        else:
            cc["action_dim_flat"] = horizon * self.critic_action_dim
        cc["siglip_channels"] = 1152
        cc["siglip_grid"] = 16
        # Everything the critic *may* condition on this run: the modalities the sampler produces
        # itself -- the state and one SigLIP patch map per camera the policy is given, the wrist
        # views as well as the head -- plus whichever sensors the task config's `data_type`
        # turned on (they are in `critic_obs_extra` because the first observation has already
        # been handed in). Which of them it actually uses is decided downstream, by the critic's
        # own `encoder_modalities` config -- this side just declares what is on offer, and the
        # critic raises if it was configured for something the sim is not producing.
        siglip_shape = (cc["siglip_grid"], cc["siglip_grid"], cc["siglip_channels"])
        cc["obs_shapes"] = {
            **{view: siglip_shape for view in SIGLIP_MODALITIES.values()},
            "state": (self.critic_action_dim,),
            **{key: tuple(np.shape(value)) for key, value in self.critic_obs_extra.items()},
        }
        self.online_critic = load_critic(cc, self._critic_ckpt)
        if self.demo_retriever is not None and not set(self.proposal_modalities) & set(self.online_critic.obs_keys):
            # The data_type flags only *offer* the proposals; the critic's own
            # `encoder_modalities` decides whether it reads them, exactly as for every other
            # modality. Retrieval is the most expensive thing in a control step, so a critic
            # that ignores them must not be charged for them -- and a config that turned the
            # flags on and forgot the encoder side should hear about it rather than pay
            # silently.
            print(f"[pi_model] critic does not read {list(self.proposal_modalities)} "
                  f"(encoder_modalities: {list(self.online_critic.obs_keys)}) -- turning demo "
                  f"retrieval off for this run")
            self.demo_retriever = None
            for name in PROPOSAL_MODALITIES:
                self.critic_obs_extra.pop(name, None)
            self.proposal_modalities = ()
        elif self.demo_retriever is not None:
            # How the critic consumes the pool: attended (the distance only shortlists) or
            # pooled (the distance is the whole selection). Per modality, since the two
            # proposal forms can be configured with different encoders.
            attends = getattr(self.online_critic, "proposal_modalities", ())
            for name in self.proposal_modalities:
                candidates = f"the {self.demo_retriever.top_k} retrieved candidate(s)"
                print(f"[pi_model] {name}: the critic "
                      + (f"cross-attends {candidates} -- keys from "
                         f"{list(self.online_critic.proposal_keys[name])}, query from all of "
                         f"{list(self.online_critic.obs_keys)}"
                         if name in attends else f"pools {candidates} (max+mean over the set)"))
        self._attach_demo_cotrain()
        # The candidates' own observations, one `(top_k, *shape)` array per key modality of
        # every proposal set the critic cross-attends. Unlike everything else here these are not
        # a modality the run offers -- the critic decides which of them it needs and how wide
        # they are (`OnlineValueCritic.obs_shapes`), and `_refresh_demo_proposals` fills them in
        # per control step from the rows retrieval returned. Seeded now so they take part in
        # `_critic_extra_shapes` like any other per-step array.
        self._proposal_key_modalities = {
            name: tuple(self.online_critic.proposal_keys[name])
            for name in getattr(self.online_critic, "proposal_modalities", ())
            if name in self.proposal_modalities
        }
        # A key has to be something a retrieved demo row can actually serve, which is a
        # property of the demo dataset (its own columns, and which of them this run kept in
        # `demo_retrieval.sensor_modalities`) -- so the critic side cannot check it and this is
        # where it lands. At startup rather than at the first control step, because the whole
        # point of opening the retriever up front is that a bad retrieval config should not cost
        # an hour of rollouts.
        servable = set(self.demo_retriever.key_modalities) if self.demo_retriever else set()
        for name, modalities in self._proposal_key_modalities.items():
            if missing := [m for m in modalities if m not in servable]:
                raise ValueError(
                    f"the critic cross-attends {name!r} with key_modalities {missing}, which a "
                    f"demonstration from {self._demo_retrieval_config.get('repo_id')} cannot "
                    f"serve. It can serve {sorted(servable)}. Either drop them from "
                    f"`key_modalities` (they stay part of the attention *query*), add them to "
                    f"`demo_retrieval.sensor_modalities` if the dataset has the columns, or "
                    f"point `repo_id` at a demo dataset that records them."
                )
        for name, modalities in self._proposal_key_modalities.items():
            for modality in modalities:
                entry = f"{name}.keys.{modality}"
                self.critic_obs_extra[entry] = np.zeros(
                    self.online_critic.obs_shapes[entry], self.online_critic.buffer_dtype(entry)
                )
        # The subset that has to be shipped into the sampler on every call (the other two are
        # built in there). Their shapes are fixed here and enforced per step: a point cloud
        # collected with `pcd_down_sample_num: 0` has a different N every step, which would
        # otherwise surface as an XLA recompile per control step.
        self._critic_extra_shapes = {
            key: self.online_critic.obs_shapes[key]
            for key in self.online_critic.buffer_keys
            if key in self.critic_obs_extra
        }
        # A warm-started critic restores its lifetime update counter from the checkpoint (an
        # offline-trained one is in the hundreds/thousands), so the ramp has to be measured
        # against where *this* run started -- otherwise it reads as already finished and
        # guidance jumps to the target on the very first chunk. The exception is a *resumed*
        # run, whose checkpoint is its own earlier self: it hands its original baseline back
        # (`critic_ramp_baseline`) so the ramp picks up where the interruption left it.
        self._critic_updates_at_start = (int(self.online_critic.num_updates)
                                         if self._critic_ramp_baseline is None
                                         else self._critic_ramp_baseline)

        # A checkpoint's architecture keys override the caller's, so a critic whose shapes do not
        # match (wrong embodiment, wrong action horizon) loads "successfully" and then fails with
        # an opaque dot_general shape error inside sample_actions. Catch it here instead. Note
        # this cannot tell a *raw-space* critic apart from a model-space one: the raw
        # observation.state / action columns have the same widths as their .model counterparts,
        # differing only in normalization. Getting that right is on whoever trains the critic.
        if self.critic_type == "dsrl":
            scored = (f"the {cc['action_dim_flat'] // cc['noise_action_dim']}x"
                      f"{cc['noise_action_dim']} noise chunk pi0.5 denoises from")
            fix = ("Only a DSRL checkpoint of the same embodiment and `noise_horizon` fits.")
        else:
            scored = f"the {chunk} normalized action chunk"
            fix = ("Collect a rollout dataset with collect_critic_obs and train on the "
                   "observation.state.model / action.model columns.")
        for key in ("action_dim_flat", "state_dim"):
            got, want = int(self.online_critic.config[key]), int(cc[key])
            if got != want:
                raise ValueError(
                    f"critic checkpoint {self._critic_ckpt!r} was trained with {key}={got}, but "
                    f"this run feeds {key}={want} (state is the {self.critic_action_dim}-dim "
                    f"model state; the action is {scored}). {fix}"
                )

        if self.critic_type == "dsrl":
            self._wire_dsrl_sampler(cc)
        else:
            self._wire_qmfm_sampler(cc, chunk)
        unused = sorted(set(cc["obs_shapes"]) - set(self.online_critic.obs_keys))
        print(f"[pi_model] critic observation: "
              + ", ".join(f"{k}{tuple(cc['obs_shapes'][k])}" for k in self.online_critic.obs_keys)
              + (f" (available but unused: {', '.join(unused)})" if unused else ""))

    def _demo_source(self, repo_id, critic_modalities=None):
        """A `DemoRetriever` over `repo_id`, reusing the run's own if it is the same dataset.

        Retrieval (§5b) and demo co-training both need the same three things -- the LeRobot
        reader, the policy's input transform and its image tower -- so a run doing both encodes
        through one object. They are separable, though: co-training on demonstrations is useful
        with no proposal modality anywhere in the critic, and the two can even name different
        datasets (the offline config's `dataset.repo_id` against `demo_retrieval.repo_id`), in
        which case a second reader is opened for it.
        """
        if self.demo_retriever is not None and self.demo_retriever.repo_id == repo_id:
            return self.demo_retriever
        # Only the keys that affect *encoding*. The bank knobs (num_demos, top_k, bank_size,
        # frame_stride, the inversion) belong to retrieval and would only constrain a retriever
        # that is never going to build a bank. `sensor_modalities` is one of the encoding keys:
        # which recorded columns are read is a property of the dataset and this critic, not of
        # the bank, and co-training needs exactly the same ones a retrieved row would serve.
        cfg = {
            key: value
            for key, value in self._demo_retrieval_config.items()
            if key in ("root", "encode_batch_size", "seed", "sensor_modalities")
            and value is not None
        }
        return DemoRetriever(
            self.policy._model,
            self.policy._input_transform,
            repo_id=repo_id,
            invert=False,
            wrench_trace_len=self.wrench_trace_len,
            # Left unset (the default), the sensors to load are derived from what the critic
            # asks for -- the same derivation `_init_demo_retriever` gets. Without it a run
            # with no proposal modality never opens the run's own retriever, so this fallback
            # would keep no recorded sensor at all and `cotrain_rows` would refuse every
            # `pointcloud` / `depth.<cam>` / `images.<cam>` the critic encodes, with a message
            # pointing at a `sensor_modalities` this path did not read.
            critic_modalities=(self._critic_wanted_modalities()
                               if critic_modalities is None else tuple(critic_modalities)),
            **cfg,
        )

    def _attach_demo_cotrain(self):
        """Load the critic's offline half from the demonstrations, if that is what it asked for.

        `offline_mix` normally points at a rollout dataset, whose rows the critic can load on its
        own. Pointing it at the **supervised fine-tuning set** instead (`dataset.kind: demo` in
        the offline config) makes the rows the policy's own encoding of demonstration frames --
        the SigLIP maps are this tower's, the state and action chunk this transform's -- so the
        critic cannot build them and defers to here, where the policy exists.

        Filtering to the task under evaluation happens inside: an SFT dataset covers every task
        the policy was fine-tuned on at once, and a demonstration of another task is a different
        MDP wearing the same observation shapes.
        """
        pending = getattr(self.online_critic, "pending_demo_cotrain", None)
        if not pending:
            return
        if not (pending["task"] or self.task_name):
            raise ValueError(
                "demo co-training needs `task_name` to pick demonstrations of the right task; "
                "the eval driver did not pass one and the offline config's `dataset.task` is "
                "unset."
            )
        task = pending["task"] or self.task_name
        # The critic's *actual* obs keys, not the config's: a warm start takes its modalities
        # from the checkpoint, and these are the exact names `cotrain_rows` will be asked for,
        # so deriving what to load from them cannot come up short.
        source = self._demo_source(pending["repo_id"], pending["modalities"])
        print(f"[pi_model] encoding demonstrations of {task!r} from {pending['repo_id']} as "
              f"critic observations {list(pending['modalities'])}, one row per "
              f"{pending['horizon']}-step chunk"
              + ("" if source is self.demo_retriever else " (own reader)"))
        self.online_critic.attach_demo_cotrain(source, run_task=self.task_name)

    def _wire_qmfm_sampler(self, cc, chunk):
        """Point the sampler at the Value critic: gradient guidance and/or best-of-N."""
        self.policy._sample_kwargs.update({
            "critic_apply": self.online_critic.critic_apply,
            # A `None` guidance scale switches the value gradient off entirely, which is what a
            # best-of-N-only run wants: with a traced 0.0 the sampler cannot tell "ramping in
            # from zero" from "never steering", and would keep paying two extra forward passes
            # per denoising step to multiply a gradient by zero. Nonzero targets keep the traced
            # scalar so the ramp costs no recompile.
            "guidance_scale": (jnp.asarray(0.0, dtype=jnp.float32)
                               if self.guidance_scale_target != 0.0 else None),
            "best_of_n": self.best_of_n,
            "critic_action_dim": self.critic_action_dim,
        })
        if not self.train_critic_online:
            ramp = "no ramp (frozen critic), "
        elif self._critic_ramp_baseline is None:
            ramp = (f"guidance_ramp_updates={self.guidance_ramp_updates} (from this run's first "
                    f"TD update), ")
        else:
            # Resumed run: the ramp is already partway along, so say where it comes back at
            # rather than implying it starts here.
            ramp = (f"guidance_ramp_updates={self.guidance_ramp_updates} (resumed at update "
                    f"{int(self.online_critic.num_updates) - self._critic_updates_at_start} of "
                    f"the ramp -> guidance {self.scheduled_guidance_scale():.4g}), ")
        if self.guidance_scale_target == 0.0:
            # Best-of-N only: the sampler is the plain pi0.5 one and the critic never enters a
            # gradient, it only ranks. Say so, rather than printing a guidance target of 0.
            ramp = ""
            steering = "no gradient guidance (guidance_scale 0), "
        else:
            steering = f"guidance_scale_target={self.guidance_scale_target}, "
        select = (f"best-of-{self.best_of_n} (highest ensemble-mean Q per control step), "
                  if self.best_of_n > 1 else "")
        print(f"[pi_model] QMFM Value critic enabled, critic {self._critic_mode()} "
              f"({self._critic_warm()}, {steering}{select}{ramp}"
              f"num_qs={cc['num_qs']}, action chunk={chunk} "
              f"-> action_dim_flat={cc['action_dim_flat']})")

    def _wire_dsrl_sampler(self, cc):
        """Point the sampler at the DSRL actor: it picks the noise, the denoising is untouched.

        No guidance kwargs and no `critic_apply` -- there is nothing for the denoising loop to
        climb, so the sampler stays byte-for-byte the frozen pi0.5 one and only the latent it
        starts from changes. `return_critic_obs` is what carries the actor's choice back out
        (`critic_noise`) along with the observation it was made against.
        """
        self.policy._sample_kwargs.update({
            "noise_apply": self.online_critic.noise_apply,
            "critic_action_dim": self.critic_action_dim,
            "return_critic_obs": True,
        })
        noise_h, noise_d = self.online_critic.action_chunk_shape
        warmup = (f"{self._noise_warmup_left} warmup chunk(s) on Gaussian noise first, "
                  if self._noise_warmup_left else "")
        print(f"[pi_model] DSRL noise-space actor enabled, agent {self._critic_mode()} "
              f"({self._critic_warm()}, {warmup}noise chunk={noise_h}x{noise_d} held out to "
              f"{self.model_config.action_horizon} denoising rows "
              f"-> action_dim_flat={cc['action_dim_flat']}, "
              f"|w| <= {float(self.online_critic.config['action_magnitude']):g})")

    def _critic_warm(self):
        return (f"warm-started from {self._critic_ckpt} at "
                f"{int(self.online_critic.num_updates)} updates" if self._critic_ckpt
                else "from scratch")

    def _critic_mode(self):
        return "trained online by TD" if self.train_critic_online else "FROZEN (no TD updates)"

    @property
    def critic_ramp_baseline(self):
        """The lifetime update count the guidance ramp is measured from.

        Written into `resume_state.json` by the eval driver and handed back on resume, so the
        ramp survives an interruption. 0 until the critic is built (see `_init_critic`).
        """
        return self._critic_updates_at_start

    def scheduled_guidance_scale(self):
        """Guidance ramps 0 -> target over the first `guidance_ramp_updates` TD updates.

        Counted from the start of this run, so a critic warm-started from `critic_ckpt` ramps
        in exactly like one trained from scratch: its values are trained on a different
        (offline) state distribution, so easing the sampler into them is worth doing even
        though the network is not random.

        A frozen critic (`train_critic_online: false`) has no TD updates to count -- the ramp
        would pin guidance at 0 for the entire run -- so it guides at the target from the first
        chunk. Its values never move either, so there is nothing to ease into.

        "This run" spans an interruption: a resumed run is the same run, and is handed the
        original's baseline (`critic_ramp_baseline`), so the ramp continues from where it
        stopped instead of dropping back to 0 for another `guidance_ramp_updates`.
        """
        if self.online_critic is None:
            return 0.0
        if not self.train_critic_online:
            return self.guidance_scale_target
        updates = self.online_critic.num_updates - self._critic_updates_at_start
        if updates <= 0:
            return 0.0
        if self.guidance_ramp_updates <= 0:
            return self.guidance_scale_target
        progress = min(1.0, updates / float(self.guidance_ramp_updates))
        return self.guidance_scale_target * progress

    @contextlib.contextmanager
    def frozen_for_eval(self):
        """Run rollouts that measure this policy without changing it.

        Used by the driver's periodic held-out evaluation (`eval_interval` in
        deploy_policy.yml), which re-runs a fixed set of `eval_seed` episodes every so often and
        makes the resulting success rate the criterion for which critic checkpoint is kept. That
        only means anything if the evaluation is a pure measurement, so for the duration of the
        block:

        * **no transition is stashed** -- the driver already skips `commit`/`train_step`, but the
          stash happens inside `get_action`, so it needs its own switch (`_collect_transitions`).
          A critic that trained on its own held-out episodes would make successive scores
          incomparable, and would leak 10 episodes of the fixed evaluation set into the buffer
          every interval.
        * **the guidance scale is left exactly where the ramp has it.** Deliberately not the
          `train_critic_online: false` path, which jumps guidance straight to the target: what is
          being scored is the policy as it behaves *right now*, not as a later frozen run would.
        * **the DSRL warmup budget and its RNG do not advance.** `_warmup_noise` draws (and
          counts down) once per control step, so without this a run with `noise_warmup_chunks`
          set would spend its warmup on evaluation episodes, and every run's latent sequence
          would depend on how many evaluations had happened.
        * **the debug recorders are switched off and their pending values cleared on the way
          out**, since `QValueRecorder.record` / `BestOfNRecorder.record` consume whatever the
          last `get_action` left -- a value from an evaluation episode would otherwise be logged
          against the first control step of the next training episode.

        What is deliberately *not* frozen is the DSRL actor's stochasticity: `noise_apply` samples
        rather than taking the distribution's mode, exactly as jaxrl2 does in its own eval, so an
        evaluation sees the same policy the rollouts do.
        """
        state = (self._collect_transitions, self._noise_warmup_left,
                 self._noise_rng.bit_generator.state,
                 self.record_q_values, self.record_demo_retrieval)
        self._collect_transitions = False
        self.record_q_values = False
        self.record_demo_retrieval = False
        if self.demo_retriever is not None:
            self.demo_retriever.record_retrieval = False
        try:
            yield
        finally:
            (self._collect_transitions, self._noise_warmup_left,
             self._noise_rng.bit_generator.state,
             self.record_q_values, self.record_demo_retrieval) = state
            if self.demo_retriever is not None:
                self.demo_retriever.record_retrieval = self.record_demo_retrieval
            # Anything the evaluation's last control step left behind. These are read-and-clear
            # on the driver's side, so a leftover would be attributed to the wrong episode.
            self.last_q_values = None
            self.last_best_scores = None
            self.last_best_index = None
            self.last_demo_retrieval = None

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")
        if self.demo_retriever is not None:
            # One draw of demonstrations for the whole run, built on the first episode and held
            # for every later one -- the proposals are a critic input, and re-drawing would move
            # what the critic is conditioned on partway through the eval (see
            # DemoRetriever.ensure_bank). The instruction plays no part in the choice:
            # demonstrations are selected by RoboTwin *task*, and one task's episodes carry
            # hundreds of different instructions (see demo_retrieval).
            self.demo_retriever.record_retrieval = self.record_demo_retrieval
            first = self.demo_retriever.bank is None
            bank = self.demo_retriever.ensure_bank(self.task_name)
            if first:
                print(f"[pi_model] demo bank (fixed for this run): {bank.describe()}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state, critic_obs=None):
        """Set the observation the next `get_action` runs on.

        `critic_obs` is the sim's sensor modalities for this control step (depth / point cloud /
        contact wrench, from `envs/utils/obs_modalities.py`). It is only passed on the call that
        precedes a `get_action`; the refreshes inside the policy's action loop leave the last
        one in place rather than paying to rebuild it for a step that never scores anything.
        """
        if critic_obs is not None:
            self.critic_obs_extra = critic_obs
        if self.critic_action_dim is None:
            # First observation of the run: the embodiment's action width is now known, and so
            # is the set of sensor modalities the task config produces.
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

    def _critic_extra_obs(self):
        """This step's sensor modalities, in the fixed shapes `_init_critic` pinned down.

        Returned unbatched (the sampler's kwargs bypass `Policy.infer`'s batching, so the caller
        adds the leading axis), in the dtype they arrive in -- camera frames stay uint8 rather
        than becoming four times the bytes to cross into XLA -- and NaN and all: an early-ended
        chunk pads its wrench trace with NaN, and mapping that to something finite is the
        encoders' job.

        The shapes checked here are the *sensor's* (`_critic_extra_shapes`, from the critic's
        `obs_shapes`), because that is what the observation is required to arrive as. A critic
        whose `encoder_modalities` downscale a camera then works at a smaller one, so the
        arrays are handed to `to_stored_obs` on the way out -- once, before the caller splits
        them between the sampler and the replay buffer, so both see identical pixels.
        """
        obs = {}
        for key, shape in self._critic_extra_shapes.items():
            value = self.critic_obs_extra.get(key)
            if value is None:
                raise KeyError(
                    f"critic modality {key!r} was present on the first observation but is "
                    f"missing now; the observation only carries "
                    f"{sorted(self.critic_obs_extra)}."
                )
            value = np.asarray(value)
            if value.shape != shape:
                raise ValueError(
                    f"critic modality {key!r} changed shape from {shape} to {value.shape}. "
                    f"A point cloud does this when `pcd_down_sample_num: 0` leaves it "
                    f"un-downsampled -- the critic needs a fixed-size observation."
                )
            obs[key] = value
        return self.online_critic.to_stored_obs(obs)

    def _proposal_width(self, name):
        """The trailing width one proposal modality is kept at. **Not the same for the two.**

        `DemoRetriever.propose` returns both at the model's padded `action_dim` (32), and only
        one of them should be narrowed back to the embodiment's `critic_action_dim`:

        * `action_proposals` -- yes, exactly as the sampler narrows the chunk it scores.
          AlohaInputs zero-pads 14 -> 32 and those trailing dims normalize to constant zero, so
          a demo chunk keeps only the embodiment's own dims and lines up term-for-term with the
          action the critic is judging.
        * `noise_proposals` -- no. This is not an action but the *seed* that denoises into one,
          and a latent has no dead dims even where an action does: all 32 go through
          `action_in_proj` and shape the 14 that come out, which is why `_init_critic` sizes
          DSRL's own latent on `action_dim` rather than on `critic_action_dim`. Whatever
          `invert_actions` recovers in the trailing dims is a real part of the latent that
          produced that chunk, so slicing them off would hand the critic a candidate it could
          not actually seed the sampler with -- and would silently disagree with the space the
          DSRL actor acts in.
        """
        if name == "noise_proposals":
            return int(self.model_config.action_dim)
        return self.critic_action_dim

    def _refresh_demo_proposals(self):
        """Retrieve this control step's demo proposals into the critic's extra observation.

        Each is narrowed on the way in to whatever `_proposal_width` says it is -- the
        embodiment's dims for a demo chunk, the model's full padded `action_dim` for an
        inverted seed.

        Costs one image tower + prefix pass on top of the sampler's own, plus -- when
        `noise_proposals` is on -- `num_steps * num_inner_steps` action-expert passes at batch
        `top_k` for the inversion. That is the dominant cost of the whole control step, which is
        why nothing builds a retriever unless a critic asked for these.
        """
        if self.demo_retriever is None:
            return
        proposals = self.demo_retriever.propose(self.observation_window)
        for name in self.proposal_modalities:
            self.critic_obs_extra[name] = proposals[name][..., : self._proposal_width(name)]
        self._refresh_proposal_keys(proposals["proposal_rows"])
        self.last_demo_retrieval = self.demo_retriever.retrieved()

    def _refresh_proposal_keys(self, rows):
        """Encode the retrieved rows' own observations, for a critic that cross-attends them.

        The keys' side of the attention: the same modalities the critic encodes the live
        observation with, run over the demo frames the distance shortlisted. They are computed
        here, per control step, rather than held for every frame of the bank -- a SigLIP patch
        map is ~0.6 MB per view, so a resident bank of them was ~0.9 GB of device memory that
        grew with `bank_size` (see `demo_retrieval.DemoRetriever.critic_keys`). The cost is one
        image-tower pass at batch `top_k`.

        One pass covers both proposal sets: they are retrieved together and so share their rows,
        and the two may still be configured with different `key_modalities`, so the union is
        encoded once and fanned out.
        """
        if not self._proposal_key_modalities:
            return
        # The offline half of a co-trained batch retrieves against this same bank, ranked once
        # (`OnlineValueCritic.attach_demo_retrieval`). Here rather than at `_init_critic`
        # because the bank is drawn on the first episode, and idempotent by the same flag the
        # proposals themselves are refreshed under.
        if self._offline_retrieval_bank is not self.demo_retriever.bank:
            self.online_critic.attach_demo_retrieval(self.demo_retriever)
            self._offline_retrieval_bank = self.demo_retriever.bank
        wanted = sorted({m for ms in self._proposal_key_modalities.values() for m in ms})
        keys = self.demo_retriever.critic_keys(rows, wanted, state_dim=self.critic_action_dim)
        for name, modalities in self._proposal_key_modalities.items():
            for modality in modalities:
                entry = f"{name}.keys.{modality}"
                # A patch map arrives as the flat `(top_k, 256, 1152)` sequence the tower emits;
                # the critic's encoder takes the `(16, 16, 1152)` grid.
                self.critic_obs_extra[entry] = keys[modality].reshape(
                    self._critic_extra_shapes[entry]
                )

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        if self.online_critic is None:
            out = self.policy.infer(self.observation_window)
            self._stash_critic_obs(out)
            return out["actions"]

        # Inject the current (traced) network params so online updates take effect without an
        # XLA recompile.
        if self.critic_type == "dsrl":
            self.policy._sample_kwargs["actor_params"] = self.online_critic.actor_params
        else:
            # The scheduled guidance scale is traced too, so changing it per call ramps guidance
            # without compiling a sampler for each scalar value.
            self.current_guidance_scale = self.scheduled_guidance_scale()
            self.policy._sample_kwargs["critic_params"] = self.online_critic.params
            if self.guidance_scale_target != 0.0:
                # Left at None for a best-of-N-only run -- see _init_critic. Writing a traced
                # zero here would silently switch the (unused) guidance path back on.
                self.policy._sample_kwargs["guidance_scale"] = jnp.asarray(
                    self.current_guidance_scale, dtype=jnp.float32
                )
        self._refresh_demo_proposals()
        extra = self._critic_extra_obs()
        # The candidates' own observations go in under the names the critic's cross attention
        # reads them by (`OnlineValueCritic.proposal_key_obs`); everything else goes in as it
        # stands, under its own modality name.
        self.policy._sample_kwargs["critic_obs_extra"] = {
            **{key: jnp.asarray(value)[None, ...]
               for key, value in extra.items() if key in self.online_critic.obs_keys},
            **self.online_critic.proposal_key_obs(extra),
        }

        warmup_noise = self._warmup_noise()
        out = self.policy.infer(
            self.observation_window,
            # Expanded by the agent, not here: the latent the warmup draws is in the actor's own
            # parameterization, so what it means as a 50x32 chunk is the agent's to say.
            noise=(None if warmup_noise is None
                   else self.online_critic.expand_noise(warmup_noise)[None]),
        )
        critic_obs = {**out["critic_obs_siglip"], "state": out["critic_obs_state"], **extra}
        if self.best_of_n > 1:
            # Which of the N candidates was executed, and what the critic scored all of them
            # at. `critic_action` below is already the winner, so everything downstream --
            # the replay transition, the Q log -- is about the chunk that actually ran.
            self.last_best_index = int(out["critic_best_index"])
            self.last_best_scores = np.asarray(out["critic_best_scores"], dtype=np.float32)
        # What this critic's Q takes as its action, and therefore what the replay transition and
        # the Q log are about: the normalized chunk the sampler scored, or -- for DSRL -- the
        # latent it denoised that chunk from, which during warmup is the draw made here.
        if self.critic_type != "dsrl":
            critic_action = out["critic_action"]
        else:
            critic_action = out["critic_noise"] if warmup_noise is None else warmup_noise
        # Stash this control step's (obs, action) for the replay buffer. Every SigLIP view
        # the sampler produced is offered; `stash` keeps only the critic's own
        # `encoder_modalities`, so an unused view costs the buffer nothing. A frozen critic
        # skips this entirely -- nothing would ever train on the transitions, and the buffer
        # is allocated lazily, so it never costs the run any memory.
        if self.train_critic_online and self._collect_transitions:
            self.online_critic.stash(critic_obs, critic_action)
        if self.record_q_values:
            self.last_q_values = self.online_critic.q_values(critic_obs, critic_action)
        self._stash_critic_obs(out)
        return out["actions"]

    def _warmup_noise(self):
        """The Gaussian latent for this control step, while a DSRL run is still warming up.

        dsrl_pi0 collects its first trajectory from the base policy's own noise before letting
        SAC choose, so the buffer's first transitions come from the distribution pi0.5 was
        trained to denoise rather than from an untrained tanh actor. Drawn here rather than
        inside the sampler because it is also what gets stashed as the transition's action.

        None once the actor takes over -- and always for every other critic -- which is what
        makes `Pi0.sample_actions` fall through to `noise_apply`. The switch costs one sampler
        recompile, since passing a `noise=` argument at all is what changes.
        """
        if self._noise_warmup_left <= 0:
            return None
        self._noise_warmup_left -= 1
        if self._noise_warmup_left == 0:
            print("[pi_model] DSRL noise warmup over -- the actor picks the latent from the "
                  "next control step (one sampler recompile)")
        return self._noise_rng.standard_normal(self.online_critic.action_chunk_shape).astype(np.float32)

    def _stash_critic_obs(self, out):
        """Keep the last control step's model-space critic view for dataset collection.

        ``critic_obs_state`` (critic_action_dim,) is the normalized model state and
        ``critic_action`` (action_horizon, critic_action_dim) the normalized chunk, both in
        embodiment dims -- the tensors the critic is scored on. ``out["actions"]`` is the same
        chunk after the output transform has unnormalized it.

        ``critic_obs_siglip`` holds the SigLIP patch maps the critic's CNN encoders read, one
        per camera view, as produced by ``Pi0.sample_actions``' own image tower -- so recording
        them here makes the separate ``multisensory_steering.create_dataset siglip`` pass
        unnecessary for every view, not just the head. Each is stored under its modality name
        (``siglip.head``, ``siglip.left_wrist``, ...) as the flat ``(256, 1152)`` patch sequence
        that pass wrote (the encoders reshape to the 16x16 grid themselves) and in fp16,
        matching the online replay buffer's ``stash``.

        ``sample_noise`` (``noise`` here) is the flow-matching latent the action expert denoised
        that chunk from: ``(action_horizon, action_dim)``, i.e. the model's **padded** 32 dims
        rather than the embodiment's 14, since all of them feed ``action_in_proj``. It rides
        along with the rest unconditionally -- at 6.4 KB/row it is ~0.6% of a row against the
        576 KB a single SigLIP view costs, and unlike the maps it is a *draw* rather than a
        function of the observation, so a second pass over the dataset could not add it later.
        """
        if not self.collect_critic_obs or "critic_action" not in out:
            return
        self.last_critic_obs = {
            "state": np.asarray(out["critic_obs_state"], dtype=np.float32),
            "action": np.asarray(out["critic_action"], dtype=np.float32),
            "noise": np.asarray(out["sample_noise"], dtype=np.float32),
        }
        maps = out.get("critic_obs_siglip") or {}
        self.last_critic_obs["siglip"] = {
            view: np.asarray(maps[view], dtype=np.float16).reshape(-1, np.shape(maps[view])[-1])
            for view in self.collect_siglip
            if view in maps
        }

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
