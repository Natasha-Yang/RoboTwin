import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# The critic's name for each camera view the model sees. These are the names RoboTwin's own
# sensor modalities and the rollout dataset's columns use (`envs/utils/obs_modalities.py`), so
# `siglip.left_wrist` means the same view online (inside the sampler) as it does in a dataset
# column a critic was pretrained on. A view the embodiment does not have is filled in by the
# input transform as a black frame -- it still gets encoded, and still reads as one.
SIGLIP_MODALITIES = {
    "base_0_rgb": "siglip.head",
    "left_wrist_0_rgb": "siglip.left_wrist",
    "right_wrist_0_rgb": "siglip.right_wrist",
}

# The same names in a fixed order, which is what an array of per-view embeddings is stacked
# along (see `embed_observation` / `propose_from_demos`). A dict's order is an implementation
# detail everywhere else in this file; here it is part of the data layout, so it gets a name.
SIGLIP_VIEWS = tuple(SIGLIP_MODALITIES.values())


def _relative_l2(dot: at.Array, query_sq: at.Array, bank_sq: at.Array) -> at.Array:
    """``||q - b|| / ||q||``, from the expanded ``||q||^2 + ||b||^2 - 2 q.b``.

    Clipped at zero before the root: the expansion can go slightly negative in float32 for
    near-identical vectors, and a NaN there would poison the whole ranking. The denominator is
    floored too -- a query vector that is genuinely zero would otherwise divide by zero, and
    flooring makes that signal read as a plain unscaled distance rather than an infinity that
    swallows the average.
    """
    distance = jnp.sqrt(jnp.maximum(query_sq + bank_sq - 2.0 * dot, 0.0))
    return distance / jnp.maximum(jnp.sqrt(query_sq), 1e-6)


def pool_siglip(image_encoded: dict[str, at.Array]) -> dict[str, at.Float[at.Array, "b emb"]]:
    """Reduce each view's patch map to one vector, keyed by its critic modality name.

    The mean over the 256 patches, and nothing else. Retrieval ranks by **relative L2 distance**
    (`propose_from_demos`), which divides by the magnitude of the *query's* own vector rather
    than putting each vector on the unit sphere -- so the magnitude is part of the signal here
    and must not be normalized away. Views the model does not expose to the critic are dropped.
    """
    return {
        SIGLIP_MODALITIES[name]: jnp.mean(encoded.astype(jnp.float32), axis=-2)
        for name, encoded in image_encoded.items()
        if name in SIGLIP_MODALITIES
    }


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def hold_noise(noise: at.Array, action_horizon: int) -> at.Array:
    """Hold a short noise chunk's last row out to the sampler's full action horizon.

    DSRL's actor acts in a latent space of its own choosing, and it is cheaper to learn than
    the sampler's: ``dsrl_pi0`` (``examples/train_utils_sim.py``) has SAC pick a single
    ``(1, 32)`` noise row per control step and repeats it across all 50 denoising rows rather
    than making the policy a 1600-dimensional one. Any chunk length up to the horizon works the
    same way; one already at full length passes straight through.
    """
    short = int(noise.shape[1])
    if short == action_horizon:
        return noise
    if short > action_horizon:
        raise ValueError(f"noise chunk is {short} rows, longer than the action horizon {action_horizon}.")
    return jnp.concatenate([noise, jnp.repeat(noise[:, -1:], action_horizon - short, axis=1)], axis=1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def embed_images(self, obs: _model.Observation) -> tuple[dict[str, at.Array], dict[str, at.Array]]:
        """Run the SigLIP image tower once per camera view.

        Returns both of the tower's outputs per view: the tokens projected to the LLM width,
        which is all the prefix needs, and the raw pre-projection patch features
        (``aux["encoded"]``, 1152-d) the critic conditions on, which ``embed_prefix`` otherwise
        throws away. Keeping both here is what lets a run that wants the critic's view of every
        camera pay for the tower exactly once (see ``sample_actions``).
        """
        tokens, encoded = {}, {}
        for name, image in obs.images.items():
            image_tokens, aux = self.PaliGemma.img(image, train=False)
            tokens[name] = image_tokens
            encoded[name] = aux["encoded"]
        return tokens, encoded

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation, *, image_tokens: dict[str, at.Array] | None = None
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images -- reusing the tower output the caller already ran, if it kept one
        # (`sample_actions` does, since the critic wants the same tower's patch features).
        if image_tokens is None:
            image_tokens, _ = self.embed_images(obs)
        for name in obs.images:
            view_tokens = image_tokens[name]

            tokens.append(view_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=view_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * view_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _prefix_pass(self, obs: _model.Observation) -> tuple[list, at.Bool[at.Array, "b p"], int, dict[str, at.Array]]:
        """Run the image tower and the prefix forward once, and return the KV cache it fills.

        The prefix depends on neither ``x_t`` nor the timestep, so every denoising path pays for
        it exactly once per call and the per-step suffix passes just attend to this cache. Also
        returns the tower's raw patch features (see ``embed_images``), which the critic
        conditions on, and the prefix length ``_action_expert_features`` checks its mask against.
        """
        image_tokens, image_encoded = self.embed_images(obs)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs, image_tokens=image_tokens)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return kv_cache, prefix_mask, prefix_tokens.shape[1], image_encoded

    def _action_expert_features(
        self, obs: _model.Observation, kv_cache, prefix_mask, prefix_len: int, x_t, time
    ) -> at.Float[at.Array, "b ah emb"]:
        """One denoising forward pass: the action-expert features, pre-``action_out_proj``.

        ``obs``/``prefix_mask``/``kv_cache`` must all be at ``x_t``'s batch -- best-of-N widens
        them together before calling in (see ``sample_actions``).
        """
        batch = x_t.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            obs, x_t, jnp.broadcast_to(time, batch)
        )
        # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
        # other
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
        # prefix tokens
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
        # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch,
            suffix_tokens.shape[1],
            prefix_len + suffix_tokens.shape[1],
        )
        # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return suffix_out[:, -self.action_horizon :]

    def _invert_from_prefix(
        self,
        obs: _model.Observation,
        kv_cache,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_len: int,
        actions: _model.Actions,
        *,
        num_steps: int,
        num_inner_steps: int,
        num_substeps: int,
    ) -> tuple[_model.Actions, at.Float[at.Array, " s"]]:
        """The inversion itself, against a prefix that has already been run.

        Split out of ``invert_actions`` so a caller that has already paid for the image tower and
        the prefix pass under this observation can invert against that cache instead of a second
        one -- ``propose_from_demos`` inverts several retrieved chunks that way. ``obs``,
        ``prefix_mask`` and ``kv_cache`` must be at ``actions``' batch, as for
        ``_action_expert_features``.

        Returns the recovered noise and the per-(sub)step fixed-point residual, in the order the
        steps were undone. See ``invert_actions`` for what the iteration is and why.
        """
        dt = -1.0 / num_steps
        dt_sub = dt / num_substeps

        # Every timestep to be undone, in the order the forward pass would visit them. The
        # denoising step's own `t_k` is accumulated the way the forward loop accumulates it -- it
        # carries `time` as a float32 scalar starting at 1.0 and adds the Python float `dt`, so
        # rebuilding the schedule in float64 would put each inverse step at a slightly different
        # t than its forward step used. With `num_substeps > 1` each step contributes its own
        # sub-grid on top of that, offset from the same `t_k`.
        times = []
        time = jnp.asarray(1.0, dtype=jnp.float32)
        for _ in range(num_steps):
            times.extend((time + m * dt_sub).astype(jnp.float32) for m in range(num_substeps))
            time = (time + dt).astype(jnp.float32)
        times = jnp.stack(times)

        def invert_step(x_next, time):
            """Undo the single forward (sub)step taken at `time`, by fixed-point iteration.

            The iterate is seeded with `x_next` itself, so the first pass is the DDIM-style
            explicit guess and each further one refines it.
            """

            def fixed_point(_, carry):
                x_t, _ = carry
                v_t = self.action_out_proj(
                    self._action_expert_features(obs, kv_cache, prefix_mask, prefix_len, x_t, time)
                )
                x_new = x_next - dt_sub * v_t
                return x_new, jnp.max(jnp.abs(x_new - x_t))

            x_t, residual = jax.lax.fori_loop(
                0, num_inner_steps, fixed_point, (x_next, jnp.asarray(jnp.inf, dtype=jnp.float32))
            )
            return x_t, residual

        # Reverse order: the last forward step is the first one undone.
        return jax.lax.scan(invert_step, actions, times[::-1])

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        return_features: bool = False,
        critic_apply=None,
        critic_params=None,
        critic_obs_extra: dict[str, jax.Array] | None = None,
        noise_apply=None,
        actor_params=None,
        guidance_scale: float | at.Float[at.Array, ""] | None = 0.0,
        best_of_n: int = 1,
        return_critic_obs: bool = False,
        critic_action_dim: int | None = None,
    ) -> _model.Actions | tuple[_model.Actions, dict[str, jax.Array]]:
        """Sample an action chunk via flow-matching denoising.

        If ``return_features`` is True, additionally return the raw action-expert features
        (``suffix_out[:, -action_horizon:]``, i.e. the hidden states feeding ``action_out_proj``,
        before the velocity projection) recorded at every denoising step. This is what SAFE
        (https://vla-safe.github.io/) consumes for uncertainty / failure detection. The return
        becomes ``(actions, {"action_features": feats})`` where ``feats`` has shape
        ``(batch, num_steps, action_horizon, feature_dim)``.

        If ``critic_apply``/``critic_params`` are provided, the denoising is steered by Universal
        Guidance (Bansal et al. 2023, adapted for flow matching), matching
        ``steering-with-failures``' ``steered_ode.py::make_universal_guidance_fn``: at each step
        the clean action chunk is estimated by Tweedie (``x1 = clip(x_t - t*v, -1, 1)``), the
        value gradient ``grad_V = d/d(x_t) mean_k Q(obs, x1)`` is taken **through** the velocity
        field, rescaled to the velocity norm (QMFM's ``steer_use_sigma_t=False``, Eq 129), and
        ``guidance_scale`` (QMFM's ``steering_coeff``) times it steers the velocity. The ``t=1``
        step is taken **unsteered**: there ``x_t`` is pure noise, so its Tweedie estimate carries
        no signal worth differentiating. The returned chunk is clipped to ``[-1, 1]`` as well --
        pi0.5 normalizes actions by quantiles, so that is the action range (see ``x1_estimate``).
        Both clips are on this path only; the unguided and best-of-N-only paths are untouched.

        ``critic_apply(params, obs, action)`` is the JAX apply of the QMFM ``Value`` ensemble
        (``multisensory_steering``); ``params``
        is a traced pytree (so online critic updates need no recompile), ``guidance_scale`` is a
        traced scalar (so online schedules do not recompile per value), and ``critic_apply`` is a
        static arg. Returns ``(actions, {"critic_obs_siglip", "critic_obs_state",
        "critic_action", "sample_noise"})`` for online replay-buffer collection, where
        ``critic_obs_siglip`` is itself a ``{modality: patch map}`` dict, one entry per camera
        view, and ``sample_noise`` is the ``(b, action_horizon, action_dim)`` latent the
        returned chunk was denoised from. This path takes precedence over ``return_features``.

        ``best_of_n > 1`` draws that many candidate chunks from independent noise and returns
        the one the critic scores highest (``mean_k Q``, the same aggregation the guidance
        ascends), one winner per batch element. It is orthogonal to the guidance: with a
        ``guidance_scale`` each candidate is steered independently and the best of the steered
        chunks is kept, so the two compose. It needs a critic for the ranking, so it requires
        ``critic_apply``/``critic_params`` even when nothing is being steered -- pass
        ``guidance_scale=None`` for that (best-of-N only), which skips the value gradient
        entirely rather than multiplying it by a constant zero. The aux dict then also carries
        ``"critic_best_scores"`` ``(b, n)`` and ``"critic_best_index"`` ``(b,)``.

        The N candidates ride along as extra batch elements, replicated *after* the prefix pass,
        so the SigLIP tower and the prefix forward are still paid exactly once per control step
        and only the KV cache and the denoising loop scale with N. ``best_of_n`` is static:
        changing it recompiles.

        ``obs`` is a ``{modality: array}`` dict. This model produces the state and one SigLIP
        map per camera view itself -- ``"state"``, ``"siglip.head"``, ``"siglip.left_wrist"``,
        ``"siglip.right_wrist"`` (see ``critic_observation``) -- and anything else the critic
        conditions on comes from outside the model, through ``critic_obs_extra``: the sensor
        modalities the sim observation carries (depth maps, point cloud, contact wrench; see
        ``envs/utils/obs_modalities.py``), already **batched** and already narrowed to the keys
        that critic actually wants. Which keys those are is the critic's business, not this
        sampler's -- it just merges the dict and hands it over. Changing the set of keys changes
        the pytree structure and therefore recompiles, so it must stay fixed for a run.

        ``noise_apply``/``actor_params`` are the third way a critic can act on sampling, and the
        only one that does not touch the denoising at all: DSRL replaces the Gaussian latent the
        chunk is denoised *from* with one a noise-space SAC actor chose
        (``multisensory_steering.critics.dsrl``). ``noise_apply(params, obs, key)`` returns
        ``(latent, noise)``: the ``(b, *action_chunk_shape)`` latent the actor chose and the
        ``(b, action_horizon, action_dim)`` chunk it expands to. The two differ because the
        sampler's latent is 1600 numbers and SAC acts in a low-rank family inside it -- a few
        rows held out to the horizon (`hold_noise`), or the factors of a low-rank product.
        `dsrl.expand_noise` does the expanding, so which family it is stays the agent's
        business rather than the sampler's. It is called after the prefix pass, so the actor
        conditions on the same SigLIP maps and state the critic does, and the *latent* comes
        back in the aux dict as ``"critic_noise"`` (what the DSRL replay buffer stores as the
        action). ``noise_apply``
        is a static arg and ``actor_params`` a traced pytree, as ``critic_apply``/
        ``critic_params`` are. An explicit ``noise=`` wins over it -- that is how a warmup phase
        feeds the plain Gaussian.

        ``return_critic_obs`` returns that same aux dict from the **unguided** sampler, so
        rollout-dataset collection records critic training data (model-space state and action
        chunk, the SigLIP patch maps, and the noise the chunk came from) in exactly the space
        the guided path scores. Both the returned ``critic_action`` (the full-horizon chunk,
        still *normalized*) and ``critic_obs_state`` are narrowed to ``critic_action_dim``
        embodiment dims -- i.e. ``(action_horizon, 14)`` and ``(14,)`` for aloha, versus the
        unnormalized chunk ``Policy.infer``'s output transform produces. ``sample_noise`` is
        **not** narrowed (see ``critic_aux``): every one of the padded ``action_dim`` noise dims
        feeds ``action_in_proj`` and shapes the embodiment dims that come out, so dropping the
        tail would leave a seed that no longer reproduces the chunk.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        has_critic = critic_apply is not None and critic_params is not None
        # Gradient guidance is on unless it is explicitly switched off with `guidance_scale=None`
        # -- a *traced* 0.0 still steers (that is what the online ramp's first chunks are), so
        # the two cases cannot be told apart by value. None is how a best-of-N-only run says
        # "rank with the critic, but do not differentiate through it".
        steer = has_critic and guidance_scale is not None
        # Best-of-N candidates, sampled as extra batch elements.
        n = 1 if best_of_n is None else max(1, int(best_of_n))
        if n > 1 and not has_critic:
            raise ValueError(
                f"best_of_n={n} needs a critic to rank the candidates with, but no "
                f"critic_apply/critic_params were given."
            )
        if n > 1 and noise_apply is not None:
            # The ranking is over denoised chunks, so the winner's *latent* would have to be
            # selected alongside it for `critic_noise` to mean anything. Nothing needs that:
            # a DSRL run ranks nothing (its Q is over the latent, see pi_model.CRITIC_TYPES).
            raise ValueError(
                f"best_of_n={n} cannot be combined with a noise-space actor: the selection is "
                f"over the chunks, so the latent reported back would not be the winner's."
            )
        sample_batch = batch_size * n

        def tile(tree):
            """Replicate each batch element n times, contiguously: [a a a b b b] for n=3.

            That is the layout `reshape(batch_size, n, ...)` undoes, which is how the candidates
            are grouped back per batch element for the argmax below.
            """
            return tree if n == 1 else jax.tree.map(lambda x: jnp.repeat(x, n, axis=0), tree)

        if noise is not None and noise.shape[0] != sample_batch:
            raise ValueError(
                f"noise has batch {noise.shape[0]}, expected {sample_batch} "
                f"(batch {batch_size} x best_of_n {n}). Best-of-N needs independent noise per "
                f"candidate -- replicating one seed would draw the same chunk n times."
            )

        # first fill KV cache with a forward pass of the prefix. The image tower runs here (once
        # per camera view) and its raw patch features are what the critic sees, so they are taken
        # from this pass rather than re-encoding a frame further down.
        kv_cache, prefix_mask, prefix_len, image_encoded = self._prefix_pass(observation)

        # Best-of-N replicates the *conditioning*, not the work that produced it: the tower and
        # the prefix pass above ran once at batch `batch_size`, and only the cache they filled is
        # widened so the N candidates can attend to it. `KVCache` is stacked over layers by the
        # scan in gemma.Module, so its batch axis is 1, not 0 (`l b t k h`).
        if n > 1:
            kv_cache = jax.tree.map(lambda x: jnp.repeat(x, n, axis=1), kv_cache)
            prefix_mask = tile(prefix_mask)
        # The observation the *suffix* is embedded against. Tiled as a whole rather than field by
        # field so it stays a well-formed Observation (its batch dims are typechecked); the
        # images in it are unused from here on and get dropped by DCE. `observation` itself stays
        # at the original batch, which is the one `critic_observation` and the aux dict want.
        suffix_obs = tile(observation)

        def action_expert_features(x_t, time):
            """Run one denoising forward pass and return the action-expert features (pre-projection)."""
            return self._action_expert_features(suffix_obs, kv_cache, prefix_mask, prefix_len, x_t, time)

        # The critic works in *embodiment* dims, not the model's padded `action_dim`: AlohaInputs
        # zero-pads 14 -> 32 on the way in, for both the state and the action chunk, and those
        # trailing dims normalize to constant zero, so feeding them to the critic only widens its
        # input with dead weights. `critic_action_dim` is the unpadded width (14 for aloha
        # agilex, taken from the sim's own joint vector); None keeps the full padded tensors.
        critic_ad = self.action_dim if critic_action_dim is None else int(critic_action_dim)

        def critic_observation():
            """The `{modality: array}` observation the Value critic conditions on.

            The modalities the model itself supplies are one SigLIP patch map per camera view
            it was given -- the raw `aux["encoded"]` (1152-d) the prefix pass above computed and
            embed_prefix discards, reshaped to the 16x16 patch grid the CNN encoder expects, and
            named `siglip.head` / `siglip.left_wrist` / `siglip.right_wrist` (SIGLIP_MODALITIES)
            -- plus the model-space (normalized) state, narrowed to the same `critic_ad`
            embodiment dims as the action chunk below. `critic_obs_extra` adds the sim's own
            sensor modalities on top.

            All views are offered whatever the critic ends up reading: they are already computed
            (the prefix needs them), and which ones are actually encoded is the critic's own
            configuration. The ones it ignores cost nothing here -- they are dead code inside the
            guidance gradient -- only the round trip in `aux`.
            """
            state = observation.state.astype(jnp.float32)[..., :critic_ad]
            siglip = {
                SIGLIP_MODALITIES[name]: einops.rearrange(
                    encoded.astype(jnp.float32), "b (h w) c -> b h w c", h=16, w=16
                )
                for name, encoded in image_encoded.items()
                if name in SIGLIP_MODALITIES
            }
            return {**siglip, "state": state, **(critic_obs_extra or {})}

        def critic_action_view(actions):
            """The normalized `(b, action_horizon, critic_ad)` chunk the critic is scored on."""
            return actions[..., :critic_ad]

        def critic_aux(critic_obs, x_0, seed=None):
            """What the caller gets back: the model-produced modalities plus the scored chunk.

            Only the model's own modalities are echoed back -- the caller passed
            `critic_obs_extra` in, so it already has the rest (and they are numpy on the host,
            not worth a round trip). The SigLIP maps come back as a `{modality: array}` dict so
            the replay buffer and the dataset collector can key them by view.

            `seed` overrides which noise chunk is reported as `sample_noise` -- best-of-N passes
            the winning candidate's, so the seed describes the chunk actually being returned.
            """
            return {
                "critic_obs_siglip": {
                    name: critic_obs[name] for name in SIGLIP_MODALITIES.values() if name in critic_obs
                },
                "critic_obs_state": critic_obs["state"],
                "critic_action": critic_action_view(x_0),
                # The `(b, action_horizon, action_dim)` chunk the flow was actually integrated
                # from: the sampler's own Gaussian draw, unless the caller supplied one or an
                # actor chose it. Everything else here is a function of the observation and
                # could be recomputed from a stored frame; this is the draw that made the chunk
                # *this* sample rather than another, and once the sampler has returned only
                # `invert_actions` recovers it -- `num_steps * num_inner_steps` action-expert
                # passes, and only for the unguided sampler. Rollout collection records it as the
                # dataset's `action.noise`, which is the action of a noise-space agent's MDP
                # (DSRL, sec 5c). Kept at the model's padded `action_dim` rather than narrowed to
                # `critic_action_dim` like the chunk above: the trailing dims of an *action*
                # normalize to constant zero, but every dim of a *noise* goes through
                # `action_in_proj` and shapes the embodiment dims that come out, so a narrowed
                # seed would no longer reproduce its chunk.
                "sample_noise": noise if seed is None else seed,
                # The same latent in the *actor's* own parameterization, when one chose it --
                # `noise_horizon` rows, or the factors of a low-rank product, which `expand_noise`
                # turned into the `sample_noise` above. That, not the expansion, is the action of
                # DSRL's MDP and what its replay buffer stores; the caller cannot recover it,
                # since the choice was made in here.
                **({} if noise_chunk is None else {"critic_noise": noise_chunk}),
            }

        # `critic_obs` is at the caller's batch (it is what the aux dict reports and what a
        # replay buffer stores); `critic_obs_n` is the same observation replicated once per
        # best-of-N candidate, which is what the sampler scores against. Both are built whether
        # or not anything reads them: they are reshapes of tensors the prefix pass has already
        # computed, so an unread one is traced and then eliminated.
        critic_obs = critic_observation()
        critic_obs_n = tile(critic_obs)

        noise_chunk = None
        if noise is None:
            if noise_apply is None:
                noise = jax.random.normal(rng, (sample_batch, self.action_horizon, self.action_dim))
            else:
                # DSRL: the latent is *chosen* by a noise-space actor rather than drawn, and the
                # frozen policy is left alone. Deliberately here, after the prefix pass: the
                # actor sees the same observation the critic scores -- SigLIP maps included --
                # which is exactly what would not be available if the noise had to be picked
                # before the sampler was called.
                rng, noise_rng = jax.random.split(rng)
                # Two different objects: the latent the actor *chose* (the transition's
                # action, returned as `critic_noise`) and the chunk it expands to, which is
                # what the flow is actually seeded with. They coincide only when the actor
                # picks all `action_horizon` rows outright.
                noise_chunk, noise = noise_apply(actor_params, critic_obs_n, noise_rng)

        def denoise(step_fn, first_step_fn=None):
            """Integrate `step_fn` from t=1 (noise) down to t=0, returning the clean chunk.

            `first_step_fn` takes the t=1 step instead, when the first step is special: the
            guided path takes it unsteered (see `guided_step`). `cond` is on the time carried
            in the loop rather than on a step counter, so the total is `num_steps` integrator
            steps either way -- one of them just runs ahead of the loop.
            """

            def cond(carry):
                _, time = carry
                # robust to floating-point error
                return time >= -dt / 2

            carry = (noise, 1.0)
            if first_step_fn is not None:
                carry = first_step_fn(carry)
            x_0, _ = jax.lax.while_loop(cond, step_fn, carry)
            return x_0

        if has_critic:

            def score(x_0):
                """mean_k Q for every candidate chunk -- `(sample_batch,)`."""
                chunk = critic_action_view(x_0).reshape(x_0.shape[0], -1)
                return critic_apply(critic_params, critic_obs_n, chunk).mean(axis=0)

            def pick_best(x_0):
                """Best-of-N: keep the highest-scoring candidate per batch element.

                Ranked by the same `mean_k Q` the guidance ascends, so with both switched on the
                selection agrees with what the steering was trying to do rather than pulling
                against it. Scored once, after denoising -- the guided path's own gradients are
                taken at the intermediate x_t, not at the chunk it lands on.
                """
                scores = score(x_0).reshape(batch_size, n)
                idx = jnp.argmax(scores, axis=-1)
                candidates = x_0.reshape(batch_size, n, *x_0.shape[1:])
                best = jnp.take_along_axis(candidates, idx[:, None, None, None], axis=1)[:, 0]
                return best, {"critic_best_scores": scores, "critic_best_index": idx}

            def finish(x_0):
                """Pick the winner (best-of-N only) and build the aux dict the caller gets."""
                if n == 1:
                    return x_0, critic_aux(critic_obs, x_0)
                best, picked = pick_best(x_0)
                # The winner's own seed, grouped and gathered exactly as the candidates were, so
                # `sample_noise` still denoises into the chunk that came back rather than into
                # some discarded candidate's.
                seeds = noise.reshape(batch_size, n, *noise.shape[1:])
                seed = jnp.take_along_axis(
                    seeds, picked["critic_best_index"][:, None, None, None], axis=1
                )[:, 0]
                return best, {**critic_aux(critic_obs, best, seed=seed), **picked}

        def step(carry):
            x_t, time = carry
            v_t = self.action_out_proj(action_expert_features(x_t, time))
            return x_t + dt * v_t, time + dt

        if steer:
            # Universal Guidance (Bansal et al. 2023) for flow matching, matching
            # steering-with-failures' steered_ode.py::make_universal_guidance_fn. The gradient
            # rescale below is QMFM's (agents/mfm.py::compute_flow_actions).

            def guided_step(carry):
                x_t, time = carry
                v_t = self.action_out_proj(action_expert_features(x_t, time))

                def x1_estimate(a):
                    # Tweedie estimate of the clean action (flow target at t=0): x1 = x_t - t*v,
                    # differentiated THROUGH the velocity field.
                    #
                    # Clipped to [-1, 1] because that IS pi0.5's action range: for a pi05 model
                    # `DataConfigFactory.create_base_config` sets `use_quantile_norm=True`
                    # (training/config.py), so actions go through `Normalize._normalize_quantile`
                    # -- q01 maps to -1 and q99 to +1. An estimate outside the box is off the
                    # manifold the critic was fitted on. The clip zeroes the gradient in a
                    # saturated dim, which is the point: the critic gets no say in pushing an
                    # action further past the range it has ever seen.
                    v = self.action_out_proj(action_expert_features(a, time))
                    return jnp.clip((a - time * v).astype(jnp.float32), -1.0, 1.0)

                def value_fn(a):
                    # grad_V = d/d(x_t) mean_k Q(obs, x1(x_t)); .sum() over batch keeps per-sample grads.
                    # Only the embodiment dims are scored, so the padded tail gets zero gradient.
                    chunk = critic_action_view(x1_estimate(a))
                    qs = critic_apply(critic_params, critic_obs_n, chunk.reshape(a.shape[0], -1))
                    return qs.mean(axis=0).sum()

                grad = jax.grad(value_fn)(x_t).astype(v_t.dtype)

                # Rescale the value gradient to the velocity norm (QMFM steer_use_sigma_t=False,
                # Eq 129) over the WHOLE flattened action chunk per sample -- matches QMFM's
                # chunked axis=-1 norm rather than normalizing each timestep independently.
                gf = grad.reshape(grad.shape[0], -1)
                vf = v_t.reshape(v_t.shape[0], -1)
                scale = jnp.linalg.norm(vf, axis=-1, keepdims=True) / (
                    jnp.linalg.norm(gf, axis=-1, keepdims=True) + 1e-9
                )
                grad = (scale * gf).reshape(grad.shape)
                # QMFM ascends Q via v + steering_coeff*grad while integrating t=0->1. Here time
                # runs t=1->0 with dt<0, so the step dt*(-grad) moves the sample along +grad
                # (uphill on the critic). `guidance_scale` is QMFM's steering_coeff (>0 ascends Q).
                return x_t + dt * (v_t - guidance_scale * grad), time + dt

            # The t=1 step runs the plain `step`: x_t is pure noise there, so `x1_estimate` is a
            # Tweedie estimate from nothing and its gradient is noise the critic would be asked
            # to follow anyway. The final chunk is clipped to the same [-1, 1] the estimate is,
            # and clipped BEFORE `finish` so the chunk that gets executed, the one best-of-N
            # ranks and the `critic_action` the replay buffer stores are all the same array.
            return finish(jnp.clip(denoise(guided_step, first_step_fn=step), -1.0, 1.0))

        if has_critic:
            # Best-of-N with no steering: plain pi0.5 denoising for every candidate, and the
            # critic is only read to rank the chunks it produced. Cheaper per candidate than the
            # guided path, which pays two extra forward passes per denoising step for the value
            # gradient.
            return finish(denoise(step))

        if not return_features:
            x_0 = denoise(step)
            if not return_critic_obs and noise_chunk is None:
                return x_0
            # Unguided sampling, but emit the critic's view of this step so rollout collection
            # can record critic training data in exactly the space the guided path scores -- and
            # so a DSRL run gets back the latent its actor picked, which only exists in here.
            return x_0, critic_aux(critic_obs, x_0)

        # Feature-recording path: accumulate the action-expert features from every denoising step.
        # `num_steps` must be a static Python int here (it is on the eval/rollout path).
        feature_dim = self.action_out_proj.in_features
        feats_init = jnp.zeros((batch_size, num_steps, self.action_horizon, feature_dim), dtype=jnp.float32)

        def step_feat(carry):
            x_t, time, feats, i = carry
            features = action_expert_features(x_t, time)
            feats = feats.at[:, i].set(features.astype(feats.dtype))
            v_t = self.action_out_proj(features)
            return x_t + dt * v_t, time + dt, feats, i + 1

        def cond_feat(carry):
            _, time, _, _ = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _, feats, _ = jax.lax.while_loop(cond_feat, step_feat, (noise, 1.0, feats_init, 0))
        return x_0, {"action_features": feats}

    def invert_actions(
        self,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        num_steps: int = 10,
        num_inner_steps: int = 10,
        num_substeps: int = 1,
        return_info: bool = False,
    ) -> _model.Actions | tuple[_model.Actions, dict[str, jax.Array]]:
        """Reverse the flow: recover the noise that ``sample_actions`` would denoise into ``actions``.

        The exact inverse of the sampler, not an approximation of it: run this on a clean chunk
        and hand the result back as ``sample_actions(..., noise=<this>)`` and the same chunk comes
        out, to the arithmetic precision of the forward pass (see the precision note). ``num_steps``
        must be the number of denoising steps the forward pass uses, and ``observation`` the same
        conditioning; the map being inverted is the one *those* pin down.

        ``actions`` is in the sampler's own space -- the **normalized**, ``action_dim``-padded
        chunk ``sample_actions`` returns, i.e. before ``Policy.infer``'s output transform. To
        invert an action recorded in robot units, push it back through that transform first.

        How it works. The sampler integrates t=1 (noise) -> t=0 with explicit Euler at a fixed
        step ``dt = -1/num_steps``::

            x_{k+1} = x_k + dt * v(x_k, t_k),    t_k = 1 + k*dt

        so undoing one step means solving ``x_k + dt*v(x_k, t_k) = x_{k+1}`` for ``x_k`` -- an
        *implicit* equation, since v is evaluated at the unknown. It has no closed form, so it is
        solved by fixed-point iteration, run ``num_inner_steps`` times per step::

            x <- x_{k+1} - dt * v(x, t_k),   starting from x = x_{k+1}

        whose fixed point is by construction exactly the ``x_k`` the forward step started from.
        The iteration contracts at rate ``|dt| * Lip(v)``, comfortably < 1 at the sampler's step
        sizes, so ~4 iterations already reach 1e-4 and 8-10 hit the float32 floor. Stopping at one
        iteration is the usual DDIM-style *approximate* inversion, whose O(dt^2)-per-step error
        is exactly what the remaining iterations remove.

        The steps are undone in the reverse of the order the sampler took them, at the timesteps
        it actually visited -- ``t_k`` is accumulated in float32 from 1.0 exactly as the forward
        while_loop accumulates it, so each inverse step is taken at the same value its forward
        step used.

        ``num_substeps`` splits each denoising step into that many sub-intervals of ``dt/num_substeps``
        and inverts each one (still by the fixed-point iteration above, at the sub-interval's own
        time and step size). This is a **different trade**, not more accuracy: the fine grid
        inverts the underlying ODE rather than the coarse Euler map ``sample_actions`` actually
        applies, and the two differ by the integrator's own O(dt^2) truncation error -- so
        ``num_substeps > 1`` makes the round trip through the *coarse* sampler worse, not better.
        Reach for it only when the fixed point at the full ``dt`` will not contract (a very small
        ``num_steps`` against a stiff velocity field): each sub-interval's contraction rate is
        ``num_substeps`` times smaller, which is what buys convergence back. The default ``1``
        is the exact inverse of the sampler and is what the round-trip guarantee above refers to.

        Cost is ``num_steps * num_substeps * num_inner_steps`` action-expert passes against one
        prefix pass (the prefix depends on neither x_t nor t, so the image tower still runs once).

        Only the plain sampler is inverted -- not the critic-guided path, whose steps depend on
        the critic's parameters at the time, and not best-of-N, which is not injective (the
        losing candidates' noise is unrecoverable from the winning chunk).

        A precision note: the round trip is only as exact as the forward pass is deterministic
        and reproducible, and on this model that floor is set by matmul precision rather than by
        the fixed point. Measured on a dummy-width pi0.5 at ``num_steps=10``, ``num_inner_steps=8``:
        ~4e-7 in float32 under ``jax_default_matmul_precision="highest"``, ~4e-4 in float32 with
        the GPU default (tf32 matmuls), ~2e-3 at the config default ``dtype="bfloat16"``. The
        fixed point converges below all three; raising ``num_inner_steps`` past the point where
        the residual stops falling buys nothing.

        With ``return_info``, also returns ``{"residual": (num_steps * num_substeps,)}`` -- the max
        absolute size of the last fixed-point update at each inverted (sub)step, in the order they
        were undone (the step at t closest to 0 first). It bounds the remaining error up to the
        contraction factor, so a residual that has not fallen to the precision floor means
        ``num_inner_steps`` was too small.
        """
        if not isinstance(num_steps, int):
            raise TypeError(f"num_steps must be a static Python int for inversion, got {type(num_steps).__name__}.")
        if num_steps < 1 or num_inner_steps < 1 or num_substeps < 1:
            raise ValueError(
                f"num_steps, num_inner_steps and num_substeps must be >= 1, got "
                f"{num_steps}, {num_inner_steps} and {num_substeps}."
            )

        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        expected = (batch_size, self.action_horizon, self.action_dim)
        if actions.shape != expected:
            raise ValueError(
                f"actions has shape {actions.shape}, expected {expected}. `invert_actions` takes the "
                f"sampler's own normalized, action_dim-padded chunk."
            )
        kv_cache, prefix_mask, prefix_len, _ = self._prefix_pass(observation)
        noise, residual = self._invert_from_prefix(
            observation,
            kv_cache,
            prefix_mask,
            prefix_len,
            actions,
            num_steps=num_steps,
            num_inner_steps=num_inner_steps,
            num_substeps=num_substeps,
        )

        if not return_info:
            return noise
        return noise, {"residual": residual}

    def embed_observation(self, observation: _model.Observation) -> dict[str, at.Float[at.Array, "b emb"]]:
        """One pooled SigLIP vector per camera view -- the retrieval key for an observation.

        Runs the image tower and nothing else: no prefix pass, no action expert. That is what
        makes encoding a whole demonstration episode affordable (one tower pass per frame),
        and it is the same tower, on the same resized frames, that ``sample_actions`` feeds the
        critic from -- so a bank encoded here is directly comparable to the embedding
        ``propose_from_demos`` takes of the live observation.

        Keyed by the ``siglip.<view>`` modality names the critic and the rollout datasets use.
        Pooling and normalization are in ``pool_siglip``.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        _, image_encoded = self.embed_images(observation)
        return pool_siglip(image_encoded)

    def embed_observation_maps(self, observation: _model.Observation) -> dict[str, at.Float[at.Array, "b p emb"]]:
        """The un-pooled SigLIP **patch map** per camera view, keyed by critic modality name.

        The same tower pass ``embed_observation`` runs, without the `pool_siglip` at the end:
        this is the array the critic's own image encoder takes (`multisensory_steering`'s
        ``siglip.<view>`` modality), so a demo frame encoded here can be pushed through the
        critic's encoder exactly as the live observation is -- which is what a retrieval that
        cross-attends with the critic's own representation needs, rather than the pooled
        vector an L2 distance is enough for.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        _, image_encoded = self.embed_images(observation)
        return {
            SIGLIP_MODALITIES[name]: encoded
            for name, encoded in image_encoded.items()
            if name in SIGLIP_MODALITIES
        }

    def propose_from_demos(
        self,
        observation: _model.Observation,
        demo_embeddings: at.Float[at.Array, "m v emb"],
        demo_actions: at.Float[at.Array, "m ah ad"],
        *,
        demo_mask: at.Bool[at.Array, " m"] | None = None,
        demo_extra: dict[str, at.Float[at.Array, "m d"]] | None = None,
        query_extra: dict[str, at.Float[at.Array, "b d"]] | None = None,
        top_k: int = 1,
        views: tuple[str, ...] = SIGLIP_VIEWS,
        invert: bool = True,
        num_steps: int = 10,
        num_inner_steps: int = 10,
        num_substeps: int = 1,
        return_info: bool = False,
    ) -> dict[str, jax.Array]:
        """Retrieve the demo frames most like this observation, and the noise behind their actions.

        The demo bank is ``m`` frames of demonstrations of the same task (see
        ``openpi.policies.demo_retrieval``), already in the model's own space:
        ``demo_embeddings`` are ``embed_observation`` vectors stacked along ``views`` in that
        order, and ``demo_actions`` the **normalized**, ``action_dim``-padded chunks that were
        the policy's training targets at those frames. Both are computed once per episode; this
        runs per control step.

        Three steps, in one pass over the prefix:

        1. Embed the live observation the same way. The image tower has to run for the prefix
           anyway, so the retrieval key is free -- ``_prefix_pass`` hands back the patch features
           ``embed_prefix`` would otherwise discard.
        2. Measure the **L2 distance** to every demo frame and take the ``top_k`` *nearest*.
           Every signal gets its own independent distance and they are then averaged with equal
           weight: one term per camera view in ``views``, plus one per key of ``demo_extra`` --
           the non-visual signals the caller measures on the host (the robot's pose; see
           ``openpi.policies.demo_retrieval``), passed alongside the query's own values as
           ``query_extra`` under the same keys.

           Each term is **relative**: ``||q - b|| / ||q||``, the distance as a fraction of the
           magnitude of the current observation's own vector for that signal. That is what
           makes terms averageable without unit-normalizing anything -- a 1152-d SigLIP view
           whose vectors have norm ~24 and a 32-d pose whose vectors have norm ~2.5 both come
           out O(0.1-1) instead of an order of magnitude apart -- and the scale comes from the
           query, so nothing here has to know what a given signal means or gather statistics
           over the bank. ``demo_mask`` marks padding slots in a fixed-size bank as ineligible.

           Unlike a cosine this is sensitive to magnitude, which is the point: two poses
           pointing the same way but ten times apart in size are the same to a cosine and far
           apart here.
        3. ``invert_actions`` each retrieved chunk **under the live observation** -- the noise
           that would make *this* observation's sampler produce a demo-like action here. The
           inversion reuses the prefix from step 1, so ``top_k`` proposals cost
           ``num_steps * num_inner_steps`` action-expert passes at batch ``top_k``, and one
           tower + prefix pass in total.

        Returns ``{"action_proposals": (b, k, ah, ad)}``, plus ``"noise_proposals"`` of the same
        shape when ``invert``. The two are the same retrieval seen from either end of the flow --
        the clean chunks, and the seeds that map to them -- and either one alone is a valid thing
        to condition a critic on, which is why ``invert`` is separate: skipping it drops by far
        the larger half of the cost.

        With ``return_info``, also returns ``"distance"`` ``(b, m)`` over the whole bank,
        ``"indices"`` and ``"scores"`` ``(b, k)`` for what was chosen (``scores`` being those
        rows' distances, smallest first), and the inversion's ``"residual"`` when it ran.

        Note the asymmetry between the two outputs: the action proposals are exactly the demo
        chunks, unchanged by anything here, while the noise proposals are only as exact as the
        inversion (``invert_actions``' precision note applies, and its fixed point is being run
        at whatever ``num_inner_steps`` the caller pays for). ``num_steps`` must match the
        sampler's, or the noise inverts a different map than the one it will be fed to.
        """
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a static Python int >= 1, got {top_k!r}.")
        if demo_embeddings.shape[0] != demo_actions.shape[0]:
            raise ValueError(
                f"demo bank is inconsistent: {demo_embeddings.shape[0]} embeddings but "
                f"{demo_actions.shape[0]} action chunks."
            )
        if top_k > demo_actions.shape[0]:
            raise ValueError(f"top_k={top_k} exceeds the {demo_actions.shape[0]}-frame demo bank.")
        if demo_embeddings.shape[1] != len(views):
            raise ValueError(
                f"demo_embeddings has {demo_embeddings.shape[1]} views but `views` names "
                f"{len(views)}: {list(views)}. The bank must be stacked in this order."
            )
        if (demo_extra is None) != (query_extra is None) or set(demo_extra or {}) != set(query_extra or {}):
            raise ValueError(
                f"demo_extra and query_extra must carry the same signals; got "
                f"{sorted(demo_extra or {})} and {sorted(query_extra or {})}."
            )
        for name, vectors in (demo_extra or {}).items():
            if vectors.shape[0] != demo_actions.shape[0]:
                raise ValueError(
                    f"demo_extra[{name!r}] has {vectors.shape[0]} rows, expected "
                    f"{demo_actions.shape[0]} to match the bank."
                )
            if vectors.shape[-1] != query_extra[name].shape[-1]:
                raise ValueError(
                    f"demo_extra[{name!r}] is {vectors.shape[-1]}-d but query_extra[{name!r}] is "
                    f"{query_extra[name].shape[-1]}-d."
                )

        expected = (self.action_horizon, self.action_dim)
        if demo_actions.shape[1:] != expected:
            raise ValueError(
                f"demo_actions has chunk shape {demo_actions.shape[1:]}, expected {expected}. The "
                f"bank holds the sampler's own normalized, action_dim-padded chunks."
            )

        observation = _model.preprocess_observation(None, observation, train=False)
        batch = observation.state.shape[0]

        kv_cache, prefix_mask, prefix_len, image_encoded = self._prefix_pass(observation)
        pooled = pool_siglip(image_encoded)
        if missing := [view for view in views if view not in pooled]:
            raise ValueError(f"observation has no {missing} view(s); it carries {sorted(pooled)}.")
        current = jnp.stack([pooled[view] for view in views], axis=1)

        # Both sides are L2-normalized per view, so this is the mean cosine similarity across
        # views -- one scalar per (batch element, demo frame).
        # One relative L2 distance per view plus one per extra signal, averaged with equal
        # weight. Summed rather than meaned per group first, so a camera view and the pose count
        # the same -- which is what "average everything" means and is not what weighting the
        # visual and non-visual halves equally would give.
        query_sq = jnp.sum(jnp.square(current), axis=-1)[:, None, :]
        total = _relative_l2(
            jnp.einsum("bvc,mvc->bmv", current, demo_embeddings),
            query_sq,
            jnp.sum(jnp.square(demo_embeddings), axis=-1)[None, :, :],
        ).sum(axis=-1)
        for name in sorted(demo_extra or {}):
            query, bank = query_extra[name], demo_extra[name]
            total = total + _relative_l2(
                jnp.einsum("bd,md->bm", query, bank),
                jnp.sum(jnp.square(query), axis=-1)[:, None],
                jnp.sum(jnp.square(bank), axis=-1)[None, :],
            )
        distance = total / (len(views) + len(demo_extra or {}))
        if demo_mask is not None:
            distance = jnp.where(demo_mask[None, :], distance, jnp.inf)
        # top_k of the negated distance = the k nearest, still ordered best-first.
        scores, indices = jax.lax.top_k(-distance, top_k)
        scores = -scores
        actions = jnp.take(demo_actions, indices, axis=0)

        out: dict[str, jax.Array] = {"action_proposals": actions}
        residual = None
        if invert:
            # The k candidates are extra batch elements against the prefix that is already
            # cached, exactly as best-of-N widens the sampler: `_prefix_pass` ran once at
            # `batch`, and only the cache is replicated. KVCache is stacked over layers, so its
            # batch axis is 1 (`l b t k h`), not 0.
            def tile(tree):
                return tree if top_k == 1 else jax.tree.map(lambda x: jnp.repeat(x, top_k, axis=0), tree)

            wide_kv = kv_cache if top_k == 1 else jax.tree.map(lambda x: jnp.repeat(x, top_k, axis=1), kv_cache)
            noise, residual = self._invert_from_prefix(
                tile(observation),
                wide_kv,
                tile(prefix_mask),
                prefix_len,
                actions.reshape(batch * top_k, *expected),
                num_steps=num_steps,
                num_inner_steps=num_inner_steps,
                num_substeps=num_substeps,
            )
            out["noise_proposals"] = noise.reshape(batch, top_k, *expected)

        if return_info:
            out |= {"distance": distance, "indices": indices, "scores": scores}
            if residual is not None:
                out["residual"] = residual
        return out
