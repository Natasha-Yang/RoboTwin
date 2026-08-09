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

        If ``critic_apply``/``critic_params`` are provided, the denoising is steered by QMFM's
        exact denoised-estimate gradient guidance (``QMFM/agents/mfm.py::compute_flow_actions``,
        ``steer_use_denoised_estimate=True``): at each step the clean action chunk is estimated
        (``x1 = x_t - t*v``), the value gradient ``grad_V = d/d(x_t) mean_k Q(obs, x1)`` is taken
        **through** the velocity field, rescaled to the velocity norm, and ``guidance_scale``
        (QMFM's ``steering_coeff``) times it steers the velocity. ``critic_apply(params, obs,
        action)`` is the JAX apply of the QMFM ``Value`` ensemble (``multisensory_steering``); ``params``
        is a traced pytree (so online critic updates need no recompile), ``guidance_scale`` is a
        traced scalar (so online schedules do not recompile per value), and ``critic_apply`` is a
        static arg. Returns ``(actions, {"critic_obs_siglip", "critic_obs_state",
        "critic_action"})`` for online replay-buffer collection, where ``critic_obs_siglip`` is
        itself a ``{modality: patch map}`` dict, one entry per camera view. This path takes
        precedence over ``return_features``.

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

        ``return_critic_obs`` returns that same aux dict from the **unguided** sampler, so
        rollout-dataset collection records critic training data (model-space state and action
        chunk, plus the SigLIP patch maps) in exactly the space the guided path scores. Both the
        returned ``critic_action`` (the full-horizon chunk, still *normalized*) and
        ``critic_obs_state`` are narrowed to ``critic_action_dim`` embodiment dims -- i.e.
        ``(action_horizon, 14)`` and ``(14,)`` for aloha, versus the unnormalized chunk
        ``Policy.infer``'s output transform produces.
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
        sample_batch = batch_size * n

        def tile(tree):
            """Replicate each batch element n times, contiguously: [a a a b b b] for n=3.

            That is the layout `reshape(batch_size, n, ...)` undoes, which is how the candidates
            are grouped back per batch element for the argmax below.
            """
            return tree if n == 1 else jax.tree.map(lambda x: jnp.repeat(x, n, axis=0), tree)

        if noise is None:
            noise = jax.random.normal(rng, (sample_batch, self.action_horizon, self.action_dim))
        elif noise.shape[0] != sample_batch:
            raise ValueError(
                f"noise has batch {noise.shape[0]}, expected {sample_batch} "
                f"(batch {batch_size} x best_of_n {n}). Best-of-N needs independent noise per "
                f"candidate -- replicating one seed would draw the same chunk n times."
            )

        # first fill KV cache with a forward pass of the prefix. The image tower runs here (once
        # per camera view) and its raw patch features are what the critic sees, so they are taken
        # from this pass rather than re-encoding a frame further down.
        image_tokens, image_encoded = self.embed_images(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, image_tokens=image_tokens)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

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
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                suffix_obs, x_t, jnp.broadcast_to(time, sample_batch)
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
                sample_batch,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
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

        def critic_aux(critic_obs, x_0):
            """What the caller gets back: the model-produced modalities plus the scored chunk.

            Only the model's own modalities are echoed back -- the caller passed
            `critic_obs_extra` in, so it already has the rest (and they are numpy on the host,
            not worth a round trip). The SigLIP maps come back as a `{modality: array}` dict so
            the replay buffer and the dataset collector can key them by view.
            """
            return {
                "critic_obs_siglip": {
                    name: critic_obs[name] for name in SIGLIP_MODALITIES.values() if name in critic_obs
                },
                "critic_obs_state": critic_obs["state"],
                "critic_action": critic_action_view(x_0),
            }

        def denoise(step_fn):
            """Integrate `step_fn` from t=1 (noise) down to t=0, returning the clean chunk."""

            def cond(carry):
                _, time = carry
                # robust to floating-point error
                return time >= -dt / 2

            x_0, _ = jax.lax.while_loop(cond, step_fn, (noise, 1.0))
            return x_0

        if has_critic:
            # `critic_obs` is at the caller's batch (it is what the aux dict reports and what the
            # replay buffer stores); `critic_obs_n` is the same observation replicated once per
            # best-of-N candidate, which is what the sampler scores against.
            critic_obs = critic_observation()
            critic_obs_n = tile(critic_obs)

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
                return best, {**critic_aux(critic_obs, best), **picked}

        def step(carry):
            x_t, time = carry
            v_t = self.action_out_proj(action_expert_features(x_t, time))
            return x_t + dt * v_t, time + dt

        if steer:
            # QMFM-exact denoised-estimate gradient guidance
            # (QMFM/agents/mfm.py::compute_flow_actions, steer_use_denoised_estimate=True).

            def guided_step(carry):
                x_t, time = carry
                v_t = self.action_out_proj(action_expert_features(x_t, time))

                def x1_estimate(a):
                    # Clean action estimate (flow target at t=0): x1 = x_t - t*v, differentiated
                    # THROUGH the velocity field (QMFM). QMFM clips to [-1, 1]; pi0.5 normalized
                    # actions are ~standardized (not hard-bounded), so we skip the clip.
                    v = self.action_out_proj(action_expert_features(a, time))
                    return (a - time * v).astype(jnp.float32)

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
                    jnp.linalg.norm(gf, axis=-1, keepdims=True) + 1e-9)
                grad = (scale * gf).reshape(grad.shape)
                # QMFM ascends Q via v + steering_coeff*grad while integrating t=0->1. Here time
                # runs t=1->0 with dt<0, so the step dt*(-grad) moves the sample along +grad
                # (uphill on the critic). `guidance_scale` is QMFM's steering_coeff (>0 ascends Q).
                return x_t + dt * (v_t - guidance_scale * grad), time + dt

            return finish(denoise(guided_step))

        if has_critic:
            # Best-of-N with no steering: plain pi0.5 denoising for every candidate, and the
            # critic is only read to rank the chunks it produced. Cheaper per candidate than the
            # guided path, which pays two extra forward passes per denoising step for the value
            # gradient.
            return finish(denoise(step))

        if not return_features:
            x_0 = denoise(step)
            if not return_critic_obs:
                return x_0
            # Unguided sampling, but emit the critic's view of this step so rollout collection
            # can record critic training data in exactly the space the guided path scores.
            return x_0, critic_aux(critic_observation(), x_0)

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
