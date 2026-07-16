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


def _top_principal_direction(centered: jax.Array, n_iter: int = 15) -> jax.Array:
    """Top right singular vector of ``centered`` ``(C, D)`` via power iteration on ``C^T C``.

    A backend-agnostic, jit-friendly stand-in for the toy example's closed-form 2-D PCA
    (``critic_ensemble_toy_example/critics.py::CriticCluster``), generalized to the flattened
    pi0.5 action dim. Deterministic init from the largest-norm centered row; degenerate
    (all-equal) gradients collapse to a zero direction, which the caller handles.
    """
    v = centered[jnp.argmax(jnp.linalg.norm(centered, axis=-1))]
    v = v / (jnp.linalg.norm(v) + 1e-12)

    def body(_, vv):
        w = centered.T @ (centered @ vv)
        return w / (jnp.linalg.norm(w) + 1e-12)

    return jax.lax.fori_loop(0, n_iter, body, v)


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

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

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
        guidance_scale: float | at.Float[at.Array, ""] = 0.0,
        cluster: bool = False,
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
        action)`` is the JAX apply of the QMFM ``Value`` ensemble (``qmfm_critic.py``); ``params``
        is a traced pytree (so online critic updates need no recompile), ``guidance_scale`` is a
        traced scalar (so online schedules do not recompile per value), and ``critic_apply`` is a
        static arg. Returns ``(actions, {"critic_obs_img", "critic_obs_state", "critic_action"})``
        for online replay-buffer collection. This path takes precedence over ``return_features``.

        If ``cluster`` is True, the ensemble gradient is instead formed with the ``CriticCluster``
        algorithm (``critic_ensemble_toy_example/critics.py::CriticCluster.action_gradient``),
        ported to JAX: the per-critic value gradients are split into k=2 clusters by the sign of
        their projection onto the top principal direction, the pessimistic-Q steering gradient
        (``mean - 0.5*std/sqrt(count)``) of EACH cluster is computed through the velocity, and the
        one whose gradient best aligns (cosine) with the base velocity ``v_t`` is kept.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def action_expert_features(x_t, time):
            """Run one denoising forward pass and return the action-expert features (pre-projection)."""
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
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
                batch_size,
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

        if critic_apply is not None and critic_params is not None:
            # QMFM-exact denoised-estimate gradient guidance
            # (QMFM/agents/mfm.py::compute_flow_actions, steer_use_denoised_estimate=True).
            # The Value critic conditions on the head-camera SigLIP patch features (the raw
            # `aux["encoded"]`, 1152-d), which the image tower produces but embed_prefix
            # discards -- so re-run it once here (constant across denoising steps) and reshape
            # to the 16x16 patch grid the CNN encoder expects.
            state = observation.state.astype(jnp.float32)
            _, img_aux = self.PaliGemma.img(observation.images["base_0_rgb"], train=False)
            siglip = img_aux["encoded"].astype(jnp.float32)  # (b, 256, 1152)
            siglip_map = einops.rearrange(siglip, "b (h w) c -> b h w c", h=16, w=16)
            critic_obs = (siglip_map, state)

            def guided_step(carry):
                x_t, time = carry
                v_t = self.action_out_proj(action_expert_features(x_t, time))

                def x1_estimate(a):
                    # Clean action estimate (flow target at t=0): x1 = x_t - t*v, differentiated
                    # THROUGH the velocity field (QMFM). QMFM clips to [-1, 1]; pi0.5 normalized
                    # actions are ~standardized (not hard-bounded), so we skip the clip.
                    v = self.action_out_proj(action_expert_features(a, time))
                    return (a - time * v).astype(jnp.float32)

                if cluster:
                    # CriticCluster gradient guidance (critic_ensemble_toy_example/critics.py::
                    # CriticCluster.action_gradient), ported to JAX.
                    bsz = x_t.shape[0]

                    # (1) Per-critic gradients d Q_c / d x1 at the current x1 (direct critic input;
                    # critic-only backward -- this is the clustering decision, so stop-gradient).
                    x1_flat = jax.lax.stop_gradient(x1_estimate(x_t)).reshape(bsz, -1)  # (b, adf)

                    def qc_sum(x1f):
                        return critic_apply(critic_params, critic_obs, x1f).sum(axis=1)  # (C,)

                    g = jax.jacrev(qc_sum)(x1_flat)  # (C, b, adf)
                    g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)  # unit per critic

                    # (2) k=2 clustering per sample: sign of the projection onto the top principal
                    # direction of the centered unit gradients.
                    def cluster_labels(gs):  # gs: (C, adf) -> (C,) bool
                        centered = gs - gs.mean(axis=0, keepdims=True)
                        return (centered @ _top_principal_direction(centered)) >= 0

                    labels = jax.vmap(cluster_labels, in_axes=1)(g).T  # (C, b)
                    mask_a = labels.astype(jnp.float32)
                    mask_b = 1.0 - mask_a

                    # (3) Pessimistic-Q steering gradient (THROUGH the velocity) over a critic subset.
                    def pessimistic_grad(mask):  # mask: (C, b), a stop-grad constant
                        def value_fn(a):
                            qs = critic_apply(critic_params, critic_obs, x1_estimate(a).reshape(a.shape[0], -1))
                            cnt = jnp.maximum(mask.sum(axis=0), 1.0)  # (b,)
                            mean = (mask * qs).sum(axis=0) / cnt
                            var = (mask * (qs - mean[None]) ** 2).sum(axis=0) / cnt
                            return (mean - 0.5 * jnp.sqrt(var + 1e-12) / jnp.sqrt(cnt)).sum()

                        return jax.grad(value_fn)(x_t).astype(v_t.dtype)

                    grad_a = pessimistic_grad(mask_a)
                    grad_b = pessimistic_grad(mask_b)

                    # (4) Keep the cluster whose steering gradient aligns best (cosine) with v_t.
                    def cos_with_v(gc):
                        gf = gc.reshape(bsz, -1)
                        vf = v_t.reshape(bsz, -1)
                        return (gf * vf).sum(-1) / (
                            jnp.linalg.norm(gf, axis=-1) * jnp.linalg.norm(vf, axis=-1) + 1e-9)  # (b,)

                    cos_a = jnp.where(mask_a.sum(axis=0) > 0, cos_with_v(grad_a), -jnp.inf)
                    cos_b = jnp.where(mask_b.sum(axis=0) > 0, cos_with_v(grad_b), -jnp.inf)
                    grad = jnp.where((cos_a >= cos_b)[:, None, None], grad_a, grad_b)
                else:
                    def value_fn(a):
                        # grad_V = d/d(x_t) mean_k Q(obs, x1(x_t)); .sum() over batch keeps per-sample grads.
                        qs = critic_apply(critic_params, critic_obs, x1_estimate(a).reshape(a.shape[0], -1))
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

            def guided_cond(carry):
                _, time = carry
                # robust to floating-point error
                return time >= -dt / 2

            x_0, _ = jax.lax.while_loop(guided_cond, guided_step, (noise, 1.0))
            aux = {
                "critic_obs_img": siglip_map,
                "critic_obs_state": state,
                "critic_action": x_0,
            }
            return x_0, aux

        if not return_features:

            def step(carry):
                x_t, time = carry
                v_t = self.action_out_proj(action_expert_features(x_t, time))
                return x_t + dt * v_t, time + dt

            def cond(carry):
                x_t, time = carry
                # robust to floating-point error
                return time >= -dt / 2

            x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
            return x_0

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
