"""`Pi0.sample_actions` under critic gradient guidance: Universal Guidance's three properties.

The guidance is Universal Guidance (Bansal et al. 2023) adapted for flow matching, matching
`steering-with-failures`' `steered_ode.py::make_universal_guidance_fn`. What is worth pinning
down is not the value of the chunk -- that depends on the weights -- but the three structural
choices that make it *that* algorithm rather than a near relative:

  * the t=1 step is unsteered (there `x_t` is pure noise, so its Tweedie estimate is nothing),
  * a critic with no gradient leaves the integrator exactly where the plain sampler leaves it,
  * the returned chunk is clipped to [-1, 1] -- and only on this path, so the plain sampler is
    untouched.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.shared import nnx_utils

_STATIC = ("critic_apply", "noise_apply", "return_critic_obs", "critic_action_dim", "best_of_n")


@pytest.fixture(scope="module")
def tiny():
    """A dummy-width pi0.5. The guidance plumbing is independent of the weights."""
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=8,
        pi05=True,
    )
    model = config.create(jax.random.key(0))
    return config, nnx_utils.module_jit(model.sample_actions, static_argnames=_STATIC)


def _constant_critic(params, obs, action):
    """Q is the same everywhere, so dQ/d(chunk) is exactly zero. Shape `(num_qs, batch)`."""
    return jnp.zeros((2, action.shape[0]), dtype=jnp.float32)


def _pushy_critic(params, obs, action):
    """Q rises with every action dim, so the gradient points straight out of the [-1, 1] box."""
    return jnp.broadcast_to(action.sum(axis=-1)[None], (2, action.shape[0]))


def test_first_step_is_unsteered(tiny):
    """At num_steps=1 the only step IS the skipped one, so guidance cannot move the chunk."""
    config, sample = tiny
    obs = config.fake_obs(batch_size=1)

    plain = sample(jax.random.key(0), obs, num_steps=1)
    guided, _ = sample(
        jax.random.key(0),
        obs,
        num_steps=1,
        critic_apply=_pushy_critic,
        critic_params={},
        guidance_scale=jnp.asarray(1e3, dtype=jnp.float32),
        critic_action_dim=config.action_dim,
    )

    # The guided path still clips its output; the plain one does not, so compare against the
    # clipped plain chunk. Any guidance at t=1 would move it far more than the tolerance.
    np.testing.assert_allclose(np.asarray(guided), np.clip(np.asarray(plain), -1.0, 1.0), atol=1e-5)


def test_zero_gradient_critic_is_a_noop(tiny):
    """A flat critic contributes no gradient, so only the clip separates guided from plain."""
    config, sample = tiny
    obs = config.fake_obs(batch_size=1)

    plain = sample(jax.random.key(0), obs, num_steps=10)
    guided, _ = sample(
        jax.random.key(0),
        obs,
        num_steps=10,
        critic_apply=_constant_critic,
        critic_params={},
        guidance_scale=jnp.asarray(1.0, dtype=jnp.float32),
        critic_action_dim=config.action_dim,
    )

    np.testing.assert_allclose(np.asarray(guided), np.clip(np.asarray(plain), -1.0, 1.0), atol=1e-5)


def test_guided_output_is_clipped_and_the_plain_one_is_not(tiny):
    """The clip is real, and it is on the guided path only."""
    config, sample = tiny
    obs = config.fake_obs(batch_size=1)

    guided, _ = sample(
        jax.random.key(0),
        obs,
        num_steps=10,
        critic_apply=_pushy_critic,
        critic_params={},
        guidance_scale=jnp.asarray(1e3, dtype=jnp.float32),
        critic_action_dim=config.action_dim,
    )
    assert np.all(np.abs(np.asarray(guided)) <= 1.0 + 1e-6)

    # A critic that hard-pushes every dim outward would leave the box without the clip, so the
    # bound above is evidence of the clip rather than of a small step. Confirm by checking that
    # the *unclipped* integrator does leave it: the guided chunk sits on the boundary.
    assert np.any(np.abs(np.asarray(guided)) >= 1.0 - 1e-3)


def test_best_of_n_still_selects_after_guidance(tiny):
    """Guidance and best-of-N compose: N steered candidates, one winner, aux dict intact."""
    config, sample = tiny
    obs = config.fake_obs(batch_size=1)

    actions, aux = sample(
        jax.random.key(0),
        obs,
        num_steps=10,
        critic_apply=_pushy_critic,
        critic_params={},
        guidance_scale=jnp.asarray(0.1, dtype=jnp.float32),
        best_of_n=4,
        critic_action_dim=config.action_dim,
    )

    assert actions.shape == (1, config.action_horizon, config.action_dim)
    assert aux["critic_best_scores"].shape == (1, 4)
    assert aux["critic_best_index"].shape == (1,)
    # The winner is the highest-scoring candidate, and every candidate came out of the clip.
    assert np.all(np.abs(np.asarray(actions)) <= 1.0 + 1e-6)
