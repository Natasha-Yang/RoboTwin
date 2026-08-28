"""`Pi0.sample_actions` with a noise-space actor: DSRL's hook into the frozen sampler.

DSRL does not change the denoising at all -- it changes the latent the chunk is denoised from.
So the properties worth pinning down are about the plumbing rather than the integrator: that
the actor's chunk is what the sampler actually starts from, that the chunk comes back out (it
is the action of DSRL's MDP, and nothing outside the sampler can recover it), that the actor
sees the same observation the critic scores, and that an explicit `noise=` still wins.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.models.pi0 import SIGLIP_MODALITIES, hold_noise
from openpi.shared import nnx_utils

_STATIC = ("critic_apply", "noise_apply", "return_critic_obs", "critic_action_dim", "best_of_n")


@pytest.fixture(scope="module")
def tiny():
    """A dummy-width pi0.5. The noise hook is independent of the weights, as the sampler is."""
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=8,
        pi05=True,
    )
    model = config.create(jax.random.key(0))
    return config, nnx_utils.module_jit(model.sample_actions, static_argnames=_STATIC)


def test_hold_noise_repeats_the_last_row():
    noise = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
    held = hold_noise(noise, 7)
    assert held.shape == (2, 7, 4)
    np.testing.assert_array_equal(held[:, :3], noise)
    for row in range(3, 7):
        np.testing.assert_array_equal(held[:, row], noise[:, -1])
    # Already at the horizon: untouched, not copied through a concatenate.
    assert hold_noise(noise, 3) is noise
    with pytest.raises(ValueError, match="longer than the action horizon"):
        hold_noise(noise, 2)


def test_actor_noise_is_what_the_sampler_denoises(tiny):
    """A chosen latent gives exactly the chunk that latent gives when passed in by hand."""
    config, sample = tiny
    obs = config.fake_obs(batch_size=1)
    chosen = jax.random.normal(jax.random.key(7), (1, 1, config.action_dim))

    actions, aux = sample(
        jax.random.key(0), obs,
        noise_apply=lambda params, observations, key: params,
        actor_params=chosen,
        return_critic_obs=True,
    )
    expected = sample(jax.random.key(0), obs, noise=hold_noise(chosen, config.action_horizon))

    np.testing.assert_allclose(actions, expected, atol=1e-6)
    # And the latent comes back at the length the actor chose, not the horizon it was held to.
    np.testing.assert_array_equal(aux["critic_noise"], chosen)


def test_actor_sees_the_observation_the_critic_scores(tiny):
    """`noise_apply` is called after the prefix pass, on the critic's own observation dict."""
    config, sample = tiny
    obs = config.fake_obs(batch_size=1)
    seen = {}

    def noise_apply(params, observations, key):
        seen.update({name: value.shape for name, value in observations.items()})
        return jnp.zeros((1, 1, config.action_dim))

    _, aux = sample(
        jax.random.key(0), obs,
        noise_apply=noise_apply, actor_params=None,
        critic_obs_extra={"wrench.left": jnp.zeros((1, 10, 6))},
        critic_action_dim=6,
        return_critic_obs=True,
    )
    # The SigLIP map per camera view, the model-space state narrowed to the embodiment's dims,
    # and whatever the sim handed in -- i.e. exactly `critic_aux`'s own modalities.
    assert seen["state"] == (1, 6)
    assert seen["wrench.left"] == (1, 10, 6)
    for view in SIGLIP_MODALITIES.values():
        assert seen[view] == (1, 16, 16, 1152)
    assert set(aux["critic_obs_siglip"]) == set(SIGLIP_MODALITIES.values())
    assert aux["critic_obs_state"].shape == (1, 6)


def test_explicit_noise_wins_over_the_actor(tiny):
    """The warmup path: a Gaussian latent passed in leaves the actor unused (and unreported)."""
    config, sample = tiny
    obs = config.fake_obs(batch_size=1)
    noise = jax.random.normal(jax.random.key(3), (1, config.action_horizon, config.action_dim))

    def noise_apply(params, observations, key):
        raise AssertionError("the actor must not be consulted when noise= is given")

    actions, aux = sample(
        jax.random.key(0), obs, noise=noise,
        noise_apply=noise_apply, actor_params=None, return_critic_obs=True,
    )
    np.testing.assert_allclose(actions, sample(jax.random.key(0), obs, noise=noise), atol=1e-6)
    assert "critic_noise" not in aux


def test_plain_sampler_is_unchanged(tiny):
    """No hook, no aux: the baseline path still returns a bare chunk from its own Gaussian."""
    config, sample = tiny
    obs = config.fake_obs(batch_size=2)
    actions = sample(jax.random.key(0), obs)
    assert actions.shape == (2, config.action_horizon, config.action_dim)
