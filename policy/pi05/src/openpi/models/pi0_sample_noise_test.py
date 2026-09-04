"""`sample_noise`: the latent a sampled chunk was denoised from, reported back to the caller.

Rollout collection records it as the dataset's `action.noise` column (`script/collect_dataset.py`),
which is what a noise-space agent (DSRL) would train an offline Q(s, w) on. The property that
makes the column worth anything is that it is the *seed of this chunk*: pass it back in as
`noise=` and the same chunk comes out. So that is what these pin down -- for the plain sampler,
for best-of-N (where the seed reported must be the winner's, not an arbitrary candidate's), and
for a chosen latent (where it is the expanded chunk, not the actor's own parameterization).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.models.pi0 import hold_noise
from openpi.shared import nnx_utils

_STATIC = ("critic_apply", "noise_apply", "return_critic_obs", "critic_state_dim",
           "best_of_n")


@pytest.fixture(scope="module")
def tiny():
    """A dummy-width pi0.5. What the seed reproduces is independent of the weights."""
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=8,
        pi05=True,
    )
    model = config.create(jax.random.key(0))
    return config, nnx_utils.module_jit(model.sample_actions, static_argnames=_STATIC)


def test_sample_noise_reproduces_the_chunk(tiny):
    """The collection path: replaying the reported seed gives back the chunk that was recorded."""
    config, sample = tiny
    obs = config.fake_obs(batch_size=2)

    actions, aux = sample(jax.random.key(0), obs, return_critic_obs=True, critic_state_dim=6)

    # Both at the model's full padded width -- the critic scores every dim of the chunk, and the
    # trailing noise dims are live inputs to `action_in_proj` anyway, so a narrowed seed would
    # not reproduce anything. Only the state is narrowed to the embodiment's 6.
    assert aux["sample_noise"].shape == (2, config.action_horizon, config.action_dim)
    assert aux["critic_action"].shape == (2, config.action_horizon, config.action_dim)
    assert aux["critic_obs_state"].shape == (2, 6)
    np.testing.assert_allclose(
        sample(jax.random.key(1), obs, noise=aux["sample_noise"]), actions, atol=1e-6
    )


def test_best_of_n_reports_the_winners_seed(tiny):
    """With N candidates the seed must denoise into the chunk that came back, not a discarded one."""
    config, sample = tiny
    obs = config.fake_obs(batch_size=2)
    # A critic that ranks candidates by a fixed linear score -- enough to make the argmax pick
    # something other than candidate 0, which is what the gather has to follow.
    weights = jax.random.normal(jax.random.key(5), (config.action_horizon * config.action_dim,))

    def critic_apply(params, observations, action):
        return (action @ params)[None]  # (1, b): one "ensemble member"

    actions, aux = sample(
        jax.random.key(0), obs,
        critic_apply=critic_apply, critic_params=weights,
        guidance_scale=None, best_of_n=4, return_critic_obs=True,
    )

    assert aux["sample_noise"].shape == (2, config.action_horizon, config.action_dim)
    assert aux["critic_best_scores"].shape == (2, 4)
    # The selection is doing something -- otherwise this would pass on the seed of candidate 0.
    assert aux["critic_best_index"].max() > 0
    # The candidates ride along as extra batch elements, so the winner was denoised at batch
    # `2 * 4` while this replay runs at batch 2, and bf16 matmuls do not agree bit for bit
    # across that. What is being checked is that this is the *same* chunk rather than another
    # candidate's, so an unrelated seed sets the scale: it lands orders of magnitude further off.
    gap = np.abs(np.asarray(sample(jax.random.key(1), obs, noise=aux["sample_noise"])) - actions).max()
    unrelated = np.abs(np.asarray(sample(jax.random.key(2), obs)) - actions).max()
    assert gap < 0.05 < unrelated


def test_sample_noise_is_the_expanded_latent_not_the_actors(tiny):
    """DSRL: `critic_noise` is what SAC acts in, `sample_noise` is what the flow started from."""
    config, sample = tiny
    chosen = jax.random.normal(jax.random.key(7), (1, 1, config.action_dim))

    _, aux = sample(
        jax.random.key(0), config.fake_obs(batch_size=1),
        noise_apply=lambda params, observations, key: (params, hold_noise(params, config.action_horizon)),
        actor_params=chosen,
        return_critic_obs=True,
    )

    np.testing.assert_array_equal(aux["critic_noise"], chosen)
    np.testing.assert_array_equal(aux["sample_noise"], hold_noise(chosen, config.action_horizon))
