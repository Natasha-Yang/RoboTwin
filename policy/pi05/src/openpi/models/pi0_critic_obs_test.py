"""How wide the observation and action the critic is handed are.

The model pads both the state and the action chunk to `action_dim` on the way in, and the two
paddings are treated differently. The state's is constant zero, so it is stripped
(`critic_state_dim`). The action's is not inert -- every dim of a row feeds `action_in_proj` and
the action tokens attend to each other -- so the critic scores the whole
`(action_horizon, action_dim)` chunk, with no option to narrow it. These pin that asymmetry,
which is what `action_dim_flat` on the critic side is derived from.
"""

import jax
import jax.numpy as jnp
import pytest

from openpi.models import pi0_config
from openpi.shared import nnx_utils

_STATIC = ("critic_apply", "noise_apply", "return_critic_obs", "critic_state_dim", "best_of_n")
_PADDED, _EMBODIMENT, _HORIZON = 8, 6, 4


@pytest.fixture(scope="module")
def tiny():
    """A dummy-width pi0.5: `action_dim` 8 padded, standing in for aloha's 14 -> 32."""
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy",
        action_horizon=_HORIZON, action_dim=_PADDED, pi05=True,
    )
    model = config.create(jax.random.key(0))
    return config, nnx_utils.module_jit(model.sample_actions, static_argnames=_STATIC)


def _aux(sample, config, **kwargs):
    _, aux = sample(jax.random.key(0), config.fake_obs(batch_size=1),
                    return_critic_obs=True, **kwargs)
    return aux


def test_the_scored_chunk_is_the_whole_padded_chunk(tiny):
    """Every dim the sampler denoises reaches the critic -- `action_dim_flat` is horizon x 8."""
    config, sample = tiny
    aux = _aux(sample, config, critic_state_dim=_EMBODIMENT)
    assert aux["critic_action"].shape == (1, _HORIZON, _PADDED)
    # And the padded tail carries real content, which is why scoring it is worth anything.
    assert float(jnp.abs(aux["critic_action"][..., _EMBODIMENT:]).max()) > 0.0


def test_the_state_is_still_narrowed(tiny):
    """The one asymmetry: the state's padding normalizes to constant zero, so it is dropped."""
    config, sample = tiny
    assert _aux(sample, config, critic_state_dim=_EMBODIMENT)["critic_obs_state"].shape == (1, _EMBODIMENT)
    # None leaves the state at the model's full width (what a caller with no embodiment gets).
    assert _aux(sample, config)["critic_obs_state"].shape == (1, _PADDED)


def test_the_state_width_does_not_touch_the_action(tiny):
    """Narrowing the state must not narrow the chunk with it -- they were one argument once."""
    config, sample = tiny
    for width in (_EMBODIMENT, _PADDED, None):
        assert _aux(sample, config, critic_state_dim=width)["critic_action"].shape == (
            1, _HORIZON, _PADDED)


def test_the_noise_matches_the_scored_chunk(tiny):
    """`sample_noise` is the padded width too: every one of those dims feeds `action_in_proj`,
    so a narrowed seed would not reproduce its chunk."""
    config, sample = tiny
    assert _aux(sample, config, critic_state_dim=_EMBODIMENT)["sample_noise"].shape == (
        1, _HORIZON, _PADDED)
