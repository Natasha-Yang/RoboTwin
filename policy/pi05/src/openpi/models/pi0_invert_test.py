"""FlowDAgger parity and round-trip tests for `Pi0.invert_actions`."""

import contextlib
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.shared import nnx_utils

# `invert_actions` builds its (sub)step schedule in Python, so these are compile-time constants.
_INVERT_STATIC = ("num_steps", "fp_per_step", "num_inner_steps", "num_substeps", "return_info")


@contextlib.contextmanager
def exact_matmuls():
    """Force full-precision float32 matmuls, so the round trip is limited by the fixed point."""
    with jax.default_matmul_precision("highest"):
        yield


def _tiny_model(dtype: str = "float32"):
    """A dummy-width pi0.5, so the round trip is testable without loading a 2B checkpoint.

    Inversion is a property of the sampler's integrator, not of the weights, so a randomly
    initialized model exercises it exactly as a trained one does.
    """
    config = pi0_config.Pi0Config(
        dtype=dtype,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=8,
        pi05=True,
    )
    return config, config.create(jax.random.key(0))


def _fns(model):
    return (
        nnx_utils.module_jit(model.sample_actions),
        nnx_utils.module_jit(model.invert_actions, static_argnames=_INVERT_STATIC),
    )


def _round_trip(config, model, *, num_steps=10, batch_size=2, **invert_kwargs):
    """noise -> actions -> recovered noise -> replayed actions, plus the inversion's own info."""
    obs = config.fake_obs(batch_size)
    sample, invert = _fns(model)

    noise = jax.random.normal(jax.random.key(1), (batch_size, config.action_horizon, config.action_dim))
    actions = sample(jax.random.key(0), obs, num_steps=num_steps, noise=noise)
    recovered, info = invert(obs, actions, num_steps=num_steps, return_info=True, **invert_kwargs)
    replayed = sample(jax.random.key(0), obs, num_steps=num_steps, noise=recovered)
    return noise, actions, recovered, replayed, info


def test_invert_actions_round_trip():
    """The recovered noise reproduces the original chunk -- and is the original noise."""
    config, model = _tiny_model()
    with exact_matmuls():
        noise, actions, recovered, replayed, info = _round_trip(config, model)

    assert recovered.shape == noise.shape
    assert jnp.all(jnp.isfinite(recovered))
    # Sanity: the sampler moved the noise a long way, so this is not testing an identity map.
    assert float(jnp.max(jnp.abs(actions - noise))) > 1e-2

    np.testing.assert_allclose(replayed, actions, atol=1e-5, rtol=0)
    np.testing.assert_allclose(recovered, noise, atol=2e-5, rtol=0)
    assert info["error"].shape == (2,)
    assert float(jnp.max(info["error"])) < 1e-10


def test_invert_actions_at_default_precision():
    """Without forcing exact matmuls the round trip still holds, at tf32's own precision."""
    config, model = _tiny_model()
    noise, actions, recovered, replayed, _ = _round_trip(config, model)
    np.testing.assert_allclose(replayed, actions, atol=1e-3, rtol=0)
    np.testing.assert_allclose(recovered, noise, atol=1e-3, rtol=0)


def test_invert_actions_recovers_an_arbitrary_chunk():
    """A chunk the sampler never produced still inverts: forward(invert(a)) == a."""
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    sample, invert = _fns(model)

    actions = jax.random.uniform(
        jax.random.key(7), (2, config.action_horizon, config.action_dim), minval=-1.0, maxval=1.0
    )
    with exact_matmuls():
        recovered = invert(obs, actions, num_steps=10)
        replayed = sample(jax.random.key(0), obs, num_steps=10, noise=recovered)
    np.testing.assert_allclose(replayed, actions, atol=1e-5, rtol=0)


def test_invert_actions_holds_across_step_counts():
    """Exactness is not tuned to num_steps=10 -- a coarser and a finer schedule both invert."""
    config, model = _tiny_model()
    for num_steps in (4, 20):
        with exact_matmuls():
            _, actions, _, replayed, info = _round_trip(
                config, model, num_steps=num_steps, fp_per_step=8
            )
        np.testing.assert_allclose(replayed, actions, atol=2e-5, rtol=0)
        assert info["error"].shape == (2,)


def test_invert_actions_is_tied_to_the_forward_step_count():
    """The inverse is of one specific integrator: inverting at the wrong num_steps misses."""
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    sample, invert = _fns(model)

    noise = jax.random.normal(jax.random.key(1), (2, config.action_horizon, config.action_dim))
    actions = sample(jax.random.key(0), obs, num_steps=10, noise=noise)

    mismatched = invert(obs, actions, num_steps=5)
    assert float(jnp.max(jnp.abs(mismatched - noise))) > 1e-2


def test_invert_actions_converges_with_more_fixed_point_refinements():
    """FlowDAgger's per-step refinements monotonically tighten the recovered latent."""
    config, model = _tiny_model()
    errors = []
    with exact_matmuls():
        for fp_per_step in (1, 2, 4, 8):
            noise, _, recovered, _, _ = _round_trip(config, model, fp_per_step=fp_per_step)
            errors.append(float(jnp.max(jnp.abs(recovered - noise))))
    assert errors[-1] < 1e-5
    assert all(a > b for a, b in itertools.pairwise(errors)), errors


def test_num_inner_steps_is_a_compatibility_alias():
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    actions = jax.random.normal(jax.random.key(1), (2, config.action_horizon, config.action_dim))
    _, invert = _fns(model)
    with exact_matmuls():
        current = invert(obs, actions, fp_per_step=3)
        legacy = invert(obs, actions, num_inner_steps=3)
    np.testing.assert_array_equal(current, legacy)


def test_default_matches_flowdagger_settings():
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    actions = jax.random.normal(jax.random.key(2), (2, config.action_horizon, config.action_dim))
    _, invert = _fns(model)
    default = invert(obs, actions)
    explicit = invert(obs, actions, num_steps=10, fp_per_step=5)
    np.testing.assert_array_equal(default, explicit)


def test_invert_actions_rejects_bad_arguments():
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    actions = jnp.zeros((2, config.action_horizon, config.action_dim))

    with pytest.raises(ValueError, match="expected"):
        model.invert_actions(obs, jnp.zeros((2, config.action_horizon + 1, config.action_dim)))
    with pytest.raises(TypeError, match="static Python int"):
        model.invert_actions(obs, actions, num_steps=jnp.int32(10))
    with pytest.raises(ValueError, match=">= 1"):
        model.invert_actions(obs, actions, fp_per_step=0)
    with pytest.raises(ValueError, match="does not implement substeps"):
        model.invert_actions(obs, actions, num_substeps=2)
