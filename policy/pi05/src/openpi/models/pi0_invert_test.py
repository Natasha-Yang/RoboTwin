"""Round-trip tests for `Pi0.invert_actions`: noise -> chunk -> noise -> chunk.

The inversion is exact up to the arithmetic precision of the forward pass, and on GPU that
floor is set by matmul precision, not by the fixed point: float32 with tf32 matmuls (the JAX
default) bottoms out around 1e-4, while `highest` reaches ~1e-7. The tight assertions below run
under `highest` so they are testing the algorithm rather than cuBLAS; `test_invert_actions_at_default_precision`
pins down what the default buys.
"""

import contextlib
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.shared import nnx_utils

# `invert_actions` builds its (sub)step schedule in Python, so these are compile-time constants.
_INVERT_STATIC = ("num_steps", "num_inner_steps", "num_substeps", "return_info")


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
    np.testing.assert_allclose(recovered, noise, atol=1e-5, rtol=0)
    # Every step's fixed point converged to the float32 floor.
    assert info["residual"].shape == (10,)
    assert float(jnp.max(info["residual"])) < 1e-5


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
            _, actions, _, replayed, info = _round_trip(config, model, num_steps=num_steps)
        np.testing.assert_allclose(replayed, actions, atol=1e-5, rtol=0)
        assert info["residual"].shape == (num_steps,)


def test_invert_actions_is_tied_to_the_forward_step_count():
    """The inverse is of one specific integrator: inverting at the wrong num_steps misses."""
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    sample, invert = _fns(model)

    noise = jax.random.normal(jax.random.key(1), (2, config.action_horizon, config.action_dim))
    actions = sample(jax.random.key(0), obs, num_steps=10, noise=noise)

    mismatched = invert(obs, actions, num_steps=5)
    assert float(jnp.max(jnp.abs(mismatched - noise))) > 1e-2


def test_invert_actions_converges_with_more_inner_steps():
    """More fixed-point iterations monotonically tighten the round trip -- it is not luck."""
    config, model = _tiny_model()
    errors = []
    with exact_matmuls():
        for num_inner_steps in (1, 2, 4, 8):
            noise, _, recovered, _, _ = _round_trip(config, model, num_inner_steps=num_inner_steps)
            errors.append(float(jnp.max(jnp.abs(recovered - noise))))
    # One iteration is the DDIM-style approximate inverse and should be visibly the worst.
    assert errors[0] > 1e-2
    assert errors[-1] < 1e-5
    assert all(a > b for a, b in itertools.pairwise(errors)), errors


def test_invert_actions_substeps_trade_exactness_for_contraction():
    """`num_substeps > 1` inverts the ODE, not the coarse Euler map -- so it round-trips worse.

    Its fixed points still converge (the residual falls just as far); what it loses is agreement
    with the sampler's own truncation error, which is the whole point of the default of 1.
    """
    config, model = _tiny_model()
    with exact_matmuls():
        _, _, _, exact_replay, exact_info = _round_trip(config, model, num_substeps=1)
        _, actions, _, fine_replay, fine_info = _round_trip(config, model, num_substeps=10)

    assert fine_info["residual"].shape == (100,)
    assert float(jnp.max(fine_info["residual"])) < 1e-5  # converged, just to a different answer
    exact_err = float(jnp.max(jnp.abs(exact_replay - actions)))
    fine_err = float(jnp.max(jnp.abs(fine_replay - actions)))
    assert exact_err < 1e-5 < fine_err, (exact_err, fine_err)
    assert float(jnp.max(exact_info["residual"])) < 1e-5


def test_invert_actions_rejects_bad_arguments():
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    actions = jnp.zeros((2, config.action_horizon, config.action_dim))

    with pytest.raises(ValueError, match="expected"):
        model.invert_actions(obs, jnp.zeros((2, config.action_horizon + 1, config.action_dim)))
    with pytest.raises(TypeError, match="static Python int"):
        model.invert_actions(obs, actions, num_steps=jnp.int32(10))
    with pytest.raises(ValueError, match=">= 1"):
        model.invert_actions(obs, actions, num_inner_steps=0)
    with pytest.raises(ValueError, match=">= 1"):
        model.invert_actions(obs, actions, num_substeps=0)
