"""Tests for demo retrieval and proposal: `Pi0.embed_observation` / `Pi0.propose_from_demos`.

The retrieval half is exact by construction (an L2 distance and a top-k), so it is tested
against a bank one of whose rows *is* the observation's own embedding. The proposal half is the
inversion, and the test that matters is the same one `pi0_invert_test` runs: feeding a
`noise_proposals` row back to `sample_actions` must reproduce the `action_proposals` row it came
from.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.pi0_invert_test import _tiny_model
from openpi.models.pi0_invert_test import exact_matmuls
from openpi.shared import nnx_utils

_PROPOSE_STATIC = (
    "top_k",
    "views",
    "invert",
    "num_steps",
    "fp_per_step",
    "num_inner_steps",
    "num_substeps",
    "return_info",
)

_BANK = 8


def _fns(model):
    return (
        nnx_utils.module_jit(model.sample_actions),
        nnx_utils.module_jit(model.embed_observation),
        nnx_utils.module_jit(model.propose_from_demos, static_argnames=_PROPOSE_STATIC),
    )


def _bank(config, model, obs, *, planted: int, seed: int = 3):
    """A random unit-norm bank whose row `planted` is batch element 0's own embedding.

    Returns the bank arrays plus the chunks, which are random normalized chunks -- retrieval
    does not look at them, and inversion works on anything in the sampler's space.
    """
    _, embed, _ = _fns(model)
    pooled = embed(obs)
    views = pi0.SIGLIP_VIEWS
    own = jnp.stack([pooled[v] for v in views], axis=1)[0]  # (v, emb)

    key = jax.random.key(seed)
    k_emb, k_act = jax.random.split(key)
    emb = jax.random.normal(k_emb, (_BANK, len(views), own.shape[-1]))
    emb = emb / jnp.linalg.norm(emb, axis=-1, keepdims=True)
    emb = emb.at[planted].set(own)
    actions = jax.random.normal(k_act, (_BANK, config.action_horizon, config.action_dim))
    return emb, actions


def test_embed_observation_keeps_the_pooled_magnitude():
    """One vector per view, at its natural magnitude.

    Not unit-normalized: retrieval divides by the *query's* magnitude
    (`propose_from_demos`), so normalizing each vector here would remove the quantity that
    scaling is measured against.
    """
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    pooled = nnx_utils.module_jit(model.embed_observation)(obs)

    assert set(pooled) == set(pi0.SIGLIP_VIEWS)
    for view, vec in pooled.items():
        assert vec.shape == (2, 1152), view
        assert jnp.isfinite(vec).all(), view
        assert not np.allclose(jnp.linalg.norm(vec, axis=-1), 1.0, atol=1e-2), view


def test_propose_retrieves_the_planted_frame():
    """A bank row equal to the observation's own embedding is the most similar row there is."""
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    emb, actions = _bank(config, model, obs, planted=5)
    _, _, propose = _fns(model)

    out = propose(obs, emb, actions, top_k=1, invert=False, return_info=True)

    assert int(out["indices"][0, 0]) == 5
    # Distance from a unit vector to itself, averaged over views. The slack is tf32: the pooled
    # vectors are 1152-d and the tower that produced them ran at the GPU's default matmul
    # precision, so the two copies of "the same" embedding differ in the last few bits.
    np.testing.assert_allclose(float(out["scores"][0, 0]), 0.0, atol=1e-1)
    assert out["distance"].shape == (2, _BANK)
    np.testing.assert_allclose(out["action_proposals"][0, 0], actions[5], atol=1e-6)


def test_noise_proposals_reproduce_the_action_proposals():
    """The point of the noise half: seed the sampler with it and the demo chunk comes back."""
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    emb, actions = _bank(config, model, obs, planted=1)
    sample, _, propose = _fns(model)

    with exact_matmuls():
        out = propose(obs, emb, actions, top_k=2, num_steps=10, fp_per_step=8, return_info=True)
        assert out["noise_proposals"].shape == (2, 2, config.action_horizon, config.action_dim)
        for k in range(2):
            replayed = sample(jax.random.key(0), obs, num_steps=10, noise=out["noise_proposals"][:, k])
            np.testing.assert_allclose(replayed, out["action_proposals"][:, k], atol=1e-5, rtol=0)

    assert "residual" not in out


def test_distance_averages_every_signal_equally():
    """Each view and each extra signal is one independent L2 distance, averaged with equal weight."""
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    emb, actions = _bank(config, model, obs, planted=4)
    _, _, propose = _fns(model)

    # A pose signal that is unit-norm on both sides, so its cosine is exactly known: bank row j
    # is basis vector j, and the query is basis vector 6.
    dim = 8
    demo_state = jnp.eye(_BANK, dim)[:, :dim]
    query_state = jnp.zeros((2, dim)).at[:, 6].set(1.0)

    plain = propose(obs, emb, actions, top_k=1, invert=False, return_info=True)["distance"]
    joint = propose(
        obs,
        emb,
        actions,
        demo_extra={"state": demo_state},
        query_extra={"state": query_state},
        top_k=1,
        invert=False,
        return_info=True,
    )["distance"]

    # 3 views before, 4 terms after. The pose distance is 0 at row 6 (identical unit vectors)
    # and sqrt(2) everywhere else (orthogonal ones).
    pose_dist = np.full(_BANK, np.sqrt(2.0))
    pose_dist[6] = 0.0
    expected = (np.asarray(plain) * 3 + pose_dist[None, :]) / 4
    np.testing.assert_allclose(np.asarray(joint), expected, atol=1e-5)


def test_pose_decides_between_visually_near_identical_frames():
    """The case the pose term exists for, and the one real banks are in.

    Real SigLIP similarities sit at 0.97-0.99 with ~0.001 between rank 1 and rank 3, so vision
    ranks confidently but by a hair. This builds a bank like that -- every row a small
    perturbation of the query's own embedding -- and checks the pose cosine is what breaks the
    tie, which a bank of *random* embeddings could never show (there the visual winner leads by
    ~1.0 and should not be overridden).
    """
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    _, embed, propose = _fns(model)
    pooled = embed(obs)
    views = pi0.SIGLIP_VIEWS
    own = jnp.stack([pooled[v] for v in views], axis=1)[0]

    jitter = jax.random.normal(jax.random.key(11), (_BANK, len(views), own.shape[-1])) * 0.02
    emb = own[None] + jitter
    emb = emb / jnp.linalg.norm(emb, axis=-1, keepdims=True)
    actions = jax.random.normal(jax.random.key(12), (_BANK, config.action_horizon, config.action_dim))

    plain = propose(obs, emb, actions, top_k=1, invert=False, return_info=True)
    visual_spread = float(plain["distance"].max() - plain["distance"].min())
    visual_winner = int(plain["indices"][0, 0])
    assert visual_spread < 0.05, visual_spread  # a hair, as in a real bank

    # Row 6 matches the pose exactly; every other row points away from it.
    dim = 8
    target = 6 if visual_winner != 6 else 7
    demo_state = jnp.full((_BANK, dim), -1.0).at[target].set(jnp.eye(1, dim)[0])
    demo_state = demo_state / jnp.linalg.norm(demo_state, axis=-1, keepdims=True)
    query_state = jnp.zeros((2, dim)).at[:, 0].set(1.0)

    out = propose(
        obs,
        emb,
        actions,
        demo_extra={"state": demo_state},
        query_extra={"state": query_state},
        top_k=1,
        invert=False,
        return_info=True,
    )
    assert int(out["indices"][0, 0]) == target


def test_propose_rejects_mismatched_extra_signals():
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    emb, actions = _bank(config, model, obs, planted=0)
    good = {"state": jnp.zeros((_BANK, 8))}

    with pytest.raises(ValueError, match="same signals"):
        model.propose_from_demos(obs, emb, actions, demo_extra=good, top_k=1)
    with pytest.raises(ValueError, match="rows, expected"):
        model.propose_from_demos(
            obs,
            emb,
            actions,
            demo_extra={"state": jnp.zeros((3, 8))},
            query_extra={"state": jnp.zeros((2, 8))},
            top_k=1,
        )
    with pytest.raises(ValueError, match="-d but"):
        model.propose_from_demos(obs, emb, actions, demo_extra=good, query_extra={"state": jnp.zeros((2, 5))}, top_k=1)


def test_top_k_is_ordered_and_distinct():
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    emb, actions = _bank(config, model, obs, planted=0)
    _, _, propose = _fns(model)

    out = propose(obs, emb, actions, top_k=4, invert=False, return_info=True)

    for row in np.asarray(out["indices"]):
        assert len(set(row.tolist())) == 4
    # Distances, so ascending: nearest first.
    scores = np.asarray(out["scores"])
    assert np.all(np.diff(scores, axis=-1) >= 0)
    # Every chosen row's chunk is the bank's chunk at that index.
    for b in range(2):
        for k in range(4):
            np.testing.assert_allclose(out["action_proposals"][b, k], actions[out["indices"][b, k]], atol=1e-6)


def test_demo_mask_excludes_padded_rows():
    """Padding a fixed-size bank must be unretrievable, not merely unlikely."""
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    emb, actions = _bank(config, model, obs, planted=5)
    _, _, propose = _fns(model)

    mask = jnp.ones(_BANK, dtype=bool).at[5].set(False)
    out = propose(obs, emb, actions, demo_mask=mask, top_k=_BANK - 1, invert=False, return_info=True)

    assert 5 not in np.asarray(out["indices"])


def test_invert_false_returns_actions_only():
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    emb, actions = _bank(config, model, obs, planted=2)
    _, _, propose = _fns(model)

    out = propose(obs, emb, actions, top_k=1, invert=False)

    assert set(out) == {"action_proposals"}


def test_propose_rejects_bad_arguments():
    config, model = _tiny_model()
    obs = config.fake_obs(2)
    emb, actions = _bank(config, model, obs, planted=0)

    with pytest.raises(ValueError, match="exceeds"):
        model.propose_from_demos(obs, emb, actions, top_k=_BANK + 1)
    with pytest.raises(ValueError, match="views"):
        model.propose_from_demos(obs, emb, actions, views=("siglip.head",))
    with pytest.raises(ValueError, match="chunk shape"):
        model.propose_from_demos(obs, emb, actions[:, :1], top_k=1)
    with pytest.raises(ValueError, match="inconsistent"):
        model.propose_from_demos(obs, emb[:-1], actions, top_k=1)


def test_resolve_views_orders_by_the_model():
    from openpi.policies import demo_retrieval

    assert demo_retrieval.resolve_views(None) == pi0.SIGLIP_VIEWS
    assert demo_retrieval.resolve_views(["right_wrist", "head"]) == ("siglip.head", "siglip.right_wrist")
    assert demo_retrieval.resolve_views("siglip.head") == ("siglip.head",)
    with pytest.raises(ValueError, match="not camera views"):
        demo_retrieval.resolve_views(["front"])


def test_pi0_config_is_untouched():
    """Guard: the tiny model used here must stay a real pi0.5, not drift into a pi0."""
    config, _ = _tiny_model()
    assert isinstance(config, pi0_config.Pi0Config)
    assert config.pi05
