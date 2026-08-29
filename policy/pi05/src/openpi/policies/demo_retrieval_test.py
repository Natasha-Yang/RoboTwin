"""Tests for `openpi.policies.demo_retrieval` against a real LeRobot dataset on disk.

The reader half is format plumbing (two on-disk LeRobot layouts) and the bank half is about
staying in the policy's space, so both are exercised against an actual dataset rather than a
mock -- a mock would agree with whatever the reader happens to do. The model is a dummy-width
pi0.5: retrieval is a property of the pipeline, not of the weights.

Skipped when no demo dataset is present.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies import aloha_policy
from openpi.policies import demo_retrieval
from openpi.shared import nnx_utils
import openpi.transforms as _transforms

_PROPOSE_STATIC = ("top_k", "views", "invert", "num_steps", "num_inner_steps", "num_substeps", "return_info")

# A v2.1 (one parquet per episode) and a v3.0 (episodes share parquet files) dataset. Both
# layouts are read by `LeRobotEpisodeReader`, and the version is the whole reason it exists:
# a v2.x lerobot install refuses to open the v3.0 one at all.
V21_REPO = "NatashaYang/robotwin_demo_clean_50_lerobot"
V30_REPO = "NatashaYang/robotwin_lerobot_dataset"


def _have(repo_id: str) -> bool:
    return (demo_retrieval.default_root() / repo_id / "meta" / "info.json").exists()


requires_v21 = pytest.mark.skipif(not _have(V21_REPO), reason=f"{V21_REPO} not downloaded")


def _tiny_aloha_model(action_horizon: int = 8):
    """A dummy-width pi0.5 with aloha's real widths, so the policy transforms fit it."""
    config = pi0_config.Pi0Config(
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=action_horizon,
        action_dim=32,
        pi05=True,
    )
    return config, config.create(jax.random.key(0))


def _input_transform(config):
    """The policy's own input chain, minus the parts that need a trained checkpoint.

    Mirrors `policy_config.create_trained_policy` (aloha data transforms, then normalization,
    then the model transforms). `Normalize(None)` is a no-op stand-in for the checkpoint's norm
    stats -- everything this test checks is about shape, space and alignment, none of which the
    normalization constants change.
    """
    return _transforms.compose(
        [
            aloha_policy.AlohaInputs(adapt_to_pi=False),
            _transforms.DeltaActions(_transforms.make_bool_mask(6, -1, 6, -1)),
            _transforms.Normalize(None),
            _transforms.ResizeImages(224, 224),
            _transforms.TokenizePrompt(
                PaligemmaTokenizer(config.max_token_len),
                discrete_state_input=config.discrete_state_input,
            ),
            _transforms.PadStatesAndActions(config.action_dim),
        ]
    )


def _absolute_input_transform(config):
    """The same chain with `use_delta_joint_actions=False`, i.e. no `DeltaActions` step.

    No config in this repo is built that way, so the only way to exercise the absolute-action
    branch is to construct the chain it would produce.
    """
    return _transforms.compose(
        [
            aloha_policy.AlohaInputs(adapt_to_pi=False),
            _transforms.Normalize(None),
            _transforms.ResizeImages(224, 224),
            _transforms.TokenizePrompt(
                PaligemmaTokenizer(config.max_token_len),
                discrete_state_input=config.discrete_state_input,
            ),
            _transforms.PadStatesAndActions(config.action_dim),
        ]
    )


def _retriever(config, model, repo_id=V21_REPO, **kwargs):
    kwargs.setdefault("frame_stride", 20)
    kwargs.setdefault("bank_size", 16)
    kwargs.setdefault("encode_batch_size", 8)
    return demo_retrieval.DemoRetriever(model, _input_transform(config), repo_id=repo_id, **kwargs)


@pytest.mark.parametrize("repo_id", [V21_REPO, V30_REPO])
def test_reader_handles_both_on_disk_layouts(repo_id):
    if not _have(repo_id):
        pytest.skip(f"{repo_id} not downloaded")
    reader = demo_retrieval.LeRobotEpisodeReader(repo_id)

    instructions = reader.instructions()
    assert len(instructions) == reader.num_episodes
    assert all(isinstance(text, str) and text for text in instructions)

    episode = reader.read_episode(1)
    length = reader.episode_length(1)
    assert episode["state"].shape == (length, 14)
    assert episode["action"].shape == (length, 14)
    for camera in demo_retrieval.DEMO_CAMERAS:
        frames = episode["images"][camera]
        # Channel-first uint8, the layout AlohaInputs reads and the eval path also hands it.
        assert frames.shape[:2] == (length, 3)
        assert frames.dtype == np.uint8


@requires_v21
def test_episodes_are_grouped_by_robotwin_task_not_instruction():
    """The point of going through `assign_episode_tasks`: one task, many instructions."""
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model)

    episodes = retriever.episodes_for_task("beat_block_hammer")
    assert len(episodes) > 1
    instructions = retriever.reader.instructions()
    assert len({instructions[i] for i in episodes}) > 1, "expected one task to span many instructions"

    with pytest.raises(ValueError, match="no episodes of task"):
        retriever.episodes_for_task("not_a_robotwin_task")


@requires_v21
def test_bank_is_fixed_size_with_padding_masked_out():
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model, num_demos=1, bank_size=16, frame_stride=20)

    bank = retriever.select_bank("beat_block_hammer")

    assert bank.task == "beat_block_hammer"
    assert len(bank.episodes) == 1
    assert bank.embeddings.shape == (16, len(pi0.SIGLIP_VIEWS), 1152)
    assert bank.actions.shape == (16, config.action_horizon, config.action_dim)
    assert bank.mask.shape == (16,)
    assert bank.mask.sum() == bank.num_frames < 16
    # Real rows hold the pooled vectors at their natural magnitude (retrieval scales by the
    # query's, not per vector); padding is exactly zero and never retrievable.
    norms = np.linalg.norm(bank.embeddings[: bank.num_frames], axis=-1)
    assert np.isfinite(norms).all()
    assert not np.allclose(norms, 1.0, atol=1e-2), "bank must not be unit-normalized"
    assert not bank.embeddings[bank.num_frames :].any()


@requires_v21
def test_retrieved_rows_are_encoded_on_demand_for_cross_attention():
    """The keys' side of the attention: a row has to be encodable the way a live frame is.

    Which means the *patch map* per view, not the pooled vector the L2 distance ranks on, and
    the pose in the model's own space -- both keyed by the modality names the critic's
    encoders are configured under, so the same encoder runs on both sides. What the bank keeps
    is the frames those maps are computed from; `critic_keys` encodes the retrieved rows.
    """
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model, num_demos=1, bank_size=16, frame_stride=20)
    bank = retriever.select_bank("beat_block_hammer")

    # The rows' model inputs, at the model's own resolution and still uint8 -- the whole point
    # of not keeping the patch maps, which are ~256x larger per view.
    assert set(bank.obs["image"]) == set(bank.obs["image_mask"])
    for frames in bank.obs["image"].values():
        assert frames.shape == (bank.num_frames, 224, 224, 3)
        assert frames.dtype == np.uint8
    assert bank.states.shape == (16, config.action_dim)

    rows = np.array([0, 2, bank.num_frames - 1])
    keys = retriever.critic_keys(rows, state_dim=14)
    assert set(keys) == {*pi0.SIGLIP_VIEWS, "state"}
    for view in pi0.SIGLIP_VIEWS:
        assert keys[view].shape == (len(rows), 256, 1152)
        assert keys[view].dtype == np.float16
        # Encoding a row now must give the same thing the bank pooled at build time. The
        # tolerance is fp16: a key is stored the way the replay buffer stores a SigLIP map,
        # while the bank pooled the fp32 maps it was built from.
        np.testing.assert_allclose(
            keys[view].astype(np.float32).mean(axis=-2),
            bank.embeddings[rows, pi0.SIGLIP_VIEWS.index(view)],
            atol=1e-2,
        )
    # Narrowed to the embodiment's own dims, exactly as the sampler narrows the state it hands
    # the critic -- the model pads both out to `action_dim`.
    assert keys["state"].shape == (len(rows), 14)
    np.testing.assert_allclose(keys["state"], bank.states[rows, :14])

    # Only the modalities asked for are encoded, and nothing else can be asked for.
    assert set(retriever.critic_keys(rows, ("state",))) == {"state"}
    with pytest.raises(KeyError, match="wrench.left"):
        retriever.critic_keys(rows, ("wrench.left",))
    with pytest.raises(IndexError, match="not frames of the demo bank"):
        retriever.critic_keys([bank.num_frames])


@requires_v21
def test_critic_keys_need_a_bank_first():
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model)
    with pytest.raises(RuntimeError, match="No demo bank yet"):
        retriever.critic_keys([0])


@requires_v21
def test_action_space_is_relative_to_the_conditioning_pose():
    """The invariant a proposal has to satisfy: motion from a pose, not an absolute target.

    Probed rather than read off the config, since it is the *transform chain* that decides.
    Aloha's is `make_bool_mask(6, -1, 6, -1)`: the twelve arm-joint dims are deltas, the two
    gripper dims (6 and 13) are absolute widths.
    """
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model)

    assert retriever.delta_dims.tolist() == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


@requires_v21
def test_bank_actions_are_in_the_models_space():
    """Padded to `action_dim` and delta-encoded against the demo's own state, like training."""
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model, bank_size=16, frame_stride=20)
    bank = retriever.select_bank("beat_block_hammer")
    episode = bank.episodes[0]

    raw = retriever.reader.read_episode(episode)
    # First bank row = frame 0 of the episode. The joint dims are deltas from that frame's
    # state; the gripper dims (6 and 13) are absolute (see make_bool_mask(6, -1, 6, -1)).
    expected = raw["action"][: config.action_horizon] - raw["state"][0]
    expected[:, 6] = raw["action"][: config.action_horizon, 6]
    expected[:, 13] = raw["action"][: config.action_horizon, 13]

    np.testing.assert_allclose(bank.actions[0, :, :14], expected, atol=1e-5)
    # Everything past the embodiment's 14 dims is the model's zero padding.
    assert not bank.actions[0, :, 14:].any()


@requires_v21
def test_absolute_action_demos_are_converted_to_deltas():
    """`use_delta_joint_actions=False` must not put another episode's absolute pose in the bank.

    With no `DeltaActions` in the chain the probe finds nothing state-dependent, so pi0.5's own
    `DeltaActions(NATIVE_DELTA_MASK)` is applied instead. The test of that is not "some
    subtraction happened" but that it lands in the *same* control mode: the bank must come out
    identical to the one the policy's own delta chain builds from the same episode.
    """
    config, model = _tiny_aloha_model()
    delta = _retriever(config, model)
    absolute = demo_retrieval.DemoRetriever(
        model,
        _absolute_input_transform(config),
        repo_id=V21_REPO,
        frame_stride=20,
        bank_size=16,
        encode_batch_size=8,
    )

    assert not delta.converts_actions
    assert absolute.converts_actions
    # The effective space is the native mask's, even though the policy's chain supplied none.
    assert absolute.delta_dims.tolist() == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    assert absolute.delta_dims.tolist() == delta.delta_dims.tolist()
    assert "applied here" in absolute.describe_action_space()

    # Same seed, same draw -- so the two banks are the same frames in (hopefully) the same space.
    delta_bank = delta.select_bank("beat_block_hammer")
    absolute_bank = absolute.select_bank("beat_block_hammer")
    assert delta_bank.episodes == absolute_bank.episodes
    np.testing.assert_allclose(absolute_bank.actions, delta_bank.actions, atol=1e-6)

    # And the conversion is not vacuous. A delta only differs from an absolute target where the
    # pose it is measured from is nonzero, and an arm this task does not use sits at qpos 0 for
    # the whole episode -- so this checks the row whose pose is furthest from home.
    raw = delta.reader.read_episode(delta_bank.episodes[0])
    frames = delta_bank.row_frame[: delta_bank.num_frames]
    row = int(np.argmax(np.linalg.norm(raw["state"][frames], axis=1)))
    frame = int(frames[row])
    window = np.arange(frame, frame + config.action_horizon).clip(max=len(raw["action"]) - 1)
    moved = delta.delta_dims
    assert np.abs(raw["action"][window][:, moved] - delta_bank.actions[row][:, moved]).max() > 1e-2


@requires_v21
def test_delta_action_demos_are_left_to_the_policys_own_transform():
    """The normal case: the chain already delta-encodes, so nothing is subtracted twice."""
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model)
    bank = retriever.select_bank("beat_block_hammer")
    raw = retriever.reader.read_episode(bank.episodes[0])

    # Deltas on the arm joints, absolute on the grippers -- one subtraction, not two.
    expected = raw["action"][: config.action_horizon] - raw["state"][0]
    expected[:, 6] = raw["action"][: config.action_horizon, 6]
    expected[:, 13] = raw["action"][: config.action_horizon, 13]
    np.testing.assert_allclose(bank.actions[0, :, :14], expected, atol=1e-5)


@requires_v21
def test_bank_carries_the_pose_signal_at_its_natural_magnitude():
    """`state` is the second distance term: the pose in the model's own normalized space.

    Stored unscaled -- the relative distance divides by the *query's* magnitude, so normalizing
    each bank vector would throw away exactly what that scaling is measured against.
    """
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model)
    assert retriever.signals == ("siglip", "state")
    assert "4 independent relative L2 distances" in retriever.describe_similarity()

    bank = retriever.select_bank("beat_block_hammer")
    poses = bank.extra["state"]

    assert poses.shape == (16, config.action_dim)
    assert not poses[bank.num_frames :].any()
    # It is the transform's state verbatim, not the raw qpos and not a normalized copy of it.
    raw = retriever.reader.read_episode(bank.episodes[0])
    expected = np.asarray(
        retriever.input_transform(
            {
                "images": {c: raw["images"][c][0] for c in demo_retrieval.DEMO_CAMERAS},
                "state": raw["state"][0],
                "actions": raw["action"][: config.action_horizon],
                "prompt": "",
            }
        )["state"],
        dtype=np.float32,
    )
    np.testing.assert_allclose(poses[0], expected, atol=1e-6)
    norms = np.linalg.norm(poses[: bank.num_frames], axis=-1)
    assert not np.allclose(norms, 1.0, atol=1e-2), "poses must not be unit-normalized"


@requires_v21
def test_siglip_only_retrieval_drops_the_pose_term():
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model, signals=["siglip"])
    bank = retriever.select_bank("beat_block_hammer")

    assert retriever.signals == ("siglip",)
    assert bank.extra == {}
    assert "3 independent relative L2 distances" in retriever.describe_similarity()


def test_resolve_signals_validates():
    assert demo_retrieval.resolve_signals(None) == ("siglip", "state")
    assert demo_retrieval.resolve_signals(["state", "siglip"]) == ("siglip", "state")
    with pytest.raises(ValueError, match="unknown similarity signal"):
        demo_retrieval.resolve_signals(["siglip", "wrench"])
    with pytest.raises(ValueError, match="`siglip` is required"):
        demo_retrieval.resolve_signals(["state"])


def test_zero_query_does_not_divide_by_zero():
    """The relative distance divides by ||query||; a zero query must not produce NaN."""
    config, model = _tiny_aloha_model()
    _, _, propose = (None, None, nnx_utils.module_jit(model.propose_from_demos, static_argnames=_PROPOSE_STATIC))
    obs = config.fake_obs(1)
    emb = jnp.ones((4, len(pi0.SIGLIP_VIEWS), 1152))
    actions = jnp.zeros((4, config.action_horizon, config.action_dim))
    out = propose(
        obs,
        emb,
        actions,
        demo_extra={"state": jnp.ones((4, 8))},
        query_extra={"state": jnp.zeros((1, 8))},
        top_k=1,
        invert=False,
        return_info=True,
    )
    assert np.isfinite(np.asarray(out["distance"])).all()


@requires_v21
def test_propose_returns_both_modalities_at_the_configured_shape():
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model, top_k=3, bank_size=16, frame_stride=20)
    bank = retriever.select_bank("beat_block_hammer")

    # An observation shaped like the eval loop's: channel-first uint8 frames plus the joint
    # vector (see policy/pi05/pi_model.py::update_observation_window).
    raw = retriever.reader.read_episode(bank.episodes[0])
    window = {
        "images": {camera: raw["images"][camera][0] for camera in demo_retrieval.DEMO_CAMERAS},
        "state": raw["state"][0],
        "prompt": "beat the block with the hammer",
    }
    out = retriever.propose(window)

    assert set(out) == {"action_proposals", "noise_proposals", "proposal_rows"}
    for name in ("action_proposals", "noise_proposals"):
        value = out[name]
        assert value.shape == retriever.proposal_shape == (3, config.action_horizon, config.action_dim)
        assert np.isfinite(value).all()
    # The observation IS a demo frame of this bank, so the closest row should be its own.
    np.testing.assert_allclose(out["action_proposals"][0], bank.actions[0], atol=1e-4)
    # And the rows say so: they are the candidates' addresses in the bank, which is what a
    # cross-attending critic looks the keys up by.
    assert out["proposal_rows"].shape == (3,)
    assert out["proposal_rows"][0] == 0
    np.testing.assert_allclose(bank.actions[out["proposal_rows"]], out["action_proposals"], atol=1e-6)


@requires_v21
def test_propose_without_inversion_skips_the_noise():
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model, invert=False, bank_size=16, frame_stride=20)
    bank = retriever.select_bank("beat_block_hammer")
    raw = retriever.reader.read_episode(bank.episodes[0])

    out = retriever.propose(
        {
            "images": {camera: raw["images"][camera][0] for camera in demo_retrieval.DEMO_CAMERAS},
            "state": raw["state"][0],
            "prompt": "beat the block with the hammer",
        }
    )

    assert set(out) == {"action_proposals", "proposal_rows"}


@requires_v21
def test_offline_ranking_matches_the_samplers():
    """The property co-training rests on: rank a *stored* observation the sampler's way.

    An offline row carries the un-pooled SigLIP map the same tower produced, so pooling it and
    ranking in numpy has to pick the rows `propose_from_demos` picks. If the two ever disagree,
    a mixed batch is conditioned on two different sets of demonstrations and nothing says so.
    """
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model, top_k=4, bank_size=16, frame_stride=20)
    bank = retriever.select_bank("beat_block_hammer")
    raw = retriever.reader.read_episode(bank.episodes[0])

    for frame in (0, 3, 7):
        window = {
            "images": {camera: raw["images"][camera][frame] for camera in demo_retrieval.DEMO_CAMERAS},
            "state": raw["state"][frame],
            "prompt": "beat the block with the hammer",
        }
        rows_from_sampler = retriever.propose(window)["proposal_rows"]

        # What an offline row holds: this observation's own patch maps and model-space pose,
        # narrowed to the embodiment's dims the way the dataset column stores them.
        inputs = retriever.input_transform(jax.tree.map(lambda x: x, window))
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        batched.pop("actions", None)
        maps = retriever._embed(_model.Observation.from_dict(batched))
        pooled = {view: np.asarray(maps[view], np.float32).mean(axis=-2) for view in retriever.views}
        state = np.asarray(inputs["state"], np.float32)[None, :14]

        rows_offline = retriever.rank_observations(pooled, state)[0]
        np.testing.assert_array_equal(rows_offline, rows_from_sampler)


@requires_v21
def test_offline_ranking_needs_the_views_it_ranks_on():
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model, bank_size=16, frame_stride=20)
    retriever.select_bank("beat_block_hammer")
    with pytest.raises(KeyError, match="missing"):
        retriever.rank_observations({"siglip.head": np.zeros((1, 1152), np.float32)},
                                    np.zeros((1, 14), np.float32))


@requires_v21
def test_the_bank_is_drawn_once_and_held_for_the_run():
    """`ensure_bank` must be idempotent: the critic's conditioning cannot move mid-run."""
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model, num_demos=1, bank_size=16, frame_stride=20)

    first = retriever.ensure_bank("beat_block_hammer")
    # Several later episodes ask again; every one gets the same bank object back.
    for _ in range(3):
        assert retriever.ensure_bank("beat_block_hammer") is first

    # Enough demonstrations exist that a re-draw would very likely land elsewhere, so the
    # stability above is not just "there was only one to pick".
    assert len(retriever.episodes_for_task("beat_block_hammer")) > 10
    # `select_bank` is still the explicit escape hatch, and does draw again.
    assert retriever.select_bank("beat_block_hammer") is not first


@requires_v21
def test_encoded_episodes_are_cached_across_banks():
    """Re-drawing an episode must not re-encode it: encoding is the expensive part."""
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model, bank_size=16, frame_stride=20)
    episode = retriever.episodes_for_task("beat_block_hammer")[0]

    first_embeddings, first_actions, *_ = retriever.encode_episode(episode)
    second_embeddings, second_actions, *_ = retriever.encode_episode(episode)

    assert second_embeddings is first_embeddings
    assert second_actions is first_actions


def test_missing_dataset_says_so():
    config, model = _tiny_aloha_model()
    with pytest.raises(FileNotFoundError, match="No LeRobot dataset"):
        demo_retrieval.DemoRetriever(model, _input_transform(config), repo_id="nobody/nothing")


def test_top_k_cannot_exceed_the_bank():
    config, model = _tiny_aloha_model()
    with pytest.raises(ValueError, match="exceeds bank_size"):
        _retriever(config, model, top_k=32, bank_size=16)


@requires_v21
def test_record_retrieval_keeps_thumbnails_and_ranks_the_whole_bank():
    """The debug path: thumbnails per bank row, and as many ranks as it asks for."""
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model, top_k=1, debug_top_k=3, bank_size=16, frame_stride=10)
    retriever.record_retrieval = True
    bank = retriever.select_bank("beat_block_hammer")

    assert bank.thumbnails is not None
    assert bank.thumbnails.shape[0] == 16
    assert bank.thumbnails.shape[-1] == 3
    assert bank.thumbnails.dtype == np.uint8
    # Padding rows are blank, and their provenance is marked absent.
    assert not bank.thumbnails[bank.num_frames :].any()
    assert (bank.row_episode[bank.num_frames :] == -1).all()
    assert set(bank.row_episode[: bank.num_frames].tolist()) == set(bank.episodes)

    raw = retriever.reader.read_episode(bank.episodes[0])
    window = {
        "images": {camera: raw["images"][camera][0] for camera in demo_retrieval.DEMO_CAMERAS},
        "state": raw["state"][0],
        "prompt": "beat the block with the hammer",
    }
    proposals = retriever.propose(window)
    info = retriever.retrieved()

    # The critic still gets top_k=1; the visualization sees three ranks of the same ordering.
    assert proposals["action_proposals"].shape[0] == 1
    assert info["num_proposals"] == 1
    assert len(info["indices"]) == 3
    assert info["thumbnails"].shape[0] == 3
    # Ranked nearest-first (distances ascend), and never a padding row.
    assert np.all(np.diff(info["distances"]) >= 0)
    assert np.all(info["indices"] < bank.num_frames)
    # Querying with a demo frame retrieves that frame first.
    assert int(info["indices"][0]) == 0


@requires_v21
def test_retrieved_is_none_without_record_retrieval():
    """The default path keeps no thumbnails -- they are the only thing otherwise discarded."""
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model, bank_size=16, frame_stride=20)
    bank = retriever.select_bank("beat_block_hammer")

    assert bank.thumbnails is None
    assert retriever.retrieved() is None


# ---------------------------------------------------------------------------------------------
# Co-training on the demonstrations themselves (`cotrain_rows`)
# ---------------------------------------------------------------------------------------------


@requires_v21
def test_read_episode_decodes_only_the_frames_asked_for():
    """`frames=` selects what is decoded; states and actions still come back whole.

    A frame's action chunk reaches `action_horizon` steps past it, so the two float columns have
    to be the episode's own -- it is the JPEGs that are worth not decoding, and a co-training
    pass keeps one frame in every `horizon`.
    """
    reader = demo_retrieval.LeRobotEpisodeReader(V21_REPO)
    length = reader.episode_length(3)
    frames = np.arange(0, length, 25)

    whole = reader.read_episode(3)
    subset = reader.read_episode(3, frames=frames)

    np.testing.assert_array_equal(subset["state"], whole["state"])
    np.testing.assert_array_equal(subset["action"], whole["action"])
    np.testing.assert_array_equal(subset["frames"], frames)
    for camera in demo_retrieval.DEMO_CAMERAS:
        assert len(subset["images"][camera]) == len(frames)
        # Indexed by position in `frames`, not by frame index.
        np.testing.assert_array_equal(subset["images"][camera], whole["images"][camera][frames])

    with pytest.raises(IndexError):
        reader.read_episode(3, frames=[length])


@requires_v21
def test_cotrain_rows_are_control_steps_in_the_critics_space():
    """One row per `horizon` frames, at the critic's own widths, and the chunk is the bank's.

    The rows are what an offline `(s, a)` for this critic *is*: the SigLIP patch maps the
    sampler feeds it, the normalized pose narrowed to the embodiment's dims, and the same
    normalized delta chunk `encode_episode` puts in the bank. If the chunk here and the chunk
    there disagreed, a co-trained batch and a retrieved proposal would be two different spaces.
    """
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model)
    episodes = retriever.episodes_for_task("beat_block_hammer")[:2]
    horizon = 40
    state_dim = 14

    rows = retriever.cotrain_rows(episodes, horizon=horizon, state_dim=state_dim)

    expected = sum(len(range(0, retriever.reader.episode_length(ep), horizon)) for ep in episodes)
    assert len(rows["frame_index"]) == expected
    assert rows["action"].shape == (expected, config.action_horizon, state_dim)
    assert rows["obs"]["state"].shape == (expected, state_dim)
    for view in pi0.SIGLIP_VIEWS:
        assert rows["obs"][view].shape == (expected, 256, 1152)
        assert rows["obs"][view].dtype == np.float16

    # Rows are the control steps, in episode order, restarting at 0 for each episode.
    for episode in episodes:
        where = rows["episode_index"] == episode
        np.testing.assert_array_equal(
            rows["frame_index"][where],
            np.arange(0, retriever.reader.episode_length(episode), horizon),
        )

    # The same frame, encoded by the bank path, gives the same chunk and pose.
    encoded = retriever.encode_episode(episodes[0])  # frame_stride=20, so frame 40 is row 2
    first = np.flatnonzero(rows["episode_index"] == episodes[0])
    np.testing.assert_allclose(
        rows["action"][first[1]], encoded["actions"][2, :, :state_dim], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        rows["obs"]["state"][first[1]], encoded["states"][2, :state_dim], rtol=1e-6, atol=1e-6
    )


@requires_v21
def test_cotrain_rows_can_be_narrowed_to_the_modalities_a_critic_reads():
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model)
    episodes = retriever.episodes_for_task("beat_block_hammer")[:1]

    rows = retriever.cotrain_rows(
        episodes, horizon=60, modalities=("siglip.head", "state"), state_dim=14
    )
    assert set(rows["obs"]) == {"siglip.head", "state"}


@requires_v21
def test_cotrain_rows_reject_a_modality_no_demonstration_carries():
    config, model = _tiny_aloha_model()
    retriever = _retriever(config, model)
    with pytest.raises(KeyError, match="wrench.left"):
        retriever.cotrain_rows([0], horizon=50, modalities=("state", "wrench.left"))
