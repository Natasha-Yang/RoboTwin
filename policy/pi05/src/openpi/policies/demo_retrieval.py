"""Retrieve demonstration frames that look like the current observation, and what they did.

The point of this is to hand a steering critic something it cannot work out for itself: what a
*successful* demonstration of this task did from a state like the one the robot is in now. Two
forms of that reach the critic, as the modalities `action_proposals` and `noise_proposals`:

    action_proposals   (top_k, action_horizon, action_dim)   the demo's own action chunk
    noise_proposals    (top_k, action_horizon, action_dim)   the seed that maps to it here

Both are in pi0.5's own **normalized model space** -- the same space as the `action.model`
column of a rollout dataset and the chunk the critic is scored on -- so a critic can compare a
proposal with the action the sampler actually drew, term for term. The noise form is the more
directly actionable of the two: it is what `sample_actions` would have to be seeded with, under
*this* observation, to produce a demo-like chunk (see `Pi0.invert_actions`), so a critic can
learn about the sampler's input rather than only its output.

The pipeline. Steps 1 and 2 run **once per run**, on the first episode; step 3 every control
step. The bank is deliberately not re-drawn per episode -- it is a critic input, and changing it
partway through would evaluate a value learned against one set of demonstrations on another (see
`DemoRetriever.ensure_bank`).

  1. Find the demonstrations. The demo dataset is a LeRobot dataset of RoboTwin demos, typically
     the same one the policy was fine-tuned on, covering many tasks. Episodes are mapped to
     RoboTwin tasks by `openpi.training.episode_selection.assign_episode_tasks` -- *not* by
     instruction string, because one task expands into hundreds of different instructions and
     the dataset's own `task_index` indexes instructions, not tasks. `num_demos` of the episodes
     belonging to the task under evaluation are then drawn at random.
  2. Encode them. Every frame becomes a bank row: the **pooled** SigLIP embedding per camera
     view (`Pi0.embed_observation_maps`, averaged over patches), the robot's pose in the model's
     own normalized space, the normalized action chunk that was the policy's training target at
     that frame, and the frame's own model inputs -- all produced by pushing the raw frame
     through the policy's own input transform, so nothing here can drift from what the policy
     sees. Only the pooled form is kept: see step 3.
  3. Per control step, `Pi0.propose_from_demos` embeds the live observation with the same tower,
     takes the `top_k` **nearest** bank rows, and inverts their chunks under the live
     observation. "Nearest" is the **average of one independent relative L2 distance per
     signal** -- one per camera view plus one for the pose (`SIMILARITY_SIGNALS`, `signals` in
     the config). Each term is `||query - row|| / ||query||`, i.e. the distance as a fraction of
     the magnitude of the current observation's own vector for that signal, which is what makes
     terms of very different dimensionality and scale averageable without normalizing the
     magnitude away.

What the distance decides, and what it no longer decides. Step 3 is a **candidate pool**: it
picks which `top_k` rows are on offer, because inverting a chunk costs a pass of the action
expert and the bank is far too large to invert whole. Which of those candidates the value
actually rests on is then the critic's, learned rather than fixed -- `propose` also returns
`proposal_rows`, and a critic configured with `encoder: attn_proposals`
(`multisensory_steering.critics.qmfm.ProposalAttentionEncoder`) **cross-attends** them: its
query is the live observation as its own encoders represent it, the keys are those same encoders
run over the candidate rows, and the values are the encoded proposals. The keys are the reason
`DemoRetriever.critic_keys` exists -- the critic's image encoder takes the un-pooled SigLIP
**patch map**, so a demo frame has to be encodable exactly the way a live one is. A critic that
pools the set instead (`encoder: action_proposals`) ignores all of this and nothing about the
retrieval changes.

**Only the pooled embedding is kept per row.** A patch map is ~0.6 MB per view even at fp16, so
holding one per bank row was ~0.9 GB of device memory for a 512-row bank over three cameras --
paid for the whole run, on the same card the sampler is allocating from, and scaling with
`bank_size`. The bank keeps each row's uint8 model inputs instead (~0.15 MB per view, host RAM)
and `critic_keys` re-runs the image tower on the `top_k` rows a control step actually retrieved.
That trades a fixed, bank-sized block of device memory for one extra tower pass per control step
at batch `top_k`, and it is what lets `bank_size` grow.

Step 2 is still the expensive one -- one image-tower pass per demo frame, for the pooled
embeddings the distance ranks on -- but it is paid once for the whole run rather than once per
episode. Encoded episodes are additionally cached by index, so an explicit re-draw
(`select_bank`) that lands on the same demonstration costs nothing.
"""

import dataclasses
import functools
import io
import json
import logging
import os
import pathlib
import random
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.pi0 import SIGLIP_VIEWS
from openpi.shared import nnx_utils
from openpi.training import episode_selection
import openpi.transforms as _transforms

logger = logging.getLogger("openpi")

# Cameras, as the demo dataset column names them, mapped to what the policy's input transform
# calls them. RoboTwin LeRobot datasets are written by
# `examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py`, which uses these three.
DEMO_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")

# The signals retrieval compares on. `siglip` expands to one term per camera view, `state` is one
# term; each is an independent relative L2 distance and they are averaged with equal weight, so
# adding one dilutes the others. (A multimodal demo dataset now carries a contact wrench too --
# see WRENCH_COLUMN_PREFIX -- but it is not a similarity signal: a trace is NaN-padded where the
# episode has nothing behind it, and an L2 over NaN is NaN. It reaches the critic as a modality,
# not as part of the metric.)
SIMILARITY_SIGNALS = ("siglip", "state")

# The contact-wrench columns of a demo dataset, and the modality names they become. One column
# per end-effector link (aloha: `fl_link7`, `fl_link8`, `fr_link7`, `fr_link8`) plus one per arm
# (`left`, `right`), each holding **one `(6,)` row per frame** -- the world-frame contact wrench
# averaged over that frame's physics steps. A demo frame is one primitive step, so that row is
# the same quantity a rollout dataset stores per primitive step and the env logs online, and the
# `(wrench_trace_len, 6)` trace a critic observes is the window of them the previous chunk
# covered (`wrench_traces`). Datasets written before `script/demo_wrench_per_control_step.py`
# store the physics-step trace inside each frame instead, `(save_freq, 6)`; the reader averages
# those on the way in so both layouts reach the critic as the same thing.
WRENCH_COLUMN_PREFIX = "observation.wrench."
WRENCH_MODALITY_PREFIX = "wrench."
# Components per row, as `envs/utils/wrench.py::WRENCH_COMPONENTS` orders them.
WRENCH_COMPONENTS = 6

# Width the head-camera frame of every bank row is kept at when retrieval is being visualized
# (`record_retrieval`). Deliberately smaller than the rollout's own `DEBUG_GIF_FRAME_WIDTH`:
# there is one of these per *bank row*, not per control step, and a 512-row bank at full width
# would be over 100 MB of host RAM per episode kept in the encoding cache.
DEMO_THUMBNAIL_WIDTH = 160

# pi0.5's native aloha control mode: the six arm joints of each arm are commanded as deltas from
# the pose the chunk is conditioned on, the gripper as an absolute width. This is the same mask
# `LeRobotAlohaDataConfig` builds its `DeltaActions` with (`use_delta_joint_actions`), reused here
# rather than restated -- it is only needed for a config that left that transform out.
NATIVE_DELTA_MASK = _transforms.make_bool_mask(6, -1, 6, -1)

# The static jit arguments of `Pi0.propose_from_demos`. Changing any of them recompiles, which
# is why they are configuration rather than per-step state.
_PROPOSE_STATIC = ("top_k", "views", "invert", "num_steps", "num_inner_steps", "num_substeps", "return_info")


def resolve_views(views) -> tuple[str, ...]:
    """Camera view names -> `siglip.<view>` modality names, in the model's canonical order.

    Accepts the short names (`head`, `left_wrist`) or the modalities themselves, and `None` /
    `true` for all of them. The order of the result is `Pi0.SIGLIP_VIEWS`, not the caller's,
    because it is the axis a bank's embeddings are stacked along.
    """
    if views is None or isinstance(views, bool):
        return SIGLIP_VIEWS if views is not False else ()
    if isinstance(views, str):
        views = [views]
    wanted = {v if str(v).startswith("siglip.") else f"siglip.{v}" for v in views}
    if unknown := sorted(wanted - set(SIGLIP_VIEWS)):
        raise ValueError(
            f"retrieval views {unknown} are not camera views the policy has. "
            f"Available: {[v.split('.', 1)[1] for v in SIGLIP_VIEWS]}."
        )
    if not wanted:
        raise ValueError("retrieval needs at least one camera view.")
    return tuple(view for view in SIGLIP_VIEWS if view in wanted)


def resolve_signals(signals) -> tuple[str, ...]:
    """Which signals the distance averages over, in `SIMILARITY_SIGNALS` order."""
    if signals is None or signals is True:
        return SIMILARITY_SIGNALS
    if isinstance(signals, str):
        signals = [signals]
    wanted = {str(name) for name in signals}
    if unknown := sorted(wanted - set(SIMILARITY_SIGNALS)):
        raise ValueError(f"unknown similarity signal(s) {unknown}; expected {list(SIMILARITY_SIGNALS)}.")
    if "siglip" not in wanted:
        raise ValueError("`siglip` is required: it is the only signal that identifies the scene.")
    return tuple(name for name in SIMILARITY_SIGNALS if name in wanted)


# ---------------------------------------------------------------------------------------------
# Reading a LeRobot dataset without going through `lerobot`
# ---------------------------------------------------------------------------------------------


class LeRobotEpisodeReader:
    """Just enough of the LeRobot on-disk format to pull whole episodes out of it.

    Deliberately not `lerobot.LeRobotDataset`: the installed lerobot refuses datasets written in
    a newer codebase version than its own (a v3.0 dataset raises `ForwardCompatibilityError`
    against a v2.x install), and the two layouts differ only in where the episode index lives.
    Both are read here:

        v2.1  one parquet per episode      + `meta/episodes.jsonl`
        v3.0  parquets shared by episodes  + `meta/episodes/**.parquet`

    Only what retrieval needs is read -- the instruction per episode, and one episode's states,
    actions, camera frames and (where the dataset has them) contact-wrench rows at a time.
    Images are stored as encoded bytes in both layouts (the RoboTwin converter writes `image`,
    not `video`, features) and are decoded on demand.
    """

    def __init__(self, repo_id: str, root: str | os.PathLike | None = None):
        self.repo_id = repo_id
        self.root = pathlib.Path(root) if root is not None else default_root() / repo_id
        if not (self.root / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"No LeRobot dataset at {self.root} (expected meta/info.json). Set the demo "
                f"dataset's `root`, or make sure {repo_id!r} has been downloaded."
            )
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.version = str(self.info.get("codebase_version", "v2.1"))
        self._episodes = self._read_episode_meta()

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def instructions(self) -> list[str | None]:
        """Each episode's instruction, in episode-index order (the order task assignment needs)."""
        return [ep["instruction"] for ep in self._episodes]

    def episode_length(self, episode_index: int) -> int:
        return int(self._episodes[episode_index]["length"])

    @functools.cached_property
    def wrench_columns(self) -> tuple[str, ...]:
        """The dataset's `observation.wrench.*` columns, or `()` for one collected without them.

        Read off `meta/info.json` rather than a parquet, so a caller can ask what a dataset
        carries before opening an episode. Which keys exist follows the embodiment's URDF link
        names plus the two arm totals, exactly as the demo HDF5 and a rollout dataset name them.
        """
        return tuple(sorted(n for n in self.info.get("features", {}) if n.startswith(WRENCH_COLUMN_PREFIX)))

    @property
    def wrench_keys(self) -> tuple[str, ...]:
        """`wrench_columns` with the `observation.wrench.` prefix stripped: `fl_link7`, `left`, ..."""
        return tuple(name[len(WRENCH_COLUMN_PREFIX) :] for name in self.wrench_columns)

    def _read_episode_meta(self) -> list[dict]:
        rows: dict[int, dict] = {}
        v3_dir = self.root / "meta" / "episodes"
        if v3_dir.is_dir():
            import pyarrow.parquet as pq

            for path in sorted(v3_dir.rglob("*.parquet")):
                table = pq.read_table(path, columns=["episode_index", "tasks", "length", *_V3_LOCATOR])
                for row in table.to_pylist():
                    rows[int(row["episode_index"])] = {
                        "instruction": _first_task(row.get("tasks")),
                        "length": row["length"],
                        "locator": (row["data/chunk_index"], row["data/file_index"]),
                    }
        else:
            with (self.root / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    rows[int(row["episode_index"])] = {
                        "instruction": _first_task(row.get("tasks")),
                        "length": row["length"],
                        "locator": None,
                    }
        if sorted(rows) != list(range(len(rows))):
            raise ValueError(f"{self.repo_id} has non-contiguous episode indices; cannot order them.")
        return [rows[i] for i in range(len(rows))]

    def read_episode(self, episode_index: int, frames=None) -> dict:
        """One episode's `{state, action, images, wrench, frames}`, states/actions `(T, 14)` f32.

        `images` maps each camera to a `(len(frames), H, W, 3)` uint8 array in the layout the
        policy's input transform expects -- channel-first, as `AlohaInputs` reads it -- decoded
        here so the caller never sees the storage format.

        `wrench` is `{key: (T, 6)}` for a dataset that has the columns and `{}` for one that
        does not -- one row per frame, the contact wrench averaged over that frame. It is
        returned whole for the same reason the states are: the trace a control step observes is
        the *previous* chunk's, so a kept frame needs the rows before it (`wrench_traces`).

        `frames` selects which frames are *decoded*, as indices into the episode's own time
        order; `None` is all of them. The states, actions and wrench are always returned whole,
        because a frame's action chunk reaches `action_horizon` steps past it and would
        otherwise be cut off -- they are float32 columns, while decoding a frame is three JPEGs.
        So a caller that only wants every horizon-th frame (`DemoRetriever.cotrain_rows`) pays
        for the frames it keeps rather than for the episode. `frames` is returned alongside,
        since the image arrays are indexed by *position in it*, not by frame index.
        """
        import pyarrow.parquet as pq

        meta = self._episodes[episode_index]
        columns = [
            "observation.state",
            "action",
            "frame_index",
            "episode_index",
            *_image_columns(),
            *self.wrench_columns,
        ]
        if meta["locator"] is not None:  # v3.0: many episodes share one file
            chunk, file = meta["locator"]
            path = self.root / self.info["data_path"].format(chunk_index=chunk, file_index=file)
            table = pq.read_table(path, columns=columns)
            table = table.filter(np.asarray(table["episode_index"]) == episode_index)
        else:  # v2.1: one file per episode
            chunk = episode_index // int(self.info.get("chunks_size", 1000))
            path = self.root / self.info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
            table = pq.read_table(path, columns=columns)

        order = np.argsort(np.asarray(table["frame_index"]))
        state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False))[order].astype(np.float32)
        action = np.stack(table["action"].to_numpy(zero_copy_only=False))[order].astype(np.float32)
        keep = np.arange(len(order)) if frames is None else np.asarray(frames, dtype=np.int64).reshape(-1)
        if keep.size and (keep.min() < 0 or keep.max() >= len(order)):
            raise IndexError(f"episode {episode_index} has {len(order)} frames; asked for {keep.tolist()}.")
        images = {}
        for camera in DEMO_CAMERAS:
            encoded = table[f"observation.images.{camera}"].to_pylist()
            decoded = np.stack([_decode_image(encoded[i]) for i in order[keep]])
            # AlohaInputs takes [channel, height, width] and transposes it back itself; the eval
            # path hands it images the same way round (see deploy_policy.encode_obs).
            images[camera] = np.transpose(decoded, (0, 3, 1, 2))
        wrench = {
            column[len(WRENCH_COLUMN_PREFIX) :]: _per_frame_wrench(table[column], order, column)
            for column in self.wrench_columns
        }
        return {
            "state": state,
            "action": action,
            "images": images,
            "wrench": wrench,
            "frames": keep,
        }


_V3_LOCATOR = ("data/chunk_index", "data/file_index")


def _image_columns() -> list[str]:
    return [f"observation.images.{camera}" for camera in DEMO_CAMERAS]


def _first_task(tasks) -> str | None:
    if tasks is None:
        return None
    if isinstance(tasks, str):
        return tasks.strip()
    tasks = list(tasks)
    return tasks[0].strip() if tasks else None


def _thumbnail(frame_chw: np.ndarray) -> np.ndarray:
    """A channel-first demo frame -> a small HWC uint8 thumbnail for the debug GIF."""
    from PIL import Image

    img = Image.fromarray(np.transpose(np.asarray(frame_chw, dtype=np.uint8), (1, 2, 0)))
    if img.width > DEMO_THUMBNAIL_WIDTH:
        height = max(1, round(img.height * DEMO_THUMBNAIL_WIDTH / img.width))
        img = img.resize((DEMO_THUMBNAIL_WIDTH, height), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _per_frame_wrench(column, order, name: str) -> np.ndarray:
    """One `(T, 6)` row per frame from a wrench column, whichever way it is stored.

    A dataset written to the current convention already holds one averaged row per frame. One
    written before `script/demo_wrench_per_control_step.py` holds that frame's own physics-step
    trace, `(save_freq, 6)`, NaN-padded where the frame had fewer samples behind it -- averaged
    here, nan-aware, so the two layouts are indistinguishable downstream. Zero is a meaningful
    reading (touching nothing), which is why the padding is NaN and why the mean has to skip it
    rather than count it.
    """
    values = np.asarray(column.to_pylist(), dtype=np.float32)[order]
    if values.ndim == 3:  # legacy (frames, physics steps, 6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN frame -> NaN, as it should
            values = np.nanmean(values, axis=1)
    if values.ndim != 2 or values.shape[-1] != WRENCH_COMPONENTS:
        raise ValueError(
            f"column {name!r} is {values.shape[1:]} per frame, not a {WRENCH_COMPONENTS}-component "
            f"wrench (or a trace of them)."
        )
    return values.astype(np.float32)


def wrench_traces(per_frame: dict, frames, trace_len: int) -> dict[str, np.ndarray]:
    """Per-frame wrench rows -> the `(trace_len, 6)` trace each of `frames` observes.

    The critic's `wrench.*` modality is the trace the **previous** chunk left: the primitive
    steps between the last observation and this one, which is the only wrench that exists before
    the current chunk has been executed (CLAUDE.md 6.2 / 7a). A demo frame is one primitive step
    and its row covers the physics steps leading up to it, so the window for frame `t` is the
    `trace_len` rows ending **at** `t` -- `t - trace_len + 1 ... t` -- which is exactly the span
    a control step at `t` would have drained had a policy been driving.

    Shorter windows are NaN-padded at the tail, matching `envs/utils/wrench.py::stack_step_wrench`:
    frame 0 has nothing behind it and carries a single sample of the contact state at that
    instant, the same as an episode's first drain online. NaN rather than zero, because zero is
    a reading.

    Returns `{f"wrench.{key}": (len(frames), trace_len, 6)}`, empty when the dataset has none.
    """
    frames = np.asarray(frames, dtype=np.int64).reshape(-1)
    trace_len = int(trace_len)
    if trace_len < 1:
        raise ValueError(f"wrench trace length must be >= 1 primitive step, got {trace_len}.")
    out = {}
    for key, values in per_frame.items():
        traces = np.full((len(frames), trace_len, WRENCH_COMPONENTS), np.nan, dtype=np.float32)
        for i, t in enumerate(frames.tolist()):
            window = values[max(0, t - trace_len + 1) : t + 1]
            traces[i, : len(window)] = window
        out[f"{WRENCH_MODALITY_PREFIX}{key}"] = traces
    return out


def _decode_image(cell) -> np.ndarray:
    from PIL import Image

    data = cell["bytes"] if isinstance(cell, dict) else cell
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


def _relative_l2_np(dot, query_sq, bank_sq):
    """``Pi0._relative_l2`` in numpy: ``||q - b|| / ||q||`` from the expanded form.

    Same clipping, same denominator floor -- the two have to agree term for term, or offline
    rows retrieve different demonstrations from the ones the live rollout retrieves and a
    co-trained batch is conditioned two different ways.
    """
    distance = np.sqrt(np.maximum(query_sq + bank_sq - 2.0 * dot, 0.0))
    return distance / np.maximum(np.sqrt(query_sq), 1e-6)


def default_root() -> pathlib.Path:
    env = os.environ.get("HF_LEROBOT_HOME")
    if env:
        return pathlib.Path(env).expanduser()
    return pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "lerobot"


# ---------------------------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DemoBank:
    """A fixed-size, model-space index of demonstration frames.

    Fixed size on purpose: the bank is a jit argument, so letting it follow the demos' own
    lengths would recompile `propose_from_demos` on every eval episode. Rows beyond the frames
    actually held are padding, marked ineligible by `mask` (the distance to a padded row is
    forced to +inf, so it can never be retrieved).
    """

    embeddings: np.ndarray  # (bank_size, num_views, 1152) float32, unit norm per view
    actions: np.ndarray  # (bank_size, action_horizon, action_dim) float32, normalized
    mask: np.ndarray  # (bank_size,) bool
    # The rows' own **model inputs**, as the policy's input transform produced them:
    # `{"image": {cam: (num_frames, 224, 224, 3) uint8}, "state": ..., "tokenized_prompt": ...}`.
    # A critic that cross-attends the bank builds its attention keys by running its own image
    # encoder over a row, which takes the SigLIP *patch map* -- and one of those is ~0.6 MB per
    # view per row, i.e. ~0.9 GB of device memory for a 512-row bank over three cameras. So the
    # bank keeps the frames the map is computed *from* (~0.15 MB per view per row, host RAM,
    # and uint8) and `DemoRetriever.critic_keys` re-encodes only the `top_k` rows a control
    # step actually retrieved. Unlike everything else here this has exactly `num_frames` rows
    # and no padding: a padded row sits at +inf distance and can never be retrieved.
    obs: dict
    states: np.ndarray  # (bank_size, action_dim) float32, the model's own normalized pose
    # The non-visual signals the distance also averages over, one (bank_size, d) array per name
    # (see SIMILARITY_SIGNALS), stored at their natural magnitude -- the scaling is per query,
    # not per vector. Empty when retrieval is on `siglip` alone.
    extra: dict[str, np.ndarray]
    # The rows' contact-wrench traces, `{"wrench.<key>": (num_frames, wrench_trace_len, 6)}`, for
    # a demo dataset that carries the columns and a run that set `wrench_trace_len`; `{}`
    # otherwise. Like `obs` these are unpadded and indexed on the host by rows a retrieval
    # returned, and for the same reason: a padded row sits at +inf distance and is never one.
    wrench: dict[str, np.ndarray]
    views: tuple[str, ...]
    task: str
    episodes: tuple[int, ...]
    num_frames: int
    # Head-camera thumbnail per row, for the debug visualization; None unless `record_retrieval`
    # was set before the bank was built. `row_episode` / `row_frame` say where each row came
    # from, so a plot can be labelled with the demo episode and frame it retrieved.
    thumbnails: np.ndarray | None = None
    row_episode: np.ndarray | None = None
    row_frame: np.ndarray | None = None

    def describe(self) -> str:
        return (
            f"{self.num_frames} frames from {len(self.episodes)} demo episode(s) "
            f"{list(self.episodes)} of {self.task!r}, padded to {len(self.mask)}"
        )

    def describe_keys(self) -> str:
        """What a row keeps for the cross attention, and its cost, for the startup banner."""
        views = [view.split(".", 1)[1] for view in self.views]
        frames = sum(v.nbytes for v in self.obs.get("image", {}).values())
        wrench = (
            ""
            if not self.wrench
            else f" + {len(self.wrench)} wrench trace(s) "
            f"{next(iter(self.wrench.values())).shape[1:]}"
        )
        return (
            f"model inputs ({', '.join(views)}) + {self.states.shape[-1]}-d pose{wrench} per row, "
            f"{(frames + self.states.nbytes) / 1e9:.2f} GB; the SigLIP patch maps a critic key "
            f"is built from are re-encoded per control step for the retrieved rows only"
        )


class DemoRetriever:
    """Builds demo banks for a policy, and turns an observation into proposals.

    Holds the (small) configuration, the reader, and a per-episode cache of encoded frames --
    encoding is the expensive part and an episode re-drawn later in the run is then free.
    """

    def __init__(
        self,
        model,
        input_transform,
        *,
        repo_id: str,
        root: str | os.PathLike | None = None,
        num_demos: int = 1,
        top_k: int = 1,
        views=None,
        signals=None,
        bank_size: int = 512,
        frame_stride: int = 1,
        num_steps: int = 10,
        num_inner_steps: int = 10,
        num_substeps: int = 1,
        invert: bool = True,
        wrench_trace_len: int | None = None,
        encode_batch_size: int = 16,
        debug_top_k: int = 3,
        seed: int = 0,
    ):
        self.model = model
        self.input_transform = input_transform
        self.reader = LeRobotEpisodeReader(repo_id, root)
        self.repo_id = repo_id
        self.num_demos = max(1, int(num_demos))
        self.top_k = max(1, int(top_k))
        self.views = resolve_views(views)
        self.signals = resolve_signals(signals)
        self.bank_size = int(bank_size)
        self.frame_stride = max(1, int(frame_stride))
        self.num_steps = int(num_steps)
        self.num_inner_steps = int(num_inner_steps)
        self.num_substeps = int(num_substeps)
        self.invert = bool(invert)
        # Rows of a `wrench.*` modality, i.e. how many primitive steps of contact history a
        # frame is paired with. It is an architecture key on the critic side (its obs shape is
        # settled before the first row arrives), so it is passed in from the run's own
        # `wrench_trace_len` rather than guessed here; None means "no wrench in the bank", and
        # `cotrain_rows` falls back to its `horizon`, which is what the key defaults to anyway.
        self.wrench_trace_len = int(wrench_trace_len) if wrench_trace_len else None
        self.encode_batch_size = max(1, int(encode_batch_size))
        # How many matches the debug visualization shows. Independent of `top_k` -- the
        # distance over the whole bank comes back anyway, so showing more neighbours than the
        # critic is given costs nothing and is usually what you want to look at: seeing ranks 2
        # and 3 is how you tell a confident match from the best of a bad set.
        self.debug_top_k = max(1, int(debug_top_k))
        self.rng = random.Random(seed)

        if self.top_k > self.bank_size:
            raise ValueError(f"top_k={self.top_k} exceeds bank_size={self.bank_size}.")

        self.action_horizon = int(model.action_horizon)
        self.action_dim = int(model.action_dim)
        # Whether the policy's own chain makes actions relative, and -- if it does not -- pi0.5's
        # own `DeltaActions` to do it here instead. See `_probe_action_space`.
        self.delta_dims = self._probe_action_space()
        self._delta_fallback = None
        if self.delta_dims.size == 0:
            self._delta_fallback = _transforms.DeltaActions(NATIVE_DELTA_MASK)
            self.delta_dims = np.flatnonzero(NATIVE_DELTA_MASK)
        self.bank: DemoBank | None = None
        # Debug visualization: keep a head-camera thumbnail per bank row and remember which rows
        # each control step retrieved, so the driver can draw the query frame beside the demo
        # frames it matched (`envs/utils/debug_vis.py::DemoRetrievalRecorder`). Off by default --
        # the thumbnails are the only part of a demo episode that would otherwise be discarded
        # after encoding. Set by the eval driver before the first bank is built.
        self.record_retrieval = False
        self.last_distance: np.ndarray | None = None
        self._encoded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
        # The un-pooled maps: the bank needs both forms and this is the one they both come from
        # (`encode_episode` pools it for the distance).
        self._embed = nnx_utils.module_jit(model.embed_observation_maps)
        self._propose = nnx_utils.module_jit(model.propose_from_demos, static_argnames=_PROPOSE_STATIC)

    def _probe_action_space(self) -> np.ndarray:
        """Which action dims the policy's transform encodes *relative to the current pose*.

        A proposal is only meaningful to a critic if it is the same kind of quantity as the
        chunk the sampler drew, and pi0.5's aloha chain (`LeRobotAlohaDataConfig`,
        `use_delta_joint_actions`) makes that a **delta**: `DeltaActions` subtracts the state the
        chunk is conditioned on from the arm-joint dims, leaving the two gripper dims absolute.
        So a bank chunk is the demo's motion *away from the demo's own pose at that frame*, not
        an absolute joint target, and `AbsoluteActions` adds the live state back on the way out.

        This is checked rather than assumed, by pushing the same actions through the transform
        under two different states and seeing which dims move. Which of the two cases holds
        decides what `encode_episode` has to do (see `_relative_actions`):

        * **some dims move** -- the chain is delta-encoding and the ones that do not are
          deliberate (aloha's two gripper dims are absolute widths, `make_bool_mask(6,-1,6,-1)`).
          The bank is left exactly as the transform produced it, which is also exactly the space
          the sampler's own chunk is in.
        * **no dim moves** -- the config was built with `use_delta_joint_actions=False`, so the
          bank would hold another episode's *absolute* joint targets. pi0.5's own
          `DeltaActions(NATIVE_DELTA_MASK)` is applied here instead, so a proposal is always
          motion relative to a pose, in pi0.5's native control mode either way.
        """
        probe_images = {camera: np.zeros((3, 224, 224), dtype=np.uint8) for camera in DEMO_CAMERAS}
        actions = np.linspace(-0.2, 0.2, self.action_horizon * 14, dtype=np.float32).reshape(-1, 14)

        def encode(state):
            return np.asarray(
                self.input_transform(
                    {
                        "images": {k: v.copy() for k, v in probe_images.items()},
                        "state": np.full(14, state, dtype=np.float32),
                        "actions": actions.copy(),
                        "prompt": "",
                    }
                )["actions"]
            )

        moved = np.abs(encode(0.0) - encode(0.3)).max(axis=0)[:14] > 1e-6
        return np.flatnonzero(moved)

    @property
    def wrench_modalities(self) -> tuple[str, ...]:
        """The `wrench.<key>` modalities this demo dataset can produce, `()` if it has none."""
        return tuple(f"{WRENCH_MODALITY_PREFIX}{key}" for key in self.reader.wrench_keys)

    def _wrench_trace_len(self, fallback: int | None = None) -> int | None:
        """The trace length to build with, or None when there is no wrench to build."""
        if not self.reader.wrench_columns:
            return None
        if self.wrench_trace_len is not None:
            return self.wrench_trace_len
        return None if fallback is None else int(fallback)

    @property
    def proposal_shape(self) -> tuple[int, int, int]:
        """The shape of one proposal modality, before it is narrowed to embodiment dims."""
        return (self.top_k, self.action_horizon, self.action_dim)

    @functools.cached_property
    def episode_tasks(self) -> list[str]:
        """Every demo episode's RoboTwin task name, in episode-index order."""
        logger.info("Resolving %d demo episodes to RoboTwin tasks...", self.reader.num_episodes)
        return episode_selection.assign_episode_tasks(self.reader.instructions())

    def episodes_for_task(self, task_name: str) -> list[int]:
        matching = [i for i, name in enumerate(self.episode_tasks) if name == task_name]
        if not matching:
            available = sorted(set(self.episode_tasks))
            raise ValueError(
                f"demo dataset {self.repo_id!r} has no episodes of task {task_name!r}. " f"It covers: {available}."
            )
        return matching

    def ensure_bank(self, task_name: str) -> DemoBank:
        """The run's demo bank -- drawn once, on the first episode, and held for all of them.

        Deliberately **not** per episode. The proposals are a critic input, so re-drawing would
        change what the critic is conditioned on partway through: a value learned against one set
        of demonstrations would then be evaluated against another, and two episodes run with the
        same seed would not be comparable. Holding one bank also makes the choice reproducible
        from `seed` alone and survives a resume, since the bank is rebuilt identically rather
        than restored.

        `select_bank` is the underlying draw, for a caller that deliberately wants a new one.
        """
        if self.bank is None or self.bank.task != task_name:
            self.select_bank(task_name)
        return self.bank

    def select_bank(self, task_name: str) -> DemoBank:
        """Draw `num_demos` demonstrations of `task_name` and build the bank for them.

        The draw comes off this object's own seeded RNG, so it is reproducible from `seed`.
        Callers in the eval loop want `ensure_bank`: this always draws a *new* bank, which
        during a run would move the critic's conditioning underneath it.
        """
        available = self.episodes_for_task(task_name)
        k = min(self.num_demos, len(available))
        if k < self.num_demos:
            logger.warning("Only %d demo episode(s) of %s available; asked for %d.", k, task_name, self.num_demos)
        chosen = sorted(self.rng.sample(available, k))
        self.bank = self._build_bank(task_name, chosen)
        return self.bank

    def _build_bank(self, task_name: str, episodes: list[int]) -> DemoBank:
        embeddings, obs, actions, states, thumbs = [], [], [], [], []
        wrench: list[dict] = []
        row_episode, row_frame = [], []
        for episode in episodes:
            encoded = self.encode_episode(episode)
            embeddings.append(encoded["embeddings"])
            obs.append(encoded["obs"])
            actions.append(encoded["actions"])
            states.append(encoded["states"])
            wrench.append(encoded["wrench"])
            if encoded["thumbnails"] is not None:
                thumbs.append(encoded["thumbnails"])
            rows = len(encoded["embeddings"])
            row_episode.append(np.full(rows, episode, dtype=np.int32))
            row_frame.append(np.arange(rows, dtype=np.int32) * self.frame_stride)
        embeddings = np.concatenate(embeddings, axis=0)
        obs = jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *obs)
        actions = np.concatenate(actions, axis=0)
        states = np.concatenate(states, axis=0)
        wrench = {name: np.concatenate([w[name] for w in wrench]) for name in wrench[0]}
        # The distance's non-visual terms are a view on the same poses, not a second copy.
        extra = {"state": states} if "state" in self.signals else {}
        thumbs = np.concatenate(thumbs, axis=0) if thumbs else None
        row_episode = np.concatenate(row_episode)
        row_frame = np.concatenate(row_frame)

        if len(embeddings) > self.bank_size:
            # Subsample uniformly rather than truncating: a demo's later frames are the part a
            # rollout reaches last, and dropping them would leave the end of the task unindexed.
            keep = np.linspace(0, len(embeddings) - 1, self.bank_size).round().astype(int)
            logger.warning(
                "Demo bank for %s holds %d frames, over bank_size=%d; subsampling uniformly.",
                task_name,
                len(embeddings),
                self.bank_size,
            )
            embeddings, actions = embeddings[keep], actions[keep]
            obs, states = jax.tree.map(lambda x: x[keep], obs), states[keep]
            wrench = {name: values[keep] for name, values in wrench.items()}
            extra = {name: values[keep] for name, values in extra.items()}
            row_episode, row_frame = row_episode[keep], row_frame[keep]
            if thumbs is not None:
                thumbs = thumbs[keep]

        num_frames = len(embeddings)
        if num_frames < self.top_k:
            # Padding rows sit at +inf distance, so a top_k wider than the bank would return
            # zero-filled proposals that look like real ones. Fail instead.
            raise ValueError(
                f"demo bank for {task_name} holds only {num_frames} frames but top_k={self.top_k}. "
                f"Lower top_k, lower frame_stride, or raise num_demos."
            )
        pad = self.bank_size - num_frames
        mask = np.concatenate([np.ones(num_frames, bool), np.zeros(pad, bool)])
        if pad:
            # `obs` is deliberately left unpadded -- it is indexed on the host by rows that came
            # back from a retrieval, and a padded row sits at +inf distance and never does.
            embeddings = np.concatenate([embeddings, np.zeros((pad, *embeddings.shape[1:]), np.float32)])
            actions = np.concatenate([actions, np.zeros((pad, *actions.shape[1:]), np.float32)])
            states = np.concatenate([states, np.zeros((pad, *states.shape[1:]), np.float32)])
            extra = {
                name: np.concatenate([values, np.zeros((pad, *values.shape[1:]), np.float32)])
                for name, values in extra.items()
            }
            row_episode = np.concatenate([row_episode, np.full(pad, -1, np.int32)])
            row_frame = np.concatenate([row_frame, np.full(pad, -1, np.int32)])
            if thumbs is not None:
                thumbs = np.concatenate([thumbs, np.zeros((pad, *thumbs.shape[1:]), np.uint8)])
        bank = DemoBank(
            embeddings=embeddings,
            actions=actions,
            mask=mask,
            obs=obs,
            states=states,
            extra=extra,
            wrench=wrench,
            views=self.views,
            task=task_name,
            episodes=tuple(episodes),
            num_frames=num_frames,
            thumbnails=thumbs,
            row_episode=row_episode,
            row_frame=row_frame,
        )
        logger.info("Demo bank: %s", bank.describe())
        logger.info("Demo bank rows carry their own observations for cross attention: %s", bank.describe_keys())
        return bank

    def encode_episode(self, episode: int) -> dict:
        """One episode as bank rows: ``embeddings`` / ``obs`` / ``actions`` / ``states``.

        All of it comes out of the policy's own input transform, one frame at a time: the same
        resize, the same aloha decoding, the same delta-action conversion against that frame's
        own state, and the same normalization the policy was trained with. That is the whole
        reason to go through the transform rather than reading the parquet's actions directly --
        the bank has to live in the space the sampler works in.

        The tower pass keeps only what it pools to: ``embeddings``, the vector the L2 distance
        ranks on. The un-pooled patch map a critic key is built from is *not* kept -- at ~0.6 MB
        per view per frame it is the whole memory cost of a bank -- so what a cached episode
        holds instead is ``obs``, the frames' own model inputs (uint8, ~0.15 MB per view), which
        `critic_keys` re-encodes for the handful of rows a control step retrieves.
        """
        if episode in self._encoded:
            return self._encoded[episode]

        frames = np.arange(0, self.reader.episode_length(episode), self.frame_stride)
        transformed = self._transform_frames(episode, frames, thumbnails=self.record_retrieval)
        obs = transformed["obs"]

        embeddings = []
        for maps in self._tower_batches(obs):
            if missing := [view for view in self.views if view not in maps]:
                raise ValueError(f"the policy does not expose {missing} to the critic.")
            stack = np.stack([np.asarray(maps[view], dtype=np.float32) for view in self.views], axis=1)
            # Pooled here and the maps dropped: the pooling is a sum over 256 terms and is done
            # once per row rather than per query, so the distance's side of the bank is fp32 and
            # 256x smaller than what it came from.
            embeddings.append(stack.mean(axis=-2))

        encoded = {
            "embeddings": np.concatenate(embeddings).astype(np.float32),
            "obs": obs,
            "actions": transformed["actions"],
            "states": transformed["states"],
            "wrench": transformed["wrench"],
            "thumbnails": transformed["thumbnails"],
        }
        self._encoded[episode] = encoded
        return encoded

    def _transform_frames(
        self, episode: int, frames, *, thumbnails: bool = False, wrench_trace_len: int | None = None
    ) -> dict:
        """Push `frames` of one demo episode through the policy's own input transform.

        The shared half of `encode_episode` and `cotrain_rows`: everything a demo frame becomes
        before a tower ever runs -- its model inputs, its normalized pose, and the normalized
        action chunk that was the policy's training target at it. Going through the transform
        rather than reading the parquet's columns directly is the whole point: the same resize,
        the same aloha decoding, the same delta-action conversion against that frame's own
        state, and the same normalization the policy was trained with, so nothing here can drift
        from what the policy sees.

        The contact wrench is the one thing here the transform has nothing to say about -- it
        is not part of an observation the policy sees -- so it is windowed straight from the
        dataset's own per-frame rows (`wrench_traces`) at `wrench_trace_len` rows, defaulting to
        the retriever's own. `{}` when the dataset has no wrench columns or no length is known.

        Returns ``{"obs", "actions", "states", "wrench", "thumbnails", "frames"}``.
        """
        frames = np.asarray(frames, dtype=np.int64).reshape(-1)
        raw = self.reader.read_episode(episode, frames=frames)
        instruction = self.reader.instructions()[episode] or ""
        length = len(raw["state"])

        batch, chunks, thumbs, states = [], [], [], []
        for i, t in enumerate(frames.tolist()):
            # The chunk that was the training target at frame t. LeRobot pads a chunk that runs
            # past the end of the episode by holding the last action, and so does this.
            window = np.arange(t, t + self.action_horizon).clip(max=length - 1)
            inputs = self.input_transform(
                {
                    # The image arrays hold only the frames that were decoded, so they are
                    # indexed by position in `frames` rather than by frame index.
                    "images": {camera: raw["images"][camera][i] for camera in DEMO_CAMERAS},
                    "state": raw["state"][t],
                    "actions": self._relative_actions(raw["action"][window], raw["state"][t]),
                    "prompt": instruction,
                }
            )
            chunks.append(np.asarray(inputs["actions"], dtype=np.float32))
            # `actions` is not part of an Observation; it rode along only to be normalized.
            inputs.pop("actions", None)
            # The pose in the model's own normalized space, the same vector the sampler is
            # conditioned on -- so a pose distance compares like with like, and the zero padding
            # past the embodiment's dims contributes nothing to it. Collected whether or not the
            # distance uses it: it is also the pose half of a cross-attention key, and there it
            # is not optional (`critic_keys`).
            states.append(np.asarray(inputs["state"], dtype=np.float32))
            # Kept as the transform produced it, which for the camera views means uint8 at the
            # model's own 224x224: `Observation.from_dict` is what turns those into floats, and
            # doing it here would quadruple what the bank holds.
            batch.append(jax.tree.map(np.asarray, inputs))
            if thumbnails:
                # The head camera as the sim shows it, not as the model sees it: the debug GIF
                # is for a human comparing the rollout with the demo, so the un-resized,
                # un-padded frame is the honest one to draw.
                thumbs.append(_thumbnail(raw["images"]["cam_high"][i]))

        trace_len = self._wrench_trace_len(wrench_trace_len)
        return {
            "obs": jax.tree.map(lambda *xs: np.stack(xs), *batch),
            "actions": np.stack(chunks),
            "states": np.stack(states),
            "wrench": {} if trace_len is None else wrench_traces(raw["wrench"], frames, trace_len),
            "thumbnails": np.stack(thumbs) if thumbs else None,
            "frames": frames,
        }

    def _tower_batches(self, obs):
        """Yield the image tower's un-pooled patch maps for `obs`, ``encode_batch_size`` at a time.

        Batched rather than run whole because a patch map is ~1.2 MB per view per frame in
        float32: a 220-frame episode over three views is most of a gigabyte, and both callers
        immediately reduce it (pooled to 1152 dims for the bank, cast to fp16 for a co-training
        row). Keeping the reduction inside the loop is what makes an episode's encoding cost
        bounded by `encode_batch_size` rather than by its length.
        """
        rows = len(jax.tree.leaves(obs)[0])
        for start in range(0, rows, self.encode_batch_size):
            group = jax.tree.map(
                lambda x: jnp.asarray(x[start : start + self.encode_batch_size]), obs  # noqa: B023
            )
            yield self._embed(_model.Observation.from_dict(group))

    def _relative_actions(self, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        """A demo's raw action chunk, made relative to the pose it was taken from.

        A no-op in the normal case: the policy's own `DeltaActions` is already inside
        `input_transform` and does exactly this a moment later, for the dims it is masked to.
        The fallback runs only when the chain delta-encodes nothing
        (`use_delta_joint_actions=False`), where the demo's actions are absolute joint targets
        that would otherwise reach the critic as "go to episode 74's pose" -- true of that demo,
        and useless from here.

        It is pi0.5's own `DeltaActions` doing the work either way, at the same point in the
        pipeline (before `Normalize`, since a delta must be normalized by the action statistics
        and differencing in normalized space would mix in the state's) and under the same mask,
        so the two paths put the bank in one control mode rather than two.
        """
        if self._delta_fallback is None:
            return actions
        # DeltaActions edits `actions` in place; the caller's array is a parquet slice.
        return self._delta_fallback(
            {"state": np.asarray(state, dtype=np.float32), "actions": np.array(actions, dtype=np.float32)}
        )["actions"]

    @property
    def converts_actions(self) -> bool:
        """Whether the bank is delta-encoded here rather than by the policy's own transform."""
        return self._delta_fallback is not None

    def describe_action_space(self) -> str:
        """One line on what space the proposals are in, for the startup banner."""
        absolute = sorted(set(range(int(self.delta_dims.max()) + 1)) - set(self.delta_dims.tolist()))
        source = (
            "policy transform"
            if self._delta_fallback is None
            else "pi0.5's native DeltaActions, applied here (policy chain has none)"
        )
        return (
            f"dims {self.delta_dims.tolist()} relative to the pose the chunk is conditioned on"
            + (f", dims {absolute} absolute (grippers)" if absolute else "")
            + f" [{source}]"
        )

    def propose(self, observation_window: dict) -> dict[str, np.ndarray]:
        """This control step's proposals, unbatched, keyed by modality name.

        `observation_window` is the raw dict `PI0.get_action` is about to infer on; it goes
        through the same input transform as everything else here. Returns `action_proposals` and
        (unless inversion is off) `noise_proposals`, each `(top_k, action_horizon, action_dim)`,
        plus `proposal_rows`: the `(top_k,)` bank rows they came from.

        Those indices are what makes the L2 ranking a *candidate pool* rather than the final
        word. A critic that cross-attends the candidates builds its attention keys from those
        rows' own observations (`critic_keys`) and decides for itself how much of each candidate
        to use; one that pools the set ignores them. Either way the
        distance still chooses which `top_k` rows are on offer -- inverting a chunk costs a
        pass of the action expert, so the pool is as wide as the caller is willing to pay for
        and no wider.
        """
        if self.bank is None:
            raise RuntimeError("No demo bank yet; call ensure_bank(task_name) first.")
        # Copied first: the transforms may modify their input in place, and this is the same
        # dict `Policy.infer` is about to be handed (which copies for the same reason).
        inputs = self.input_transform(jax.tree.map(lambda x: x, observation_window))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        out = self._propose(
            _model.Observation.from_dict(inputs),
            jnp.asarray(self.bank.embeddings),
            jnp.asarray(self.bank.actions),
            demo_mask=jnp.asarray(self.bank.mask),
            demo_extra={name: jnp.asarray(v) for name, v in self.bank.extra.items()},
            query_extra=self._query_extra(inputs),
            top_k=self.top_k,
            views=self.views,
            invert=self.invert,
            num_steps=self.num_steps,
            num_inner_steps=self.num_inner_steps,
            num_substeps=self.num_substeps,
            return_info=True,
        )
        if self.record_retrieval:
            # The distance over the *whole* bank, so the debug view can rank as deep as it
            # likes without the sampler retrieving more than the critic asked for. Kept here
            # rather than returned so the return value stays "the modalities the critic gets",
            # which is what every caller but the visualization wants.
            self.last_distance = np.asarray(out["distance"][0], dtype=np.float32)
        return {
            **{
                name: np.asarray(out[name][0], dtype=np.float32)
                for name in ("action_proposals", "noise_proposals")
                if name in out
            },
            "proposal_rows": np.asarray(out["indices"][0], dtype=np.int32),
        }

    def rank_observations(self, pooled, states=None, top_k=None) -> np.ndarray:
        """Rank the bank for observations that are **not** the live one: ``(b, top_k)`` rows.

        The same metric `Pi0.propose_from_demos` applies -- one relative L2 distance per view
        plus one for the pose, averaged with equal weight -- reimplemented in numpy over already
        pooled embeddings, so a caller with a stored observation can retrieve without a policy.
        That is what lets offline rollouts be co-trained on alongside the live ones: a dataset's
        SigLIP column is the same tower's patch map, so pooling it gives exactly the query
        vector the sampler would have built (`multisensory_steering.critics.offline_replay`).

        ``pooled`` is ``{siglip.<view>: (b, 1152)}`` over at least this retriever's `views`, and
        ``states`` the model-space pose, which may be narrowed to the embodiment's own dims --
        it is zero-padded back to the bank's width here, which is exactly what the sampler
        compares since those trailing dims are the transform's own zero padding.

        Ties go to the lower row index, as `jax.lax.top_k` does, so a row retrieved here is the
        row the sampler would have retrieved for the same observation.
        """
        if self.bank is None:
            raise RuntimeError("No demo bank yet; call ensure_bank(task_name) first.")
        bank = self.bank
        k = int(self.top_k if top_k is None else top_k)
        if missing := [view for view in self.views if view not in pooled]:
            raise KeyError(
                f"ranking needs a pooled embedding for every view the distance averages over; "
                f"missing {missing}. This retriever ranks on {list(self.views)}."
            )
        query = np.stack([np.asarray(pooled[view], np.float32) for view in self.views], axis=1)
        if query.shape[-1] != bank.embeddings.shape[-1]:
            raise ValueError(
                f"pooled embeddings are {query.shape[-1]}-d but the bank's are "
                f"{bank.embeddings.shape[-1]}-d. Pool a patch map over its patch axis, not its "
                f"channels."
            )
        total = _relative_l2_np(
            np.einsum("bvc,mvc->bmv", query, bank.embeddings),
            np.square(query).sum(-1)[:, None, :],
            np.square(bank.embeddings).sum(-1)[None, :, :],
        ).sum(-1)
        terms = len(self.views)
        for name in sorted(n for n in self.signals if n != "siglip"):
            if name != "state":
                raise NotImplementedError(f"ranking does not implement the {name!r} signal.")
            if states is None:
                raise ValueError(
                    "this retriever's distance includes the pose, so `states` is required. "
                    "Drop `state` from `signals` to rank on the camera views alone."
                )
            q = np.asarray(states, np.float32)
            width = bank.states.shape[-1]
            if q.shape[-1] > width:
                raise ValueError(f"states are {q.shape[-1]}-d, wider than the bank's {width}.")
            if q.shape[-1] < width:  # the transform's own trailing zero padding
                q = np.pad(q, [(0, 0)] * (q.ndim - 1) + [(0, width - q.shape[-1])])
            total = total + _relative_l2_np(
                q @ bank.states.T,
                np.square(q).sum(-1)[:, None],
                np.square(bank.states).sum(-1)[None, :],
            )
            terms += 1
        distance = np.where(bank.mask[None, :], total / terms, np.inf)
        return np.argsort(distance, axis=-1, kind="stable")[:, :k].astype(np.int32)

    def critic_keys(
        self,
        rows,
        modalities: tuple[str, ...] | None = None,
        state_dim: int | None = None,
    ) -> dict[str, np.ndarray]:
        """The retrieved rows' own observations: ``{modality: (len(rows), *shape)}``.

        The keys' side of a cross-attending critic's attention, keyed by the same modality names
        the *live* observation uses -- because that is the point: the critic runs its own
        `siglip.<view>` and `state` encoders over these rows to build the keys it compares its
        query against, so the two sides have to be the same kind of array.

        Computed **now**, for these rows only. The bank holds each row's model inputs rather
        than the patch map they encode to (`DemoBank.obs`), so this is one image-tower pass at
        batch ``len(rows)`` per control step -- the price of not keeping a bank-sized block of
        patch maps resident for the whole run. `state_dim` narrows the model-space pose to the
        embodiment's own dims, exactly as the sampler narrows the state it hands the critic.

        A `wrench.<key>` is the exception: it needs no encoding at all. The row already carries
        the `(wrench_trace_len, 6)` trace the demo frame observed, reconstructed from the
        dataset's per-frame rows when the bank was built, so it is served straight out of
        `DemoBank.wrench`.
        """
        if self.bank is None:
            raise RuntimeError("No demo bank yet; call ensure_bank(task_name) first.")
        # Every camera the policy sees, not just the `views` the *distance* ranks on: a row
        # keeps its whole model input, so a view can be a key without being part of the metric.
        # Plus whichever wrench columns this demo dataset carries, if a trace length is set.
        available = (*SIGLIP_VIEWS, "state", *sorted(self.bank.wrench))
        wanted = available if modalities is None else tuple(modalities)
        if unknown := [name for name in wanted if name not in available]:
            raise KeyError(
                f"a demo frame has no {unknown}: the bank carries a demonstration's camera "
                f"views, pose"
                + (f" and {sorted(self.bank.wrench)}" if self.bank.wrench else "")
                + f", i.e. {list(available)}. Drop them from the critic's "
                f"`key_modalities` (the query side can still use them)."
                + (
                    ""
                    if self.bank.wrench or not self.reader.wrench_columns
                    else f" ({self.repo_id} does carry {list(self.wrench_modalities)}, but this "
                    f"run built the bank with no `wrench_trace_len`.)"
                )
            )
        rows = np.asarray(rows, dtype=np.int32).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= self.bank.num_frames):
            # Padding rows sit at +inf distance and are never retrieved, and they are the rows
            # `DemoBank.obs` does not have -- so this is a caller indexing something else.
            raise IndexError(
                f"rows {rows.tolist()} are not frames of the demo bank, which holds "
                f"{self.bank.num_frames} (padded to {len(self.bank.mask)})."
            )
        out = {}
        # Only the camera views go through the tower: a pose and a wrench trace are already
        # arrays on the bank row.
        if views := [
            name
            for name in wanted
            if name != "state" and not name.startswith(WRENCH_MODALITY_PREFIX)
        ]:
            group = jax.tree.map(lambda x: jnp.asarray(x[rows]), self.bank.obs)
            maps = self._embed(_model.Observation.from_dict(group))
            if missing := [view for view in views if view not in maps]:
                raise ValueError(f"the policy does not expose {missing} to the critic.")
            # fp16, the dtype the critic's replay buffer stores a SigLIP map in: these go
            # straight into a stored transition, and the encoder casts on the way in anyway.
            out.update({view: np.asarray(maps[view], dtype=np.float16) for view in views})
        if "state" in wanted:
            states = self.bank.states[rows]
            out["state"] = np.asarray(
                states if state_dim is None else states[..., : int(state_dim)], dtype=np.float32
            )
        for name in wanted:
            if name.startswith(WRENCH_MODALITY_PREFIX):
                out[name] = np.asarray(self.bank.wrench[name][rows], dtype=np.float32)
        return out

    # -----------------------------------------------------------------------------------------
    # Co-training on the demonstrations themselves
    # -----------------------------------------------------------------------------------------

    #: What a demo frame can be to a critic **whatever dataset it came from**: what the policy
    #: makes of the frame, which is exactly the space the online transitions are in. A multimodal
    #: demo dataset adds its `wrench.<key>` columns on top of these (`cotrain_modalities`) -- the
    #: only recorded sensor that survives the demo pipeline. Depth, point clouds and privileged
    #: task state do not, and `images.<cam>` is missing for a different reason again: a demo
    #: dataset stores 480x640 (or, for the multimodal converter, its own) frames while the sim
    #: hands the critic 240x320, so the two are not the same array.
    COTRAIN_MODALITIES = (*SIGLIP_VIEWS, "state")

    @property
    def cotrain_modalities(self) -> tuple[str, ...]:
        """`COTRAIN_MODALITIES` plus whatever wrench columns this demo dataset actually has."""
        return (*self.COTRAIN_MODALITIES, *self.wrench_modalities)

    def cotrain_rows(
        self,
        episodes,
        *,
        horizon: int,
        modalities: tuple[str, ...] = COTRAIN_MODALITIES,
        state_dim: int | None = None,
    ) -> dict:
        """Whole demo episodes as critic-space rows, one per control step.

        This is the demonstrations turned into the same kind of object a rollout dataset's rows
        are (`multisensory_steering.critics.offline_replay`), so the supervised fine-tuning set
        the policy was trained on can be co-trained into the critic's TD batch alongside the
        rollouts it is steering. What makes that possible without collecting anything is that
        every column a critic reads off a rollout dataset is *derived from the frame by the
        policy*: the SigLIP patch map is this tower's, the state and the action chunk are this
        transform's output. So they can be produced here, from the demo parquet, at the cost of
        one tower pass per kept frame.

        Rows are the frames at ``frame_index % horizon == 0`` -- the control steps the online
        critic actually takes, so a demo row and a live transition are the same distance apart
        in time and the same discount applies to both. Only those frames are decoded.

        A `wrench.<key>` row is not encoded but **reconstructed**: a multimodal demo dataset
        stores one averaged `(6,)` per frame, and the trace a control step observes is the
        `wrench_trace_len` (defaulting to `horizon`) of them the previous chunk covered, ending
        at the row's own frame -- the same window, and the same NaN padding at an episode's
        start, that `pop_step_wrench` produces online.

        Args:
            episodes: which demo episodes to encode (already filtered to one task by the caller,
                see `episodes_for_task`).
            horizon: primitive sim steps per control step (`pi0_step`).
            modalities: which of `cotrain_modalities` to produce. A view not asked for costs
                nothing but the tower pass that produced it.
            state_dim: narrow the model-space pose and action chunk to the embodiment's own dims,
                exactly as the sampler narrows what it hands the critic. None keeps the model's
                padded width.

        Returns:
            ``{"obs": {modality: (N, ...)}, "action": (N, action_horizon, d),
            "episode_index": (N,), "frame_index": (N,)}``. A SigLIP map is the flat
            ``(256, 1152)`` patch sequence the tower emits, at fp16 -- the same form (and dtype)
            a rollout dataset's `siglip.<view>` column stores, so the consumer reshapes it to the
            critic's grid the same way for both. A wrench is ``(N, wrench_trace_len, 6)``
            float32 -- ``horizon`` rows when the retriever was given no length of its own -- the
            same shape (and NaN padding) as the online modality and a rollout dataset's column.
        """
        if unknown := [name for name in modalities if name not in self.cotrain_modalities]:
            raise KeyError(
                f"a demonstration from {self.repo_id} has no {unknown}: a demo frame is a camera "
                f"view, a pose"
                + (f" and {list(self.wrench_modalities)}" if self.wrench_modalities else "")
                + f", i.e. {list(self.cotrain_modalities)}. Drop them from the critic's "
                f"`encoder_modalities`, or co-train on a rollout dataset instead."
            )
        views = tuple(
            name for name in modalities if name != "state" and not name.startswith(WRENCH_MODALITY_PREFIX)
        )
        wanted_wrench = tuple(name for name in modalities if name.startswith(WRENCH_MODALITY_PREFIX))
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1 primitive step, got {horizon}.")

        maps = {view: [] for view in views}
        wrench = {name: [] for name in wanted_wrench}
        chunks, states, episode_index, frame_index = [], [], [], []
        for episode in episodes:
            frames = np.arange(0, self.reader.episode_length(episode), horizon)
            transformed = self._transform_frames(episode, frames, wrench_trace_len=horizon)
            for group in self._tower_batches(transformed["obs"]):
                if missing := [view for view in views if view not in group]:
                    raise ValueError(f"the policy does not expose {missing} to the critic.")
                for view in views:
                    # fp16, the dtype the critic's replay buffer stores a SigLIP map in and the
                    # dtype a collected dataset's column has -- these go straight into a stored
                    # transition and the encoder casts on the way in anyway.
                    maps[view].append(np.asarray(group[view], dtype=np.float16))
            for name in wanted_wrench:
                wrench[name].append(transformed["wrench"][name])
            chunks.append(transformed["actions"])
            states.append(transformed["states"])
            episode_index.append(np.full(len(frames), episode, dtype=np.int64))
            frame_index.append(frames.astype(np.int64))

        if not episode_index:
            raise ValueError("cotrain_rows was given no episodes to encode.")
        narrow = slice(None) if state_dim is None else slice(0, int(state_dim))
        obs = {view: np.concatenate(parts) for view, parts in maps.items()}
        obs.update({name: np.concatenate(parts) for name, parts in wrench.items()})
        if "state" in modalities:
            obs["state"] = np.concatenate(states)[..., narrow].astype(np.float32)
        return {
            "obs": obs,
            "action": np.concatenate(chunks)[..., narrow].astype(np.float32),
            "episode_index": np.concatenate(episode_index),
            "frame_index": np.concatenate(frame_index),
        }

    def _query_extra(self, inputs) -> dict:
        """This observation's own non-visual signals, unit-norm and batched, keyed like the bank.

        `inputs` is the already-transformed observation, so the pose is read from exactly the
        vector the sampler will be conditioned on rather than re-derived from the raw qpos.
        """
        if "state" not in self.signals:
            return {}
        return {"state": jnp.asarray(inputs["state"], dtype=jnp.float32)}

    def describe_similarity(self) -> str:
        """One line on what the retrieval ranks by, for the startup banner."""
        terms = [view.split(".", 1)[1] for view in self.views]
        terms += [name for name in self.signals if name != "siglip"]
        return (
            f"mean of {len(terms)} independent relative L2 distances "
            f"(||q-b||/||q||: {', '.join(terms)}), nearest wins"
        )

    def retrieved(self) -> dict | None:
        """What the last `propose` matched: thumbnails, similarities and where they came from.

        None until a step has been proposed with `record_retrieval` set. Ranks the whole bank
        and reports the nearest `max(top_k, debug_top_k)` rows, so the visualization can show
        neighbours the critic was not given -- the first `num_proposals` of them are the ones it
        was. The thumbnails are the demo frames themselves, so a viewer can see whether
        "nearest" means what it should.
        """
        if not self.record_retrieval or self.last_distance is None or self.bank is None:
            return None
        n = min(max(self.top_k, self.debug_top_k), self.bank.num_frames)
        # Ascending: nearest first. Padding rows sit at +inf, so a plain argsort cannot
        # surface one.
        idx = np.argsort(self.last_distance, kind="stable")[:n]
        return {
            "indices": idx.astype(np.int32),
            "distances": self.last_distance[idx],
            "thumbnails": None if self.bank.thumbnails is None else self.bank.thumbnails[idx],
            "episode": self.bank.row_episode[idx],
            "frame": self.bank.row_frame[idx],
            "bank_frames": self.bank.num_frames,
            "num_proposals": self.top_k,
        }
