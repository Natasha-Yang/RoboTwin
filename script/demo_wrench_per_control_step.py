"""Collapse a demo LeRobot dataset's per-physics-step wrench columns to one row per frame.

Demo collection logs the contact wrench after **every** `scene.step()` and `pkl2hdf5` stacks
the `save_freq` samples that led up to each saved frame, so a demo dataset built by
`examples/aloha_real/convert_robotwin_multimodal_to_lerobot.py` carries
`observation.wrench.<key>` as a `(save_freq, 6)` trace per frame. Everything that *consumes* a
wrench -- the critic's live `wrench.*` modality, a rollout dataset's column -- works one row per
**primitive step**, that step's physics steps averaged (CLAUDE.md 6.1 / 7a). A demo frame is one
primitive step, so its column should be a single `(6,)` vector, not the trace inside it.

This rewrites an existing dataset in place to that convention:

    observation.wrench.<key>   (save_freq, 6)  ->  (6,)     nanmean over the physics steps

NaN-aware because the trace is NaN-padded, not zero-padded: a frame at the start of a motion
segment has fewer than `save_freq` samples behind it, and zero is a meaningful reading (the arm
touching nothing), so the padding must not drag the mean toward zero.

The `(pi0_step, 6)` trace a critic observes is then reassembled from these per-frame rows at
read time, by `openpi.policies.demo_retrieval` -- one row per frame over the `pi0_step` frames
the previous chunk covered. Which is the point of storing the average rather than the trace: the
demo's own frame cadence is the primitive-step cadence, and the control-step window is a
consumer's choice (`wrench_trace_len`), not something to bake into the dataset.

Usage:
    python script/demo_wrench_per_control_step.py <repo_id> [<repo_id> ...] [--root DIR]
    python script/demo_wrench_per_control_step.py <repo_id> --dry-run

Idempotent: a column that is already one vector per frame is left alone, so a re-run after an
interruption finishes the job rather than averaging twice. The parquet files are rewritten
atomically (temp file + rename), episode by episode; `meta/episodes_stats.jsonl` is recomputed
for the changed columns and `meta/info.json` is rewritten last, so an interrupted run is
resumable and never leaves a half-written file.
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

WRENCH_PREFIX = "observation.wrench."
# The (6,) row's own component order, matching envs/utils/wrench.py::WRENCH_COMPONENTS.
COMPONENT_NAME = "FxFyFzTxTyTz"


def default_root() -> pathlib.Path:
    """Where LeRobot datasets live, the same way `demo_retrieval.default_root` resolves it."""
    if env := os.environ.get("HF_LEROBOT_HOME"):
        return pathlib.Path(env).expanduser()
    return pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "lerobot"


def episode_parquets(root: pathlib.Path, info: dict) -> list[pathlib.Path]:
    """Every episode parquet of a v2.1 dataset, in episode order."""
    if (root / "meta" / "episodes").is_dir():
        raise SystemExit(
            f"{root} is a v3.0 dataset (episodes share parquet files). This script only "
            f"rewrites the v2.1 one-file-per-episode layout."
        )
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info["data_path"]
    out = []
    for episode in range(int(info["total_episodes"])):
        path = root / data_path.format(episode_chunk=episode // chunks_size, episode_index=episode)
        if not path.exists():
            raise SystemExit(f"missing episode file {path}")
        out.append(path)
    return out


def trace_columns(schema: pa.Schema) -> list[str]:
    """The wrench columns still stored as a per-frame trace (a list of lists), if any."""
    out = []
    for name in schema.names:
        if not name.startswith(WRENCH_PREFIX):
            continue
        field = schema.field(name)
        # (S, 6) arrives as list<list<float>>; an already-collapsed (6,) is a flat
        # fixed_size_list<float>[6], whose value type is a plain float.
        if pa.types.is_nested(field.type.value_type):
            out.append(name)
    return out


def collapse(table: pa.Table, columns: list[str]) -> tuple[pa.Table, dict[str, np.ndarray]]:
    """Replace each trace column with its nan-mean over the physics-step axis."""
    collapsed = {}
    for name in columns:
        trace = np.asarray(
            [np.asarray(row, dtype=np.float32) for row in table[name].to_pylist()], dtype=np.float32
        )
        if trace.ndim != 3 or trace.shape[-1] != 6:
            raise SystemExit(f"{name}: expected (frames, steps, 6), got {trace.shape}")
        # An all-NaN frame would warn and produce NaN, which is the honest answer (nothing was
        # sampled) -- but it does not happen on collected data, so say so if it ever does.
        empty = np.isnan(trace).all(axis=(1, 2))
        with np.errstate(invalid="ignore"):
            mean = np.where(
                empty[:, None], np.nan, np.nanmean(np.where(empty[:, None, None], 0.0, trace), axis=1)
            ).astype(np.float32)
        if empty.any():
            print(f"    {name}: {int(empty.sum())} frame(s) had no wrench sample at all -> NaN")
        collapsed[name] = mean
        array = pa.FixedSizeListArray.from_arrays(pa.array(mean.reshape(-1), pa.float32()), 6)
        table = table.set_column(
            table.schema.get_field_index(name),
            pa.field(name, array.type),
            pa.chunked_array([array]),
        )
    return table.replace_schema_metadata(patch_hf_metadata(table.schema.metadata, columns)), collapsed


def patch_hf_metadata(metadata, columns: list[str]):
    """Rewrite the `huggingface` schema metadata's feature spec for the collapsed columns.

    A parquet written by `datasets` carries the feature dict that produced it; leaving it
    claiming `Array2D(shape=[save_freq, 6])` would make the file unreadable through `datasets`
    even though the arrow data is fine.
    """
    if not metadata or b"huggingface" not in metadata:
        return metadata
    meta = dict(metadata)
    spec = json.loads(meta[b"huggingface"].decode())
    features = spec.get("info", {}).get("features", {})
    for name in columns:
        if name in features:
            features[name] = {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": 6,
                "_type": "Sequence",
            }
    meta[b"huggingface"] = json.dumps(spec).encode()
    return meta


def rewrite_episode(path: pathlib.Path, dry_run: bool) -> dict[str, np.ndarray]:
    """Collapse one episode file in place; returns the new values, for the stats pass."""
    parquet = pq.ParquetFile(path)
    columns = trace_columns(parquet.schema_arrow)
    if not columns:
        return {}
    table = parquet.read()
    compression = parquet.metadata.row_group(0).column(0).compression.lower()
    del parquet
    table, collapsed = collapse(table, columns)
    if dry_run:
        return collapsed
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression=compression)
    os.replace(tmp, path)
    return collapsed


def column_stats(values: np.ndarray) -> dict:
    """LeRobot's per-episode stats for one collapsed column, in its own json layout."""
    return {
        "min": np.nanmin(values, axis=0).tolist(),
        "max": np.nanmax(values, axis=0).tolist(),
        "mean": np.nanmean(values, axis=0).tolist(),
        "std": np.nanstd(values, axis=0).tolist(),
        "count": [len(values)],
    }


def rewrite_stats(root: pathlib.Path, stats: dict[int, dict[str, np.ndarray]], dry_run: bool):
    """Replace the wrench entries of `meta/episodes_stats.jsonl`, streaming (the file is GBs)."""
    path = root / "meta" / "episodes_stats.jsonl"
    if not path.exists():
        return
    tmp = path.with_suffix(".jsonl.tmp")
    written = 0
    with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            episode = int(row["episode_index"])
            for name, values in stats.get(episode, {}).items():
                if name in row.get("stats", {}):
                    row["stats"][name] = column_stats(values)
                    written += 1
            dst.write(json.dumps(row) + "\n")
    if dry_run:
        tmp.unlink()
    else:
        os.replace(tmp, path)
    print(f"  episodes_stats.jsonl: {written} column stats recomputed")


def rewrite_info(root: pathlib.Path, columns: list[str], dry_run: bool):
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text())
    for name in columns:
        info["features"][name] = {"dtype": "float32", "shape": [6], "names": [COMPONENT_NAME]}
    if not dry_run:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(info, indent=4))
        os.replace(tmp, path)
    print(f"  info.json: {len(columns)} feature(s) now shape [6]")


def convert(repo_id: str, root: pathlib.Path, dry_run: bool):
    dataset_root = root / repo_id
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"no LeRobot dataset at {dataset_root} (expected meta/info.json)")
    info = json.loads(info_path.read_text())
    wrench = [name for name in info["features"] if name.startswith(WRENCH_PREFIX)]
    if not wrench:
        raise SystemExit(f"{repo_id} has no {WRENCH_PREFIX}* columns; nothing to collapse.")
    shapes = {name: tuple(info["features"][name]["shape"]) for name in wrench}
    print(f"{repo_id}: {len(wrench)} wrench column(s) {sorted(shapes.values())[0]} "
          f"over {info['total_episodes']} episodes{' [dry run]' if dry_run else ''}")

    paths = episode_parquets(dataset_root, info)
    stats: dict[int, dict[str, np.ndarray]] = {}
    changed: list[str] = []
    for episode, path in enumerate(paths):
        collapsed = rewrite_episode(path, dry_run)
        if collapsed:
            stats[episode] = collapsed
            changed = changed or sorted(collapsed)
        if (episode + 1) % 25 == 0 or episode + 1 == len(paths):
            print(f"  {episode + 1}/{len(paths)} episodes", flush=True)
            sys.stdout.flush()
    if not changed:
        print("  every wrench column is already one row per frame; nothing to do.")
        return
    rewrite_stats(dataset_root, stats, dry_run)
    rewrite_info(dataset_root, changed, dry_run)
    if dry_run:
        example = stats[min(stats)][changed[0]]
        print(f"  would write {changed[0]} as {example.shape}, first row {example[0].tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repo_id", nargs="+", help="LeRobot dataset(s), e.g. NatashaYang/foo")
    parser.add_argument("--root", type=pathlib.Path, default=None,
                        help="where the datasets live (default: $HF_LEROBOT_HOME)")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()
    root = args.root or default_root()
    for repo_id in args.repo_id:
        convert(repo_id, root, args.dry_run)


if __name__ == "__main__":
    main()
