"""Histograms of the contact wrench columns of a RoboTwin rollout dataset.

Eight panels: the six components on their own axes (forces left column, torques right),
then the magnitude of the force vector and of the torque vector, with one overlaid
histogram per `observation.wrench.*` column.

A dataset collected since 2026-08-28 carries two families of those columns: one per
end-effector link (aloha: `fl_link7`, `fl_link8`, `fr_link7`, `fr_link8`), so a finger
squeezing against its opposite is visible instead of cancelling, and one per arm (`left`,
`right`), each arm being the sum of its own links. Plotting both at once would draw that
sum twice, so `--family` picks one; links win by default, since the arm total is
recoverable from them and not the other way round. An older dataset has only the arm
columns and is plotted as-is whatever is asked for.

Reads the dataset's parquet shards directly (only the wrench columns, discovered from the
schema), so it does not depend on the `datasets` version that wrote them.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

COMPONENTS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

# Categorical slots 1 and 2 of the reference palette, plus that mode's chart ink.
# Validated on both surfaces: CVD dE 24.7 light / 26.8 dark (>= 8), normal-vision
# dE 33.6 / 31.8 (>= 15), contrast 4.30 / 3.12 light and 4.79 / 4.48 dark (>= 3).
# `series` is walked in column order, which `load_wrench` keeps sorted -- for aloha that is
# the left arm's links (`fl_*`) then the right's (`fr_*`), so the cool half of the palette
# lands on one gripper and the warm half on the other, as in the debug plots. The count of
# series is the embodiment's link count rather than a fixed two, so the palette is walked
# cyclically rather than assumed to cover it.
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  series=("#2a78d6", "#4aa3b8", "#eb6834", "#c4432c",
                          "#3f8f4a", "#8455b8", "#8a6a3c", "#7e8a2f")),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 series=("#3987e5", "#5cb8cd", "#d95926", "#e0705a",
                         "#4fa85c", "#9b6bd6", "#a8834b", "#96a338")),
}

WRENCH_PREFIX = "observation.wrench."


# The two arm-total columns, by the names `envs/utils/wrench.py::ARM_TAGS` gives them. Every
# other `observation.wrench.*` column is one end-effector link, under its URDF name.
ARM_LABELS = ("left", "right")


def load_wrench(pattern, family="links"):
    """``({label: (num_samples, 6)}, dropped, rows)``, NaN-padded steps dropped.

    The columns are discovered from the first shard's schema rather than named here, so this
    follows the embodiment — a dual-arm aloha yields four links, another gripper yields however
    many `envs/utils/wrench.py::ee_link_labels` reports for it — and the same script reads a
    dataset with both families, one with only the per-link columns, and an older per-arm one;
    the label is the column's own suffix. `family` is "links", "arms" or "all"; a family the
    dataset does not have falls back to what it does have, with a note.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet shards match {pattern!r}")
    cols = sorted(c for c in pq.read_schema(files[0]).names if c.startswith(WRENCH_PREFIX))
    if not cols:
        raise SystemExit(f"no {WRENCH_PREFIX}* columns in {files[0]} — "
                         "was the dataset collected with data_type.wrench on?")
    arms = [c for c in cols if c[len(WRENCH_PREFIX):] in ARM_LABELS]
    links = [c for c in cols if c not in arms]
    if family != "all":
        wanted = arms if family == "arms" else links
        if not wanted:
            have = "arm totals" if arms else "links"
            print(f"this dataset carries only {have}; plotting those instead of {family!r}")
        else:
            cols = wanted
    chunks = {c: [] for c in cols}
    for path in files:
        table = pq.read_table(path, columns=cols)
        for c in cols:
            chunks[c].append(np.asarray(table[c].to_pylist(), dtype=np.float32))
    # (rows, pi0_step, 6) -> one row per executed primitive step.
    series = {c[len(WRENCH_PREFIX):]: np.concatenate(chunks[c]).reshape(-1, len(COMPONENTS))
              for c in cols}
    rows = sum(len(c) for c in chunks[cols[0]])
    # A step is NaN-padded in every column at once (a chunk that ended early ran no step for
    # any of them), so one shared mask keeps the series index-aligned.
    keep = np.logical_and.reduce([np.isfinite(a).all(1) for a in series.values()])
    return {k: v[keep] for k, v in series.items()}, int((~keep).sum()), rows


def panel(ax, series, title, unit, theme, bins=72):
    """`series` is ``{label: (num_samples,)}``, one histogram each on shared bin edges."""
    edges = np.histogram_bin_edges(np.concatenate(list(series.values())), bins=bins)
    colors = theme["series"]
    for i, (label, values) in enumerate(series.items()):
        color = colors[i % len(colors)]
        ax.hist(values, bins=edges, color=color, alpha=0.35, zorder=2)
        # The outline is what stays readable where the distributions overlap.
        ax.hist(values, bins=edges, histtype="step", color=color, lw=2.0, label=label, zorder=3)
    ax.set_yscale("log")
    ax.set_title(title, color=theme["ink"], fontsize=12, fontweight="semibold", loc="left", pad=8)
    ax.set_xlabel(unit, color=theme["ink2"], fontsize=9.5)
    ax.grid(axis="y", color=theme["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=3, width=0.8)
    ax.set_facecolor(theme["surface"])


def plot(series, dropped, rows, dataset, out_path, mode):
    theme = THEME[mode]
    stacked = np.vstack(list(series.values()))
    zero_share = float((np.abs(stacked).sum(1) < 1e-9).mean())
    samples = len(next(iter(series.values())))

    fig, axes = plt.subplots(4, 2, figsize=(12, 13.5), facecolor=theme["surface"])
    # Column 1 forces, column 2 torques, so each row shares an axis; magnitudes last.
    layout = [("Fx", 0, "N"), ("Tx", 3, "N·m"),
              ("Fy", 1, "N"), ("Ty", 4, "N·m"),
              ("Fz", 2, "N"), ("Tz", 5, "N·m")]
    for ax, (name, idx, unit) in zip(axes.ravel(), layout):
        panel(ax, {k: v[:, idx] for k, v in series.items()}, name, unit, theme)
    panel(axes[3, 0], {k: np.linalg.norm(v[:, :3], axis=1) for k, v in series.items()},
          "|F|  force magnitude", "N", theme)
    panel(axes[3, 1], {k: np.linalg.norm(v[:, 3:], axis=1) for k, v in series.items()},
          "|T|  torque magnitude", "N·m", theme)
    for ax in axes[:, 0]:
        ax.set_ylabel("samples", color=theme["ink2"], fontsize=9.5)

    fig.suptitle("End-effector contact wrench, world frame", color=theme["ink"], fontsize=17,
                 fontweight="semibold", x=0.055, y=0.988, ha="left")
    fig.text(0.055, 0.957, dataset, color=theme["ink2"], fontsize=11, ha="left")
    fig.text(0.055, 0.940,
             f"{samples:,} executed steps per series over {rows:,} action chunks   ·   "
             f"{zero_share:.0%} read exactly zero (no contact) — counts on a log scale"
             + (f"   ·   {dropped:,} NaN-padded steps dropped" if dropped else ""),
             color=theme["muted"], fontsize=10, ha="left")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.965, 0.995),
                        frameon=False, fontsize=11.5, ncol=min(len(series), 4),
                        handlelength=1.6, columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(theme["ink2"])

    fig.tight_layout(rect=(0.03, 0.01, 0.98, 0.925), h_pad=2.6, w_pad=3.2)
    fig.savefig(out_path, dpi=150, facecolor=theme["surface"])
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="NatashaYang/robotwin_clean_ep25_lift_pot_rollouts_dataset")
    p.add_argument("--shards", default=None, help="glob of parquet shards (default: the HF hub cache)")
    p.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--family", default="links", choices=("links", "arms", "all"),
                   help="which wrench columns to plot; an arm is the sum of its links, so "
                        "'all' draws that sum twice (default: links)")
    args = p.parse_args()

    shards = args.shards or os.path.expanduser(
        f"~/.cache/huggingface/hub/datasets--{args.dataset.replace('/', '--')}"
        "/snapshots/*/data/train-*.parquet")
    series, dropped, rows = load_wrench(shards, args.family)
    stem = args.dataset.split("/")[-1]
    for mode in ("light", "dark"):
        plot(series, dropped, rows, args.dataset,
             os.path.join(args.out_dir, f"wrench_hist_{stem}_{mode}.png"), mode)
