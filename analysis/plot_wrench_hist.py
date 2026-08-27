"""Histograms of the end-effector contact wrench columns of a RoboTwin rollout dataset.

Eight panels: the six components on their own axes (forces left column, torques right),
then the magnitude of the force vector and of the torque vector. Every gripper link is
overlaid in each panel — the dataset stores the wrench per link (aloha: ``fl_link7``,
``fl_link8``, ``fr_link7``, ``fr_link8``), not summed per arm, so a finger squeezing against
its opposite is visible instead of cancelling.

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

# One categorical colour per gripper link, in the order the columns are read (the left arm's
# links first, as `wrench.link_wrench_vector` reports them). Okabe-Ito, which stays separable
# under the common colour-vision deficiencies -- the number of series is now the embodiment's
# link count rather than a fixed two, so a hand-tuned pair no longer covers it. The dark row is
# the same hue order lightened for the dark surface.
SERIES = {
    "light": ("#0072b2", "#56b4e9", "#d55e00", "#e69f00", "#009e73", "#cc79a7"),
    "dark": ("#56b4e9", "#9ad4f5", "#e8834a", "#f0bc55", "#3fc79b", "#e2a2c2"),
}
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835"),
}

WRENCH_PREFIX = "observation.wrench."


def load_wrench(pattern):
    """``({link: (num_samples, 6)}, dropped, rows)``, NaN-padded steps dropped.

    The link columns are discovered from the first shard's schema rather than named here, so
    this follows the embodiment: a dual-arm aloha yields four, another gripper yields however
    many links `envs/utils/wrench.py::ee_link_labels` reports for it.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet shards match {pattern!r}")
    cols = [n for n in pq.read_schema(files[0]).names if n.startswith(WRENCH_PREFIX)]
    if not cols:
        raise SystemExit(f"no {WRENCH_PREFIX}* columns in {files[0]} — "
                         "was the dataset collected with data_type.wrench on?")
    chunks = {c: [] for c in cols}
    for path in files:
        table = pq.read_table(path, columns=cols)
        for c in cols:
            chunks[c].append(np.asarray(table[c].to_pylist(), dtype=np.float32))
    # (rows, pi0_step, 6) -> one row per executed primitive step.
    series = {c: np.concatenate(chunks[c]).reshape(-1, len(COMPONENTS)) for c in cols}
    rows = sum(len(c) for c in chunks[cols[0]])
    # A step is NaN-padded in every link at once (a chunk that ended early ran no step for any
    # of them), so one shared mask keeps the links index-aligned.
    keep = np.logical_and.reduce([np.isfinite(a).all(1) for a in series.values()])
    return ({c[len(WRENCH_PREFIX):]: a[keep] for c, a in series.items()},
            int((~keep).sum()), rows)


def panel(ax, series, colors, title, unit, theme, bins=72):
    """One component's histogram, every link overlaid. `series` is ``{link: (n,)}``."""
    edges = np.histogram_bin_edges(np.concatenate(list(series.values())), bins=bins)
    for (label, values), color in zip(series.items(), colors):
        ax.hist(values, bins=edges, color=color, alpha=0.28, zorder=2)
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


def plot(links, dropped, rows, dataset, out_path, mode):
    theme = THEME[mode]
    colors = SERIES[mode]
    stacked = np.vstack(list(links.values()))
    zero_share = float((np.abs(stacked).sum(1) < 1e-9).mean())
    steps = len(next(iter(links.values())))

    fig, axes = plt.subplots(4, 2, figsize=(12, 13.5), facecolor=theme["surface"])
    # Column 1 forces, column 2 torques, so each row shares an axis; magnitudes last.
    layout = [("Fx", 0, "N"), ("Tx", 3, "N·m"),
              ("Fy", 1, "N"), ("Ty", 4, "N·m"),
              ("Fz", 2, "N"), ("Tz", 5, "N·m")]
    for ax, (name, idx, unit) in zip(axes.ravel(), layout):
        panel(ax, {k: v[:, idx] for k, v in links.items()}, colors, name, unit, theme)
    panel(axes[3, 0], {k: np.linalg.norm(v[:, :3], axis=1) for k, v in links.items()},
          colors, "|F|  force magnitude", "N", theme)
    panel(axes[3, 1], {k: np.linalg.norm(v[:, 3:], axis=1) for k, v in links.items()},
          colors, "|T|  torque magnitude", "N·m", theme)
    for ax in axes[:, 0]:
        ax.set_ylabel("samples", color=theme["ink2"], fontsize=9.5)

    fig.suptitle("End-effector contact wrench per gripper link, world frame", color=theme["ink"],
                 fontsize=17, fontweight="semibold", x=0.055, y=0.988, ha="left")
    fig.text(0.055, 0.957, dataset, color=theme["ink2"], fontsize=11, ha="left")
    fig.text(0.055, 0.940,
             f"{steps:,} executed steps per link over {rows:,} action chunks   ·   "
             f"{zero_share:.0%} read exactly zero (no contact) — counts on a log scale"
             + (f"   ·   {dropped:,} NaN-padded steps dropped" if dropped else ""),
             color=theme["muted"], fontsize=10, ha="left")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.965, 0.995),
                        frameon=False, fontsize=11.5, ncol=len(labels), handlelength=1.6,
                        columnspacing=1.6)
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
    args = p.parse_args()

    shards = args.shards or os.path.expanduser(
        f"~/.cache/huggingface/hub/datasets--{args.dataset.replace('/', '--')}"
        "/snapshots/*/data/train-*.parquet")
    links, dropped, rows = load_wrench(shards)
    stem = args.dataset.split("/")[-1]
    for mode in ("light", "dark"):
        plot(links, dropped, rows, args.dataset,
             os.path.join(args.out_dir, f"wrench_hist_{stem}_{mode}.png"), mode)
