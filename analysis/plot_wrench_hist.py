"""Histograms of the TCP contact wrench columns of a RoboTwin rollout dataset.

Eight panels: the six components on their own axes (forces left column, torques right),
then the magnitude of the force vector and of the torque vector. Left and right arm are
overlaid in each panel.

Reads the dataset's parquet shards directly (only the two wrench columns), so it does not
depend on the `datasets` version that wrote them.
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
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", left="#2a78d6", right="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", left="#3987e5", right="#d95926"),
}


def load_wrench(pattern):
    """(left, right) arrays of shape (num_samples, 6), NaN-padded steps dropped."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet shards match {pattern!r}")
    cols = ["observation.wrench.left", "observation.wrench.right"]
    chunks = {c: [] for c in cols}
    for path in files:
        table = pq.read_table(path, columns=cols)
        for c in cols:
            chunks[c].append(np.asarray(table[c].to_pylist(), dtype=np.float32))
    # (rows, pi0_step, 6) -> one row per executed primitive step.
    out = []
    for c in cols:
        arr = np.concatenate(chunks[c]).reshape(-1, len(COMPONENTS))
        out.append(arr)
    rows = sum(len(c) for c in chunks[cols[0]])
    keep = np.isfinite(out[0]).all(1) & np.isfinite(out[1]).all(1)
    return out[0][keep], out[1][keep], int((~keep).sum()), rows


def panel(ax, left, right, title, unit, theme, bins=72):
    edges = np.histogram_bin_edges(np.concatenate([left, right]), bins=bins)
    for values, color, label in ((left, theme["left"], "left arm"),
                                 (right, theme["right"], "right arm")):
        ax.hist(values, bins=edges, color=color, alpha=0.35, zorder=2)
        # The outline is what stays readable where the two distributions overlap.
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


def plot(left, right, dropped, rows, dataset, out_path, mode):
    theme = THEME[mode]
    both = np.vstack([left, right])
    zero_share = float((np.abs(both).sum(1) < 1e-9).mean())

    fig, axes = plt.subplots(4, 2, figsize=(12, 13.5), facecolor=theme["surface"])
    # Column 1 forces, column 2 torques, so each row shares an axis; magnitudes last.
    layout = [("Fx", 0, "N"), ("Tx", 3, "N·m"),
              ("Fy", 1, "N"), ("Ty", 4, "N·m"),
              ("Fz", 2, "N"), ("Tz", 5, "N·m")]
    for ax, (name, idx, unit) in zip(axes.ravel(), layout):
        panel(ax, left[:, idx], right[:, idx], name, unit, theme)
    panel(axes[3, 0], np.linalg.norm(left[:, :3], axis=1), np.linalg.norm(right[:, :3], axis=1),
          "|F|  force magnitude", "N", theme)
    panel(axes[3, 1], np.linalg.norm(left[:, 3:], axis=1), np.linalg.norm(right[:, 3:], axis=1),
          "|T|  torque magnitude", "N·m", theme)
    for ax in axes[:, 0]:
        ax.set_ylabel("samples", color=theme["ink2"], fontsize=9.5)

    fig.suptitle("TCP contact wrench, world frame", color=theme["ink"], fontsize=17,
                 fontweight="semibold", x=0.055, y=0.988, ha="left")
    fig.text(0.055, 0.957, dataset, color=theme["ink2"], fontsize=11, ha="left")
    fig.text(0.055, 0.940,
             f"{len(left):,} executed steps per arm over {rows:,} action chunks   ·   "
             f"{zero_share:.0%} read exactly zero (no contact) — counts on a log scale"
             + (f"   ·   {dropped:,} NaN-padded steps dropped" if dropped else ""),
             color=theme["muted"], fontsize=10, ha="left")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.965, 0.995),
                        frameon=False, fontsize=11.5, ncol=2, handlelength=1.6, columnspacing=1.6)
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
    left, right, dropped, rows = load_wrench(shards)
    stem = args.dataset.split("/")[-1]
    for mode in ("light", "dark"):
        plot(left, right, dropped, rows, args.dataset,
             os.path.join(args.out_dir, f"wrench_hist_{stem}_{mode}.png"), mode)
