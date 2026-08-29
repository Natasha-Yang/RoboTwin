#!/bin/bash
# ---------------------------------------------------------------------------
# Compute π0.5 normalization stats for a training config (CLAUDE.md section 5.1).
# Killarney (Alliance Canada / SLURM).
#
# Submit:   sbatch cluster/compute_norm_stats.sh <train_config_name> [episodes_per_task]
#   e.g.    sbatch cluster/compute_norm_stats.sh pi05_base_aloha_lora_multimodal_50x10
#
# Writes to policy/pi05/assets/<train_config_name>/<repo_id>/norm_stats.json.
# Must be run once per config before finetune.sh, and re-run if the dataset or
# the episode subset changes.
#
# Notes:
#   * Only `state` and `actions` feed the statistics, but the torch dataloader
#     still decodes every frame's images, so this is I/O bound on the dataset.
#   * Killarney has no CPU-only partition, so this asks for a GPU it barely uses.
#   * Pass episodes_per_task only to override what the config bakes in -- it must
#     match what train.py is given, or the stats describe different data than the
#     run they normalize.
#   * Do NOT fold this into `sbatch --wrap`: that runs under /bin/sh (dash), and
#     `set -o pipefail` fails there with "Illegal option -o pipefail".
# ---------------------------------------------------------------------------
#SBATCH --account=aip-florian7
#SBATCH --job-name=norm_stats
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --exclude=kn117
#SBATCH --output=logs/lerobot/%x-%j.out

set -euo pipefail

CONFIG="${1:?usage: sbatch cluster/compute_norm_stats.sh <train_config_name> [episodes_per_task]}"
EPT="${2:-}"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
if [ -z "$ROBOTWIN_ROOT" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/setup_env.sh" ]; then
        ROBOTWIN_ROOT="$SLURM_SUBMIT_DIR"
    else
        ROBOTWIN_ROOT="/project/6101811/natashay/RoboTwin"
    fi
fi

# The CVMFS profile injects a PYTHONPATH and a PIP_CONFIG_FILE pointing at a
# wheelhouse with a stub opencv; both leak into any interpreter we run here.
unset PYTHONPATH PIP_CONFIG_FILE
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1     # compute nodes have no internet
export HF_HOME="${HF_HOME:-/home/natashay/projects/aip-florian7/natashay/hf_cache}"

cd "$ROBOTWIN_ROOT/policy/pi05"
PY="$PWD/.venv/bin/python"
[ -x "$PY" ] || { echo "no venv interpreter at $PY" >&2; exit 1; }

echo "=== node: $(hostname) | config: $CONFIG | episodes_per_task: ${EPT:-<from config>} ==="
# tyro exposes every parameter as a keyword: the config name is --config-name,
# NOT a positional (CLAUDE.md 5.1 showed it positionally, which just prints a
# "the following arguments are required: --config-name" usage block and exits 2).
if [ -n "$EPT" ]; then
    "$PY" scripts/compute_norm_stats.py --config-name "$CONFIG" --episodes-per-task "$EPT"
else
    "$PY" scripts/compute_norm_stats.py --config-name "$CONFIG"
fi
