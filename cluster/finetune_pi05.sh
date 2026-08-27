#!/bin/bash
# ---------------------------------------------------------------------------
# pi0.5 fine-tuning GPU job (Alliance Canada / SLURM).
#
# Submit:  sbatch cluster/finetune_pi05.sh <episodes_per_task> [config] [model_name] [train.py args...]
#   e.g.   sbatch cluster/finetune_pi05.sh 10
#          sbatch cluster/finetune_pi05.sh 25 pi05_base_aloha_lora_50 my_run
#
# Anything after <model_name> is forwarded verbatim to scripts/train.py. To pick
# up an interrupted run from its newest checkpoint instead of starting over:
#          sbatch cluster/finetune_pi05.sh 25 pi05_base_aloha_lora_clean_50x25 \
#                 robotwin_clean_ep25 --resume
# (finetune.sh then drops its default --overwrite, which would otherwise wipe the
# very checkpoints being resumed.)
#
# Unlike cluster/robotwin_gpu.sh this does NOT use the SAPIEN/conda env or
# Vulkan -- training runs in the separate policy/pi05 uv venv (JAX). Compute
# nodes have no internet, so the pi05_base checkpoint, the LeRobot dataset, and
# the norm stats must all be prepared on a login node first (see CLAUDE.md §5.1).
# ---------------------------------------------------------------------------
#SBATCH --account=aip-florian7
#SBATCH --job-name=pi05_finetune
#SBATCH --gpus-per-node=l40s:1
# kn117's GPU has uncorrectable ECC errors / pending row remaps but Slurm still
# schedules onto it; JAX then finds no CUDA device and the job dies at startup.
# Re-check with `nvidia-smi -q -d ECC,ROW_REMAPPER` and drop this once it's fixed.
# Override for a single run with: sbatch --exclude=<nodes> cluster/finetune_pi05.sh ...
#SBATCH --exclude=kn117
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune/%x-%j.out

set -euo pipefail

N="${1:?usage: sbatch cluster/finetune_pi05.sh <episodes_per_task> [config] [model_name] [train.py args...]}"
CONFIG="${2:-pi05_base_aloha_lora_50}"
MODEL="${3:-robotwin_clean_ep${N}}"
# Everything past the three positionals is passed straight through to train.py.
if [ "$#" -gt 3 ]; then shift 3; else shift "$#"; fi
EXTRA=("$@")

# Resolve repo root: under sbatch, $0 is a spooled copy, so dirname "$0" fails.
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
if [ -z "$ROBOTWIN_ROOT" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/setup_env.sh" ]; then
        ROBOTWIN_ROOT="$SLURM_SUBMIT_DIR"
    else
        ROBOTWIN_ROOT="/project/6101811/natashay/RoboTwin"
    fi
fi

# Guard against the Alliance CVMFS python leak (dummy opencv wheel, pip config).
unset PYTHONPATH PIP_CONFIG_FILE 2>/dev/null || true

# Offline compute node: resolve the LeRobot dataset locally and block hub calls.
export HF_LEROBOT_HOME=/project/6101811/natashay/hf_cache/lerobot
export HF_HUB_OFFLINE=1
# Compute nodes have no internet, so wandb.init() online-mode times out (90s) and
# kills the job. Log to disk instead; `wandb sync <run_dir>` from a login node later.
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

echo "=== node: $(hostname) | N=$N | config=$CONFIG | model=$MODEL | extra=${EXTRA[*]-} ==="
nvidia-smi -L || true

cd "$ROBOTWIN_ROOT/policy/pi05"
# GPU 0 within the job's SLURM-scoped CUDA_VISIBLE_DEVICES.
bash finetune.sh "$CONFIG" "$MODEL" 0 --episodes-per-task="$N" ${EXTRA[@]+"${EXTRA[@]}"}
