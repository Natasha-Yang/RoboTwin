#!/bin/bash
# ---------------------------------------------------------------------------
# pi0.5 fine-tuning GPU job (Alliance Canada / SLURM).
#
# Submit:  sbatch cluster/finetune_pi05.sh <episodes_per_task> [config] [model_name]
#   e.g.   sbatch cluster/finetune_pi05.sh 10
#          sbatch cluster/finetune_pi05.sh 25 pi05_base_aloha_lora_50 my_run
#
# Unlike cluster/robotwin_gpu.sh this does NOT use the SAPIEN/conda env or
# Vulkan -- training runs in the separate policy/pi05 uv venv (JAX). Compute
# nodes have no internet, so the pi05_base checkpoint, the LeRobot dataset, and
# the norm stats must all be prepared on a login node first (see CLAUDE.md §5.1).
# ---------------------------------------------------------------------------
#SBATCH --account=def-florian7_gpu
#SBATCH --job-name=pi05_finetune
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune/%x-%j.out

set -euo pipefail

N="${1:?usage: sbatch cluster/finetune_pi05.sh <episodes_per_task> [config] [model_name]}"
CONFIG="${2:-pi05_base_aloha_lora_50}"
MODEL="${3:-robotwin_clean_ep${N}}"

# Resolve repo root: under sbatch, $0 is a spooled copy, so dirname "$0" fails.
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
if [ -z "$ROBOTWIN_ROOT" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/setup_env.sh" ]; then
        ROBOTWIN_ROOT="$SLURM_SUBMIT_DIR"
    else
        ROBOTWIN_ROOT="/project/6028519/natashay/RoboTwin"
    fi
fi

# Guard against the Alliance CVMFS python leak (dummy opencv wheel, pip config).
unset PYTHONPATH PIP_CONFIG_FILE 2>/dev/null || true

# Offline compute node: resolve the LeRobot dataset locally and block hub calls.
export HF_LEROBOT_HOME=/home/natashay/links/projects/def-florian7/natashay/hf_cache/lerobot
export HF_HUB_OFFLINE=1
# Compute nodes have no internet, so wandb.init() online-mode times out (90s) and
# kills the job. Log to disk instead; `wandb sync <run_dir>` from a login node later.
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

echo "=== node: $(hostname) | N=$N | config=$CONFIG | model=$MODEL ==="
nvidia-smi -L || true

cd "$ROBOTWIN_ROOT/policy/pi05"
# GPU 0 within the job's SLURM-scoped CUDA_VISIBLE_DEVICES.
bash finetune.sh "$CONFIG" "$MODEL" 0 --episodes-per-task="$N"
