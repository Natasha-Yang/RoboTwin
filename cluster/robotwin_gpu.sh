#!/bin/bash
# ---------------------------------------------------------------------------
# RoboTwin GPU batch job for the Killarney cluster (Alliance Canada / SLURM).
#
# Submit:   sbatch cluster/robotwin_gpu.sh <command...>
#   e.g.    sbatch cluster/robotwin_gpu.sh bash collect_data.sh beat_block_hammer demo_randomized 0
#   default (no args) runs the headless render smoke-test.
#
# Notes:
#   * SAPIEN ray tracing AND rasterization both render on these GPUs. (This was
#     first verified on H100, where the lack of RT cores does not block ray
#     tracing -- the driver/OptiX handles it; the L40S has RT cores outright.)
#   * Compute nodes have no internet; do all pip/git installs on a login node.
#   * Submit from the repo root, or set ROBOTWIN_ROOT to override its location.
# ---------------------------------------------------------------------------
#SBATCH --account=aip-florian7
#SBATCH --job-name=robotwin
#SBATCH --gpus-per-node=l40s:1
# No --partition: Killarney's job_submit lua filter routes a GPU job to the
# right gpubase_l40s_b* band from --gres + --time (3h -> gpubase_l40s_b1).
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Resolve the RoboTwin repo root. Under `sbatch`, $0 is a spooled copy of the
# script (not the real path), so dirname "$0" does NOT work -- hence an explicit
# override / submit-dir / known-location chain instead.
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
if [ -z "$ROBOTWIN_ROOT" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/setup_env.sh" ]; then
        ROBOTWIN_ROOT="$SLURM_SUBMIT_DIR"
    else
        ROBOTWIN_ROOT="/project/6101811/natashay/RoboTwin"
    fi
fi
cd "$ROBOTWIN_ROOT"

# --- activate the conda env (unsets the CVMFS PYTHONPATH/PIP leaks) ---
source "$ROBOTWIN_ROOT/setup_env.sh"

# --- headless Vulkan rendering (SAPIEN) ---
# SAPIEN uses its own bundled Vulkan ICD; the missing libnvidia-gpucomp
# dependency is handled by the shim in setup_env.sh (sourced above).
export PYOPENGL_PLATFORM=egl   # offscreen GL for open3d/other tools
unset DISPLAY                  # ensure no X11 dependency
export PYTHONUNBUFFERED=1      # stream Python stdout live to the .out log
                              # (otherwise it block-buffers to the file and you
                              #  see nothing until the job flushes/exits)

# Warp's default cache is shared by every compute node.  Reusing JIT output
# produced under a different driver/runtime can make a valid kernel fail with
# "CUDA error: an illegal instruction".  Compile into this job's node-local
# temporary directory instead; it is both safer and faster than the networked
# home-directory cache.
if [ -n "${SLURM_TMPDIR:-}" ]; then
    export WARP_CACHE_PATH="$SLURM_TMPDIR/warp-cache"
    mkdir -p "$WARP_CACHE_PATH"
fi

echo "=== node: $(hostname) ==="
nvidia-smi -L || true

if [ "$#" -eq 0 ]; then
    echo "=== running headless render smoke-test ==="
    python cluster/test_render_headless.py
else
    echo "=== running: $* ==="
    "$@"
fi
