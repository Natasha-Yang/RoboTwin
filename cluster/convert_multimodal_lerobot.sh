#!/bin/bash
# ---------------------------------------------------------------------------
# Build a LeRobot dataset that KEEPS the extra modalities (depth, point cloud,
# wrench, endpose, camera matrices). See CLAUDE.md section 4.
#
# Submit: sbatch cluster/convert_multimodal_lerobot.sh <task_config> <data_dir> <repo_id>
#   HF_LEROBOT_HOME picks where it lands (default $HF_HOME/lerobot).
#
# Traces total RSS across the process tree against episodes written, into
# logs/lerobot/convert-<job>.trace. LeRobot's _save_episode_table concatenates every
# episode into an in-memory hf_dataset this converter never reads; with depth columns
# that grew ~168 MB/episode and OOM-killed two 64G jobs partway through. The converter
# drops it after each save_episode now, and the trace is what keeps that honest.
# ---------------------------------------------------------------------------
#SBATCH --account=aip-florian7
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --exclude=kn117

set -eu
TASK_CONFIG="${1:?usage: sbatch cluster/convert_multimodal_lerobot.sh <task_config> <data_dir> <repo_id>}"
DATA_DIR="${2:?missing <data_dir>}"
REPO_ID="${3:?missing <repo_id>}"
ROOT=/project/6101811/natashay/RoboTwin

unset PYTHONPATH PIP_CONFIG_FILE
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME=/home/natashay/projects/aip-florian7/natashay/hf_cache
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
cd "$ROOT/policy/pi05"

TRACE="$ROOT/logs/lerobot/convert-${SLURM_JOB_ID}.trace"
echo "=== $REPO_ID  from $DATA_DIR ($TASK_CONFIG)  ->  $HF_LEROBOT_HOME ==="

.venv/bin/python examples/aloha_real/convert_robotwin_multimodal_to_lerobot.py \
    --repo-id "$REPO_ID" --data-dir "$DATA_DIR" --task-config "$TASK_CONFIG" \
    --episodes-per-task 10 --depth-dtype uint16 &
PID=$!

while kill -0 $PID 2>/dev/null; do
    rss=$(ps -o rss= --ppid $PID -p $PID 2>/dev/null | awk '{s+=$1} END {printf "%.2f", s/1048576}')
    eps=$(find "$HF_LEROBOT_HOME/$REPO_ID/data" -name '*.parquet' 2>/dev/null | wc -l)
    echo "$(date '+%H:%M:%S') episodes=${eps} tree_rss_GiB=${rss}" >> "$TRACE"
    sleep 30
done
wait $PID; rc=$?
echo "converter exit=$rc" >> "$TRACE"
echo "=== peak tree RSS: $(sed 's/.*tree_rss_GiB=//' "$TRACE" | sort -rn | head -1) GiB ==="
tail -3 "$TRACE"
exit $rc
