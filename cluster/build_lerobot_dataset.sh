#!/bin/bash
# ---------------------------------------------------------------------------
# Build a LeRobot dataset from collected RoboTwin demos (CLAUDE.md section 4),
# for pi0.5 / pi0 fine-tuning.  Killarney (Alliance Canada / SLURM).
#
# Submit:   sbatch cluster/build_lerobot_dataset.sh <task_config> <episodes> <repo_id>
#   e.g.    sbatch cluster/build_lerobot_dataset.sh demo_clean_multimodal 10 \
#                  NatashaYang/robotwin_demo_clean_multimodal_50x10_lerobot
#
# Two stages, both CPU-only, both run in policy/pi05/.venv:
#   (a) scripts/process_data.py     -> policy/pi05/processed_data/<task>-<cfg>-<n>/
#   (b) examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py
#                                   -> $HF_HOME/lerobot/<repo_id>
# Stage (a) is skipped per task when it already has <episodes> complete episodes,
# so a timed-out or killed job can simply be resubmitted.  Stage (b) has NO
# resume of its own -- it always rebuilds from scratch -- hence the 12h band.
#
# Notes:
#   * Killarney has no CPU-only partition, so this asks for a GPU it never uses.
#   * Stage (a) buffers a whole episode of decoded 640x480 frames per worker
#     (~2 GB on the longest tasks).  That is what makes this a batch job rather
#     than something to run on a login node -- 8 workers there hit the per-user
#     memory cap and get SIGKILLed with no message at all.
#   * Compute nodes have no internet: this never pushes to the Hub.  Training
#     reads the dataset straight out of $HF_HOME/lerobot, so a push is optional.
# ---------------------------------------------------------------------------
#SBATCH --account=aip-florian7
#SBATCH --job-name=lerobot_build
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=kn117
#SBATCH --output=logs/lerobot/%x-%j.out

set -euo pipefail

TASK_CONFIG="${1:?usage: sbatch cluster/build_lerobot_dataset.sh <task_config> <episodes> <repo_id>}"
EPISODES="${2:?missing <episodes>}"
REPO_ID="${3:?missing <repo_id>}"
NPROC="${BUILD_NPROC:-8}"
# BUILD_STAGE=a runs only the processing, b only the conversion, all (default) both.
# Splitting lets stage (a) go to a short, fast-backfilling band while the long,
# non-resumable stage (b) waits in a 12h one behind an --dependency=afterok.
STAGE="${BUILD_STAGE:-all}"

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
export HF_HOME="${HF_HOME:-/home/natashay/projects/aip-florian7/natashay/hf_cache}"

cd "$ROBOTWIN_ROOT/policy/pi05"
PY="$ROBOTWIN_ROOT/policy/pi05/.venv/bin/python"
[ -x "$PY" ] || { echo "no venv interpreter at $PY" >&2; exit 1; }

mapfile -t TASKS < <(ls -d "$ROBOTWIN_ROOT/data"/*/"$TASK_CONFIG" 2>/dev/null | awk -F/ '{print $(NF-1)}' | sort)
[ "${#TASKS[@]}" -gt 0 ] || { echo "no tasks found with config '$TASK_CONFIG' under $ROBOTWIN_ROOT/data" >&2; exit 1; }
echo "=== ${#TASKS[@]} tasks with config '$TASK_CONFIG', $EPISODES episodes each ==="

# --- stage (a): raw demo HDF5 -> intermediate aloha format ------------------
LOGDIR="$ROBOTWIN_ROOT/logs/lerobot/process-${SLURM_JOB_ID:-local}"
mkdir -p "$LOGDIR"

count_episodes() {  # a complete episode dir has both the hdf5 and its instructions
    local d="processed_data/$1-$TASK_CONFIG-$EPISODES" n=0 i
    for (( i = 0; i < EPISODES; i++ )); do
        [ -f "$d/episode_$i/episode_$i.hdf5" ] && [ -f "$d/episode_$i/instructions.json" ] && n=$(( n + 1 ))
    done
    echo "$n"
}

todo=()
for t in "${TASKS[@]}"; do
    [ "$(count_episodes "$t")" -eq "$EPISODES" ] || todo+=( "$t" )
done
echo "stage (a): ${#todo[@]} task(s) to process, $(( ${#TASKS[@]} - ${#todo[@]} )) already complete"

if [ "$STAGE" = b ]; then
    todo=()
    echo "stage (a): skipped (BUILD_STAGE=b)"
fi

if [ "${#todo[@]}" -gt 0 ]; then
    printf '%s\n' "${todo[@]}" \
        | xargs -P "$NPROC" -I{} sh -c \
            "$PY scripts/process_data.py \"\$1\" $TASK_CONFIG $EPISODES > $LOGDIR/\$1.log 2>&1 \
             || echo \"FAILED \$1\" >> $LOGDIR/failures.txt" _ {}
fi

# --- gate: every task must be complete before the (non-resumable) conversion -
bad=0
for t in "${TASKS[@]}"; do
    n="$(count_episodes "$t")"
    if [ "$n" -ne "$EPISODES" ]; then echo "  incomplete: $t ($n/$EPISODES) -- see $LOGDIR/$t.log" >&2; bad=1; fi
done
if [ "$bad" -ne 0 ]; then
    echo "stage (a) incomplete; not starting the conversion. Resubmit to retry only the missing tasks." >&2
    exit 1
fi
echo "stage (a) complete: $(( ${#TASKS[@]} * EPISODES )) episodes"

if [ "$STAGE" = a ]; then
    echo "stage (b) skipped (BUILD_STAGE=a)"
    exit 0
fi

# --- stage (b): intermediate aloha format -> LeRobot dataset ----------------
# The converter walks raw_dir recursively for *.hdf5, so it needs a directory
# holding exactly this run's tasks and nothing else.  processed_data/ is shared
# across task configs, so link the ones we want into a per-config staging dir.
# os.walk() defaults to followlinks=False, so a symlinked *directory* would be
# listed and never descended into -- the staging tree has to be real dirs with
# symlinked files (the converter reads instructions.json next to each hdf5).
RAW_DIR="processed_data/_lerobot_raw-$TASK_CONFIG-$EPISODES"
rm -rf "$RAW_DIR"
for t in "${TASKS[@]}"; do
    src="$PWD/processed_data/$t-$TASK_CONFIG-$EPISODES"
    for (( i = 0; i < EPISODES; i++ )); do
        mkdir -p "$RAW_DIR/$t/episode_$i"
        ln -sf "$src/episode_$i/episode_$i.hdf5"   "$RAW_DIR/$t/episode_$i/episode_$i.hdf5"
        ln -sf "$src/episode_$i/instructions.json" "$RAW_DIR/$t/episode_$i/instructions.json"
    done
done
found=$(find -L "$RAW_DIR" -name '*.hdf5' | wc -l)
expected=$(( ${#TASKS[@]} * EPISODES ))
[ "$found" -eq "$expected" ] || { echo "staged $found hdf5, expected $expected" >&2; exit 1; }

echo "=== stage (b): converting $found episodes -> $HF_HOME/lerobot/$REPO_ID ==="
"$PY" examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
    --raw_dir "$RAW_DIR" --repo_id "$REPO_ID"

rm -rf "$RAW_DIR"
echo "=== done: $HF_HOME/lerobot/$REPO_ID ==="
