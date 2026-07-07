#!/bin/bash

# Submit RoboTwin collection as a fixed number of Slurm jobs (batches). Tasks are
# spread round-robin across the batches; each batch job then runs its assigned
# tasks. Two execution modes:
#
#   parallel (default): the batch runs as many tasks concurrently as fit on its
#             single allocated H100. RoboTwin's render load is tiny (3 D435
#             cameras, 320x240 RGB, ~2-4 GB VRAM), so the binding limit is CPU,
#             not the GPU. A Fir H100 node has 48 CPUs / 4 H100s -> ~12 cores per
#             GPU, and each worker needs ~3, so the job packs 12/3 = 4 workers
#             onto the GPU and requests that GPU's full CPU/RAM fair share
#             (overriding cluster/robotwin_gpu.sh's defaults).
#   sequential (--sequential): the batch runs its tasks one after another on the
#             GPU, using the job's default CPU/RAM request.
#
# Usage:
#   bash submit_all_data.sh <task_config> [num_batches] [--parallel | --sequential]
#
#   num_batches = number of Slurm jobs (i.e. number of GPUs used).
#                 Default: enough jobs to pack the GPUs (parallel), or one task
#                 per job (sequential).
#
# Examples:
#   bash submit_all_data.sh demo_randomized                # pack GPUs, default (parallel)
#   bash submit_all_data.sh demo_randomized 8              # 8 jobs, packed
#   bash submit_all_data.sh demo_randomized 5 --sequential # 5 jobs, one task at a time

set -euo pipefail
shopt -s nullglob

# --- how many workers fit on one H100 ---
# Fir H100 node = 48 CPUs / 4 GPUs -> 12 cores/GPU fair share; collection is
# CPU-bound (curobo planning + PhysX), ~3 cores/worker. VRAM/RAM are not the
# limit (~2-4 GB VRAM, ~6 GB RAM per worker on an 80 GB / ~288 GB-per-GPU node).
cpus_per_gpu=12
cpus_per_worker=3
mem_per_worker_gb=12
workers=$(( cpus_per_gpu / cpus_per_worker ))   # -> 4

# --- parse args (positional task_config, optional num_batches, mode flag) ---
task_config=""
num_batches=""
parallel=1

for arg in "$@"; do
    case "$arg" in
        --parallel)               parallel=1 ;;
        --sequential|--seq)       parallel=0 ;;
        -*)             echo "Unknown option: $arg" >&2; exit 1 ;;
        *)
            if [[ -z "$task_config" ]]; then
                task_config=$arg
            elif [[ -z "$num_batches" ]]; then
                num_batches=$arg
            else
                echo "Unexpected argument: $arg" >&2; exit 1
            fi
            ;;
    esac
done

if [[ -z "$task_config" ]]; then
    echo "Usage: bash submit_all_data.sh <task_config> [num_batches] [--parallel | --sequential]" >&2
    exit 1
fi

if [[ ! -f "task_config/${task_config}.yml" ]]; then
    echo "Task config not found: task_config/${task_config}.yml" >&2
    exit 1
fi

task_files=(description/task_instruction/*.json)
task_count=${#task_files[@]}

if (( task_count == 0 )); then
    echo "No task descriptions found in description/task_instruction/" >&2
    exit 1
fi

# Default number of batches: one task per job in sequential mode, or enough jobs
# to pack the GPUs (workers tasks each) in parallel mode.
if [[ -z "$num_batches" ]]; then
    if (( parallel )); then
        num_batches=$(( (task_count + workers - 1) / workers ))
    else
        num_batches=$task_count
    fi
elif ! [[ "$num_batches" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_batches must be a positive integer" >&2
    exit 1
fi
if (( num_batches > task_count )); then
    num_batches=$task_count
fi

mkdir -p logs/data_collection

# Spread tasks round-robin across the batches so batch sizes differ by at most
# one when task_count is not a multiple of num_batches.
declare -a batch_tasks
for (( i = 0; i < task_count; i++ )); do
    task_name=$(basename "${task_files[$i]}" .json)
    b=$(( i % num_batches ))
    batch_tasks[$b]+=" ${task_name}"
done

if (( parallel )); then
    echo "Submitting ${num_batches} parallel batch job(s) for ${task_count} tasks (<=${workers} concurrent/GPU)."
else
    echo "Submitting ${num_batches} sequential batch job(s) for ${task_count} tasks."
fi

# Only these Fir nodes have successfully initialized SAPIEN's Vulkan renderer
# in this environment. Other H100 nodes can run CUDA while exposing no usable
# Vulkan device, which makes collection fail before the task starts.
render_nodes=fc10508,fc10519,fc10604,fc10612

for (( b = 0; b < num_batches; b++ )); do
    tasks=${batch_tasks[$b]# }   # strip the leading space
    task_n=$(wc -w <<< "$tasks")

    if (( parallel )); then
        # Cap concurrency at the batch size and scale the job's CPU/RAM request
        # to the number of concurrent workers (never above the GPU fair share).
        workers_eff=$(( workers < task_n ? workers : task_n ))
        cpus=$(( workers_eff * cpus_per_worker ))
        (( cpus > cpus_per_gpu )) && cpus=$cpus_per_gpu
        mem_gb=$(( workers_eff * mem_per_worker_gb ))
        (( mem_gb < 32 )) && mem_gb=32
        echo "  batch ${b}: ${task_n} task(s), ${workers_eff} concurrent, ${cpus} cpus, ${mem_gb}G"

        # Run the batch's tasks with bounded concurrency on the single allocated
        # GPU. Each worker gets its own Warp cache dir (concurrent JIT into a
        # shared cache can race) and its own log file; launches are staggered so
        # the heavy Vulkan/Warp init does not all fire at once.
        sbatch \
            --job-name="collect-batch-${b}" \
            --output="logs/data_collection/%x-%j.out" \
            --nodelist="$render_nodes" \
            --cpus-per-task="$cpus" \
            --mem="${mem_gb}G" \
            cluster/robotwin_gpu.sh \
            bash -c '
                set -uo pipefail
                max=$1; task_config=$2; gpu_id=$3; shift 3
                logdir=logs/data_collection
                warp_base="${WARP_CACHE_PATH:-${SLURM_TMPDIR:-/tmp}/warp-cache}"
                running=0; fail=0
                for task_name in "$@"; do
                    log="${logdir}/parallel-${SLURM_JOB_ID:-0}-${task_name}.log"
                    echo "=== launching ${task_name} (log: ${log}) ==="
                    (
                        export WARP_CACHE_PATH="${warp_base}/${task_name}"
                        mkdir -p "$WARP_CACHE_PATH"
                        bash collect_data.sh "${task_name}" "${task_config}" "${gpu_id}"
                    ) > "${log}" 2>&1 &
                    if (( ++running >= max )); then
                        wait -n || fail=1
                        running=$(( running - 1 ))
                    fi
                    sleep 2
                done
                wait
                (( fail )) && echo "One or more tasks in this batch failed; see per-task logs." >&2
                exit 0
            ' collect-batch "$workers_eff" "$task_config" 0 $tasks
    else
        echo "  batch ${b}: ${task_n} task(s)"

        # Pass the batch's task names as separate arguments after the task_config
        # and gpu_id, then run them one at a time. A failed task is logged but
        # does not abort the rest of the batch.
        sbatch \
            --job-name="collect-batch-${b}" \
            --output="logs/data_collection/%x-%j.out" \
            --nodelist="$render_nodes" \
            cluster/robotwin_gpu.sh \
            bash -c '
                set -uo pipefail
                task_config=$1; gpu_id=$2; shift 2
                for task_name in "$@"; do
                    echo "=== Collecting: ${task_name} ==="
                    bash collect_data.sh "${task_name}" "${task_config}" "${gpu_id}" \
                        || echo "Collection failed for ${task_name}; continuing." >&2
                done
            ' collect-batch "$task_config" 0 $tasks
    fi
done
