#!/bin/bash

# Submit a pi0.5 evaluation (policy/pi05/eval.sh) as a Slurm job for each task in
# a list. One GPU job per task by default; --per-job N packs N tasks into a job
# and runs them one after another.
#
# Evals are NOT packed concurrently onto a GPU the way collection is
# (submit_all_data.sh). eval.sh sets XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 (~32 GB
# of an 80 GB H100 for JAX) on top of ~7 GB for SAPIEN's OptiX renderer, so two
# concurrent evals would sit at ~78 GB and OOM the renderer. Tasks inside a job
# therefore run sequentially; use more jobs, not more workers, to go faster.
#
# Usage:
#   bash eval_tasks.sh <task_config> <train_config_name> <model_name> [seed]
#                      [--tasks t1,t2,...] [--per-job N]
#                      [--guidance-scale S] [--guidance-ramp-episodes N]
#                      [--train-online true|false]
#                      [--time HH:MM:SS] [--cpus N] [--mem 48G]
#                      [--exclude n1,n2,...] [--dry-run]
#
#   --tasks     = evaluate only the given tasks (comma- or space-separated),
#                 instead of every task in description/task_instruction/.
#                 May be repeated.
#   --per-job   = tasks per Slurm job, run sequentially (default 1).
#   --guidance-* / --train-online
#               = eval.sh's optional 7th/8th/9th args; omit them to take
#                 policy/pi05/deploy_policy.yml's values. Pass
#                 --guidance-scale 0 to force the plain pi0.5 baseline.
#   --exclude   = Slurm --exclude node list (default $EVAL_EXCLUDE). For nodes
#                 whose GPU is broken in a way Slurm does not notice: a card with
#                 pending ECC row remaps stays "healthy" in sinfo but fails
#                 vkCreateDevice AND plain CUDA, so every eval routed to it dies
#                 in eval_policy.py's render pre-flight within a minute. Diagnose
#                 a suspect node with:
#                   nvidia-smi -q -d ECC,ROW_REMAPPER | grep -A2 "Remapped Rows"
#                 ("Pending: Yes" / nonzero uncorrectable = report it to support
#                 and blacklist it here meanwhile).
#   --dry-run   = print the sbatch commands instead of submitting them.
#
# Examples:
#   bash eval_tasks.sh demo_clean pi05_base_aloha_lora_clean_50x25 run0
#   bash eval_tasks.sh demo_clean pi05_base_aloha_lora_clean_50x25 run0 0 \
#        --tasks beat_block_hammer,lift_pot
#   bash eval_tasks.sh demo_clean pi05_base_aloha_lora_clean_50x25 run0 0 \
#        --tasks beat_block_hammer --guidance-scale 0.3 --guidance-ramp-episodes 10
#   bash eval_tasks.sh demo_clean pi05_base_aloha_lora_clean_50x25 run0 0 \
#        --tasks stack_blocks_three --exclude rg12501

set -euo pipefail
shopt -s nullglob

# --- defaults (all overridable by flag) ---
seed=0
per_job=1
guidance_scale=""        # empty -> eval.sh falls through to deploy_policy.yml
guidance_ramp_episodes=""
train_online=""
time_limit=${EVAL_TIME:-12:00:00}   # test_num: 100 successful episodes is long;
                                    # robotwin_gpu.sh's 3h default is not enough
cpus=${EVAL_CPUS:-8}
mem=${EVAL_MEM:-48G}
exclude=${EVAL_EXCLUDE:-}   # nodes with known-bad GPUs; see --exclude above
dry_run=0

# --- parse args ---
task_config=""
train_config_name=""
model_name=""
seed_set=0
declare -a requested_tasks=()

need_value() { [[ $2 -gt 1 ]] || { echo "$1 requires a value" >&2; exit 1; }; }

while (( $# )); do
    arg=$1
    case "$arg" in
        --tasks)
            need_value --tasks $#; shift
            IFS=', ' read -r -a _t <<< "$1"
            requested_tasks+=("${_t[@]}")
            ;;
        --tasks=*)
            IFS=', ' read -r -a _t <<< "${arg#--tasks=}"
            requested_tasks+=("${_t[@]}")
            ;;
        --per-job)                need_value --per-job $#; shift; per_job=$1 ;;
        --per-job=*)              per_job=${arg#--per-job=} ;;
        --guidance-scale)         need_value --guidance-scale $#; shift; guidance_scale=$1 ;;
        --guidance-scale=*)       guidance_scale=${arg#--guidance-scale=} ;;
        --guidance-ramp-episodes)  need_value --guidance-ramp-episodes $#; shift; guidance_ramp_episodes=$1 ;;
        --guidance-ramp-episodes=*) guidance_ramp_episodes=${arg#--guidance-ramp-episodes=} ;;
        --train-online)           need_value --train-online $#; shift; train_online=$1 ;;
        --train-online=*)         train_online=${arg#--train-online=} ;;
        --time)                   need_value --time $#; shift; time_limit=$1 ;;
        --time=*)                 time_limit=${arg#--time=} ;;
        --cpus)                   need_value --cpus $#; shift; cpus=$1 ;;
        --cpus=*)                 cpus=${arg#--cpus=} ;;
        --mem)                    need_value --mem $#; shift; mem=$1 ;;
        --mem=*)                  mem=${arg#--mem=} ;;
        # Repeatable, so several bad nodes accumulate instead of overwriting.
        --exclude)                need_value --exclude $#; shift
                                  exclude="${exclude:+$exclude,}$1" ;;
        --exclude=*)              exclude="${exclude:+$exclude,}${arg#--exclude=}" ;;
        --dry-run|-n)             dry_run=1 ;;
        # Print the header comment block (skipping the shebang) as the help text.
        -h|--help)                awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; seen=1; next}
                                       seen {exit}' "$0"
                                  exit 0 ;;
        -*)                       echo "Unknown option: $arg" >&2; exit 1 ;;
        *)
            if   [[ -z "$task_config"       ]]; then task_config=$arg
            elif [[ -z "$train_config_name" ]]; then train_config_name=$arg
            elif [[ -z "$model_name"        ]]; then model_name=$arg
            elif (( ! seed_set ));            then seed=$arg; seed_set=1
            else echo "Unexpected argument: $arg" >&2; exit 1
            fi
            ;;
    esac
    shift
done

if [[ -z "$model_name" ]]; then
    echo "Usage: bash eval_tasks.sh <task_config> <train_config_name> <model_name> [seed]" \
         "[--tasks t1,t2,...] [--per-job N] [--guidance-scale S]" \
         "[--guidance-ramp-episodes N] [--train-online true|false]" \
         "[--time HH:MM:SS] [--cpus N] [--mem 48G]" \
         "[--exclude n1,n2,...] [--dry-run]" >&2
    exit 1
fi

if [[ ! -f "task_config/${task_config}.yml" ]]; then
    echo "Task config not found: task_config/${task_config}.yml" >&2
    exit 1
fi
if ! [[ "$per_job" =~ ^[1-9][0-9]*$ ]]; then
    echo "--per-job must be a positive integer" >&2
    exit 1
fi

# The checkpoint is only read once the job starts running, which can be hours
# after submission -- so check the path now rather than discovering a typo then.
# checkpoint_id itself lives in deploy_policy.yml, so only the run dir is checked.
ckpt_dir="policy/pi05/checkpoints/${train_config_name}/${model_name}"
if [[ ! -d "$ckpt_dir" ]]; then
    echo "WARNING: no checkpoint dir at ${ckpt_dir} -- submitting anyway." >&2
fi

# Build the list of tasks: either the explicit --tasks list (validated against
# description/task_instruction/) or every task found there.
declare -a task_names=()
if (( ${#requested_tasks[@]} )); then
    for task_name in "${requested_tasks[@]}"; do
        if [[ ! -f "description/task_instruction/${task_name}.json" ]]; then
            echo "Task not found: description/task_instruction/${task_name}.json" >&2
            exit 1
        fi
        task_names+=("$task_name")
    done
else
    for f in description/task_instruction/*.json; do
        task_names+=("$(basename "$f" .json)")
    done
fi
task_count=${#task_names[@]}

if (( task_count == 0 )); then
    echo "No task descriptions found in description/task_instruction/" >&2
    exit 1
fi

mkdir -p logs/eval

num_jobs=$(( (task_count + per_job - 1) / per_job ))
echo "Submitting ${num_jobs} eval job(s) for ${task_count} task(s):" \
     "config=${task_config} train_config=${train_config_name} model=${model_name} seed=${seed}"
[[ -n "$guidance_scale"         ]] && echo "  guidance_scale=${guidance_scale}"
[[ -n "$guidance_ramp_episodes" ]] && echo "  guidance_ramp_episodes=${guidance_ramp_episodes}"
[[ -n "$train_online"           ]] && echo "  train_online=${train_online}"
[[ -n "$exclude"                ]] && echo "  exclude=${exclude}"

# The body each job runs: activate nothing here (cluster/robotwin_gpu.sh has
# already sourced setup_env.sh); just cd into policy/pi05 and call eval.sh per
# task. eval.sh runs as a child process, so its own `cd ../..` and venv
# activation do not leak back into this loop. A failed task is reported but does
# not abort the rest of the job.
job_body='
    set -uo pipefail
    task_config=$1; train_config_name=$2; model_name=$3; seed=$4
    gs=$5; ramp=$6; online=$7; shift 7
    cd policy/pi05
    fail=0
    for task_name in "$@"; do
        echo "=== Evaluating: ${task_name} ==="
        bash eval.sh "$task_name" "$task_config" "$train_config_name" "$model_name" \
            "$seed" 0 "$gs" "$ramp" "$online" \
            || { echo "Eval failed for ${task_name}; continuing." >&2; fail=1; }
    done
    (( fail )) && echo "One or more evals in this job failed; see the log above." >&2
    exit 0
'

for (( i = 0; i < task_count; i += per_job )); do
    chunk=("${task_names[@]:i:per_job}")
    # Name the job after its first task (plus a count when it carries several).
    job_name="eval-${chunk[0]}"
    (( ${#chunk[@]} > 1 )) && job_name="${job_name}+$(( ${#chunk[@]} - 1 ))"
    echo "  ${job_name}: ${chunk[*]}"

    sbatch_cmd=(
        sbatch
        --job-name="$job_name"
        --output="logs/eval/%x-%j.out"
        --cpus-per-task="$cpus"
        --mem="$mem"
        --time="$time_limit"
    )
    # Omitted entirely when empty -- `--exclude=` with no value is a Slurm error.
    [[ -n "$exclude" ]] && sbatch_cmd+=(--exclude="$exclude")
    sbatch_cmd+=(
        cluster/robotwin_gpu.sh
        bash -c "$job_body" eval-job
        "$task_config" "$train_config_name" "$model_name" "$seed"
        "$guidance_scale" "$guidance_ramp_episodes" "$train_online"
        "${chunk[@]}"
    )

    if (( dry_run )); then
        printf '    [dry-run]'; printf ' %q' "${sbatch_cmd[@]}"; printf '\n'
    else
        "${sbatch_cmd[@]}"
    fi
done
