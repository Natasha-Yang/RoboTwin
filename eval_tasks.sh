#!/bin/bash

# Submit a pi0.5 evaluation (policy/pi05/eval.sh) as a Slurm job for each task in
# a list. One GPU job per task by default; --per-job N packs N tasks into a job
# and runs them one after another.
#
# Evals are NOT packed concurrently onto a GPU the way collection is
# (submit_all_data.sh). eval.sh sets XLA_PYTHON_CLIENT_MEM_FRACTION=0.4, which on
# Killarney's 46 GB L40S reserves ~18 GB for JAX, on top of ~7 GB for SAPIEN's
# OptiX renderer -> ~25 GB for one eval. Two concurrent would want ~50 GB and OOM
# the 46 GB card. Tasks inside a job therefore run sequentially; use more jobs,
# not more workers, to go faster. (The same held on Rorqual's 80 GB H100, where
# 0.4 meant ~32 GB and two evals came to ~78 GB.)
#
# Killarney time bands: Slurm routes a GPU job to a gpubase_l40s_b* partition by
# --time. <=3h lands in b1 (168 nodes), <=12h in b2 (126), <=1d in b3 (84), <=3d
# in b4 (42), <=7d in b5 (17); the default here is 12h. A shorter --time gets a
# bigger node pool but NOT necessarily a sooner start -- each band queues
# separately, and b1 is often the more contended one (measured 2026-08-07: a 12h
# job was scheduled ~8h earlier than an otherwise identical 3h job). Check before
# assuming, with:
#   sbatch --test-only --time=<HH:MM:SS> --cpus-per-task=8 --mem=48G \
#          cluster/robotwin_gpu.sh bash -c true
#
# Usage:
#   bash eval_tasks.sh <task_config> <train_config_name> <model_name> [seed]
#                      [--tasks t1,t2,...] [--per-job N]
#                      [--guidance-scale S] [--guidance-ramp-updates N]
#                      [--train-online true|false] [--critic-ckpt PATH]
#                      [--time HH:MM:SS] [--cpus N] [--mem 48G]
#                      [--exclude n1,n2,...] [--dry-run]
#
#   bash eval_tasks.sh --resume "<run dir>" [--time ...] [--cpus N] [--mem 48G]
#                      [--exclude n1,n2,...] [--dry-run]
#
#   --resume    = continue one interrupted run, in its existing directory. It is the
#                 ONLY experiment argument the command takes: the task, task config,
#                 train config, model and seed are read out of the deploy_policy.yml
#                 snapshot in that directory, and so are the critic settings -- that
#                 snapshot is what the run was actually launched with, whereas the
#                 working-tree ymls describe whatever you are setting up next.
#                 Passing --tasks, --critic-ckpt, --guidance-*, --train-online or the
#                 positional arguments alongside it is an error rather than an
#                 override, because none of them can apply to a run that is already
#                 half done: its episodes, seed sequence and critic were produced
#                 under the old settings. Only the Slurm knobs below still apply.
#                 Resume one directory per command; there is no batch form, so a
#                 mistyped path cannot fan out across tasks.
#
#   --tasks     = evaluate only the given tasks (comma- or space-separated),
#                 instead of every task in description/task_instruction/.
#                 May be repeated.
#   --per-job   = tasks per Slurm job, run sequentially (default 1).
#   --guidance-* / --train-online
#               = eval.sh's optional 7th/8th/9th args; omit them to take
#                 policy/pi05/deploy_policy.yml's values. Pass
#                 --guidance-scale 0 to force the plain pi0.5 baseline.
#   --critic-ckpt
#               = warm-start the critic from this pickle instead of the one named
#                 in deploy_policy.yml (it has no positional slot in eval.sh, so
#                 it rides through eval.sh's pass-through tail as
#                 `--critic_ckpt <path>`). Resolved to an absolute path here, so
#                 it means the same thing whatever directory the job runs from.
#                 NOTE it applies to *every* task in the submission, and a critic
#                 is trained per task -- with --tasks naming more than one, this
#                 is only right for a checkpoint meant to transfer across them.
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
#        --tasks beat_block_hammer --guidance-scale 0.3 --guidance-ramp-updates 256
#   bash eval_tasks.sh demo_clean pi05_base_aloha_lora_clean_50x25 run0 0 \
#        --tasks stack_blocks_three --exclude kn117
#   bash eval_tasks.sh demo_clean pi05_base_aloha_lora_clean_50x25 run0 0 \
#        --tasks turn_switch --guidance-scale 0.3 --train-online false \
#        --critic-ckpt "eval_result/turn_switch/pi05/demo_clean/robotwin_clean_ep10/2026-09-04 06:27:31/online_value_critic_best.pkl"
#   bash eval_tasks.sh --resume "eval_result/lift_pot/pi05/demo_clean/robotwin_clean_ep10/2026-09-04 12:55:11"

set -euo pipefail
shopt -s nullglob

# --- defaults (all overridable by flag) ---
seed=0
per_job=1
guidance_scale=""        # empty -> eval.sh falls through to deploy_policy.yml
guidance_ramp_updates=""
train_online=""
critic_ckpt=""           # empty -> eval.sh falls through to deploy_policy.yml
resume_dir=""            # non-empty -> continue that run and take everything from it
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
        --guidance-ramp-updates)  need_value --guidance-ramp-updates $#; shift; guidance_ramp_updates=$1 ;;
        --guidance-ramp-updates=*) guidance_ramp_updates=${arg#--guidance-ramp-updates=} ;;
        --train-online)           need_value --train-online $#; shift; train_online=$1 ;;
        --train-online=*)         train_online=${arg#--train-online=} ;;
        --critic-ckpt)            need_value --critic-ckpt $#; shift; critic_ckpt=$1 ;;
        --critic-ckpt=*)          critic_ckpt=${arg#--critic-ckpt=} ;;
        --resume)                 need_value --resume $#; shift; resume_dir=$1 ;;
        --resume=*)               resume_dir=${arg#--resume=} ;;
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

# --- resume: take the whole experiment from the run directory ---------------
# A resumed run continues episodes, a seed sequence and a critic that were produced under the
# settings in its own snapshot, so nothing here may reconfigure it -- the arguments that would
# are refused rather than ignored. Deriving the positionals from the snapshot (instead of asking
# for them again) is also what keeps `--resume` from being pointed at another task's directory:
# there is no second place for the task name to come from and disagree.
snap_get() {  # snap_get <yml> <key> -- top-level scalar, quotes and trailing comment stripped
    local v
    v=$(sed -n -E "s/^$2:[[:space:]]+(.*[^[:space:]])[[:space:]]*\$/\1/p" "$1" | head -1)
    v=${v%%$'\t'#*}
    [[ "$v" != \'* && "$v" != \"* ]] && v=${v%% #*}
    v=${v#\"}; v=${v%\"}; v=${v#\'}; v=${v%\'}
    printf '%s' "$v"
}

if [[ -n "$resume_dir" ]]; then
    conflicts=()
    [[ -n "$task_config"           ]] && conflicts+=("positional arguments")
    (( ${#requested_tasks[@]} ))      && conflicts+=("--tasks")
    [[ -n "$critic_ckpt"           ]] && conflicts+=("--critic-ckpt")
    [[ -n "$guidance_scale"        ]] && conflicts+=("--guidance-scale")
    [[ -n "$guidance_ramp_updates" ]] && conflicts+=("--guidance-ramp-updates")
    [[ -n "$train_online"          ]] && conflicts+=("--train-online")
    (( per_job != 1 ))                && conflicts+=("--per-job")
    if (( ${#conflicts[@]} )); then
        echo "--resume takes no experiment arguments; drop: ${conflicts[*]}" >&2
        echo "The run's settings come from the deploy_policy.yml snapshot in its own" >&2
        echo "directory -- that is what it was launched with. To evaluate something" >&2
        echo "different, start a new run instead of resuming this one." >&2
        exit 1
    fi

    resume_dir=${resume_dir%/}
    if [[ ! -d "$resume_dir" ]]; then
        echo "--resume: no such directory: ${resume_dir}" >&2; exit 1
    fi
    if [[ ! -f "$resume_dir/resume_state.json" ]]; then
        echo "--resume: ${resume_dir} has no resume_state.json -- nothing to continue." >&2
        echo "(A run that predates crash recovery can have one rebuilt from its Slurm log:" >&2
        echo " python script/resume_state_from_log.py <log> \"${resume_dir}\")" >&2
        exit 1
    fi
    snap="$resume_dir/deploy_policy.yml"
    if [[ ! -f "$snap" ]]; then
        echo "--resume: ${resume_dir} has no deploy_policy.yml snapshot." >&2
        echo "Runs from before config snapshotting cannot be resumed this way." >&2
        exit 1
    fi

    task_name=$(snap_get "$snap" task_name)
    task_config=$(snap_get "$snap" task_config)
    train_config_name=$(snap_get "$snap" train_config_name)
    model_name=$(snap_get "$snap" model_name)
    seed=$(snap_get "$snap" seed)
    for kv in "task_name=$task_name" "task_config=$task_config" \
              "train_config_name=$train_config_name" "model_name=$model_name" "seed=$seed"; do
        if [[ -z "${kv#*=}" || "${kv#*=}" == "null" ]]; then
            echo "--resume: ${snap} does not record ${kv%%=*}; cannot rebuild the command." >&2
            exit 1
        fi
    done

    # The snapshot and the path it sits in have to agree. They are written by the same process
    # and can only diverge if the directory was moved, or if a run wrote into a directory that
    # was never its own -- which is the failure this flag exists to make impossible.
    dir_task=$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$resume_dir")")")")")
    dir_cfg=$(basename "$(dirname "$(dirname "$resume_dir")")")
    if [[ "$dir_task" != "$task_name" || "$dir_cfg" != "$task_config" ]]; then
        echo "--resume: ${resume_dir} disagrees with its own snapshot." >&2
        echo "  path says   : task=${dir_task} config=${dir_cfg}" >&2
        echo "  snapshot says: task=${task_name} config=${task_config}" >&2
        echo "This directory holds more than one experiment; do not resume it." >&2
        exit 1
    fi

    requested_tasks=("$task_name")
    echo "Resuming ${resume_dir}"
    echo "  task=${task_name} config=${task_config} train_config=${train_config_name}" \
         "model=${model_name} seed=${seed}"
fi

if [[ -z "$model_name" ]]; then
    echo "Usage: bash eval_tasks.sh <task_config> <train_config_name> <model_name> [seed]" \
         "[--tasks t1,t2,...] [--per-job N] [--guidance-scale S]" \
         "[--guidance-ramp-updates N] [--train-online true|false]" \
         "[--critic-ckpt PATH]" \
         "[--time HH:MM:SS] [--cpus N] [--mem 48G]" \
         "[--exclude n1,n2,...] [--dry-run]" >&2
    echo "   or: bash eval_tasks.sh --resume \"<run dir>\"" \
         "[--time HH:MM:SS] [--cpus N] [--mem 48G] [--exclude n1,n2,...] [--dry-run]" >&2
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

# Same reasoning for the critic checkpoint, plus one more: the job body cds into
# policy/pi05 and eval.sh cds back to the repo root, so a relative path would be
# read from a directory the caller never typed it against. Pin it here, where the
# cwd is still the one the user is looking at. A path that does not exist is
# passed through verbatim (it may be created before the job starts running).
if [[ -n "$critic_ckpt" ]]; then
    if [[ -e "$critic_ckpt" ]]; then
        critic_ckpt=$(realpath "$critic_ckpt")
    else
        echo "WARNING: no critic checkpoint at ${critic_ckpt} -- submitting anyway." >&2
    fi
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
[[ -n "$guidance_ramp_updates"  ]] && echo "  guidance_ramp_updates=${guidance_ramp_updates}"
[[ -n "$train_online"           ]] && echo "  train_online=${train_online}"
[[ -n "$critic_ckpt"            ]] && echo "  critic_ckpt=${critic_ckpt}"
[[ -n "$resume_dir"             ]] && echo "  resume=${resume_dir}"
[[ -n "$exclude"                ]] && echo "  exclude=${exclude}"

# The body each job runs: activate nothing here (cluster/robotwin_gpu.sh has
# already sourced setup_env.sh); just cd into policy/pi05 and call eval.sh per
# task. eval.sh runs as a child process, so its own `cd ../..` and venv
# activation do not leak back into this loop. A failed task is reported but does
# not abort the rest of the job.
job_body='
    set -uo pipefail
    task_config=$1; train_config_name=$2; model_name=$3; seed=$4
    gs=$5; ramp=$6; online=$7; ckpt=$8; resume=$9; shift 9
    # Neither critic_ckpt nor resume has a positional slot in eval.sh; both go
    # through the pass-through tail, which starts at the 13th argument -- hence
    # the three empty placeholders for use_step_reward / best_of_n /
    # critic_config_path. They are mutually exclusive at the parser above: a
    # resumed run takes its critic from its own directory.
    # (No apostrophes in here: job_body is single-quoted.)
    extra=()
    [ -n "$ckpt" ] && extra+=(--critic_ckpt "$ckpt")
    [ -n "$resume" ] && extra+=(--resume "$resume")
    [ ${#extra[@]} -gt 0 ] && extra=("" "" "" "${extra[@]}")
    cd policy/pi05
    fail=0
    for task_name in "$@"; do
        echo "=== Evaluating: ${task_name} ==="
        bash eval.sh "$task_name" "$task_config" "$train_config_name" "$model_name" \
            "$seed" 0 "$gs" "$ramp" "$online" ${extra[@]+"${extra[@]}"} \
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
        "$guidance_scale" "$guidance_ramp_updates" "$train_online" "$critic_ckpt"
        "$resume_dir"
        "${chunk[@]}"
    )

    if (( dry_run )); then
        printf '    [dry-run]'; printf ' %q' "${sbatch_cmd[@]}"; printf '\n'
    else
        "${sbatch_cmd[@]}"
    fi
done
