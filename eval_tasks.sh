#!/bin/bash

# Submit a pi0.5 evaluation (policy/pi05/eval.sh) as a Slurm job for each task in
# a list. Each task's eval is split across --shards parallel jobs (default 4), so
# what used to be one 12h job is four 3h ones running side by side; --per-job N
# packs N tasks into a job and runs them one after another.
#
# Sharding needs a *reference seed list* per task -- the seeds to evaluate, one per
# line, by default eval_result/<task>/seed_<seed>_list.txt (--seed-list overrides
# the pattern). An unsharded eval cannot be split because it does not know its own
# seed sequence in advance: it walks upward from 100000*(1+seed) until `test_num`
# seeds have passed the expert check, and which ones those are is not a function of
# the episode index. Given a fixed list, this script cuts it into --shards
# contiguous chunks, writes each to
# eval_result/<task>/seed_shards/<submit_time>/, and hands one to each job
# (eval.sh's 12th arg -> eval_policy.py's `seed_list`). A shard whose seed fails the
# expert check skips it, so a shard can finish with slightly fewer episodes than
# seeds; `_result.txt` divides by the episodes actually run.
#
# A reference list is just the seeds some earlier run of that task turned out to
# use -- lift them out of a finished run with:
#   python -c "import pandas,sys; print(pandas.read_csv(sys.argv[1]).seed.to_csv(index=False,header=False))" \
#       "eval_result/<task>/pi05/<cfg>/<ckpt>/<ts>/_episode_results.csv" \
#       > eval_result/<task>/seed_<seed>_list.txt
# (or `suc_test_seed_list` in that run's resume_state.json). They are only valid
# for the same task config and embodiment -- the expert check is what accepted
# them, and it is stochastic enough that the odd seed will be rejected on a rerun.
#
# Each shard also gets `run_tag=shard<k>of<N>` (eval.sh's 13th arg), which inserts
# that level above the run timestamp in eval_result/. Without it the concurrent
# shards would be candidates for each other's resume state and would overwrite one
# another's critic checkpoint. Each shard also opens its own W&B run, so to score
# the task as a whole, concatenate the four _episode_results.csv files. Use
# --shards 1 for the old single-job behavior (no seed list, no run tag).
#
# Sharding is REFUSED when the critic trains online (`train_online` true with a
# critic actually running, i.e. guidance_scale != 0 or best_of_n > 1): N processes
# would each train their own critic on their own slice and steer differently from
# one another partway through, so the shards would not be evaluating the same
# policy. Freeze the critic (--train-online false, needs a critic_ckpt) to shard a
# guided eval, or use --shards 1. The check resolves train_online the way
# eval_policy.py does -- CLI > deploy_policy.yml > critic_config_path -- and errs
# toward refusing when it cannot read those files.
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
# in b4 (42), <=7d in b5 (17); the default here is 3h with sharding on, 12h with
# --shards 1. A shorter --time gets a bigger node pool but NOT necessarily a
# sooner start -- each band queues separately, and b1 is often the more contended
# one (measured 2026-08-07: a 12h job was scheduled ~8h earlier than an otherwise
# identical 3h job). Four shards win on wall-clock once they are *running*, but
# they are also four separate queue waits in the band that measured slower, so
# check before assuming, with:
#   sbatch --test-only --time=<HH:MM:SS> --cpus-per-task=8 --mem=48G \
#          cluster/robotwin_gpu.sh bash -c true
#
# Usage:
#   bash eval_tasks.sh <task_config> <train_config_name> <model_name> [seed]
#                      [--tasks t1,t2,...] [--shards N] [--seed-list PATTERN]
#                      [--per-job N]
#                      [--guidance-scale S] [--guidance-ramp-updates N]
#                      [--train-online true|false]
#                      [--time HH:MM:SS] [--cpus N] [--mem 48G]
#                      [--exclude n1,n2,...] [--dry-run]
#
#   --tasks     = evaluate only the given tasks (comma- or space-separated),
#                 instead of every task in description/task_instruction/.
#                 May be repeated.
#   --shards    = parallel jobs per task, each evaluating its own slice of the
#                 reference seed list (default 4). 1 = one job over the whole
#                 eval, with no seed list at all.
#   --seed-list = where to find each task's reference seed list; `{task}` and
#                 `{seed}` are substituted (default
#                 eval_result/{task}/seed_{seed}_list.txt). Ignored at --shards 1.
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
#   # 4 x 3h jobs over eval_result/open_microwave/seed_1_list.txt, 25 seeds each
#   bash eval_tasks.sh demo_clean pi05_base_aloha_lora_clean_50x25 run0 1 \
#        --tasks open_microwave
#   # the old shape: one 12h job, seeds searched rather than listed
#   bash eval_tasks.sh demo_clean pi05_base_aloha_lora_clean_50x25 run0 --shards 1
#   bash eval_tasks.sh demo_clean pi05_base_aloha_lora_clean_50x25 run0 0 \
#        --tasks beat_block_hammer,lift_pot
#   bash eval_tasks.sh demo_clean pi05_base_aloha_lora_clean_50x25 run0 0 \
#        --tasks beat_block_hammer --guidance-scale 0.3 --guidance-ramp-updates 256
#   bash eval_tasks.sh demo_clean pi05_base_aloha_lora_clean_50x25 run0 0 \
#        --tasks stack_blocks_three --exclude kn117

set -euo pipefail
shopt -s nullglob

# --- defaults (all overridable by flag) ---
seed=0
per_job=1
shards=${EVAL_SHARDS:-4}            # parallel jobs per task; 1 = the old single job
seed_list_pattern=${EVAL_SEED_LIST:-'eval_result/{task}/seed_{seed}_list.txt'}
guidance_scale=""        # empty -> eval.sh falls through to deploy_policy.yml
guidance_ramp_updates=""
train_online=""
# Resolved after parsing, because it depends on --shards: a quarter of the episodes
# fits in the 3h band, the whole 100 needs 12h (robotwin_gpu.sh's own 3h default is
# not enough for that).
time_limit=${EVAL_TIME:-}
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
        --shards)                 need_value --shards $#; shift; shards=$1 ;;
        --shards=*)               shards=${arg#--shards=} ;;
        --seed-list)              need_value --seed-list $#; shift; seed_list_pattern=$1 ;;
        --seed-list=*)            seed_list_pattern=${arg#--seed-list=} ;;
        --guidance-scale)         need_value --guidance-scale $#; shift; guidance_scale=$1 ;;
        --guidance-scale=*)       guidance_scale=${arg#--guidance-scale=} ;;
        --guidance-ramp-updates)  need_value --guidance-ramp-updates $#; shift; guidance_ramp_updates=$1 ;;
        --guidance-ramp-updates=*) guidance_ramp_updates=${arg#--guidance-ramp-updates=} ;;
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
         "[--tasks t1,t2,...] [--shards N] [--seed-list PATTERN]" \
         "[--per-job N] [--guidance-scale S]" \
         "[--guidance-ramp-updates N] [--train-online true|false]" \
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
if ! [[ "$shards" =~ ^[1-9][0-9]*$ ]]; then
    echo "--shards must be a positive integer" >&2
    exit 1
fi
if [[ -z "$time_limit" ]]; then
    if (( shards > 1 )); then time_limit=3:00:00; else time_limit=12:00:00; fi
fi

# --- a critic that learns online cannot be sharded ---
# Sharding runs N independent processes, and an online critic is per-process state: each shard
# would start from the same `critic_ckpt`, train on its own slice of the episodes, and write its
# own online_value_critic.pkl. That is N short TD runs, not the one long one the eval is
# supposed to be -- and worse, the guidance each shard applies then diverges from every other
# shard's partway through, so the "same" policy is not being evaluated across the run. Refuse it
# rather than produce results that silently mean something else. A *frozen* critic
# (`train_online: false`) shards fine: every shard steers with the identical checkpoint.
#
# The effective value has to be resolved the way eval_policy.py resolves it -- CLI override >
# deploy_policy.yml > the critic config it includes -- and only matters when a critic exists at
# all (guidance_scale != 0 or best_of_n > 1); with no critic, train_online is inert.
if (( shards > 1 )); then
    deploy_yml=policy/pi05/deploy_policy.yml
    critic_state=$(
        python3 - "$deploy_yml" "$guidance_scale" "$train_online" <<'PY' || echo "?"
import pathlib, sys, yaml

deploy_path, gs_override, online_override = sys.argv[1:4]
cfg = yaml.safe_load(pathlib.Path(deploy_path).read_text(encoding="utf-8")) or {}
included_path = cfg.get("critic_config_path")
if included_path and pathlib.Path(included_path).is_file():
    included = yaml.safe_load(pathlib.Path(included_path).read_text(encoding="utf-8")) or {}
    cfg = {**included, **cfg}   # the deploy config wins, as parse_args_and_config has it


def as_bool(value, default):
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "on", "1")
    return bool(value)


guidance = float(gs_override) if gs_override else float(cfg.get("guidance_scale") or 0)
best_of_n = int(cfg.get("best_of_n") or 1)
train_online = as_bool(online_override, as_bool(cfg.get("train_online"), True))
print(int(guidance != 0 or best_of_n > 1), int(train_online))
PY
    )
    if [[ "$critic_state" == "?" ]]; then
        echo "WARNING: could not read ${deploy_yml} -- not checking train_online against" \
             "--shards ${shards}." >&2
    else
        read -r critic_on online_on <<< "$critic_state"
        if (( critic_on && online_on )); then
            echo "train_online is true and --shards is ${shards}: an online-trained critic cannot" \
                 "be split across parallel jobs (each shard would train its own critic on its own" \
                 "slice and steer differently from the others)." >&2
            echo "  Either freeze it -- --train-online false, which needs a critic_ckpt in" \
                 "${deploy_yml} -- or run the eval as one job with --shards 1." >&2
            exit 1
        fi
    fi
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

# Where shard k of `task`'s split seed list lives. Called with the literal string
# `{task}` as well, to build the pattern the job body expands per task. Absolute,
# since the job reads it from inside policy/pi05.
# Stamped with the submission time so a later resubmission writes new files instead
# of rewriting the ones a still-queued job is going to read.
submit_stamp=$(date +%Y%m%d-%H%M%S)
shard_path() {  # <task> <shard index>
    echo "${PWD}/eval_result/${1}/seed_shards/${submit_stamp}/seed${seed}_${2}of${shards}.txt"
}

# Split each task's reference seed list into `shards` contiguous chunks, up front:
# a missing list should fail before anything is submitted, not after three of four
# jobs are queued. (Remainder seeds go to the low-numbered shards, so sizes differ
# by at most one.)
if (( shards > 1 )); then
    for task_name in "${task_names[@]}"; do
        ref=${seed_list_pattern//\{task\}/$task_name}
        ref=${ref//\{seed\}/$seed}
        if [[ ! -f "$ref" ]]; then
            echo "No reference seed list for ${task_name}: ${ref}" >&2
            echo "  Sharding needs one (a seed per line) -- point --seed-list at it," \
                 "or pass --shards 1 to run the task as a single unsharded job." >&2
            exit 1
        fi
        # Tolerate comments, blank lines, CRLF and trailing junk on a line.
        mapfile -t seeds < <(sed 's/#.*//' "$ref" | tr -d '\r' | awk 'NF {print $1}')
        seed_count=${#seeds[@]}
        if (( seed_count < shards )); then
            echo "${ref} holds only ${seed_count} seed(s), fewer than the ${shards} shards" >&2
            exit 1
        fi
        base=$(( seed_count / shards ))
        rem=$(( seed_count % shards ))
        offset=0
        for (( k = 1; k <= shards; k++ )); do
            size=$base
            (( k <= rem )) && size=$(( base + 1 ))
            slice=("${seeds[@]:offset:size}")
            out=$(shard_path "$task_name" "$k")
            echo "  ${task_name} shard ${k}/${shards}: ${size} seed(s)" \
                 "${slice[0]}..${slice[size-1]}"
            if (( ! dry_run )); then
                mkdir -p "$(dirname "$out")"
                printf '%s\n' "${slice[@]}" > "$out"
            fi
            offset=$(( offset + size ))
        done
    done
    (( dry_run )) && echo "  [dry-run] shard files not written"
fi

num_jobs=$(( (task_count + per_job - 1) / per_job * shards ))
echo "Submitting ${num_jobs} eval job(s) for ${task_count} task(s) x ${shards} shard(s):" \
     "config=${task_config} train_config=${train_config_name} model=${model_name} seed=${seed}"
[[ -n "$guidance_scale"         ]] && echo "  guidance_scale=${guidance_scale}"
[[ -n "$guidance_ramp_updates"  ]] && echo "  guidance_ramp_updates=${guidance_ramp_updates}"
[[ -n "$train_online"           ]] && echo "  train_online=${train_online}"
[[ -n "$exclude"                ]] && echo "  exclude=${exclude}"

# The body each job runs: activate nothing here (cluster/robotwin_gpu.sh has
# already sourced setup_env.sh); just cd into policy/pi05 and call eval.sh per
# task. eval.sh runs as a child process, so its own `cd ../..` and venv
# activation do not leak back into this loop. A failed task is reported but does
# not abort the rest of the job.
# `seed_pat` carries a literal `{task}` rather than a resolved path, because a job
# packing several tasks (--per-job) needs a different shard file for each of them.
# Both it and `run_tag` are empty at --shards 1, and eval.sh ignores empty args.
job_body='
    set -uo pipefail
    task_config=$1; train_config_name=$2; model_name=$3; seed=$4
    gs=$5; ramp=$6; online=$7; seed_pat=$8; run_tag=$9; shift 9
    cd policy/pi05
    fail=0
    for task_name in "$@"; do
        echo "=== Evaluating: ${task_name} ==="
        seed_list=${seed_pat//\{task\}/$task_name}
        [ -n "$seed_list" ] && echo "    seeds: ${seed_list}"
        bash eval.sh "$task_name" "$task_config" "$train_config_name" "$model_name" \
            "$seed" 0 "$gs" "$ramp" "$online" "" "" "$seed_list" "$run_tag" \
            || { echo "Eval failed for ${task_name}; continuing." >&2; fail=1; }
    done
    (( fail )) && echo "One or more evals in this job failed; see the log above." >&2
    exit 0
'

for (( k = 1; k <= shards; k++ )); do
    if (( shards > 1 )); then
        seed_pat=$(shard_path '{task}' "$k")
        run_tag="shard${k}of${shards}"
    else
        seed_pat=""      # no list: the run searches seeds itself, as it always did
        run_tag=""
    fi

    for (( i = 0; i < task_count; i += per_job )); do
        chunk=("${task_names[@]:i:per_job}")
        # Name the job after its first task (plus a count when it carries several,
        # and the shard when the eval is split).
        job_name="eval-${chunk[0]}"
        (( ${#chunk[@]} > 1 )) && job_name="${job_name}+$(( ${#chunk[@]} - 1 ))"
        (( shards > 1 ))       && job_name="${job_name}-s${k}of${shards}"
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
            "$guidance_scale" "$guidance_ramp_updates" "$train_online"
            "$seed_pat" "$run_tag"
            "${chunk[@]}"
        )

        if (( dry_run )); then
            printf '    [dry-run]'; printf ' %q' "${sbatch_cmd[@]}"; printf '\n'
        else
            "${sbatch_cmd[@]}"
        fi
    done
done
