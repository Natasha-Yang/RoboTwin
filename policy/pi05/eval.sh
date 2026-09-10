#!/bin/bash
# pi0.5 evaluation in the RoboTwin sim.
#
# `guidance_scale` is the switch between the two modes:
#   0        plain pi0.5 baseline -- no critic, no replay collection, no W&B.
#   nonzero  an ensemble QMFM `Value` critic steers the frozen pi0.5 flow sampler
#            (multisensory_steering + Pi0.sample_actions), warm-started from `critic_ckpt`
#            and kept training by TD during the rollouts. Guidance ramps
#            0 -> guidance_scale over this run's first `guidance_ramp_updates` TD updates,
#            warm-started or not.
# `best_of_n` (11th arg) is the other way the critic can act on sampling: N candidate chunks are
# drawn per control step and the highest-Q one is executed. It composes with `guidance_scale`
# (each candidate is steered, then the best steered chunk wins) and also works on its own --
# `best_of_n > 1` with `guidance_scale 0` builds the critic anyway, and samples with the plain
# pi0.5 sampler.
# `train_online` (9th arg, normally set in the critic config) turns that TD training off: the
# critic still guides, but stays frozen at `critic_ckpt` -- no replay collection, no updates,
# no ramp.
# `use_step_reward` (10th arg) picks the reward the run scores itself -- and trains the critic --
# with: true adds the task's shaped per-step progress term, false leaves the sparse 1.0 on
# success. Unlike the args above it applies to the baseline too (it is what the csv's per-episode
# `reward` column and the W&B reward curves measure).
# It defaults to whatever policy/pi05/deploy_policy.yml says; the 7th arg overrides it.
# The critic's own hyperparameters come from the `adaptation_config_path` file in that yml, and the
# 12th arg swaps that file -- which is also how the *other* critic family is selected:
#   cfgs/qmfm.yaml  (default)  an ensemble Q over action chunks; `guidance_scale` / `best_of_n`.
#   cfgs/dsrl.yaml             SAC over the sampler's latent noise. The denoising is untouched
#                              and the actor chooses the noise chunk it starts from, so there is
#                              nothing to guide or rank: leave args 7 and 11 at 0 and 1.

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 # ensure GPU < 24G
# Only needed when guidance_scale != 0: the QMFM checkout multisensory_steering imports
# `ReplayBuffer` from, by explicit path ($QMFM_ROOT/utils/datasets.py).
export QMFM_ROOT="${QMFM_ROOT:-/home/natasha/QMFM}"
# Compute nodes have no internet, and the guided path opens a W&B run per eval -- an online
# wandb.init() there times out (90s) and can take the job with it. Log to disk instead and
# `wandb sync` from a login node afterwards (cluster/wandb_sync.sh).
export WANDB_MODE="${WANDB_MODE:-offline}"

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

# Optional 7th..12th args override deploy_policy.yml, and anything past the 12th is passed
# through verbatim. Omit them to take the yml's values;
# pass `0` as guidance_scale and `1` as best_of_n to force the plain pi0.5 baseline.
overrides=()
[ -n "${7}" ] && overrides+=(--guidance_scale "${7}")          # target QMFM steering_coeff
[ -n "${8}" ] && overrides+=(--guidance_ramp_updates "${8}")   # TD updates to ramp 0 -> target
[ -n "${9}" ] && overrides+=(--train_online "${9}")            # false = guide with a frozen critic
[ -n "${10}" ] && overrides+=(--use_step_reward "${10}")       # false = sparse success reward only
[ -n "${11}" ] && overrides+=(--best_of_n "${11}")             # candidate chunks per control step
[ -n "${12}" ] && overrides+=(--adaptation_config_path "${12}")   # which critic family + its hyperparameters
# Anything after those is passed through to deploy_policy.yml verbatim, for the keys that have no
# positional slot -- above all `--resume "<run dir>"`, which continues that one interrupted run.
# A resume takes its whole configuration from the deploy_policy.yml and critic-config snapshots
# in that directory rather than from the working tree, so the args above have nothing to override
# and the identity args (task_name, seed, ...) are checked against the snapshot instead of applied.
# `bash eval_tasks.sh --resume "<run dir>"` is the launcher form, and derives every positional
# argument from the same snapshot; reach for this one only when running eval.sh by hand.
# The tail also carries the periodic held-out evaluation that picks the best critic checkpoint
# (`--eval_interval 20 --eval_episodes 10 --eval_seed 7`; `--eval_interval 0` turns it off).
# `env_seed` is deliberately NOT among them: it is a task-config key, so that one file decides
# the environment for collection and eval alike (see task_config/_config_template.yml).
# `--profile true` (or `cprofile` / `both`) is also passed this way: it turns on the wall-clock
# breakdown in envs/utils/eval_profiler.py and writes `_profile.txt` into the run's result dir.
# Off by default, and off means the hooks are never installed.
[ "$#" -gt 12 ] && overrides+=("${@:13}")

# Required. An empty gpu_id would `export CUDA_VISIBLE_DEVICES=`, which CUDA reads as "no
# devices" -- and curobo builds a CUDA tensor as a default argument in motion_gen.py, i.e. at
# import, so it then dies with "No CUDA GPUs are available". envs/robot/planner.py swallows that
# and reports "check if Curobo is installed correctly", which sends you after the wrong problem
# entirely. Fail here instead, where the cause is still legible.
if [ -z "${gpu_id}" ]; then
    echo -e "\033[31meval.sh: missing gpu_id (6th argument)\033[0m" >&2
    echo "usage: bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> [...]" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

source .venv/bin/activate
cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    "${overrides[@]}"
