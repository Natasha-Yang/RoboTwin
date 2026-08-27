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
# The critic's own hyperparameters come from the `critic_config_path` file in that yml.

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G
# Only needed when guidance_scale != 0: the QMFM checkout multisensory_steering imports
# `ReplayBuffer` from, by explicit path ($QMFM_ROOT/utils/datasets.py).
export QMFM_ROOT="${QMFM_ROOT:-/home/natashay/links/projects/def-florian7/natashay/QMFM}"
# Compute nodes have no internet, and the guided path opens a W&B run per eval -- an online
# wandb.init() there times out (90s) and can take the job with it. Log to disk instead and
# `wandb sync` from a login node afterwards (cluster/wandb_sync.sh).
export WANDB_MODE="${WANDB_MODE:-offline}"
# Same reason: `offline_mix` calls datasets.load_dataset(), which otherwise re-resolves the repo
# against the Hub on every eval and re-downloads any shard the cache is missing (gigabytes, and
# it will hang rather than fail on a node with no route out). Force the local $HF_HOME cache.
# Pre-warm it from a login node first -- both the parquet snapshot *and* the arrow builder cache:
#   policy/pi05/.venv/bin/python -c "import datasets; datasets.load_dataset('<repo_id>', split='train')"
# (that venv's python specifically, so nothing rewrites the cache with a newer datasets -- see
# build_dataset()'s note in multisensory_steering/utils.py).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

# Optional 7th..11th args override deploy_policy.yml. Omit them to take the yml's values;
# pass `0` as guidance_scale and `1` as best_of_n to force the plain pi0.5 baseline.
overrides=()
[ -n "${7}" ] && overrides+=(--guidance_scale "${7}")          # target QMFM steering_coeff
[ -n "${8}" ] && overrides+=(--guidance_ramp_updates "${8}")   # TD updates to ramp 0 -> target
[ -n "${9}" ] && overrides+=(--train_online "${9}")            # false = guide with a frozen critic
[ -n "${10}" ] && overrides+=(--use_step_reward "${10}")       # false = sparse success reward only
[ -n "${11}" ] && overrides+=(--best_of_n "${11}")             # candidate chunks per control step
# Anything after those is passed through to deploy_policy.yml verbatim, for the keys that have no
# positional slot -- above all `--resume "<run dir>"`, which continues one specific interrupted run
# without editing the yml (and so without steering a concurrent eval into that run's directory).
[ "$#" -gt 11 ] && overrides+=("${@:12}")

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
