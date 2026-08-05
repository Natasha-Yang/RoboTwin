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
# `train_online` (9th arg, normally set in the critic config) turns that TD training off: the
# critic still guides, but stays frozen at `critic_ckpt` -- no replay collection, no updates,
# no ramp.
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

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

# Optional 7th/8th/9th args override deploy_policy.yml. Omit them to take the yml's values;
# pass `0` as guidance_scale to force the plain pi0.5 baseline.
overrides=()
[ -n "${7}" ] && overrides+=(--guidance_scale "${7}")          # target QMFM steering_coeff
[ -n "${8}" ] && overrides+=(--guidance_ramp_updates "${8}")   # TD updates to ramp 0 -> target
[ -n "${9}" ] && overrides+=(--train_online "${9}")            # false = guide with a frozen critic

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
