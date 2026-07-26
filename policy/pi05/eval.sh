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
# It defaults to whatever policy/pi05/deploy_policy.yml says; the 7th arg overrides it.
# The critic's own hyperparameters come from the `critic_config_path` file in that yml.

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G
export PATH="/home/natasha/miniconda3/envs/cuda128/bin:$PATH" # CUDA 12.8 for curobo on RTX 5090
export QMFM_ROOT="${QMFM_ROOT:-/home/natasha/QMFM}" # QMFM repo (ReplayBuffer is imported from here)

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

# Optional 7th/8th args override deploy_policy.yml. Omit them to take the yml's values;
# pass `0` as guidance_scale to force the plain pi0.5 baseline.
overrides=()
[ -n "${7}" ] && overrides+=(--guidance_scale "${7}")          # target QMFM steering_coeff
[ -n "${8}" ] && overrides+=(--guidance_ramp_updates "${8}")   # TD updates to ramp 0 -> target

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
