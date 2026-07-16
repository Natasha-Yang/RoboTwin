#!/bin/bash
# Online QMFM Value-critic guided pi0.5 evaluation.
# Trains an ensemble Q ("Value") critic ONLINE during the rollouts and steers the frozen
# pi0.5 flow sampler via QMFM denoised-estimate gradient guidance. Config knobs live in
# policy/pi05/deploy_policy_online.yml. See qmfm_critic.py and Pi0.sample_actions.

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
guidance_scale=${7:-0.3} # optional: target QMFM steering_coeff after the online ramp
guidance_ramp_updates=${8:-256} # optional: TD updates to ramp 0 -> target; 0 jumps after first update
cluster=${9:-false} # optional: CriticCluster gradient guidance (true/false)

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

source .venv/bin/activate
cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_online.py --config policy/$policy_name/deploy_policy_online.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --guidance_scale ${guidance_scale} \
    --guidance_ramp_updates ${guidance_ramp_updates} \
    --cluster ${cluster} \
    --policy_name ${policy_name}
