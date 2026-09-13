#!/bin/bash
# Zero-shot evaluation of the released pi05-DROID checkpoint in the RoboTwin sim.
#
#   bash eval.sh <task_name> <task_config> <seed> <gpu_id> [extra --key value ...]
#
# e.g.
#   bash eval.sh open_microwave demo_clean_franka_panda 0 0
#   bash eval.sh open_microwave demo_clean_franka_panda 0 0 --velocity_dt 0.1 --pi0_step 5
#
# The task config MUST use the two-panda embodiment form:
#   embodiment: [franka-panda, franka-panda, 0.6]
# See deploy_policy.yml for why the one-entry form does not work.
#
# There is no train_config_name / model_name / checkpoint_id here the way there is for
# policy/pi05: the checkpoint is a fixed bucket path (`checkpoint_dir` in deploy_policy.yml),
# not one of this repo's own fine-tuning runs.

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65

policy_name=pi05_droid
task_name=${1}
task_config=${2}
seed=${3}
gpu_id=${4}

# Required. An empty gpu_id would `export CUDA_VISIBLE_DEVICES=`, which CUDA reads as "no
# devices" -- and curobo builds a CUDA tensor as a default argument at import, so it then dies
# with "No CUDA GPUs are available" and envs/robot/planner.py misreports it as a bad curobo
# install. Fail here, where the cause is still legible.
if [ -z "${gpu_id}" ]; then
    echo -e "\033[31meval.sh: missing gpu_id (4th argument)\033[0m" >&2
    echo "usage: bash eval.sh <task_name> <task_config> <seed> <gpu_id> [--key value ...]" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# openpi and its deps live in the pi05 venv; this adapter adds no dependencies of its own.
source ../pi05/.venv/bin/activate
cd ../.. # move to root

# `ckpt_setting` names the result directory under eval_result/<task>/<policy>/<config>/. The
# checkpoint is fixed, so what distinguishes one run from another is the conversion settings --
# override it (`--ckpt_setting dt0.1`) when sweeping velocity_dt.
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting pi05_droid \
    --seed ${seed} \
    --policy_name ${policy_name} \
    "${@:5}"
