#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3:-0}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config

# The env clears each episode's cache as it goes (_base_task.py); this sweeps the
# parent dir afterwards. Read the root from the config rather than assuming ./data,
# so a config with a relocated save_path still gets cleaned.
data_root=$(grep -m1 '^save_path:' "task_config/${task_config}.yml" \
            | sed 's/^save_path:[[:space:]]*//; s/[[:space:]]*$//; s/^["'"'"']//; s/["'"'"']$//')
: "${data_root:=./data}"
rm -rf "${data_root}/${task_name}/${task_config}/.cache"
