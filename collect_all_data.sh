#!/bin/bash

set -euo pipefail
shopt -s nullglob

task_config=${1:-}
gpu_id=${2:-}

if [[ -z "$task_config" || -z "$gpu_id" ]]; then
    echo "Usage: bash collect_all_data.sh <task_config> <gpu_id>"
    exit 1
fi

if [[ ! -f "task_config/${task_config}.yml" ]]; then
    echo "Task config not found: task_config/${task_config}.yml"
    echo "Create it with: bash task_config/create_task_config.sh ${task_config}"
    exit 1
fi

task_files=(description/task_instruction/*.json)
if (( ${#task_files[@]} == 0 )); then
    echo "No task descriptions found in description/task_instruction/" >&2
    exit 1
fi

for task_file in "${task_files[@]}"; do
    task_name=$(basename "$task_file" .json)
    echo "Collecting: ${task_name}"
    if bash collect_data.sh "$task_name" "$task_config" "$gpu_id"; then
        :
    else
        status=$?
        echo "Collection failed for ${task_name}; stopping." >&2
        exit "$status"
    fi
done
