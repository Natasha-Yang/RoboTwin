setting=${1}
expert_data_num=${2}

for task_path in ../../data/*/; do
    task_name=$(basename $task_path)
    python scripts/process_data.py $task_name $setting $expert_data_num
done
