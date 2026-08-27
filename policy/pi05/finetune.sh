train_config_name=$1
model_name=$2
gpu_use=$3
shift 3   # any remaining args are forwarded to train.py (e.g. --episodes-per-task=10, --resume)

export CUDA_VISIBLE_DEVICES=$gpu_use
echo $CUDA_VISIBLE_DEVICES

# Fresh runs pass --overwrite so a re-launch wipes the old checkpoint dir. That is
# mutually exclusive with --resume (TrainConfig.__post_init__ raises if both are
# set), so drop it when the caller asked to resume -- otherwise --resume would
# both fail the check and, if it didn't, rmtree the checkpoints we want to load.
start_mode=--overwrite
for arg in "$@"; do
    if [ "$arg" = "--resume" ]; then
        start_mode=""
        break
    fi
done

# uv is not installed on Killarney (see CLAUDE.md §8); the venv it built elsewhere
# is still here, so call its interpreter directly when uv is missing.
if command -v uv >/dev/null 2>&1; then
    runner=(uv run)
else
    runner=(.venv/bin/python)
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 "${runner[@]}" scripts/train.py $train_config_name --exp-name=$model_name $start_mode "$@"
