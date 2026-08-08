data_dir=${1}
repo_id=${2}
# num_episodes=${3}   # optional: convert only a random subset of this many episodes
# seed=${4:-42}       # optional: seed for the random subset (default 42)

extra_args=""
if [ -n "$num_episodes" ]; then
    extra_args="--num-episodes $num_episodes --seed $seed"
fi

uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir $data_dir --repo_id $repo_id --push-to-hub
