#!/bin/bash
# Convert a native MolmoAct2 training checkpoint into the Hugging Face layout
# consumed by policy/MolmoAct/molmoact_model.py.

set -euo pipefail

checkpoint_dir="${1:?usage: $0 CHECKPOINT_DIR OUTPUT_DIR}"
output_dir="${2:?usage: $0 CHECKPOINT_DIR OUTPUT_DIR}"

repo_root="/project/6101811/natashay/RoboTwin"
molmo_root="/project/6101811/natashay/molmoact2"
python="/project/6101811/natashay/miniforge3/envs/MolmoActConvert/bin/python"

export PYTHONPATH="${repo_root}/cluster/converter_stubs"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-molmo"
export HF_HOME="/project/6101811/natashay/.cache/huggingface"
export PYTHONUNBUFFERED=1

mkdir -p "${MPLCONFIGDIR}"
cd "${molmo_root}/experiments"

echo "Converting ${checkpoint_dir}"
echo "Writing ${output_dir}"
"${python}" -m olmo.hf_model.convert_molmoact2_to_hf \
    "${checkpoint_dir}" \
    "${output_dir}" \
    --attn_implementation sdpa \
    --max_shard_size 5GB
