# RoboTwin runtime environment for the Fir cluster (Alliance Canada).
# Usage:  source setup_env.sh
#
# Notes specific to Fir / Alliance:
#   * We use a Miniforge conda env (not the module python) so the pinned
#     torch/sapien/open3d stack installs cleanly from PyPI.
#   * The CVMFS profile exports PYTHONPATH and PIP_CONFIG_FILE (their
#     wheelhouse). Both are unset here so the conda env stays self-contained
#     and pip/python don't pick up cluster site-packages or the dummy opencv.
#   * torch ships its own CUDA runtime, so we do NOT load the cuda module at
#     run time (that is only needed for compiling curobo/pytorch3d).

export ROBOTWIN_CONDA=/project/6028519/natashay/miniforge3
export ROBOTWIN_ENV=RoboTwin

# Keep the cluster's Python/pip config from leaking into the conda env.
unset PYTHONPATH
unset PIP_CONFIG_FILE

source "$ROBOTWIN_CONDA/etc/profile.d/conda.sh"
conda activate "$ROBOTWIN_ENV"

# H100 architecture (compute capability 9.0) for any on-the-fly CUDA JIT.
export TORCH_CUDA_ARCH_LIST="9.0"

# --- SAPIEN / Vulkan rendering on Fir GPU nodes ---
# The NVIDIA Vulkan driver (libGLX_nvidia.so.0, resolved by ldconfig from
# /usr/lib64/nvidia) needs libnvidia-gpucomp.so.<driver>, which on Fir lives
# ONLY in /usr/lib64 -- and we cannot put /usr/lib64 on LD_LIBRARY_PATH because
# its system glibc is incompatible with the CVMFS toolchain (grep/python break
# with "GLIBC_PRIVATE __rtld_libc_freeres"). So expose just that one lib via a
# private symlink dir. No-op on the login node (no graphics driver libs there).
if compgen -G "/usr/lib64/libnvidia-gpucomp.so*" >/dev/null 2>&1; then
    _rt_nvshim="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/robotwin_nvlibs"
    mkdir -p "$_rt_nvshim"
    ln -sf /usr/lib64/libnvidia-gpucomp.so* "$_rt_nvshim"/ 2>/dev/null
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$_rt_nvshim"
    unset _rt_nvshim

    # Point SAPIEN at the node's SYSTEM NVIDIA ICD. Left to itself, SAPIEN uses
    # its bundled ICD (it only recognizes the exact name nvidia_icd.json, but
    # Fir's system ICD is nvidia_icd.x86_64.json) -- and that bundled ICD fails
    # vk::PhysicalDevice::createDevice on this driver. The system ICD + the
    # gpucomp shim above render correctly on the H100.
    if [ -z "${VK_ICD_FILENAMES:-}" ]; then
        for _icd in /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
                    /usr/share/vulkan/icd.d/nvidia_icd.json \
                    /etc/vulkan/icd.d/nvidia_icd.x86_64.json; do
            if [ -f "$_icd" ]; then export VK_ICD_FILENAMES="$_icd"; break; fi
        done
        unset _icd
    fi
fi

echo "RoboTwin env active: $(python --version 2>&1) @ $(which python)"
