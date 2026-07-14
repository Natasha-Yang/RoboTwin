# RoboTwin runtime environment for the Killarney cluster (Alliance Canada).
# Usage:  source setup_env.sh
#
# Notes specific to Killarney / Alliance:
#   * We use a Miniforge conda env (not the module python) so the pinned
#     torch/sapien/open3d stack installs cleanly from PyPI.
#   * The CVMFS profile exports PYTHONPATH and PIP_CONFIG_FILE (their
#     wheelhouse). Both are unset here so the conda env stays self-contained
#     and pip/python don't pick up cluster site-packages or the dummy opencv.
#   * torch ships its own CUDA runtime, so we do NOT load the cuda module at
#     run time (that is only needed for compiling curobo/pytorch3d).

export ROBOTWIN_CONDA=/project/6101811/natashay/miniforge3
export ROBOTWIN_ENV=RoboTwin

# Keep the cluster's Python/pip config from leaking into the conda env.
unset PYTHONPATH
unset PIP_CONFIG_FILE

# ffmpeg for the pi0.5 / LeRobot video-encoding path. Killarney's module is
# ffmpeg 7.1.1, --enable-shared with libx264/libx265 (matches the Pi05 doc's
# from-source build), so there is no need to build ffmpeg ourselves. Loaded
# before conda activate so the conda env's bin stays at the front of PATH.
# No-op if the module system isn't present (non-Alliance host).
if command -v module >/dev/null 2>&1; then
    module load ffmpeg/7.1.1 2>/dev/null || true
    # PyAV (`av`) in the pi05 uv venv is built from source (uv sync) against
    # this module's shared ffmpeg, so its compiled _core.so dynamically loads
    # libav*.so.61 -- expose the module's lib/ so that resolves (vs the stale
    # system ffmpeg 4.4). NOTE: the conda env's own `av` is a bundled wheel and
    # does not need any of this; the ffmpeg *binary* (already on PATH) is all
    # the LeRobot subprocess encoding needs. This block matters for the venv.
    if [ -n "${EBROOTFFMPEG:-}" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$EBROOTFFMPEG/lib"

        # libav*'s codec deps (libx264/libx265/libSDL2/libvidstab/libmp3lame)
        # live in the gentoo CVMFS usr/lib64. Gentoo-interpreter binaries find
        # them via ld.so.cache, but uv's standalone python 3.11 loader does not
        # -> `import av` dies with "libx264.so.164: cannot open". Expose ONLY
        # those ffmpeg-exclusive add-on codecs via a private symlink dir. We use
        # an ALLOWLIST (never generic libs like libz/libpng/libfreetype, and
        # never glibc/libstdc++ core): LD_LIBRARY_PATH is searched before the
        # system default dirs, so shimming a generic lib would shadow the
        # version the torch/sapien/jax stack expects -- the exact hazard the
        # gpucomp note below warns about. Regenerate the list for a new ffmpeg
        # build with: ldd $EBROOTFFMPEG/lib/lib*.so.* | grep usr/lib64
        _rt_ff_gentoo=/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64
        _rt_ff_allow='libx264|libx265|libvidstab|libSDL2|libmp3lame|libfdk-aac|libopus|libvpx|libass|libaom|libdav1d|libSvtAv1|libsvtav1|libvorbis|libtheora|libwebp|libzimg|libxvidcore|libspeex|libgsm|libtwolame|librav1e|libopenh264'
        if [ -d "$_rt_ff_gentoo" ]; then
            _rt_ffshim="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/robotwin_ffmpeglibs"
            mkdir -p "$_rt_ffshim"
            for _dep in $(ldd "$EBROOTFFMPEG"/lib/lib*.so.* 2>/dev/null \
                          | grep -oE "$_rt_ff_gentoo/[^ ]+\.so[.0-9]*" | sort -u); do
                echo "${_dep##*/}" | grep -qE "^($_rt_ff_allow)[.-]" || continue
                ln -sf "$_dep" "$_rt_ffshim/${_dep##*/}" 2>/dev/null
            done
            export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$_rt_ffshim"
            unset _rt_ffshim _dep
        fi
        unset _rt_ff_gentoo _rt_ff_allow
    fi
fi

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
