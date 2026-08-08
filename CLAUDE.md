# CLAUDE.md

Guide for setting up and running this RoboTwin fork on an HPC cluster (SLURM). It
covers environment setup, **data collection**, **building LeRobot / HF datasets**,
**fine-tuning** (π0.5 and MolmoAct), **inference / evaluation** (including
critic-guided sampling), and **rollout-dataset collection**.

This repo is a fork of RoboTwin 2.0 (`RoboTwin-Platform/RoboTwin`) with additions
for cluster batch jobs (`cluster/`, `submit_all_data.sh`), a contact-wrench
observation (`envs/utils/wrench.py`), a named observation-modality layer
(`envs/utils/obs_modalities.py`), a policy-rollout → HuggingFace dataset pipeline
(`script/collect_dataset.py`), and a QMFM value-critic guided-inference path inside
π0.5's flow sampler. It was developed on the **Fir** cluster (Alliance Canada) and is
currently run on **Rorqual**. The sections below give the concrete cluster setup
**and** what to change to port it elsewhere.

> The upstream project docs live at https://robotwin-platform.github.io/doc/ —
> refer there for task/config semantics. This file is about running it on a cluster.

---

## Agent working notes (Claude — read first)

- **Before a non-trivial edit to the simulation code**, invoke the **`/framework`**
  skill first. It's a fast orientation to how the actual code fits together (task
  envs in `envs/`, the base task, the motion/actor API, actor point system, configs,
  and the collect/eval + policy-adapter contracts) so your edit matches existing
  contracts instead of reinventing them. Load it whenever a request means writing or
  substantially changing: a task env (`envs/*.py`), `envs/_base_task.py`, the
  motion/actor utilities (`envs/utils/`, `envs/robot/`), a policy adapter
  (`policy/*/deploy_policy.py`), a task config, or the collect/eval drivers
  (`script/collect_data.py`, `script/eval_policy.py`). Skip it for pure
  cluster/ops/sync changes — this file already covers those.
- **Log files can be huge — never blindly read one into context.** Before opening
  anything under `logs/` (or any `*.log` / `*.out` / `slurm-*` output), check its
  size first (`ls -lh <file>`, `wc -l <file>`). Only read the whole file if it is
  trivially small or the user explicitly asks for all of it; otherwise `tail` /
  `head` / `grep` the relevant slice. Loading a multi-MB log burns the context
  window for no benefit.

---

## 0. Porting checklist (read first)

When moving to a **new cluster**, the things that are environment-specific and
must be re-checked are:

1. **Paths** — the repo currently hardcodes `/project/6028519/natashay/RoboTwin`
   (repo root) and `/project/6028519/natashay/miniforge3` (conda). Grep and update:
   `grep -rn "6028519/natashay" setup_env.sh cluster/ policy/`.
2. **SLURM account + partition** — `--account=rrg-florian7_gpu` and
   `--gpus-per-node=l40s:1` in `cluster/robotwin_gpu.sh` and `cluster/finetune_pi05.sh`.
3. **GPU arch** — `TORCH_CUDA_ARCH_LIST` in `setup_env.sh` is `8.9` (L40S / Ada /
   sm_89). Set to your GPU's compute capability (A100 = `8.0`, H100 = `9.0`, …).
   This is **not** just a JIT hint: curobo and pytorch3d are compiled ahead of time
   against it, so changing GPU generation means clean-rebuilding both (§1.2) — a
   binary built for `sm_90` will not run on an L40S.
4. **Vulkan / SAPIEN rendering** — the `setup_env.sh` gpucomp-shim + `VK_ICD_FILENAMES`
   block is Fir-specific (see §1.3). On a cluster where SAPIEN renders out of the
   box (e.g. a normal workstation with an NVIDIA driver), it no-ops harmlessly; on
   a different broken-headless cluster you may need a different fix.
5. **Render-capable nodes** — `submit_all_data.sh` pins
   `render_nodes=fc10508,fc10519,fc10604,fc10612` and both job scripts
   `--exclude=fc10512`. These node names are Fir-specific; remove or replace them.
6. **Cluster Python leakage** — the `unset PYTHONPATH / PIP_CONFIG_FILE` in
   `setup_env.sh` guards against Alliance's CVMFS profile (a dummy `opencv` wheel).
   Harmless elsewhere, keep it.

---

## 1. Environment setup

RoboTwin's simulator (SAPIEN 3.0.0b1 / PhysX / Vulkan) plus the pinned
torch/open3d stack install cleanest from PyPI, so we use a **Miniforge conda env**
(Python 3.10) rather than the cluster module Python.

> **Internet vs GPU nodes.** On most clusters login nodes have internet but no GPU,
> and compute nodes have a GPU but no internet. Do **all** `pip`/`git`/`uv`/asset
> downloads on a **login node**, then run GPU work via SLURM. Pre-cache any
> HuggingFace models (`HF_HOME`) on the login node too.

### 1.1 Create the conda env and install RoboTwin

On a **login node**:

```bash
# 1. Miniforge (if not already installed) — put it on /project, home is often tiny
#    https://github.com/conda-forge/miniforge

conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin

# 2. Core deps + patches + curobo + pytorch3d (this is the upstream installer)
bash script/_install.sh
```

`script/_install.sh` does the following (all captured so you know what to re-run
if it half-fails):
- `pip install -r script/requirements.txt` (torch==2.4.1, sapien==3.0.0b1,
  mplib==0.2.1, open3d==0.18.0, curobo deps, …)
- installs `pytorch3d` from source (`--no-build-isolation`)
- patches `sapien/wrapper/urdf_loader.py` (utf-8 open) and `mplib/planner.py`
  (drops a `collide` early-return) via `sed`
- clones + installs **curobo v0.7.8** (`envs/curobo`), `warp-lang==1.12.0`
- pins **`setuptools==69.5.1`** — SAPIEN 3.0.0b1 imports `pkg_resources`, which
  newer setuptools (≥ 70) removed. Do not let this get upgraded.

### 1.2 CUDA-compiled deps (curobo / pytorch3d)

curobo and pytorch3d compile CUDA kernels. On the login node (no GPU) they will
**skip** the CUDA build unless forced. On Fir:

```bash
module load cuda/12.2              # do NOT pipe `module load` — a pipe subshells it
export CUDA_HOME=$EBROOTCUDA
export TORCH_CUDA_ARCH_LIST=8.9    # L40S (Ada / sm_89); change per §0.3
export FORCE_CUDA=1                # force the CUDA build on the GPU-less login node
```

Then run the curobo/pytorch3d `pip install` steps. `torch` ships its own CUDA
runtime, so you do **not** load the cuda module at *run* time — only for compiling.

> **Rorqual only — build curobo with `cuda/12.6`, not `cuda/12.2`.** On Rorqual
> (H100, driver 580.x) curobo kernels compiled with `cuda/12.2` throw
> `CUDA error: an illegal instruction was encountered` (CUDA error 715) at kernel
> launch during `motion_gen.warmup()` — even though the SASS is the correct `sm_90`
> and plain torch CUDA works fine. Rebuilding the curobo CUDA extensions with
> `cuda/12.6` fixes it. So on Rorqual, `module load cuda/12.6` for the curobo build
> above. (This is Rorqual-specific; on Fir `cuda/12.2` was fine. See §7.) To rebuild
> just curobo in place:
>
> ```bash
> cd envs/curobo
> module load cuda/12.6; export CUDA_HOME=$EBROOTCUDA
> export TORCH_CUDA_ARCH_LIST=9.0 FORCE_CUDA=1
> rm -f src/curobo/curobolib/*.so && rm -rf build   # clean, so no stale kernels linger
> pip install -e . --no-build-isolation --no-deps --force-reinstall
> ```

### 1.3 SAPIEN / Vulkan headless rendering

RoboTwin renders every camera frame with SAPIEN over Vulkan, including ray tracing
(`envs/_base_task.py` hardcodes `rt` + OptiX). Getting headless Vulkan working on
Fir's H100 nodes required fixes that now live in `setup_env.sh` (guarded so they
no-op on the login node / non-Fir hosts):

- **gpucomp shim**: the NVIDIA Vulkan ICD needs `libnvidia-gpucomp.so.<driver>`,
  which on Fir only exists in `/usr/lib64` (whose glibc clashes with the CVMFS
  toolchain). `setup_env.sh` symlinks just that one lib into a private dir on
  `LD_LIBRARY_PATH`.
- **system ICD**: SAPIEN's bundled ICD fails `vk::createDevice` on this driver;
  `setup_env.sh` points `VK_ICD_FILENAMES` at the node's system
  `nvidia_icd.x86_64.json`.
- **pin the render GPU**: `envs/_base_task.py` constructs the renderer with
  `sapien.Device("cuda:0")` so it doesn't probe every GPU on a busy shared node.

**Ray-tracing note:** the L40S has RT cores, so `rt` is native there. It also worked
on H100 despite that chip having none — the driver/OptiX runs it. Either way, no
`_base_task.py` edit is needed to fall back to rasterization.

**Verify rendering on a GPU node** before collecting anything:

```bash
sbatch cluster/robotwin_gpu.sh          # no args -> runs the headless render smoke-test
# or directly on an salloc'd GPU node:
python cluster/test_render_headless.py  # runs rt + raster in isolated subprocesses
```

### 1.4 Download assets

RoboTwin needs embodiment/object/background assets (~several GB) from HuggingFace.
On the **login node**:

```bash
bash script/_download_assets.sh   # downloads + unzips assets, then fixes paths
```

### 1.5 Activate for every session / job

```bash
source setup_env.sh
```

This unsets the CVMFS `PYTHONPATH`/`PIP_CONFIG_FILE`, `module load ffmpeg/7.1.1`
(see below), activates the conda env, sets `TORCH_CUDA_ARCH_LIST`, and installs the
Vulkan shim. **Every** SLURM job script sources it (`cluster/robotwin_gpu.sh`,
`cluster/rollout.sh`).

**ffmpeg (do NOT build from source).** The π0.5 doc's §1.1 tells you to compile
ffmpeg 7.1 — but that step only exists as a fallback for when `uv sync` fails
building **PyAV (`av`)** ("if error occured while build av, you should update
ffmpeg"). On Killarney you don't build anything: `module load ffmpeg/7.1.1` gives
ffmpeg 7.1.1 built `--enable-shared` with **libx264/libx265** *and* the dev headers
+ pkg-config `.pc` files. `setup_env.sh` loads it (and puts its pkgconfig dir on
`PKG_CONFIG_PATH`), so `uv sync` in `policy/pi05` compiles `av` cleanly against 7.1.

If you ignore this and try the from-source build, the `nasm/yasm not found or too
old` error is a **misleading PATH problem, not a version one**: nasm 2.15 / yasm 1.3
live in the gentoo CVMFS `usr/bin` and are new enough — `module load StdEnv/2023`
restores them. Just use the module instead.

There is one runtime wrinkle `setup_env.sh` handles for you. The pi05 `av` is built
`--enable-shared`, so at import its `_core.so` dlopens the module's `libav*.so.61`,
which in turn need ffmpeg codec libs (`libx264/libx265/libSDL2/libvidstab/libmp3lame`)
that live in the gentoo `usr/lib64`. Gentoo-interpreter binaries find those via
`ld.so.cache`, but **uv's standalone Python 3.11 loader does not** → `import av` dies
with `libx264.so.164: cannot open`. So `setup_env.sh`:
1. appends `$EBROOTFFMPEG/lib` to `LD_LIBRARY_PATH` (the `libav*.so.61` themselves), and
2. symlinks **only** those ffmpeg-exclusive codec libs (via an allowlist — never
   generic libs like `libz`/`libpng`/`libfreetype`, never glibc core) into a private
   `robotwin_ffmpeglibs` shim on `LD_LIBRARY_PATH`. It's an allowlist because
   `LD_LIBRARY_PATH` is searched before the system dirs, so shimming a *generic* lib
   would shadow the version torch/sapien/jax expect (the same hazard as the gpucomp
   shim below). Regenerate the list for a new ffmpeg build with
   `ldd $EBROOTFFMPEG/lib/lib*.so.* | grep usr/lib64`.

(The main conda env's own `av` is a bundled wheel and needs none of this; the shim is
harmless there since it only supplies codec sonames nothing else uses.)

---

## 2. SLURM job scripts

| Script | Purpose |
|---|---|
| `cluster/robotwin_gpu.sh` | Generic single-GPU (L40S) job. `sbatch cluster/robotwin_gpu.sh <cmd...>`; no args → render smoke-test. This is also how you launch an eval or a rollout collection (§6, §7). |
| `cluster/finetune_pi05.sh` | π0.5 fine-tuning job. Runs in the `policy/pi05` uv venv (JAX), **not** the SAPIEN conda env — no Vulkan needed. |
| `submit_all_data.sh` | Fan out data collection over many SLURM jobs (batches). |
| `cluster/convert_molmoact_checkpoint.sh` | Convert a native MolmoAct checkpoint into the HF layout (§5.2). |
| `cluster/wandb_sync.sh` | Push offline W&B runs from a login node. |

The GPU/collection jobs set `WARP_CACHE_PATH` to node-local `$SLURM_TMPDIR` (a shared
Warp JIT cache across nodes/drivers can produce `CUDA error: an illegal instruction`),
set `PYTHONUNBUFFERED=1` for live logs, and resolve the repo root via
`ROBOTWIN_ROOT` / `SLURM_SUBMIT_DIR` (under `sbatch`, `$0` is a spooled copy, so
`dirname $0` does not work).

**Edit before use on a new cluster:** `--account`, `--gpus-per-node`,
`--exclude`/`--nodelist`, `--time`, `--cpus-per-task`, `--mem`.

---

## 3. Data collection

Collection searches for random seeds that satisfy the task, then replays them to
record expert demonstrations into `data/<task>/<task_config>/` (per-episode HDF5 +
per-episode instruction JSON).

**Task configs** live in `task_config/*.yml` (e.g. `demo_randomized`, `demo_clean`).
They set embodiment, cameras, domain randomization, episode count, etc. Create a
new one from the template:

```bash
bash task_config/create_task_config.sh <my_config>   # copies _config_template.yml
# then edit task_config/<my_config>.yml
```

### 3.1 Single task (interactive / one GPU)

```bash
bash collect_data.sh <task_name> <task_config> <gpu_id>
# e.g.
bash collect_data.sh beat_block_hammer demo_randomized 0
```

Under SLURM:

```bash
sbatch cluster/robotwin_gpu.sh bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 3.2 All tasks, one after another (single job)

```bash
bash collect_all_data.sh <task_config> <gpu_id>
```

Iterates every task in `description/task_instruction/*.json`.

### 3.3 All tasks, fanned out across GPUs (recommended on a cluster)

`submit_all_data.sh` spreads tasks round-robin across N SLURM jobs and, in
`--parallel` mode, packs multiple concurrent collection workers onto each GPU
(collection is CPU-bound: curobo planning + PhysX, ~3 cores/worker; render VRAM is
tiny). Each worker gets its own Warp cache dir and log.

```bash
bash submit_all_data.sh <task_config> [num_batches] [--parallel | --sequential]

# examples
bash submit_all_data.sh demo_randomized                 # pack GPUs (default parallel)
bash submit_all_data.sh demo_randomized 8               # 8 jobs, packed
bash submit_all_data.sh demo_randomized 5 --sequential  # 5 jobs, one task at a time
```

Logs land in `logs/data_collection/`. **Fir-specific:** it pins collection to
`render_nodes=fc10508,fc10519,fc10604,fc10612` (the nodes whose Vulkan works) —
change or drop this on another cluster (see §0.5).

---

## 4. Building a LeRobot dataset (for fine-tuning)

> This is the **training-data** pipeline: collected expert demos → LeRobot dataset.
> For turning *policy rollouts* into a HuggingFace dataset, see §7 — a different
> pipeline with a different driver.

Converts collected demonstrations (`data/<task>/<config>/*.hdf5`) into a LeRobot
dataset that π0.5 / π0 fine-tuning consumes. Run from inside `policy/pi05`:

```bash
cd policy/pi05

# (a) repack collected HDF5 into the intermediate aloha format
#     -> processed_data/<task>-<setting>-<num>/
bash process_data_pi05.sh <task_name> <task_config> <expert_data_num>
#   e.g. bash process_data_pi05.sh beat_block_hammer demo_randomized 50

# (b) convert that into a LeRobot dataset (written to $HF_LEROBOT_HOME/<repo_id>)
bash generate.sh <processed_data_dir> <repo_id>
#   e.g. bash generate.sh processed_data/beat_block_hammer-demo_randomized-50 \
#                          NatashaYang/robotwin_lerobot_dataset
```

`generate.sh` calls `examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py`.
The `<repo_id>` you pick here is what you reference in the training config (§5.1).


---

## 5. Fine-tuning

### 5.1 π0.5 / π0 (openpi, uv-managed venv)

The `policy/pi05` tree is an openpi checkout. It has its **own** environment,
completely separate from the RoboTwin conda env — pinned to **Python 3.11**
(`.python-version`), JAX, and a **CUDA 12.8 / torch 2.7.0** stack (the
`pytorch-cu128` uv index in `pyproject.toml`), whereas the main RoboTwin env is
torch 2.4.1. Managed by [`uv`](https://docs.astral.sh/uv/). On the **login node**:

```bash
cd policy/pi05
# install uv if needed:  curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv python install    # fetch the pinned Python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync              # build policy/pi05/.venv from uv.lock
```

`GIT_LFS_SKIP_SMUDGE=1` skips large LFS blobs during dependency resolution (this is
what the openpi CI does). `uv sync` reads `uv.lock` and resolves the workspace
member `packages/openpi-client` plus the git-pinned `lerobot` rev. Add `--dev` if
you want the lint/test tools. You don't need the `third_party/aloha` /
`third_party/libero` git submodules for RoboTwin training — they're for real-robot
ALOHA and LIBERO. The resulting `.venv` is what `eval.sh` activates and `finetune.sh`
runs through `uv run`.

> **`uv sync` gotchas on a CVMFS/Gentoo cluster (hit on Rorqual).** A few deps have
> no prebuilt wheel for this platform and build from source, and the CVMFS toolchain
> trips them up. If `uv sync` fails, check these:
>
> - **`evdev` (via `lerobot → pynput → evdev`) — needs kernel headers.** `evdev`
>   is sdist-only, so it compiles against `linux/input.h`, which isn't on the default
>   include path here (the failure prints *"The 'linux/input.h' … include files are
>   missing"*). The headers exist in CVMFS — export before syncing (do this **every**
>   sync; there is no evdev wheel to avoid the source build):
>   ```bash
>   export CPATH=/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/include:$CPATH
>   export C_INCLUDE_PATH=$CPATH
>   ```
> - **`av==14.4.0` is a broken release — sdist-only, zero wheels on PyPI.** With no
>   wheel, uv builds PyAV from source against the CVMFS system ffmpeg (4.x), which
>   lacks `ch_layout` → build error *"'AVCodecParameters' has no member named
>   'ch_layout'"*. Fix: pin `av` to a version whose manylinux wheels exist (they
>   bundle their own ffmpeg 7.x, so no system ffmpeg is needed). **14.2.0** is the
>   newest 14.x with wheels and still satisfies lerobot's `av>=14.2.0`. This repo
>   pins it via `[tool.uv].override-dependencies` in `pyproject.toml` **and** the
>   `av` entry in `uv.lock` was edited to 14.2.0 + wheels (see next bullet for why
>   the lock was hand-edited).
> - **IPv6 is broken on Rorqual login nodes → `uv lock` hangs.** `curl -6` can't
>   resolve/route but `curl -4` works; uv (reqwest) stalls on IPv6 while fetching the
>   `download.pytorch.org` index during resolution, so `uv lock` times out with
>   *"operation timed out"* on `download.pytorch.org/whl/cu128/...`. Direct wheel
>   **downloads** still succeed over IPv4, so `uv sync --frozen` (install the lock
>   as-is, no re-resolution) works fine — which is why the `av` bump was applied by
>   hand-editing `uv.lock` rather than running `uv lock`. If you must re-resolve,
>   force IPv4 first.

Training configs are registered in
`policy/pi05/src/openpi/training/config.py`. Registered entries are
`pi05_base_aloha_lora` plus the episode-budget variants
`pi05_base_aloha_lora_clean_50x25` / `_50x10` / `_50x5` (same data, different
`episodes_per_task` and `num_train_steps`). To fine-tune on
your data, set that config's `repo_id` to the LeRobot dataset from §4.1
(e.g. `pi05_base_aloha_lora` already points at
`NatashaYang/robotwin_lerobot_dataset`), and adjust `num_train_steps`, `batch_size`,
LoRA vs full, `weight_loader` base checkpoint, etc.

Then, from `policy/pi05`:

```bash
# (a) compute normalization stats for the config
uv run scripts/compute_norm_stats.py <train_config_name>

# (b) fine-tune  ->  checkpoints/<train_config_name>/<model_name>/<step>/
bash finetune.sh <train_config_name> <model_name> <gpu_id>
#   e.g. bash finetune.sh pi05_base_aloha_lora robotwin_run0 0
```

`finetune.sh` runs `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py
<config> --exp-name=<model_name> --overwrite`. Wrap it in a GPU SLURM job (adapt
`cluster/robotwin_gpu.sh`, or write a training-specific `sbatch` script — training
wants more time/VRAM than collection).

### 5.2 MolmoAct

MolmoAct is fine-tuned in a **separate** repo (`molmoact2`), not here. This repo
only **runs inference** on a converted checkpoint. After training there, convert
the native checkpoint into the HuggingFace layout that
`policy/MolmoAct/molmoact_model.py` loads:

```bash
bash cluster/convert_molmoact_checkpoint.sh <checkpoint_dir> <output_dir>
```

This uses a dedicated `MolmoActConvert` conda env and
`olmo.hf_model.convert_molmoact2_to_hf`. Push the `<output_dir>` to the Hub (or
point `ckpt_path` at it locally) and pre-cache it in `HF_HOME` for offline compute
nodes. The eval config `policy/MolmoAct/deploy_policy.yml` currently loads
`ckpt_path: allenai/MolmoAct2-BimanualYAM`; point it at your converted `<output_dir>`
(or its Hub repo) to evaluate a fine-tune.

> Note the hardcoded `repo_root`/`molmo_root`/`python` paths in
> `cluster/convert_molmoact_checkpoint.sh` — update them on a new cluster.

---

## 6. Inference / evaluation

Evaluation is driven by `script/eval_policy.py`, configured by each policy's
`policy/<name>/deploy_policy.yml`, and launched by that policy's `eval.sh`. The eval
loop does an expert-feasibility check per seed, then runs the policy and optionally
logs video (`eval_video_log`). Results go to
`eval_result/<task>/<policy>/<config>/<ckpt>/<timestamp>/` (`_result.txt`,
`_episode_results.csv`, videos, and `debug_vis/` when the task config sets `debug`).

Alongside those, each run snapshots `deploy_policy.yml` and the `critic_config_path`
file it includes into the result dir (`script/eval_policy.py::snapshot_config`). The
copies keep their comments but carry the values **actually used** — `eval.sh`'s
positional args are written in, so `task_name`, `seed`, `guidance_scale` etc. read as
resolved rather than as the `null`/default in the source yml.

> To collect a *dataset* of policy rollouts (rather than just score them), use the
> separate `script/collect_dataset.py` pipeline in §7.

### 6.1 π0.5

Under SLURM (the eval needs the sim, so it goes through the rendering-capable GPU job):

```bash
sbatch cluster/robotwin_gpu.sh \
    bash -c 'cd policy/pi05 && bash eval.sh <task_name> <task_config> \
             <train_config_name> <model_name> <seed> 0'
```

Or directly on an `salloc`'d GPU node:

```bash
cd policy/pi05
bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> \
             [guidance_scale] [guidance_ramp_updates] [train_online]
# baseline (no critic guidance)
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora_clean_50x25 run0 0 0
# critic-guided (see §6.2 — needs QMFM + a critic checkpoint staged first)
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora_clean_50x25 run0 0 0 0.3 256
```

- `eval.sh` activates `policy/pi05/.venv` and calls `script/eval_policy.py` with
  `policy/pi05/deploy_policy.yml`.
- `deploy_policy.py` → `pi_model.py::PI0` loads the trained policy from
  `policy/pi05/checkpoints/<train_config_name>/<model_name>/<checkpoint_id>` and runs
  inference. `checkpoint_id` (default 15000) and `pi0_step` (how many of the 50 chunk
  steps are executed per call, default 10) come from `deploy_policy.yml`.
- Camera → model mapping (in `deploy_policy.py::encode_obs` / `pi_model.py`):
  `head_camera → cam_high`, `left_camera → cam_left_wrist`, `right_camera → cam_right_wrist`.

Setting `debug: true` in the **task config** turns on extra per-episode diagnostics
(`script/eval_policy.py::visualize_debug_obs`), all written into `debug_vis/episode<N>/`
under that same result dir:

| Output | File (under `debug_vis/episode<N>/`) | Notes |
|---|---|---|
| TCP wrench histograms | `wrench_hist_episode<N>.png` | one histogram per component (Fx/Fy/Fz/Tx/Ty/Tz), left and right arm overlaid |
| Rollout + wrench GIF | `wrench_episode<N>.gif` | head camera on the left with the **world** axes drawn as labelled x/y/z arrows, projected into the camera and anchored at each arm's TCP (the axes the components are resolved in, at the point they act); the wrench traces with a step cursor on the right |
| Raw series | `wrench_episode<N>.npz` | `step`, `left`, `right` — `(num_samples, 6)` each |
| Depth / segmentation tiles, point clouds | `obs_step<S>.png` / `pcd_step<S>.ply` | only for the modalities the task config's `data_type` enables |

The wrench is computed by `envs/utils/wrench.py` (shared with the rollout-dataset collector, §7a).
It is the **net contact wrench on the end-effector links** (wrist link + gripper
fingers + any `fix_gripper_name` links), summed from `scene.get_contacts()` impulses divided
by the sim timestep, with torque taken about the TCP origin. Both vectors are resolved in the
**world** frame (N and N·m) — only the moment arm is TCP-relative, so a trace stays comparable
across steps as the gripper rotates. It is contact-only: an arm moving through free space reads exactly
zero — this is not a joint-torque estimate. One sample is taken per policy call (so
`pi0_step` sim frames apart), paired with that call's head-camera frame.

Note `analysis/plot_wrench_hist.py` plots the same histograms straight off a collected
rollout dataset, for when you want the distribution over a whole run rather than per episode.

### 6.2 Critic gradient guidance (`guidance_scale` is the on/off switch)

> **Status on Rorqual: both dependencies are staged.** `multisensory_steering` is checked
> out at `/lustre09/project/6028519/natashay/multisensory-steering`, and the **QMFM repo**
> it imports `ReplayBuffer` from (by explicit path, `$QMFM_ROOT/utils/datasets.py`) at
> `/home/natashay/links/projects/def-florian7/natashay/QMFM`. `eval.sh` exports that as the
> `QMFM_ROOT` default — override the env var to point elsewhere. It also forces
> `WANDB_MODE=offline`: compute nodes have no internet, and the guided path opens a W&B run
> per eval, so an online `wandb.init()` times out (90 s) and can take the job down. Sync the
> offline runs from a login node afterwards (`cluster/wandb_sync.sh`). To turn guidance on,
> stage a critic checkpoint and set `guidance_scale` (or pass it as `eval.sh`'s 7th arg).
>
> Note `critic_config_path` is read **unconditionally** by `parse_args_and_config`, even
> at `guidance_scale: 0.0` — so if that path doesn't exist, *baseline* eval crashes too.
> It points at the `multisensory-steering` checkout above; repoint it when porting.

There is **one** eval script and **one** config. `guidance_scale` in `deploy_policy.yml`
(overridable as `eval.sh`'s 7th positional arg) decides whether a critic steers the frozen
pi0.5 flow sampler in `Pi0.sample_actions`:

| `guidance_scale` | Behavior |
|---|---|
| `0.0` (default on this branch) | plain pi0.5 baseline — no critic is built, no replay collection, no TD updates, no W&B |
| nonzero | an ensemble QMFM `Value` critic steers the sampler; by default it is also trained **online** by TD during the rollouts, and guidance ramps `0 → guidance_scale` over `guidance_ramp_updates` TD updates (`0` jumps to target after the first update) |


How much of the critic online TD is then allowed to move is decided by two keys in the
**critic config** (`critic_config_path`, i.e. `cfgs/qmfm.yaml` — not `deploy_policy.yml`, though
that file and the CLI can override them like any other critic key). Both only mean anything once
guidance is on:

| Key | Default | Behavior |
|---|---|---|
| `train_online: true` | ✔ | as above — transitions are stashed into the replay buffer after every control step, TD updates run, `save_critic` writes the result (see the checkpointing note below) |
| `train_online: false` | | the critic is **frozen** at `critic_ckpt`: it still steers the sampler, but nothing is stashed, no TD update runs, and the ramp is skipped (guidance sits at `guidance_scale` from the first chunk, since there are no updates to count). W&B still opens and logs the eval metrics |
| `freeze_encoder: true` | | the half-way point: TD still runs, but only on the value head — the observation encoder (`MultiModalEncoder`, or the legacy single SigLIP CNN) keeps the checkpoint's weights |

Freezing is for warm starts. `train_online: false` evaluates an offline-trained critic as-is,
with no eval-time drift in its values; `freeze_encoder` keeps the representation a large offline
dataset paid for — the part a few thousand online transitions are least able to improve — while
letting TD refit the MLP on top of it.

`train_online: false` requires a `critic_ckpt`: a frozen critic never leaves its initialization,
so with none it would guide on the gradients of a random network, and `pi_model.py` raises at
startup rather than running it. The replay buffer is allocated lazily on the first stash, so a
frozen run never pays its memory either, and `save_critic` is skipped — the critic is
byte-identical to the checkpoint the snapshotted config names. `eval.sh`'s 9th positional arg
overrides `train_online` for a one-off run.

Under `save_critic`, that critic is written to `online_value_critic.pkl` in the run's result dir
**after every episode**, not once at the end — a 100-episode guided eval is many hours, and
losing all of its TD training to a SLURM time limit is the expensive failure. There is one file
per run, overwritten each time, so nothing accumulates; to resume, point the next run's
`critic_ckpt` at it (the checkpoint carries `num_updates`, so the guidance ramp picks up where
it left off — `PI0.scheduled_guidance_scale` still counts only *this* run's updates, §6.2).
`script/eval_policy.py::save_online_critic` pickles into a sibling `.tmp` and `os.replace`s it
into position, so a kill mid-write leaves the previous complete checkpoint rather than a
truncated one — writing in place at this rate would otherwise make the interrupt this is meant
to survive destroy the checkpoint too. There is **no** separate end-of-run save: a run that
finishes got its last write from its last episode, so `main` only prints where the file is.

`freeze_encoder` is implemented in `qmfm.py::freeze_encoder_tx` as an `optax.multi_transform`
that zeroes the updates of everything under the params tree's `encoder` key, rather than as a
`stop_gradient` inside the module: `critic_def` stays one pure function, so the guidance path —
which differentiates Q with respect to the *action*, a branch that never enters the encoder — is
unaffected either way, and the zeroed gradients are dead code in the jitted update. Only the
online params are pinned; the target network keeps its Polyak update, which against a constant
encoder just converges it to the same frozen weights. Unlike `encoder_modalities` it is **not**
an architecture key — it changes no shapes, so the config's value wins over a checkpoint's and
either setting loads either checkpoint.

The ramp counts TD updates **performed in this run**. A critic warm-started from `critic_ckpt`
restores the checkpoint's lifetime `num_updates` (an offline-trained one is in the hundreds), so
using that counter directly would read as "ramp already finished" and apply full guidance from
the first chunk; `PI0.scheduled_guidance_scale` subtracts the value at load time instead. W&B's
`critic/update` still reports the lifetime count.

`guidance_scale` defaults to whatever `deploy_policy.yml` says; the positional arg only
overrides it (pass `0` to force the baseline). The critic's own hyperparameters are **not** in
`deploy_policy.yml` — it carries `critic_config_path`, and `parse_args_and_config` merges that
file in underneath, so precedence is **CLI > deploy_policy.yml > critic_config_path**. The
critic implementation and its config both come from the `multisensory_steering` package
(editable install at `/lustre09/project/6028519/natashay/multisensory-steering`, config at
`cfgs/qmfm.yaml`), which imports QMFM's `ReplayBuffer` from `$QMFM_ROOT` — see the status note
at the top of this section for where both are staged. Only the guided path logs to W&B, collects replay transitions, and honors
`save_critic` / `critic_ckpt` / the TD hyperparameters; `script/eval_policy.py` keys all of it
off whether the policy object exposes an `online_critic`, and the collection/updates
additionally off `model.train_critic_online` (`_trains_online_critic`), which — unlike the
critic object — is known before the first observation.

**The critic must be trained in pi0.5's model space.** `Pi0.sample_actions` scores the chunk
it is sampling, *before* the output transform runs: **normalized** state `(14,)` and a
**normalized** action chunk `(50, 14)` → flat `700`. The `14` is `critic_action_dim` — the
model pads *both* state and actions to `action_dim=32`, but the trailing dims are constant
zero for aloha, so the padding is stripped back off and only the embodiment's own dims reach
the critic (`state_dim=14`, not 32).

`critic_action_dim` is the width of `observation["joint_action"]["vector"]` (both arms plus
grippers), so it follows the embodiment automatically. Nothing knows that width until the sim
produces its first observation, so `PI0` builds the critic lazily in `_init_critic`, called
from the first `update_observation_window` — `model.online_critic` is `None` until then.
Anything needing to know *before* a rollout starts (e.g. whether to open a W&B run) must read
`model.uses_online_critic` instead; `script/eval_policy.py` re-reads `model.online_critic`
fresh at each use site for exactly this reason.

Since architecture keys are taken *from* the checkpoint, a critic with the wrong shapes loads
"successfully" and then fails with an opaque shape error mid-sampler, so `pi_model.py` checks
`state_dim` / `action_dim_flat` up front and raises. That check catches a wrong embodiment or
action horizon, but **not** a critic trained on the *raw* columns: those have the same widths
as their `.model` counterparts and differ only in normalization, so nothing downstream can tell
them apart. See §7 for collecting the right columns.

#### What the critic observes

The critic is not restricted to the SigLIP map and the state. Everything the sim produces this
run is offered to it, and **which modalities it uses is decided in the critic's own config**
(`multisensory_steering`'s `cfgs/qmfm.yaml`, key `encoder_modalities`) — not here:

| Modality | Source | Shape |
|---|---|---|
| `siglip.{head,left_wrist,right_wrist}` | pi0.5's own image tower, inside `sample_actions` | `(16, 16, 1152)` each |
| `state` | normalized model state, embodiment dims | `(14,)` |
| `images.{head,left_wrist,right_wrist}` | task config `data_type.rgb` | `(240, 320, 3)` uint8 |
| `images.third_view` | task config `data_type.third_view` | `(H, W, 3)` uint8 |
| `depth.{head,left_wrist,right_wrist}` | task config `data_type.depth` | `(240, 320)`, mm |
| `pointcloud` | task config `data_type.pointcloud` | `(pcd_down_sample_num, 6)` |
| `wrench.{left,right}` | per-step contact wrench, logged by the env | `(pi0_step, 6)` |

The names are the §7a dataset columns minus their `observation.` prefix, so a critic trained
offline on those columns lines up with what it sees online. `envs/utils/obs_modalities.py`
flattens an observation into them (shared by both paths); `deploy_policy.py::critic_obs_modalities`
calls it once per control step and only when a critic exists, so the baseline and collection
runs copy nothing. `pi_model.py::_init_critic` declares the whole set as `obs_shapes`, the
critic picks its subset (`OnlineValueCritic.obs_keys`), and only that subset is shipped into
`Pi0.sample_actions(critic_obs_extra=...)` and into the replay buffer. Naming a modality the
task config does not enable raises at startup, listing what *is* available.

Three things to keep in mind:

- **All three SigLIP views come free, but only in compute.** The image tower already runs once
  per camera to build the prefix, so `Pi0.embed_images` keeps its raw `aux["encoded"]` for every
  view and `embed_prefix` reuses the tokens — the wrist maps cost no extra tower pass, and the
  head map no longer costs the second one it used to. What they do cost is the replay buffer:
  each configured view is another ~1.15 MB/transition (see the last bullet). Unconfigured views
  are dropped by `OnlineValueCritic.stash`, so they only ride back from the device in `aux`.
- **The wrench a control step sees is the previous chunk's trace** — the steps between the last
  observation and this one, which is the only wrench that exists before the current chunk has
  been executed. Rollout collection drains the log at the same point in its loop, so a dataset's
  `observation.wrench.*` holds exactly this against exactly this row's state and action, and a
  critic trained offline on it needs no realignment. (Datasets collected before 2026-07-30 store
  the *following* chunk's trace instead — see §7a.) The env's per-step logging is switched by the
  task config's `data_type.wrench` (§7a); a critic configured for `wrench.*` against a config
  that has it off fails at startup.
- **Modalities are an architecture key.** They go into the checkpoint, and a warm start rebuilds
  the same encoder stack; changing the list makes an existing critic checkpoint refuse to load
  (loudly, leaf by leaf). Offline pretraining takes the same names — `multisensory_steering`'s
  `cfgs/train_offline.yaml` maps them to dataset columns under `dataset.modalities` — and a
  checkpoint only warm-starts a run whose list matches. Each modality is also stored twice per
  transition (obs and next_obs) — each `siglip.*` view is ~1.15 MB/transition (so all three come
  to ~3.5 MB), a depth camera ~0.6 MB, so `buffer_size` needs revisiting when adding one.

### 6.3 MolmoAct

```bash
cd policy/MolmoAct
bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
```

`ckpt_setting` is used as the MolmoAct `norm_tag`. The other knobs — `ckpt_path`
(currently `allenai/MolmoAct2-BimanualYAM`), `molmoact_step`, `instruction_type` — are
in `policy/MolmoAct/deploy_policy.yml`.

### 6.4 Other baselines

`policy/` also ships DP, ACT, DP3, RDT, pi0, openvla-oft, TinyVLA, DexVLA,
LLaVA-VLA, GO1. Each follows the same `deploy_policy.yml` + `eval.sh` convention;
see the upstream docs for their specific fine-tuning setup.

For remote / server-based π0.5 inference see `policy/pi05/docs/remote_inference.md`
(`scripts/serve_policy.py`).

---

## 7. Collecting a rollout dataset (policy rollouts → HF Hub)

This is the pipeline added in commit `1aa5b89`. It rolls out a trained policy in the
sim and builds a HuggingFace dataset of the resulting trajectories.

Under SLURM:

```bash
sbatch cluster/robotwin_gpu.sh \
    bash -c 'cd policy/pi05 && bash collect_dataset.sh <task_name> <task_config> \
             <train_config_name> <model_name> <seed> 0'
```

Or directly on a GPU node:

```bash
cd policy/pi05
bash collect_dataset.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id>
```

> The compute nodes have **no internet**. Either set `push_to_hub: false` and push from
> a login node afterwards, or make sure your HF token is cached and accept that the push
> step at the end of the run will fail (the dataset is still written to `output_dir`).

- Driver: `script/collect_dataset.py` with `policy/pi05/collect_dataset.yml`.
- Behavior is controlled by `collect_dataset.yml`:
  - `num_episodes: 100` — successful, expert-checked rollouts to collect.
  - `expert_check: true` — only roll out on seeds the expert can solve.
  - `output_dir: ./rollout_datasets` — local `save_to_disk` location.
  - `push_to_hub: true`, `hub_repo_id: NatashaYang/robotwin_clean_ep25_lift_pot_rollouts_dataset`.
  - `hub_private: true` — create the hub repo private. This is the **default when the key is
    absent**, so a run cannot publish a dataset by omission; set it `false` to deliberately
    create a public repo. It is honored only when the push *creates* the repo — an existing
    repo keeps whatever visibility it already has, so this cannot retroactively hide (or
    expose) a dataset that has been pushed before. To change an existing repo, use
    `HfApi().update_repo_settings(repo_id, repo_type="dataset", private=...)`.
  - `checkpoint_id: 15000`, `pi0_step: 10`, `instruction_type: unseen`.
  - `collect_critic_obs: true` — also record the policy's **model-space** view of each step.
  - `collect_siglip: true` — within that, also record the SigLIP patch features of **every**
    camera the policy sees (`siglip.head`, `siglip.left_wrist`, `siglip.right_wrist`). Set
    `false` to keep the dataset small, or list the views you want (`[head, left_wrist]`).
  - `resume: true` — see below.

Everything **beyond** rgb + qpos is decided by the **task config**, not by `collect_dataset.yml`:
whatever its `data_type` block turns on reaches `envs/_base_task.py::get_obs`, and
`extra_obs_columns` records all of it (§7b). The configs that ship here (`demo_clean`,
`demo_clean_25`, `demo_randomized`) enable only rgb + endpose + qpos + wrench, so they yield
just the columns in §7a's table; a privileged config that also turns on depth / pointcloud /
segmentation roughly doubles the bytes per row.

Each episode is flushed to its own shard in `<output_dir>/<task>/<config>/<ckpt>_shards` the
moment it finishes, and the shards are memory-mapped and concatenated into the final dataset at
the end. A row costs ~1 MB resident (three uncompressed camera frames) plus ~576 KB per SigLIP
view `collect_siglip` records and ~1 MB more under a privileged task config, and a long task
(`put_bottles_dustbin`, `step_lim` 1700, at `pi0_step`
10) can reach five figures of rows, so accumulating a whole run in memory would run to tens of
GB. Sharding also means a crashed run keeps its episodes: `progress.json` in the shard dir
records the seed to resume from — which cannot be recomputed, since the seed sequence depends
on which seeds the expert check rejected. Rerunning the same command continues from there
(`resume: false` discards the shards and starts over). The shard dir is removed only after the
dataset is saved and pushed.

### 7a. Columns, and which ones a steering critic needs

Each row is one policy call (one action chunk), in two different spaces:

| Column | Space | Shape |
|---|---|---|
| `frame_index` | primitive sim step (`take_action_cnt`), not a row counter | scalar |
| `observation.state` | raw env qpos | `(14,)` |
| `action` | raw robot, unnormalized by the output transform | `(50, 14)` |
| `observation.state.model` | **normalized** model state, embodiment dims | `(14,)` |
| `action.model` | **normalized**, embodiment dims | `(50, 14)` |
| `siglip.{head,left_wrist,right_wrist}` | per-camera SigLIP patch features, fp16 | `(256, 1152)` each |
| `observation.wrench.{left,right}` | world-frame TCP contact wrench, one row per executed step | `(pi0_step, 6)` |

Each raw column and its `.model` counterpart have the same width and hold the same quantity in
different spaces: the raw ones have been unnormalized by the output transform, the `.model` ones
have not. Neither carries the model's internal zero padding to `action_dim=32` — it is stripped
before the tensors leave the sampler, so `state_dim` is `14`, not `32`. Only `pi0_step` of the
50 chunk steps are actually executed before the next inference call; the whole chunk is recorded
because that is what the sampler scores.

`frame_index` is the sim's own `take_action_cnt` at the moment the row's observation was taken,
so rows are `pi0_step` frames apart (0, 10, 20, … at `pi0_step: 10`) rather than 1. Elapsed time
is what consumers key off: `multisensory_steering`'s offline trainer discounts by the
`frame_index` gap between consecutive rows and subsamples on `frame_index % horizon == 0`, so a
row counter would under-discount by exactly `pi0_step`. Datasets collected this way need no
`create_dataset.py rescale-frames` pass — and that pass will now refuse to run on them, since it
checks that the gap is 1 first.

The `.model` columns come from `Pi0.sample_actions(..., return_critic_obs=True)` — the same
tensors the guided sampler scores — and are only present when `collect_critic_obs` is set. A
critic intended to steer inside the sampler (§6.2) **must** be trained on these; point
`multisensory_steering`'s `cfgs/train_offline.yaml` at
`state_col: observation.state.model` / `action_col: action.model`. The raw columns remain for
behavior cloning and for critics that score executed robot actions.

The `siglip.*` columns are the visual thing the critic conditions on, and are written **during
collection** (`collect_siglip`) from `critic_obs_siglip` — the exact patch maps
`Pi0.sample_actions` feeds the critic, computed by the same `PaliGemma.img` tower on the same
resized frames. There is one per camera the policy sees, under the same names the critic uses
online (`Pi0.SIGLIP_MODALITIES`), each stored as the flat `(256, 1152)` patch sequence (the CNN
encoder reshapes to the 16×16 grid itself) in fp16, matching the online replay buffer. They are
drop-in replacements for the columns `python -m multisensory_steering.create_dataset siglip`
used to add in a second pass — point `siglip_cols: [siglip.head, siglip.left_wrist,
siglip.right_wrist]` at them and skip that pass entirely. At ~576 KB/row **each**, these columns
dominate dataset size: `collect_siglip` takes a list of views (`[head]`) as well as
`true`/`false`, and dropping the wrists is the cheapest way to shrink a run.

`observation.wrench.left` / `.right` are the **same quantity** §6.1's debug GIF plots — net contact
wrench on that arm's end-effector links, `[Fx, Fy, Fz, Tx, Ty, Tz]` in the world frame, torque
about the TCP — computed by the shared `envs/utils/wrench.py::tcp_wrench_vector` so the eval and
collection paths cannot drift apart. The rate differs: `eval_policy.py` samples once per policy
call, while collection samples after **every** primitive step, so a row carries a whole
`(pi0_step, 6)` trace rather than a single vector.

Which trace matters: it is the one the **previous** chunk produced — the steps between the
previous row's observation and this one. The env logs a sample after each `take_action`
(`_base_task.py::_log_step_wrench`, so the reading belongs to the action that just executed),
and the rollout loop drains the log with `pop_step_wrench()` **before** running the chunk, at
the same point in the loop the guided eval path drains it (§6.2). The column is therefore an
observation — something the policy could have conditioned on — not the outcome of the row's own
action, and offline training pairs it with the row's state/action as-is.

> Datasets collected before 2026-07-30 drained the log *after* the chunk, so their
> `observation.wrench.*` is the trace of the row's **own** chunk. To train on one, shift the
> column a row later within each episode — `multisensory_steering`'s `dataset.modalities`
> takes `{column: ..., shift: 1}` for exactly this.

`envs/utils/wrench.py::stack_step_wrench` does the stacking, shared with the critic's online
view of the same modality. A trace can be short — an episode's first row has no previous chunk
and carries a single sample of the contact state at that instant, and a chunk cut off by success
or `step_lim` contributes only the steps it ran — so the tail is padded with **NaN**, not zeros,
since zero is a meaningful reading (the arm touching nothing). At 480 B/row they are the
cheapest column here.

`wrench` is a `data_type` like the others, but it is the one the env cannot pick up from the
flag itself: contacts are a scene query, not part of `get_obs`. So each driver reads
`data_type.wrench` and passes `record_step_wrench` into the env — `collect_data.py`,
`collect_dataset.py` and `eval_policy.py` all do, and nothing logs a wrench with the flag off.
Turning it off drops these columns from the dataset entirely (and makes a `wrench.*` critic
modality unavailable, §6.2).


### 7b. Extra data types (privileged task configs)

`script/collect_dataset.py::extra_obs_columns` records everything else the observation carries,
so the dataset follows the task config's `data_type` block automatically. **No privileged config
ships in this fork** — make one with `bash task_config/create_task_config.sh <name>` and turn on
`depth` / `pointcloud` / `third_view` / `mesh_segmentation` / `actor_segmentation`. With all of
them on, a row gains, per camera `<cam>` ∈ `head` / `left_wrist` / `right_wrist`:

| Column | Type | Shape |
|---|---|---|
| `observation.depth.<cam>` | float32, **millimetres** | `(240, 320)` |
| `observation.mesh_segmentation.<cam>` | PNG image, palette-colored labels | `(240, 320, 3)` |
| `observation.actor_segmentation.<cam>` | PNG image, palette-colored labels | `(240, 320, 3)` |
| `observation.camera.<cam>.{intrinsic_cv,extrinsic_cv,cam2world_gl}` | float32 | `(3,3)` / `(4,4)` |
| `observation.images.third_view` | PNG image, observer camera | `(H, W, 3)` |
| `observation.pointcloud` | float32, world-frame xyz + rgb | `(pcd_down_sample_num, 6)` |
| `observation.endpose.{left,right}_endpose` | float32, xyz + quat | `(7,)` |
| `observation.endpose.{left,right}_gripper` | float32, normalized width | scalar |

Notes:
- Sim camera names (`head_camera` / `left_camera` / `right_camera`) are shortened to the same
  suffixes the rgb columns use (`head` / `left_wrist` / `right_wrist`).
- The camera matrices ride along whenever depth or a point cloud does — depth is not
  unprojectable without them, and the wrist extrinsics change every step. They are not a
  `data_type` of their own; `get_obs` always returns them.
- The schema is inferred from the first collected row (`build_features` / `infer_feature`), so
  nothing has to be enumerated per data type. ndarrays become fixed-shape `ArrayND` columns; a
  point cloud with `pcd_down_sample_num: 0` is ragged and falls back to a nested `Sequence`.
- Measured on-disk cost: **~1.1 MB/row** for `demo_clean` (with `siglip.head` only; each further
  SigLIP view adds ~0.56 MB, so all three make it ~2.2 MB) vs **~2.2 MB/row** with the
  privileged data types on. Depth is the bulk of the difference — the segmentation and
  third-view columns are PNG-compressed. There is no per-column switch here: to collect less,
  use a task config with fewer `data_type` flags.

---

## 8. Cluster gotchas (worth re-checking anywhere)

- **Dummy opencv**: Alliance's CVMFS profile exports `PIP_CONFIG_FILE` pointing at a
  wheelhouse with a stub `opencv` that breaks `pip install`. `setup_env.sh` unsets
  it (and `PYTHONPATH`). Keep this if your cluster injects a Python env via a module
  system.
- **No `sudo`**: Vulkan/ffmpeg come from the module system / CVMFS, not `apt`. Use
  `module load ffmpeg/7.1.1` — don't build ffmpeg from source (the "nasm/yasm not
  found or too old" configure error is a stripped-PATH red herring, not a real
  version problem). See §1.5.
- **`setuptools==69.5.1`**: pinned for SAPIEN's `pkg_resources`. Don't upgrade.
- **Warp cache**: always node-local (`$SLURM_TMPDIR`); a shared cache across
  drivers causes illegal-instruction CUDA crashes.
- **Rorqual: build curobo with `cuda/12.6`** (H100, driver 580.x). curobo kernels
  built with `cuda/12.2` throw `CUDA error: an illegal instruction was encountered`
  (error 715) at launch inside `motion_gen.warmup()` (e.g. `lbfgs_step_cu.forward`),
  while plain torch CUDA is fine — the `12.2` ptxas emits `sm_90` SASS this driver
  rejects. Fix = clean-rebuild curobo's CUDA extensions with `cuda/12.6` (§1.2).
  Note this is a *different* illegal-instruction cause than the Warp cache one above.
  **Rorqual-specific**; Fir built fine with `cuda/12.2`.
- **`module load` in a pipe**: never `module load ... | ...` — the pipe subshells it
  and the env is lost.
- **`$0` under sbatch** is a spooled copy — resolve the repo root via
  `SLURM_SUBMIT_DIR` or `ROBOTWIN_ROOT`, not `dirname $0`.
- **Offline compute nodes**: pre-download HF checkpoints/assets and set `HF_HOME` on
  the login node; only *record* rollouts on the compute node, build/push datasets on
  the login node.
- **pytorch3d is needed only for point clouds — and on Killarney it is installed in
  *neither* env.** `fps` in `envs/camera/camera.py` (point-cloud downsampling) imports a
  compiled `_C.so` pinned to a torch ABI, and the two envs run different torch (conda
  RoboTwin = 2.4.1, `policy/pi05/.venv` = 2.7.0), so if you do need it, **both** need
  their own build — collection runs in the first, eval/rollout-collection in the second.
  But `fps` is only reached from `camera.py`'s point-cloud path behind
  `pcd_down_sample_num > 0`, and every task config that ships here sets
  `data_type.pointcloud: false`, so nothing calls it and its absence is harmless. Two
  traps if you *do* enable `pointcloud`:
  - the fallback does **not** degrade gracefully — it prints `fps error: missing
    pytorch3d` and calls `exit()`, killing the run;
  - `camera.py` swallows a torch-ABI mismatch in a bare `except:` and prints the same
    `missing pytorch3d` message, so don't read that as "not installed".
  (The `fps` call higher up in `get_pcd` is dead code — an unconditional `return`
  precedes it.)
- **Switching GPU generation means rebuilding curobo.** Its CUDA extensions are compiled
  ahead of time for whatever `TORCH_CUDA_ARCH_LIST` said at build time; an `sm_90` build
  fails on an L40S at kernel launch ("no kernel image is available"). Check what you have
  with `cuobjdump --list-elf envs/curobo/src/curobo/curobolib/*.so | grep -o 'sm_[0-9]*'`,
  and clean-rebuild per §1.2. **Build it on a compute node, not the login node** — the
  login node's per-user memory cap kills `nvcc` mid-file, and the failure prints a bare
  `FAILED:` line with *no* diagnostic, which reads like a source error but isn't. The
  build needs no internet (`--no-build-isolation --no-deps`), so a compute node is fine.
- **The critic path needs two out-of-repo checkouts.** `guidance_scale != 0` needs the
  `multisensory_steering` package (`/lustre09/project/6028519/natashay/multisensory-steering`)
  **and** the QMFM repo it imports `ReplayBuffer` from via `$QMFM_ROOT`
  (`/home/natashay/links/projects/def-florian7/natashay/QMFM`). Both are staged on Rorqual and
  `policy/pi05/eval.sh` exports the `QMFM_ROOT` default; on a new cluster re-point that export
  and `critic_config_path`. See §6.2 before turning guidance on.
- **W&B must be offline on compute nodes.** No internet there, so `wandb.init()` blocks for 90 s
  and can kill the job. `cluster/finetune_pi05.sh` and `policy/pi05/eval.sh` both export
  `WANDB_MODE=offline`; push the runs later with `cluster/wandb_sync.sh` from a login node.

---

## 9. Claude Code setup & state sync (this fork)

This fork carries its own **project-level Claude Code config** in `.claude/`
(checked into git, so every clone on every machine behaves the same). See
`.claude/README.md` for the full write-up. Summary:

- **`.claude/settings.json`** — default model `opus`, default mode `acceptEdits`
  ("auto", edits apply without a prompt; `Shift+Tab` cycles modes), a custom status
  line, and `SessionStart`/`SessionEnd` hooks that import/export both sync stores.
- **Status line** (`.claude/statusline.sh` → `.claude/statusline.py`) renders
  `mode · model · ctx <used>/200k (pct) · /sync-claude`. Context usage is read from
  the transcript's latest token counts; the trailing `/sync-claude` reminds you of
  the everyday sync skill.
- **Skills** live in `.claude/skills/` (auto-discovered by Claude Code — a root
  `skills/` would *not* be picked up): `framework` (fast orientation to the
  simulation codebase; auto-loaded before non-trivial code edits — see "Agent
  working notes" at the top), `sync-claude` (both stores), `sync-conversations`,
  `sync-memory`.

### State sync across machines (kept out of the public fork)

`origin` is a **public** GitHub fork, and both conversation transcripts and memory
leak paths/output/secrets, so they must never enter its history. Each portable store
is its **own separate PRIVATE git repo**, nested here and **gitignored** by the main
repo (no submodule, no URL leak, no gitlink churn):

| Store | Nested path (gitignored) | Live store per machine | Private remote (SSH) |
|---|---|---|---|
| conversations | `.claude/conversations/` | `~/.claude/projects/<path-hash>/*.jsonl` | `Natasha-Yang/RoboTwinConvos` |
| memory | `.claude/memory/` | `~/.claude/projects/<path-hash>/memory/*.md` | `Natasha-Yang/RoboTwinMemory` |

The `<path-hash>` is derived from the repo's absolute path (differs per machine).
`.claude/hooks/sync-store.sh` is the engine; `sync-conversations.sh` / `sync-memory.sh`
are thin wrappers. Each supports:
  - `sync` — **bidirectional (default for the sync skills)**: export local + commit +
    pull/merge remote + import + **push**. Every call ends with a push.
  - `save` — push-only (export + commit + push).
  - `pull` — pull-only (fetch + import into the live store).
  - `export` / `import` — plain file copies (network-free; the SessionEnd/SessionStart
    hooks call these automatically for both stores).

**Everyday sync from a login node** (or just run the `/sync-claude` skill):

```bash
bash .claude/hooks/sync-conversations.sh sync   # bidirectional: pull + push
bash .claude/hooks/sync-memory.sh sync          # bidirectional: pull + push
bash .claude/hooks/sync-claude-config.sh        # .claude/** + CLAUDE.md -> main AND cluster
```

The third command is the **config propagator**: the two private stores above never
enter the public fork, but the tracked Claude utility files (`.claude/**` skills /
hooks / `settings.json` / statusline, plus `CLAUDE.md`) *do* live in the public repo
and were only ever committed on the `cluster` branch — so `main` drifted behind.
`sync-claude-config.sh` commits **only** those config paths onto **both** `main` and
`cluster` (using a throwaway git worktree for whichever branch isn't checked out) and
pushes each, without merging unrelated branch work or disturbing your other
uncommitted changes. `/sync-claude` runs all three. Pass `--no-push` to stage without
pushing.

**Fresh clone on a new machine** — the main clone contains NEITHER store; clone each
private repo into place, then pull:

```bash
git clone git@github.com:Natasha-Yang/RoboTwin.git && cd RoboTwin
cd .claude
git clone git@github.com:Natasha-Yang/RoboTwinConvos.git conversations
git clone git@github.com:Natasha-Yang/RoboTwinMemory.git  memory
cd .. && bash .claude/hooks/sync-conversations.sh pull && bash .claude/hooks/sync-memory.sh pull
```

> **Porting note:** the remote URLs are Natasha's private repos. On a new
> account/cluster, create your own private repos and point each nested repo's
> `origin` at them. `gh` isn't a cluster module (the `gh/0.18.0` module is a
> different tool) — install the static binary into `~/bin` from
> https://github.com/cli/cli/releases if you want the `gh` CLI. A fine-grained PAT
> can't create repos or be seen by `gh`'s API here; git-over-SSH is what the sync
> uses.

---


## 10. Quick reference

```bash
# --- setup (login node) ---
source setup_env.sh                     # every session/job

# --- claude state sync (login node) ---  (or run the /sync-claude skill)
bash .claude/hooks/sync-conversations.sh sync   # transcripts: bidirectional pull+push
bash .claude/hooks/sync-memory.sh sync          # memory:      bidirectional pull+push
bash .claude/hooks/sync-claude-config.sh        # .claude/** + CLAUDE.md -> main AND cluster

# --- data collection ---
bash collect_data.sh beat_block_hammer demo_randomized 0
bash submit_all_data.sh demo_randomized              # fan out over SLURM

# --- lerobot dataset (for finetuning) ---
cd policy/pi05
bash process_data_pi05.sh beat_block_hammer demo_randomized 50
bash generate.sh processed_data/beat_block_hammer-demo_randomized-50 <repo_id>

# --- finetune pi0.5 ---
uv run scripts/compute_norm_stats.py pi05_base_aloha_lora_clean_50x25
sbatch cluster/finetune_pi05.sh 25                   # or: bash finetune.sh <cfg> <name> 0

# --- eval / inference (needs the render-capable GPU job) ---
sbatch cluster/robotwin_gpu.sh bash -c \
  'cd policy/pi05 && bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora_clean_50x25 run0 0 0'
# MolmoAct:
sbatch cluster/robotwin_gpu.sh bash -c \
  'cd policy/MolmoAct && bash eval.sh beat_block_hammer demo_clean <norm_tag> 0 0'

# --- collect a rollout dataset (compute node records; push from a login node) ---
sbatch cluster/robotwin_gpu.sh bash -c \
  'cd policy/pi05 && bash collect_dataset.sh beat_block_hammer demo_clean pi05_base_aloha_lora_clean_50x25 run0 0 0'
```
