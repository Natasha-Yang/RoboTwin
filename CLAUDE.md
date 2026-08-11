# CLAUDE.md

Guide for setting up and running this RoboTwin fork on an HPC cluster (SLURM). It
covers environment setup, **data collection**, **building LeRobot / HF datasets**,
**fine-tuning** (π0.5 and MolmoAct), **inference / evaluation** (including the
critic-guided / critic-selected sampling path), and **rollout-dataset collection**.

This repo is a fork of RoboTwin 2.0 (`RoboTwin-Platform/RoboTwin`) with additions
for cluster batch jobs (`cluster/`, `submit_all_data.sh`), **rollout-dataset
collection** from a trained policy (`script/collect_dataset.py` → HuggingFace Hub),
and a **critic-guided / critic-selected inference** path built on the
`multisensory_steering` package (`guidance_scale` / `best_of_n` in `policy/pi05`; see
§6.7). It was developed on the **Fir** cluster (Alliance Canada). The sections below
give the concrete Fir setup **and** what to change to port it to a new cluster.

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
  (`script/collect_data.py`, `script/collect_dataset.py`, `script/eval_policy.py`).
  Skip it for pure cluster/ops/sync changes — this file already covers those.
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
   `--gpus-per-node=h100:1` in `cluster/robotwin_gpu.sh` and `cluster/rollout.sh`.
3. **GPU arch** — `TORCH_CUDA_ARCH_LIST` in `setup_env.sh` is `9.0` (H100 / sm_90).
   Set to your GPU's compute capability (A100 = `8.0`, RTX 4090 = `8.9`,
   RTX 5090 = `12.0`, …).
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
7. **pi05 policy / critic paths** — `policy/pi05/eval.sh` and
   `policy/pi05/collect_dataset.sh` prepend a workstation CUDA path
   (`/home/natasha/miniconda3/envs/cuda128/bin`, for curobo on an RTX 5090) and
   default `QMFM_ROOT=/home/natasha/QMFM`; `policy/pi05/deploy_policy.yml`'s
   `critic_config_path` points at a `multisensory-steering` checkout (on Fir,
   `/home/natashay/links/projects/def-florian7/natashay/multisensory-steering/cfgs/qmfm.yaml`).
   Update these to your machine (see §6.2, §6.7, §7).

---

## 1. Environment setup

RoboTwin's simulator (SAPIEN 3.0.0b1 / PhysX / Vulkan) plus the pinned
torch/open3d stack install cleanest from PyPI, so we use a **Miniforge conda env**
(Python 3.10) rather than the cluster module Python. π0.5 (`policy/pi05`) has its
**own** uv-managed venv (Python 3.11) — see §5.1. Do not mix the two.

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
export TORCH_CUDA_ARCH_LIST=9.0    # H100; change per §0.3
export FORCE_CUDA=1                # force the CUDA build on the GPU-less login node
```

Then run the curobo/pytorch3d `pip install` steps. `torch` ships its own CUDA
runtime, so you do **not** load the cuda module at *run* time — only for compiling.
**pytorch3d must be rebuilt whenever torch changes, in *each* env that uses it**
(the conda env and `policy/pi05/.venv`) — see §8.

> **Rorqual only — build curobo with `cuda/12.6`, not `cuda/12.2`.** On Rorqual
> (H100, driver 580.x) curobo kernels compiled with `cuda/12.2` throw
> `CUDA error: an illegal instruction was encountered` (CUDA error 715) at kernel
> launch during `motion_gen.warmup()` — even though the SASS is the correct `sm_90`
> and plain torch CUDA works fine. Rebuilding the curobo CUDA extensions with
> `cuda/12.6` fixes it. So on Rorqual, `module load cuda/12.6` for the curobo build
> above. (This is Rorqual-specific; on Fir `cuda/12.2` was fine. See §8.) To rebuild
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

**H100 note:** ray tracing works even though H100 has no RT cores — the driver/OptiX
runs it. No `_base_task.py` edit needed to fall back to rasterization. (The `rt`
shader uses the **`optix`** denoiser; if the renderer *wedges*, that is the first
thing to vary — see §6.5.)

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

This unsets the CVMFS `PYTHONPATH`/`PIP_CONFIG_FILE`, activates the conda env, sets
`TORCH_CUDA_ARCH_LIST`, and installs the Vulkan shim. **Every** SLURM job script
sources it (`cluster/robotwin_gpu.sh`, `cluster/rollout.sh`).

---

## 2. SLURM job scripts

| Script | Purpose |
|---|---|
| `cluster/robotwin_gpu.sh` | Generic single-H100 GPU job. `sbatch cluster/robotwin_gpu.sh <cmd...>`; no args → render smoke-test. |
| `cluster/rollout.sh` | Policy rollout / evaluation job (calls a policy's `eval.sh`). |
| `submit_all_data.sh` | Fan out data collection over many SLURM jobs (batches). |

All three set `WARP_CACHE_PATH` to node-local `$SLURM_TMPDIR` (a shared Warp JIT
cache across nodes/drivers can produce `CUDA error: an illegal instruction`), set
`PYTHONUNBUFFERED=1` for live logs, and resolve the repo root via
`ROBOTWIN_ROOT` / `SLURM_SUBMIT_DIR` (under `sbatch`, `$0` is a spooled copy, so
`dirname $0` does not work).

**Edit before use on a new cluster:** `--account`, `--gpus-per-node`,
`--exclude`/`--nodelist`, `--time`, `--cpus-per-task`, `--mem`.

---

## 3. Data collection

Collection searches for random seeds that satisfy the task, then replays them to
record expert demonstrations into `data/<task>/<task_config>/` (per-episode HDF5 +
per-episode instruction JSON).

**Task configs** live in `task_config/*.yml` (e.g. `demo_randomized`, `demo_clean`,
`demo_clean_privileged`). They set embodiment, cameras, domain randomization,
episode count, and the `data_type` block (rgb / depth / pointcloud / segmentation /
third_view / **wrench**) that decides what each observation carries. Create a new one
from the template:

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

## 4. Building LeRobot / HuggingFace datasets

### 4.1 Training data → LeRobot dataset (for fine-tuning)

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

`process_data_pi05.sh` writes `processed_data/<task>-<setting>-<num>/episode_*` —
each episode is an HDF5 with `observations/qpos`, `left/right_arm_dim`, and
JPEG-encoded `cam_high` / `cam_left_wrist` / `cam_right_wrist`. `generate.sh` calls
`examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py`, writing the LeRobot
dataset under `$HF_LEROBOT_HOME/<repo_id>` (an existing dir at that repo_id is
deleted and rebuilt). The `<repo_id>` you pick here **must match** `data.repo_id` in
the training config (§5.1), or training loads the wrong / no dataset.

### 4.2 Rollout dataset from a trained policy → HuggingFace Hub

A **separate** pipeline (not the training dataset above) rolls a trained policy out
in the sim and records each policy call as a HuggingFace `datasets` row — used to
train the **offline steering critic** (§6.7) and for behavior cloning. Driver:
`script/collect_dataset.py` with `policy/pi05/collect_dataset.{sh,yml}`. See **§7**
for the command, the `collect_dataset.yml` knobs, and the full column schema.

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
`policy/pi05/src/openpi/training/config.py`. Existing RoboTwin entries include
`pi05_base_aloha_lora`, `pi05_aloha_full_base`, `pi0_base_aloha_robotwin_lora`,
`pi0_fast_aloha_robotwin_lora`, `pi0_base_aloha_robotwin_full`. The active config for
this setup is **`pi05_base_aloha_lora`** (a LoRA fine-tune of `pi05_base`:
`Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora",
action_expert_variant="gemma_300m_lora")`, `prompt_from_task=True`,
`batch_size=32` total, `num_train_steps=30000`, `fsdp_devices=1`, `ema_decay=None`).
To fine-tune on your data, set that config's `repo_id` to the LeRobot dataset from
§4.1 (e.g. `pi05_base_aloha_lora` already points at
`NatashaYang/robotwin_lerobot_dataset`), and adjust `num_train_steps`, `batch_size`,
LoRA vs full, `weight_loader` base checkpoint, etc.

> **First run vs. resume.** For a clean first fine-tune, point `weight_loader` at the
> S3 base — `CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi05_base/params")` —
> set `resume=False`, and pass `--overwrite`. To *resume*/continue-train from a local
> checkpoint instead, point `weight_loader` at that step's `params` dir
> (`.../checkpoints/<cfg>/<model>/<step>/params`), set `resume=True`, and drop
> `--overwrite` from `finetune.sh` (so the run is resumed, not wiped).

Then, from `policy/pi05`:

```bash
# (a) compute normalization stats for the config (required before first train)
uv run scripts/compute_norm_stats.py <train_config_name>

# (b) fine-tune  ->  checkpoints/<train_config_name>/<model_name>/<step>/
bash finetune.sh <train_config_name> <model_name> <gpu_id>
#   e.g. bash finetune.sh pi05_base_aloha_lora robotwin_run0 0
```

`finetune.sh` runs `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py
<config> --exp-name=<model_name> [--overwrite]`. `params` lives inside each step dir;
logs go to Weights & Biases (`policy/pi05/wandb/`). Wrap it in a GPU SLURM job (adapt
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
`ckpt_path: NatashaYang/molmoact2-no-depth-aloha-loraft`.

> Note the hardcoded `repo_root`/`molmo_root`/`python` paths in
> `cluster/convert_molmoact_checkpoint.sh` — update them on a new cluster.

---

## 6. Inference / evaluation

Evaluation is driven by `script/eval_policy.py`, configured by each policy's
`policy/<name>/deploy_policy.yml`, and launched by that policy's `eval.sh`. The eval
loop does an expert-feasibility check per seed, then runs the policy, optionally
logs video (`eval_video_log`) and per-episode debug diagnostics (`debug: true`,
§6.6). Results go to
`eval_result/<task>/<policy>/<config>/<ckpt>/<timestamp>/` (`_result.txt`,
`_episode_results.csv`, videos, `debug_vis/` if `debug`, and — for a critic-guided
run — the online critic checkpoint + W&B). Collecting an HF **rollout dataset** is a
separate pipeline (§7).

### 6.1 Via the cluster rollout job (recommended)

```bash
sbatch cluster/rollout.sh <policy_name> <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
```

`cluster/rollout.sh` sources `setup_env.sh`, sets up headless rendering, then
`cd policy/<policy_name> && bash eval.sh ...`.

### 6.2 π0.5 directly

```bash
cd policy/pi05
bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> \
             [guidance_scale] [guidance_ramp_updates] [train_online] [use_step_reward] [best_of_n]
# baseline (no critic)
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0
# online critic-guided
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0 0.3 256
# best-of-8 selection with no gradient guidance
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0 0 "" "" "" 8
```

`eval.sh` activates `policy/pi05/.venv`, exports the critic env (`QMFM_ROOT`), caps
GPU memory (`XLA_PYTHON_CLIENT_MEM_FRACTION`, so JAX shares the GPU with the sim),
and calls `eval_policy.py` with `deploy_policy.yml`. The first six args are required;
the trailing ones override the critic knobs (§6.7) for a one-off run.

- `deploy_policy.py` → `pi_model.py::PI0` loads the trained policy from
  `policy/pi05/checkpoints/<train_config_name>/<model_name>/<checkpoint_id>` and runs
  inference. `checkpoint_id` (default 30000) and `pi0_step` (action-chunk length, 50)
  come from `deploy_policy.yml`.
- Camera → model mapping (`deploy_policy.py::encode_obs` / `pi_model.py`):
  `head_camera → cam_high`, `left_camera → cam_left_wrist`,
  `right_camera → cam_right_wrist`.
- Results: `eval_result/<task_name>/<policy_name>/<task_config>/<ckpt_setting>/<timestamp>/`.
  Alongside `_result.txt` / `_episode_results.csv`, each run snapshots
  `deploy_policy.yml` and the `critic_config_path` file it includes into that dir
  (`eval_policy.py::snapshot_config`). The copies keep their comments but carry the
  values **actually used** — `eval.sh`'s positional args are written in, so
  `task_name`, `seed`, `guidance_scale` etc. read as resolved rather than as the
  `null`/default in the source yml.
- `use_step_reward` (`deploy_policy.yml`, or `eval.sh`'s 10th arg) picks **which
  reward the run scores itself with** — what `online_critic.commit()` is fed, what
  the W&B reward curves and the `reward` column of `_episode_results.csv` measure.
  `true` (default) is 1.0 on the control step that first reaches success plus the
  task's own shaped progress term (`step_reward()`, a delta — `envs/lift_pot.py`,
  `envs/open_microwave.py`, `envs/put_object_cabinet.py`); `false` is sparse: 1.0 on
  success and 0.0 everywhere else, even for a task that defines shaping
  (`step_reward()` is then never called at all, so its own delta state never
  advances). Unlike the guidance knobs this applies to the plain baseline too.
  `eval_policy.py::control_step_reward` is the one place it acts, so a guided run's TD
  targets and the printout agree by construction; the equivalent for a collected
  dataset's `reward` column is §7.1, which is always shaped.

> `policy/pi05/eval.sh` prepends a workstation CUDA path
> (`/home/natasha/miniconda3/envs/cuda128/bin`, for curobo on an RTX 5090) and
> defaults `QMFM_ROOT=/home/natasha/QMFM` — remove/adjust both on a new cluster
> (§0.7).

### 6.3 MolmoAct directly

```bash
cd policy/MolmoAct
bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
```

`ckpt_setting` is used as the MolmoAct `norm_tag`. Other knobs (`ckpt_path`,
`molmoact_step`, `test_num`, rollout recording) are in
`policy/MolmoAct/deploy_policy.yml`.

### 6.4 Other baselines

`policy/` also ships DP, ACT, DP3, RDT, pi0, openvla-oft, TinyVLA, DexVLA,
LLaVA-VLA, GO1. Each follows the same `deploy_policy.yml` + `eval.sh` convention;
see the upstream docs for their specific fine-tuning setup.

### 6.5 Crash recovery (π0.5)

An eval run is hours long and can end early two ways: it can crash, and it can wedge.

> **The wedge, for reference.** The renderer hangs:
> `SapienRenderCameraInternal::waitForRender` → `libnvidia-eglcore` → `poll()`, a
> render fence that never signals, with the GPU completely idle and every thread
> asleep. It surfaces as a run that neither progresses nor dies. Diagnose with
> `py-spy dump --pid <pid> --native` (needs `ptrace` access — `ptrace_scope=1`, i.e.
> effectively root on a workstation; on the cluster you can attach to your own
> process). `_base_task.py` uses the `rt` shader with the **`optix`** denoiser
> (`script/test_render.py` uses `oidn` for the same block), which is the first thing
> to vary if it recurs (§1.3). Nothing detects this automatically — a hung run has to
> be noticed and killed by hand.

Either way the episodes already finished are safe. Every one is committed to the run
dir in strict order — its row appended to `_episode_results.csv`, the critic
checkpointed, then `resume_state.json` written atomically. The json is the commit
marker, so an interruption rewinds to the last episode that completed all three; a
csv row one ahead of it is trimmed on resume.

`resume: true` (`deploy_policy.yml`) then continues the newest interrupted run for
this task/policy/config/ckpt **in its existing directory** rather than opening a new
timestamped one:

| Restored | From | Why it cannot be recomputed |
|---|---|---|
| `now_seed`, `now_id`, `suc_test_seed_list` | `resume_state.json` | which seeds the expert check rejected is not a function of the episode index |
| global numpy RNG state | `resume_state.json` | the episode's instruction is drawn with `np.random.choice` |
| `test_num`, `suc`, `chunk_count` | `resume_state.json` | counters the guidance ramp and MA windows key off |
| episode rows + MA windows | `_episode_results.csv` | reloaded so the averages continue across the break |
| critic params, target, **Adam state, LR schedule position** | `online_value_critic.pkl` | see below |
| guidance ramp position | `critic_ramp_baseline` in `resume_state.json` | see below |
| W&B run | `wandb_run_id` → `resume="allow"` | keeps the critic curves one continuous series |

The optimizer half needed a change in `multisensory_steering`: `OnlineValueCritic.save`
now records `opt_state` and `lr_step`, and restores them when `restore_optimizer` is
set — which `eval_policy.py` sets only on a resume. A plain warm start from
`critic_ckpt` still gets a fresh optimizer, which is the right default for pointing an
offline critic at a new task but was silently wrong for resuming: the run came back at
the *bottom of warmup* (lr 0) and climbed the schedule again. The state is checked
against the optimizer actually built (`clip_grad` and `freeze_encoder` change its
tree) and dropped with a message rather than crashing if it no longer fits;
checkpoints written before this load exactly as they did.

The **guidance ramp** had the same shape of bug one level up. A resume points
`critic_ckpt` at the run's own `online_value_critic.pkl`, and `PI0._init_critic`
re-bases the ramp at whatever update count a checkpoint restores (§6.7 — deliberate
for an offline critic, so a warm start eases in like a fresh one). Applied to a run's
own checkpoint that means the ramp restarts: a run that had reached full
`guidance_scale` comes back at **0** and climbs the whole `guidance_ramp_updates`
again. So `resume_state.json` now carries `critic_ramp_baseline` — the count *that*
run's ramp was measured from (0 when its critic was trained from scratch here) — and
`eval_policy.py` hands it back through `deploy_policy.py` as
`PI0(critic_ramp_baseline=...)`, which uses it instead of re-deriving one. The startup
banner prints where the ramp resumes. It is not a user-facing config key: nothing but
the resume path sets it, and it is applied only when the critic checkpoint is actually
reloaded (against a critic starting at 0 updates it would pin guidance at 0 instead).
State files written before 2026-08-08 have no such key: a run whose critic trained
from scratch is reconstructed exactly (its baseline was 0), one that warm-started from
an offline `critic_ckpt` cannot be, and keeps the old restart-the-ramp behavior with a
printed note.

**Not** restored: the replay buffer (gigabytes of SigLIP features, §6.7), so a
resumed critic keeps its weights and optimizer but refills the buffer from empty and
runs no TD update until `start_training` transitions are back in it.

`_episode_results.csv` gained explicit `episode` and `seed` columns and lost its
unnamed index.

### 6.6 Debug visualization (`debug: true`)

Setting `debug: true` in the **task config** turns on extra per-episode diagnostics
(`eval_policy.py::visualize_debug_obs`), all written into `debug_vis/episode<N>/`
under that same result dir:

| Output | File (under `debug_vis/episode<N>/`) | Notes |
|---|---|---|
| TCP wrench histograms | `wrench_hist_episode<N>.png` | one histogram per component (Fx/Fy/Fz/Tx/Ty/Tz), left and right arm overlaid |
| Rollout + wrench GIF | `wrench_episode<N>.gif` | head camera on the left with the **world** axes drawn as labelled x/y/z arrows, projected into the camera and anchored at each arm's TCP (the axes the components are resolved in, at the point they act); the wrench traces with a step cursor on the right |
| Raw series | `wrench_episode<N>.npz` | `step`, `left`, `right` — `(num_samples, 6)` each |
| Critic Q trace | `q_episode<N>.png` | guided runs only (see below) |
| Rollout + Q GIF | `q_episode<N>.gif` | head camera on the left, the Q and reward traces with a step cursor on the right |
| Raw series | `q_episode<N>.npz` | `step`, `q` `(num_samples, num_qs)`, `reward`, `guidance_scale`, `return_to_go`, plus the scalars `return_mean` / `return_std` / `gamma_h` |
| Depth / segmentation tiles, point clouds | `obs_step<S>.png` / `pcd_step<S>.ply` | only for the modalities the task config's `data_type` enables |

Both GIFs animate the same rollout, so `envs/utils/debug_vis.py::RolloutFrameLog`
downscales and holds the head-camera frames once for all of them, keyed by sim step.
That module holds both recorders (`TCPWrenchRecorder`, `QValueRecorder`) and the
plotting/GIF plumbing they share; `eval_policy.py` keeps `visualize_debug_obs` and the
code that constructs, feeds and flushes them.

The wrench is computed by `envs/utils/wrench.py` (shared with the rollout-dataset
collector, §7.1). It is the **net contact wrench on the end-effector links** (wrist
link + gripper fingers + any `fix_gripper_name` links), summed from
`scene.get_contacts()` impulses divided by the sim timestep, with torque taken about
the TCP origin. Both vectors are resolved in the **world** frame (N and N·m) — only
the moment arm is TCP-relative, so a trace stays comparable across steps as the
gripper rotates. It is contact-only: an arm moving through free space reads exactly
zero — this is not a joint-torque estimate. One sample is taken per policy call (so
`pi0_step` sim frames apart), paired with that call's head-camera frame.

The **Q outputs need a critic**, so they appear only when `debug: true` meets a
nonzero `guidance_scale` or a `best_of_n > 1` (§6.7); a baseline run prints
`critic Q logging OFF` and writes none. Each row is one control step:
`pi_model.py::PI0.get_action` scores the chunk it is about to execute with the critic
that produced it — the same normalized `(50, 14)` chunk the guidance climbed and/or
best-of-N selected, against the same observation, via `OnlineValueCritic.q_values` —
and `debug_vis.py::QValueRecorder` pairs it with the reward that chunk earned and the
guidance scale in force at the time. Unlike the wrench it is recorded *after* the
control step (the value does not exist until the chunk has been drawn), against the
step index of the observation it was drawn from, so it lines up with the same frame.
The extra critic forward per chunk is why `PI0.record_q_values` is off unless the
driver turns it on.

Both plots draw the **realized discounted return-to-go** (at the critic's own
`discount ** horizon`) against Q, in return units — Q is un-normalized by the
checkpoint's `return_mean`/`return_std` first, which is the identity for a critic
trained online from scratch. Q tracking that curve is a calibrated critic; a flat Q
means it is not distinguishing the states it is steering through, a persistent gap
means it over- or under-values them, and an ensemble range that stays wide means the
members disagree about states the guidance follows anyway. The `.npz` keeps `q`
**raw** (as the guidance sees it) plus the constants to convert.

### 6.7 Critic-conditioned sampling (`guidance_scale` and `best_of_n`)

There is **one** eval script and **one** config. Two independent keys in
`deploy_policy.yml` decide whether a critic acts on `Pi0.sample_actions`, and
**either one** on its own builds the critic (and, by default, TD-trains it online):

| Key | Off | On |
|---|---|---|
| `guidance_scale` (7th positional arg) | `0.0` | an ensemble QMFM `Value` critic steers each denoising step by gradient guidance, ramping `0 → guidance_scale` over `guidance_ramp_updates` TD updates (`0` jumps to target after the first update) |
| `best_of_n` (11th arg) | `1` | `n` candidate chunks are drawn per control step and the highest-Q one is executed |

Both off is the plain pi0.5 baseline — no critic is built, no replay collection, no TD
updates, no W&B.

#### Best-of-N selection

`best_of_n: n` draws `n` chunks from independent noise in a **single**
`sample_actions` call and executes the one with the highest ensemble-mean Q — the same
aggregation the guidance ascends, so with both switched on the selection agrees with
what the steering was trying to do instead of pulling against it. The winner is the
chunk that goes into the replay buffer, the `_episode_results.csv` reward and the
debug Q log: everything downstream is about the action that actually ran, and the
losing candidates are discarded inside the sampler.

The two knobs compose but are not the same thing. Guidance moves a *single* sample
toward higher Q and can walk it off the policy's own distribution if the critic is
wrong there; best-of-N only ever returns something the frozen pi0.5 sampler drew on
its own, so a bad critic costs it nothing beyond the wasted compute — with an
untrained critic it degrades to picking a candidate at random, which is exactly the
baseline. That is why guidance needs `guidance_ramp_updates` and best-of-N needs no
ramp.

Cost is `n` denoising loops, not `n` policy calls: the candidates ride along as extra
batch elements, replicated **after** the SigLIP tower and the prefix pass, so those
still run once per control step and only the loop and the prefix KV cache scale with
`n` (the cache is ~15 MB/candidate at this prefix length). `guidance_scale: 0` with
`best_of_n > 1` is meaningfully cheaper per candidate than the guided path, which pays
two extra forward passes per denoising step for the value gradient — `pi_model.py`
passes `guidance_scale=None` in that case so the gradient branch is compiled out
rather than multiplied by a constant zero. `best_of_n` is a static jit arg: changing
it recompiles.

Per-episode W&B (`eval_policy.py::BestOfNRecorder`) reports `bestofn/q_gain` — the
winner's Q minus the mean over its candidates, i.e. what the selection bought over
executing an arbitrary one of them — and `bestofn/q_spread`, the range it chose over.
A `q_gain` near zero means the critic cannot separate the chunks the sampler draws and
the extra `n`-fold denoising is buying nothing.

#### Online TD training of the critic

How much of the critic online TD is then allowed to move is decided by two keys in the
**critic config** (`critic_config_path`, i.e. `cfgs/qmfm.yaml` — not
`deploy_policy.yml`, though that file and the CLI can override them like any other
critic key). Both only mean anything once a critic is running:

| Key | Default | Behavior |
|---|---|---|
| `train_online: true` | ✔ | as above — transitions are stashed into the replay buffer after every control step, TD updates run, `save_critic` writes the result |
| `train_online: false` | | the critic is **frozen** at `critic_ckpt`: it still steers the sampler, but nothing is stashed, no TD update runs, and the ramp is skipped (guidance sits at `guidance_scale` from the first chunk, since there are no updates to count). W&B still opens and logs the eval metrics |
| `freeze_encoder: true` | | the half-way point: TD still runs, but only on the value head — the observation encoder (`MultiModalEncoder`, or the legacy single SigLIP CNN) keeps the checkpoint's weights |

Freezing is for warm starts. `train_online: false` evaluates an offline-trained critic
as-is, with no eval-time drift in its values; `freeze_encoder` keeps the
representation a large offline dataset paid for — the part a few thousand online
transitions are least able to improve — while letting TD refit the MLP on top of it.

`train_online: false` requires a `critic_ckpt`: a frozen critic never leaves its
initialization, so with none it would steer on the gradients of a random network (or
rank best-of-N candidates by one), and `pi_model.py` raises at startup rather than
running it. The replay buffer is allocated lazily on the first stash, so a frozen run
never pays its memory either, and `save_critic` is skipped — the critic is
byte-identical to the checkpoint the snapshotted config names. `eval.sh`'s 9th
positional arg overrides `train_online` for a one-off run.

`freeze_encoder` is implemented in `qmfm.py::freeze_encoder_tx` as an
`optax.multi_transform` that zeroes the updates of everything under the params tree's
`encoder` key, rather than as a `stop_gradient` inside the module: `critic_def` stays
one pure function, so the guidance path — which differentiates Q with respect to the
*action*, a branch that never enters the encoder — is unaffected either way, and the
zeroed gradients are dead code in the jitted update. Only the online params are
pinned; the target network keeps its Polyak update, which against a constant encoder
just converges it to the same frozen weights. Unlike `encoder_modalities` it is
**not** an architecture key — it changes no shapes, so the config's value wins over a
checkpoint's and either setting loads either checkpoint.

The ramp counts TD updates **performed in this run**. A critic warm-started from
`critic_ckpt` restores the checkpoint's lifetime `num_updates` (an offline-trained one
is in the hundreds), so using that counter directly would read as "ramp already
finished" and apply full guidance from the first chunk;
`PI0.scheduled_guidance_scale` subtracts the value at load time instead. W&B's
`critic/update` still reports the lifetime count.

`guidance_scale` and `best_of_n` default to whatever `deploy_policy.yml` says; the
positional args only override them (pass `0` and `1` to force the baseline). The
critic's own hyperparameters are **not** in `deploy_policy.yml` — it carries
`critic_config_path`, and `parse_args_and_config` merges that file in underneath, so
precedence is **CLI > deploy_policy.yml > critic_config_path**. The critic
implementation and its config both come from the `multisensory_steering` package
(editable install; `critic_config_path` points at its `cfgs/qmfm.yaml` — on Fir
`/home/natashay/links/projects/def-florian7/natashay/multisensory-steering/cfgs/qmfm.yaml`),
which imports QMFM's `ReplayBuffer` from `$QMFM_ROOT` (exported by `eval.sh`; the
default is a workstation path — override it, §0.7). Only a run with a critic logs to
W&B, collects replay transitions, and honors `save_critic` / `critic_ckpt` / the TD
hyperparameters; `eval_policy.py` keys all of it off whether the policy object exposes
an `online_critic`, and the collection/updates additionally off
`model.train_critic_online` (`_trains_online_critic`), which — unlike the critic
object — is known before the first observation.

**The critic must be trained in pi0.5's model space.** `Pi0.sample_actions` scores the
chunk it is sampling, *before* the output transform runs: **normalized** state `(14,)`
and a **normalized** action chunk `(50, 14)` → flat `700`. The `14` is
`critic_action_dim` — the model pads *both* state and actions to `action_dim=32`, but
the trailing dims are constant zero for aloha, so the padding is stripped back off and
only the embodiment's own dims reach the critic (`state_dim=14`, not 32).

`critic_action_dim` is the width of `observation["joint_action"]["vector"]` (both arms
plus grippers), so it follows the embodiment automatically. Nothing knows that width
until the sim produces its first observation, so `PI0` builds the critic lazily in
`_init_critic`, called from the first `update_observation_window` — `model.online_critic`
is `None` until then. Anything needing to know *before* a rollout starts (e.g. whether
to open a W&B run) must read `model.uses_online_critic` instead; `eval_policy.py`
re-reads `model.online_critic` fresh at each use site for exactly this reason.

Since architecture keys are taken *from* the checkpoint, a critic with the wrong
shapes loads "successfully" and then fails with an opaque shape error mid-sampler, so
`pi_model.py` checks `state_dim` / `action_dim_flat` up front and raises. That check
catches a wrong embodiment or action horizon, but **not** a critic trained on the
*raw* columns: those have the same widths as their `.model` counterparts and differ
only in normalization, so nothing downstream can tell them apart. See §7.1 for
collecting the right columns.

#### What the critic observes

The critic is not restricted to the SigLIP map and the state. Everything the sim
produces this run is offered to it, and **which modalities it uses is decided in the
critic's own config** (`multisensory_steering`'s `cfgs/qmfm.yaml`, key
`encoder_modalities`) — not here:

| Modality | Source | Shape |
|---|---|---|
| `siglip.{head,left_wrist,right_wrist}` | pi0.5's own image tower, inside `sample_actions` | `(16, 16, 1152)` each |
| `state` | normalized model state, embodiment dims | `(14,)` |
| `images.{head,left_wrist,right_wrist}` | task config `data_type.rgb` | `(240, 320, 3)` uint8 |
| `images.third_view` | task config `data_type.third_view` | `(H, W, 3)` uint8 |
| `depth.{head,left_wrist,right_wrist}` | task config `data_type.depth` | `(240, 320)`, mm |
| `pointcloud` | task config `data_type.pointcloud` | `(pcd_down_sample_num, 6)` |
| `wrench.{left,right}` | per-step contact wrench, logged by the env | `(pi0_step, 6)` |

The names are the §7.1 dataset columns minus their `observation.` prefix, so a critic
trained offline on those columns lines up with what it sees online.
`envs/utils/obs_modalities.py` flattens an observation into them (shared by both
paths); `deploy_policy.py::critic_obs_modalities` calls it once per control step and
only when a critic exists, so the baseline and collection runs copy nothing.
`pi_model.py::_init_critic` declares the whole set as `obs_shapes`, the critic picks
its subset (`OnlineValueCritic.obs_keys`), and only that subset is shipped into
`Pi0.sample_actions(critic_obs_extra=...)` and into the replay buffer. Naming a
modality the task config does not enable raises at startup, listing what *is*
available.

Three things to keep in mind:

- **All three SigLIP views come free, but only in compute.** The image tower already
  runs once per camera to build the prefix, so `Pi0.embed_images` keeps its raw
  `aux["encoded"]` for every view and `embed_prefix` reuses the tokens — the wrist
  maps cost no extra tower pass, and the head map no longer costs the second one it
  used to. What they do cost is the replay buffer: each configured view is another
  ~1.15 MB/transition (see the last bullet). Unconfigured views are dropped by
  `OnlineValueCritic.stash`, so they only ride back from the device in `aux`.
- **The wrench a control step sees is the previous chunk's trace** — the steps between
  the last observation and this one, which is the only wrench that exists before the
  current chunk has been executed. Rollout collection drains the log at the same point
  in its loop, so a dataset's `observation.wrench.*` holds exactly this against
  exactly this row's state and action, and a critic trained offline on it needs no
  realignment. (Datasets collected before 2026-07-30 store the *following* chunk's
  trace instead — see §7.1.) The env's per-step logging is switched by the task
  config's `data_type.wrench` (§7.1); a critic configured for `wrench.*` against a
  config that has it off fails at startup.
- **Modalities are an architecture key.** They go into the checkpoint, and a warm
  start rebuilds the same encoder stack; changing the list makes an existing critic
  checkpoint refuse to load (loudly, leaf by leaf). Offline pretraining takes the same
  names — `multisensory_steering`'s `cfgs/train_offline.yaml` maps them to dataset
  columns under `dataset.modalities` — and a checkpoint only warm-starts a run whose
  list matches. Each modality is also stored twice per transition (obs and next_obs) —
  each `siglip.*` view is ~1.15 MB/transition (so all three come to ~3.5 MB), a depth
  camera ~0.6 MB, so `buffer_size` needs revisiting when adding one.

For remote / server-based inference see `policy/pi05/docs/remote_inference.md`
(`scripts/serve_policy.py`).

---

## 7. Collecting a rollout dataset (policy rollouts → HF Hub)

This pipeline rolls a trained policy out in the sim and builds a HuggingFace dataset
of the resulting trajectories — the training data for an offline steering critic
(§6.7), and for behavior cloning.

```bash
cd policy/pi05
bash collect_dataset.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id>
# (wrap in a GPU SLURM job like §6.1 on a cluster)
```

- Driver: `script/collect_dataset.py` with `policy/pi05/collect_dataset.yml`.
- Behavior is controlled by `collect_dataset.yml`:
  - `num_episodes: 100` — successful, expert-checked rollouts to collect.
  - `expert_check: true` — only roll out on seeds the expert can solve.
  - `output_dir: ./rollout_datasets` — local `save_to_disk` location.
  - `push_to_hub: true`, `hub_repo_id: NatashaYang/robotwin_pi05_rollouts_dataset`.
  - `hub_private: true` — create the hub repo private. This is the **default when the
    key is absent**, so a run cannot publish a dataset by omission; set it `false` to
    deliberately create a public repo. It is honored only when the push *creates* the
    repo — an existing repo keeps whatever visibility it already has, so this cannot
    retroactively hide (or expose) a dataset that has been pushed before. To change an
    existing repo, use
    `HfApi().update_repo_settings(repo_id, repo_type="dataset", private=...)`.
  - `checkpoint_id: 30000`, `pi0_step: 50`, `instruction_type: unseen`.
  - `collect_critic_obs: true` — also record the policy's **model-space** view of each
    step.
  - `collect_siglip: true` — within that, also record the SigLIP patch features of
    **every** camera the policy sees (`siglip.head`, `siglip.left_wrist`,
    `siglip.right_wrist`). Set `false` to keep the dataset small, or list the views you
    want (`[head, left_wrist]`).
  - `resume: true` — see below.

Everything **beyond** rgb + qpos is decided by the **task config**, not by
`collect_dataset.yml`: whatever its `data_type` block turns on reaches
`envs/_base_task.py::get_obs`, and `extra_obs_columns` records all of it (§7.2).
`demo_clean` therefore yields only the columns in §7.1's table, while
`demo_clean_privileged` roughly doubles the bytes per row.

Each episode is flushed to its own shard in `<output_dir>/<task>/<config>/<ckpt>_shards`
the moment it finishes, and the shards are memory-mapped and concatenated into the
final dataset at the end. A row costs ~1 MB resident (three uncompressed camera
frames) plus ~576 KB per SigLIP view `collect_siglip` records and ~1 MB more under a
privileged task config, and a long task (`put_bottles_dustbin`, `step_lim` 1700, at
`pi0_step` 10) can reach five figures of rows, so accumulating a whole run in memory
would run to tens of GB. Sharding also means a crashed run keeps its episodes:
`progress.json` in the shard dir records the seed to resume from — which cannot be
recomputed, since the seed sequence depends on which seeds the expert check rejected.
Rerunning the same command continues from there (`resume: false` discards the shards
and starts over). The shard dir is removed only after the dataset is saved and pushed.

### 7.1 Columns, and which ones a steering critic needs

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
| `reward` | reward earned by this row's own chunk | scalar |

Each raw column and its `.model` counterpart have the same width and hold the same
quantity in different spaces: the raw ones have been unnormalized by the output
transform, the `.model` ones have not. Neither carries the model's internal zero
padding to `action_dim=32` — it is stripped before the tensors leave the sampler, so
`state_dim` is `14`, not `32`. Only `pi0_step` of the 50 chunk steps are actually
executed before the next inference call; the whole chunk is recorded because that is
what the sampler scores.

`frame_index` is the sim's own `take_action_cnt` at the moment the row's observation
was taken, so rows are `pi0_step` frames apart (0, 10, 20, … at `pi0_step: 10`) rather
than 1. Elapsed time is what consumers key off: `multisensory_steering`'s offline
trainer discounts by the `frame_index` gap between consecutive rows and subsamples on
`frame_index % horizon == 0`, so a row counter would under-discount by exactly
`pi0_step`. Datasets collected this way need no `create_dataset.py rescale-frames`
pass — and that pass will now refuse to run on them, since it checks that the gap is 1
first.

The `.model` columns come from `Pi0.sample_actions(..., return_critic_obs=True)` — the
same tensors the guided sampler scores — and are only present when `collect_critic_obs`
is set. A critic intended to steer inside the sampler (§6.7) **must** be trained on
these; point `multisensory_steering`'s `cfgs/train_offline.yaml` at
`state_col: observation.state.model` / `action_col: action.model`. The raw columns
remain for behavior cloning and for critics that score executed robot actions.

The `siglip.*` columns are the visual thing the critic conditions on, and are written
**during collection** (`collect_siglip`) from `critic_obs_siglip` — the exact patch
maps `Pi0.sample_actions` feeds the critic, computed by the same `PaliGemma.img` tower
on the same resized frames. There is one per camera the policy sees, under the same
names the critic uses online (`Pi0.SIGLIP_MODALITIES`), each stored as the flat
`(256, 1152)` patch sequence (the CNN encoder reshapes to the 16×16 grid itself) in
fp16, matching the online replay buffer. They are drop-in replacements for the columns
`python -m multisensory_steering.create_dataset siglip` used to add in a second pass —
point `siglip_cols: [siglip.head, siglip.left_wrist, siglip.right_wrist]` at them and
skip that pass entirely. At ~576 KB/row **each**, these columns dominate dataset size:
`collect_siglip` takes a list of views (`[head]`) as well as `true`/`false`, and
dropping the wrists is the cheapest way to shrink a run.

`observation.wrench.left` / `.right` are the **same quantity** §6.6's debug GIF plots —
net contact wrench on that arm's end-effector links, `[Fx, Fy, Fz, Tx, Ty, Tz]` in the
world frame, torque about the TCP — computed by the shared
`envs/utils/wrench.py::tcp_wrench_vector` so the eval and collection paths cannot drift
apart. The rate differs: `eval_policy.py` samples once per policy call, while
collection samples after **every** primitive step, so a row carries a whole
`(pi0_step, 6)` trace rather than a single vector.

Which trace matters: it is the one the **previous** chunk produced — the steps between
the previous row's observation and this one. The env logs a sample after each
`take_action` (`_base_task.py::_log_step_wrench`, so the reading belongs to the action
that just executed), and the rollout loop drains the log with `pop_step_wrench()`
**before** running the chunk, at the same point in the loop the guided eval path
drains it (§6.7). The column is therefore an observation — something the policy could
have conditioned on — not the outcome of the row's own action, and offline training
pairs it with the row's state/action as-is.

> Datasets collected before 2026-07-30 drained the log *after* the chunk, so their
> `observation.wrench.*` is the trace of the row's **own** chunk. To train on one,
> shift the column a row later within each episode — `multisensory_steering`'s
> `dataset.modalities` takes `{column: ..., shift: 1}` for exactly this.

`reward` is the **same quantity the online critic is fed** during a guided eval: 1.0
on the control step that first reaches success, and otherwise the task's own
`step_reward()` — the shaped, *delta*-valued progress term (`envs/lift_pot.py`,
`envs/open_microwave.py`, `envs/put_object_cabinet.py`); tasks that define none leave
the column sparse (0 everywhere but the successful step). Both drivers call
`eval_policy.py::control_step_reward` at the same point in their loop so the two cannot
drift — `eval_policy.py` passes the result to `online_critic.commit()`,
`collect_dataset.py` stores it. Unlike the wrench, it belongs to the row's **own**
chunk (it is the outcome of this row's action, not an observation preceding it), which
is exactly what an offline `(s, a, r, s')` needs. It cannot be recomputed after the
fact: `step_reward()` is a difference against its own previous call, so it only exists
while the episode runs. Train on it with `reward_col: reward` in
`multisensory_steering`'s `cfgs/train_offline.yaml` (the alternative, `terminal_reward`,
derives the sparse signal from `success` and ignores the shaping). Datasets collected
before 2026-08-07 have no such column.

`envs/utils/wrench.py::stack_step_wrench` does the stacking, shared with the critic's
online view of the same modality. A trace can be short — an episode's first row has no
previous chunk and carries a single sample of the contact state at that instant, and a
chunk cut off by success or `step_lim` contributes only the steps it ran — so the tail
is padded with **NaN**, not zeros, since zero is a meaningful reading (the arm touching
nothing). At 480 B/row they are the cheapest column here.

`wrench` is a `data_type` like the others, but it is the one the env cannot pick up
from the flag itself: contacts are a scene query, not part of `get_obs`. So each driver
reads `data_type.wrench` and passes `record_step_wrench` into the env —
`collect_data.py`, `collect_dataset.py` and `eval_policy.py` all do, and nothing logs a
wrench with the flag off. Turning it off drops these columns from the dataset entirely
(and makes a `wrench.*` critic modality unavailable, §6.7).

### 7.2 Extra data types (privileged task configs)

`script/collect_dataset.py::extra_obs_columns` records everything else the observation
carries, so the dataset follows the task config's `data_type` block automatically.
With `demo_clean_privileged` (depth / pointcloud / third_view / mesh + actor
segmentation all `true`) a row gains, per camera `<cam>` ∈ `head` / `left_wrist` /
`right_wrist`:

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
- Sim camera names (`head_camera` / `left_camera` / `right_camera`) are shortened to
  the same suffixes the rgb columns use (`head` / `left_wrist` / `right_wrist`).
- The camera matrices ride along whenever depth or a point cloud does — depth is not
  unprojectable without them, and the wrist extrinsics change every step. They are not
  a `data_type` of their own; `get_obs` always returns them.
- The schema is inferred from the first collected row (`build_features` /
  `infer_feature`), so nothing has to be enumerated per data type. ndarrays become
  fixed-shape `ArrayND` columns; a point cloud with `pcd_down_sample_num: 0` is ragged
  and falls back to a nested `Sequence`.
- Measured on-disk cost: **~1.1 MB/row** for `demo_clean` (with `siglip.head` only;
  each further SigLIP view adds ~0.56 MB, so all three make it ~2.2 MB) vs
  **~2.2 MB/row** for `demo_clean_privileged`. Depth is the bulk of the difference —
  the segmentation and third-view columns are PNG-compressed. There is no per-column
  switch here: to collect less, use a task config with fewer `data_type` flags.

---

## 8. Cluster gotchas (Fir-derived, worth re-checking anywhere)

- **Dummy opencv**: Alliance's CVMFS profile exports `PIP_CONFIG_FILE` pointing at a
  wheelhouse with a stub `opencv` that breaks `pip install`. `setup_env.sh` unsets
  it (and `PYTHONPATH`). Keep this if your cluster injects a Python env via a module
  system.
- **No `sudo`**: Vulkan/ffmpeg come from the module system / CVMFS, not `apt`. (Also
  why `py-spy` in §6.5 can only attach to your own processes.)
- **`setuptools==69.5.1`**: pinned for SAPIEN's `pkg_resources`. Don't upgrade.
- **pytorch3d must be rebuilt whenever torch changes — in *both* envs.** pytorch3d
  ships a compiled `_C.so` (used by `fps` in `envs/camera/camera.py` for point-cloud
  downsampling) linked against a specific torch ABI. The eval path (`eval.sh`) runs in
  `policy/pi05/.venv` (torch 2.7.0 / py3.11), while collection runs in the RoboTwin
  conda env (torch 2.4.1 / py3.10) — pytorch3d must be built into **each**, against
  *that* env's torch, with the CUDA toolkit + arch from §1.2. Otherwise `camera.py`
  swallows the ABI error in a bare `except:` and prints only
  `fps error: missing pytorch3d` — the same message whether pytorch3d is ABI-broken
  *or* simply absent (the real error is an `undefined symbol` on `_C...so`). Pin the
  target interpreter by absolute path so the CUDA env's own python doesn't shadow it:
  ```bash
  <env>/bin/python -m pip install --no-build-isolation --no-deps \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  ```
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
- **HF Hub / W&B auth**: dataset/checkpoint push-pull (§4, §7) needs
  `huggingface-cli login` (a token for the `NatashaYang/...` repos); training and
  critic-guided eval log to Weights & Biases (`wandb login`, or `WANDB_MODE=offline`).
- **Offline compute nodes**: pre-download HF checkpoints/assets and set `HF_HOME` on
  the login node; only *record* rollouts on the compute node, build/push datasets on
  the login node.

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
uv run scripts/compute_norm_stats.py pi05_base_aloha_lora
bash finetune.sh pi05_base_aloha_lora robotwin_run0 0

# --- eval / inference ---
sbatch cluster/rollout.sh pi05 beat_block_hammer demo_randomized robotwin_run0 0 0
# or MolmoAct:
sbatch cluster/rollout.sh MolmoAct beat_block_hammer demo_randomized <norm_tag> 0 0
# critic-guided pi0.5 directly (see §6.7):
cd policy/pi05 && bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora <model> 0 0 0.3 256

# --- collect a rollout dataset (→ HF Hub; see §7) ---
cd policy/pi05
bash collect_dataset.sh beat_block_hammer demo_clean pi05_base_aloha_lora <model> 0 0
```
