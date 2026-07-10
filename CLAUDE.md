# CLAUDE.md

Guide for setting up and running this RoboTwin fork on an HPC cluster (SLURM). It
covers environment setup, **data collection**, **building LeRobot / HF datasets**,
**fine-tuning** (π0.5 and MolmoAct), and **inference / evaluation**.

This repo is a fork of RoboTwin 2.0 (`RoboTwin-Platform/RoboTwin`) with additions
for cluster batch jobs (`cluster/`, `submit_all_data.sh`), rollout recording +
HuggingFace dataset export (`script/rollout_recorder.py`,
`script/build_hf_dataset.py`), and a reward-critic guided-inference path
(`rewards/`). It was developed on the **Fir** cluster (Alliance Canada). The
sections below give the concrete Fir setup **and** what to change to port it to a
new cluster.

> The upstream project docs live at https://robotwin-platform.github.io/doc/ —
> refer there for task/config semantics. This file is about running it on a cluster.

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
   Set to your GPU's compute capability (A100 = `8.0`, RTX 4090 = `8.9`, …).
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
export TORCH_CUDA_ARCH_LIST=9.0    # H100; change per §0.3
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

**H100 note:** ray tracing works even though H100 has no RT cores — the driver/OptiX
runs it. No `_base_task.py` edit needed to fall back to rasterization.

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

## 4. Building LeRobot / HuggingFace datasets

There are **two distinct dataset pipelines** in this repo — don't confuse them:

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

`generate.sh` calls `examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py`.
The `<repo_id>` you pick here is what you reference in the training config (§5.1).

### 4.2 Rollout eval data → HuggingFace dataset

During **evaluation** the repo can record every rollout step (3 camera views +
proprioceptive state + frame index + episode length + success) and export it as a
`datasets` dataset. This is policy-agnostic — `script/rollout_recorder.py` wraps
`TASK_ENV.get_obs`, and a policy may customize which cameras it stores via a
`rollout_export.py` (see `policy/MolmoAct/rollout_export.py`).

Enable it in the policy's `deploy_policy.yml`:

```yaml
record_rollout: true
rollout_save_dir: "eval_result/rollouts/<name>"   # deterministic output location
build_hf_dataset: false   # on a no-internet compute node, only RECORD here
hf_success_only: false
hf_repo_id: null
```

Then build (and optionally push) the dataset from a **login node** afterwards:

```bash
python script/build_hf_dataset.py \
    --rollout_dir eval_result/rollouts/<name> \
    --output_dir  eval_result/rollouts/<name>/hf_dataset \
    [--push_to_hub <repo_id>] [--success_only]
```

(`script/eval_policy.py` will also auto-build at the end of eval if
`build_hf_dataset: true` and internet is available.)

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
`pi0_fast_aloha_robotwin_lora`, `pi0_base_aloha_robotwin_full`. To fine-tune on
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
`ckpt_path: NatashaYang/molmoact2-no-depth-aloha-loraft`.

> Note the hardcoded `repo_root`/`molmo_root`/`python` paths in
> `cluster/convert_molmoact_checkpoint.sh` — update them on a new cluster.

---

## 6. Inference / evaluation

Evaluation is driven by `script/eval_policy.py`, configured by each policy's
`policy/<name>/deploy_policy.yml`, and launched by that policy's `eval.sh`. The eval
loop does an expert-feasibility check per seed, then runs the policy, optionally
logs video (`eval_video_log`) and records rollouts (§4.2). Results go to
`eval_result/<task>/<policy>/<config>/<ckpt>/<timestamp>/` (`_result.txt`,
`_episode_results.csv`, videos, and `rollouts/` + `hf_dataset/` if recording).

### 6.1 Via the cluster rollout job (recommended)

```bash
sbatch cluster/rollout.sh <policy_name> <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
```

`cluster/rollout.sh` sources `setup_env.sh`, sets up headless rendering, then
`cd policy/<policy_name> && bash eval.sh ...`.

### 6.2 π0.5 directly

```bash
cd policy/pi05
bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id>
```

`eval.sh` activates `policy/pi05/.venv` and calls `eval_policy.py` with
`deploy_policy.yml`. Tunables in `policy/pi05/deploy_policy.yml`:
`checkpoint_id` (default 30000), `pi0_step` (action chunk), `instruction_type`, and
the optional reward-critic guidance (`critic_name`, `critic_checkpoint_path`,
`guidance_strength` — see `rewards/critic.py`; guided inference nudges predicted
actions along a critic gradient).

> `policy/pi05/eval.sh` currently prepends a Fir-workstation CUDA path
> (`/home/natasha/miniconda3/envs/cuda128/bin`) for curobo — remove/adjust it on a
> new cluster.

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

---

## 7. Cluster gotchas (Fir-derived, worth re-checking anywhere)

- **Dummy opencv**: Alliance's CVMFS profile exports `PIP_CONFIG_FILE` pointing at a
  wheelhouse with a stub `opencv` that breaks `pip install`. `setup_env.sh` unsets
  it (and `PYTHONPATH`). Keep this if your cluster injects a Python env via a module
  system.
- **No `sudo`**: Vulkan/ffmpeg come from the module system / CVMFS, not `apt`.
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

---

## 8. Claude Code setup & state sync (this fork)

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
  `skills/` would *not* be picked up): `sync-claude` (both stores), `sync-conversations`,
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
```

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

## 9. Quick reference

```bash
# --- setup (login node) ---
source setup_env.sh                     # every session/job

# --- claude state sync (login node) ---  (or run the /sync-claude skill)
bash .claude/hooks/sync-conversations.sh sync   # transcripts: bidirectional pull+push
bash .claude/hooks/sync-memory.sh sync          # memory:      bidirectional pull+push

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

# --- build HF rollout dataset (login node, after recording) ---
python script/build_hf_dataset.py --rollout_dir eval_result/rollouts/<name> --push_to_hub <repo_id>
```
