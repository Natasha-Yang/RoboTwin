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
currently run on **Rorqual** (**H100**, account `def-florian7_gpu`, repo root
`/project/6028519/natashay/RoboTwin`). The sections below give the concrete cluster setup
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
   `grep -rn "natashay/RoboTwin\|natashay/miniforge3" setup_env.sh cluster/ policy/`
   (and `grep -rn 6028519` for the allocation id itself).
2. **SLURM account + partition** — `--account=rrg-florian7_gpu` and
   `--gpus-per-node=h100:1` in `cluster/robotwin_gpu.sh` and `cluster/finetune_pi05.sh`.
3. **GPU arch** — `TORCH_CUDA_ARCH_LIST` in `setup_env.sh` is `9.0` (H100 / sm_90).
   Set to your GPU's compute capability (A100 = `8.0`, L40S / Ada = `8.9`, …).
   This is **not** just a JIT hint: curobo and pytorch3d are compiled ahead of time
   against it, so changing GPU generation means clean-rebuilding both (§1.2) — a
   binary built for `sm_90` will not run on an L40S, and vice versa.
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
export TORCH_CUDA_ARCH_LIST=9.0    # H100 (sm_90); change per §0.3
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

**The peg-insertion socket is generated, not downloaded.** `assets/*` is gitignored, so the
`121_peg-socket` mesh used by `insert_peg_socket_{loose,med,tight}` (§3.5) is produced by a
checked-in script rather than shipped. Run it once per clone / cluster, on a login node or
anywhere — it needs no GPU:

```bash
python script/gen_peg_socket_asset.py --verify
```

`--verify` loads each variant in a headless SAPIEN scene and asserts the geometry survived.
That check is not optional paranoia: SAPIEN's actor builder swallows collision-shape cook
failures in a bare `except RuntimeError: continue`, so a mesh that fails to cook yields an
actor with **zero collision shapes and no error message** — the peg would pass straight
through the socket and the task would report success. The task module also raises at import
if the asset is missing, because `create_actor` merely prints a warning and returns `None`
while the seed-search loop in `collect_data.py` is uncapped — a missing asset would
otherwise spin forever instead of failing.

### 1.5 Activate for every session / job

```bash
source setup_env.sh
```

This unsets the CVMFS `PYTHONPATH`/`PIP_CONFIG_FILE`, `module load ffmpeg/7.1.1`
(see below), activates the conda env, sets `TORCH_CUDA_ARCH_LIST`, and installs the
Vulkan shim. **Every** SLURM job script sources it (`cluster/robotwin_gpu.sh`,
`submit_all_data.sh`).

**ffmpeg (do NOT build from source).** The π0.5 doc's §1.1 tells you to compile
ffmpeg 7.1 — but that step only exists as a fallback for when `uv sync` fails
building **PyAV (`av`)** ("if error occured while build av, you should update
ffmpeg"). On Alliance clusters you don't build anything: `module load ffmpeg/7.1.1` gives
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
| `cluster/robotwin_gpu.sh` | Generic single-H100 GPU job. `sbatch cluster/robotwin_gpu.sh <cmd...>`; no args → render smoke-test. This is also how you launch an eval or a rollout collection (§6, §7). |
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

**`env_seed` — one scene for a whole run.** Every episode is normally seeded by its own
`seed`, and *everything* random is drawn from that one stream: not just where the objects
land but what the room looks like. `env_seed` (top-level in the task config, next to
`episode_num`) splits the two. With it set, the draws that decide **scene appearance** —
wall and table texture, directional/point light colors and the crazy-light coin flip, table
height, head-camera jitter — come from a generator seeded with `env_seed` instead, rebuilt
identically at the start of every episode, while `load_actors` keeps drawing object poses
from the episode's own `seed`. So a run holds the environment fixed and varies only the task
objects, which is the comparison you usually want when the background is randomized.

| | drawn from |
|---|---|
| wall / table texture, `clean_background_rate` flips | `env_seed` |
| directional + point light colors, `crazy_random_light_rate` flip, and the per-frame crazy-light jitter | `env_seed` |
| `random_table_height` → `table_z_bias` | `env_seed` |
| `random_head_camera_dis` jitter | `env_seed` |
| object poses (`load_actors`), cluttered-table objects | the episode `seed` |

Clutter is deliberately on the episode stream: `get_cluttered_table` rejection-samples
against `prohibited_area`, which moves with the task objects, so it could not be held fixed
even in principle.

`null` (the default everywhere) is the original behaviour to the byte — with no `env_seed`
the generator *is* `np.random`, so the draw order and every value are unchanged. Setting it
does shift the global stream, since the scene draws no longer consume it, so a given episode
`seed` places objects differently than the same seed would without `env_seed`. That is
inherent to splitting the streams; it means an `env_seed` run is not comparable episode-by-
episode with a non-`env_seed` one, only within itself.

**It lives in the task config and nowhere else**, and that is the point: `collect_data.py`,
`eval_policy.py` and `collect_dataset.py` all read the same key out of the same file (`args`
→ `_init_task_env_`), so collecting demos and then evaluating under one task config puts the
policy in the environment its demonstrations were recorded in. No driver has a flag of its own
that could drift from it — there is deliberately no `--env_seed`, and no key in
`deploy_policy.yml` / `collect_dataset.yml`. Both banners print the value in force.

**`env_seed` fixes *which* texture is drawn, not which pool it comes from** — that is
`domain_randomization.background_texture_pool`, and it is the one thing you must set as well to
make an eval render the same background as collection. `create_table_and_wall` normally draws
from `assets/background_texture/seen/` when collecting and from `unseen/` under `eval_mode`,
RoboTwin's held-out split, to test generalisation to backgrounds the policy never trained on.
The two pools are not even the same size (10000 vs 1000), so the same generator state cannot
name the same file across them.

| `background_texture_pool` | collecting | eval |
|---|---|---|
| `null` (default) | `seen/` | `unseen/` — the held-out split, untouched |
| `seen` | `seen/` | `seen/` |
| `unseen` | `unseen/` | `unseen/` |

Naming a pool applies it to **both** sides, which is what makes one `env_seed` pick the same
file on each. Anything else raises at startup rather than falling back. Both banners print the
resolved pool (through the env's own `resolve_background_texture_pool`, so they cannot disagree
with the scene), and the eval banner additionally flags the one combination that surprises: a
pinned `env_seed` with the split still in force, where the background differs from the demos'
even though everything else matches.

Measured, `beat_block_hammer` at `env_seed: 7`, collect path vs eval path:

| `background_texture_pool` | wall / table texture |
|---|---|
| `null` | collect `seen/6015`,`seen/8583` vs eval `unseen/895`,`unseen/391` — **differ** |
| `seen` | `seen/6015`, `seen/8583` on both sides — match |
| `unseen` | `unseen/895`, `unseen/391` on both sides — match |

Table height, every light color, the crazy-light flip and the head-camera pose match at equal
`env_seed` in all three rows — the pool is the only thing this key decides. So `env_seed: 7`
plus `background_texture_pool: seen` makes the eval environment the collection environment
exactly; `env_seed` alone holds everything but the background fixed.

`collect_data.py` additionally writes an `env_seed.txt` marker into the run's `save_path` and
**refuses to start** if a later run into that same directory asks for a different `env_seed`
(`check_env_seed_marker`). Collection resumes from `seed.txt` and replays *cached* joint
trajectories, and `env_seed` moves the table under them — the result would be a directory of
demos that silently disagree with each other rather than a crash. A directory collected before
the marker existed has none and is left alone.

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

### 3.4 What ends up in the demo HDF5

The per-episode HDF5 mirrors whatever `get_obs` returned, so the task config's `data_type`
block decides its columns (`envs/utils/pkl2hdf5.py` infers the layout from the first frame —
nothing is enumerated per data type). One row per saved frame, i.e. every `save_freq` physics
steps. `rgb: true` / `depth` / `pointcloud` / `endpose` / `qpos` / the segmentations all flow
through that path automatically.

**SigLIP features are not collected here and cannot be.** They are π0.5's own image-tower
output, so they exist only where a policy is loaded — that is `script/collect_dataset.py`
(§7), not `script/collect_data.py`. They are also the single most expensive column in that
pipeline (~576 KB/row per view).

**The contact wrench is the one exception to the `get_obs` rule**, because contacts are a scene
query rather than an observation. With `data_type.wrench: true`:

| HDF5 path | Shape | Contents |
|---|---|---|
| `wrench/<link>` | `(num_frames, save_freq, 6)` float32 | one group member per end-effector link — aloha gives `fl_link7`, `fl_link8`, `fr_link7`, `fr_link8` |
| `wrench/<arm>` | `(num_frames, save_freq, 6)` float32 | `left` and `right`, each the sum of that arm's links |

Both granularities, from the one contact query (§6.1). Each row is the trace of the `save_freq` primitive
steps that led up to that frame, one `[Fx, Fy, Fz, Tx, Ty, Tz]` sample per step, world frame,
torque about that arm's TCP. `_base_task._log_step_wrench` samples after every `scene.step()` of
`take_dense_action` / `together_move_to_pose`, and `_take_picture` drains the log while building
the frame — so this is exactly the `pi0_step`-rate column §7a records, at the demo cadence.
Short traces are **NaN**-padded (zero is a meaningful reading): a motion segment's first frame
has no steps behind it and carries a single sample of the contact state at that instant, its
last frame carries however many steps ran since the previous one.

Two consequences worth knowing:

- The HDF5 keeps the **trace**, one sample per physics step; the LeRobot conversion is where it
  becomes one averaged `(6,)` per frame — a demo frame is one primitive step, so that is the row
  a critic's `wrench.*` modality is windowed out of (§4, §5b). Nothing on this path averages, so
  the raw trace stays recoverable from the HDF5.
- Logging is gated on `save_data`, so it costs nothing during the seed-search phase, whose
  trajectories are thrown away. It does add a `scene.get_contacts()` scan to **every** physics
  step of the replay phase, which is not free on a CPU-bound collection run — turn
  `data_type.wrench` off if you don't want the columns.
- A config with `save_freq: null` has no frame cadence to stack against, so the wrench is left
  out rather than stored ragged.

### 3.5 The peg-insertion ladder (`insert_peg_socket_{loose,med,tight}`)

Three registered tasks added by this fork, and the only **clearance fit** in the task set —
every shipped RoboTwin task is a pick-and-place, hang, press or stack, where contact force is
incidental. Here it is the signal, which is what makes it the task to point a `wrench.*`
critic at (§5a, §6.1).

One arm grasps a standing 40 x 40 x 120 mm peg (a `create_box` primitive, so its contact and
functional points come for free) and inserts it into a static socket's chamfered square blind
bore. The socket is the generated asset `121_peg-socket` (§1.4); the three tasks are thin
subclasses of `envs/_peg_insertion_base.py` differing only in `socket_model_id`:

| task | model_id | clearance/side | 45° lead-in | capture radius | expert yield |
|---|---|---|---|---|---|
| `insert_peg_socket_loose` | 0 | 6.0 mm | 14 mm | 19 mm | 10/14 |
| `insert_peg_socket_med` | 1 | 3.0 mm | 6 mm | 8 mm | 11/14 |
| `insert_peg_socket_tight` | 2 | 1.5 mm | 2 mm | 3 mm | 11/14 |

**Both numbers have to scale, and the lead-in is the one that matters.** The first revision
varied only the bore and held the chamfer mouth fixed: all three rungs then had the same
~15 mm capture, a released peg self-centred on the chamfer regardless of bore, and the three
tasks produced *byte-identical* expert results. Clearance alone only bites once the peg is
already aligned. Capture radius — measured by releasing a peg 5 mm above the mouth at
increasing lateral offset — is the error budget an imprecise agent actually has, and it is
what the ladder varies (19 / 8 / 3 mm, a 6x spread). `tight` is additionally impossible to
insert at 15° of yaw, where `loose` still tolerates 15 mm of offset.

The expert is deliberately *not* the thing that separates: yields are within noise of each
other, and the residual failures are grasp-phase plan failures on the same seeds for all
three rungs. That is the point — every rung yields demos at a usable rate, and eval's
expert-feasibility gate does not select seeds differently per rung, so a policy's success
rate across the three is comparable.

Two implementation details that are load-bearing, both in
`envs/_peg_insertion_base.py::insert_peg`:

- **The descent is built by hand, not with `place_actor`.** `place_actor` emits bare
  `Action(arm, "move", ...)` with no `constraint_pose`, and the planner is a trajectory
  optimizer that knows nothing about the peg or the socket, so it is free to bow and rotate
  mid-path. The plan here passes `constraint_pose=[1,1,1,0,0,0]` (orientation held, position
  free), the same mask `grasp_actor` uses for its own final approach.
- **Two-stage approach.** A waypoint 15 mm above the mouth splits a 100 mm constrained
  descent into 55 + 45 mm. Measured: this cut the expert's placement error from ~3 mm to
  ~1.3 mm and took `tight` from 4/14 to 11/14. Without it the tight rungs are limited by
  planner drift rather than by the tolerance being studied.

`constrain="align"` is given the bore's **four** symmetry axes rather than the default single
one. With `align_axis=None` the peg's +X is forced onto the socket's +X, which for a 4-fold
symmetric peg is up to a 90° wrist swing for a geometric no-op — and near 180°
`get_align_matrix` hits its `||v1 x v2|| < 1e-6` branch and silently returns identity, i.e.
no correction at all.

`step_reward()` gives a critic shaped progress as a sum of four terms: a **one-time 0.1** the
first time the peg is correctly grasped (gripper commanded closed, in contact with the peg, on
its upper half, peg still upright), plus three clipped deltas — closing on the **bore axis**,
depth **into** the bore, and uprightness **while inserted**. Every delta is symmetric, so undoing
progress refunds it and nothing ratchets. Three details carry it:

- **Depth only counts inside the bore** (tip within `ALIGN_RADIUS` = the 10 mm `SUCCESS_LATERAL`
  of the bore axis). `depth` on its own is the signed distance below the mouth *plane*, which
  spans the whole table, so a peg standing anywhere beside the socket reads a full bore's worth
  — which used to make the shaping penalise lifting the peg and pay for setting it back down.
- **Approach is the lateral offset only** — height is deliberately left out, so lifting the peg
  is worth exactly 0 rather than reading as moving away from the mouth. The cost is a dead zone:
  the descent from the pre-insert waypoint down to the mouth plane earns nothing, since the depth
  term does not switch on until the tip is inside the bore.
- **Uprightness is only live inside the bore**, and its previous value is dropped on the way out,
  so re-entering measures from the value on entry rather than paying the whole cosine at once.

`check_success` is unchanged: depth > 30 mm of the 38 mm bore, which is unreachable outside the
hole.

**Reading the wrench on this task — the two families mean different things, and the arm sums
are the ones that see the insertion.** The fingers squeeze in opposition, so grip preload
cancels in the arm totals but not in the links. Measured over 5 demos per rung:

- `wrench/<link>` sits at a flat **~29 N plateau for the whole carry**, identical across all
  three rungs — that is the gripper holding the peg at a constant commanded position, not
  contact with the socket. It swamps the insertion.
- `wrench/<arm>` (the sum) is near zero while carrying and rises only on **external** contact.
  Peak over an episode: **0.6 N (loose) / 5.8 N (med) / 23.5 N (tight)**; within the insertion
  window alone, **0.01 / 3.8 / 19.0 N**. A ~40x monotonic spread, which is the signal the
  ladder exists to produce.

So `loose` is effectively a **contact-free control condition** — at 6 mm clearance the peg never
touches the bore — while `tight` binds hard (one demo peaked at 97 N). A critic given only
`wrench.{fl,fr}_link*` will mostly see grip preload; give it `wrench.{left,right}` too, or
instead, if what you want is the interaction force.

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

**This path keeps rgb + qpos only.** Depth, point clouds, endposes, camera matrices and the
contact wrench are dropped by `process_data.py` before the converter ever runs, so a dataset
built this way can train the policy but cannot co-train a critic on anything the critic senses
(§5d). The **multimodal** converter reads the collected HDF5 directly and carries all of it
(`examples/aloha_real/convert_robotwin_multimodal_to_lerobot.py`, on the `killarney` branch —
it is what built `robotwin_demo_{clean,randomized}_multimodal_50x10_lerobot`).

One convention it does **not** yet apply: a demo HDF5 holds the wrench as one `(save_freq, 6)`
physics-step trace per frame (§3.4), and a demo frame is one primitive step, so the dataset
column should be that trace's **average** — one `(6,)` per frame, the same quantity a rollout
dataset stores per primitive step. `script/demo_wrench_per_control_step.py <repo_id>` collapses
an already-built dataset in place (nan-aware, idempotent, atomic per episode; it also fixes
`meta/info.json` and the per-episode stats). Run it after converting, or let the reader average
on the fly — `demo_retrieval` handles both layouts — but the stored form is what an offline
trainer reading the columns directly will see.


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
             [guidance_scale] [guidance_ramp_updates] [train_online] [use_step_reward] [best_of_n] \
             [critic_config_path]
# baseline (no critic guidance)
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0
# online critic-guided
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0 0.3 256
# ... with the shaped step reward off (sparse success reward only)
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0 0.3 256 "" false
# best-of-8 selection with no gradient guidance
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0 0 "" "" "" 8
# both: every candidate steered, the best steered chunk executed
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0 0.3 256 "" "" 8
# DSRL: the sampler is untouched and a SAC actor picks the noise it denoises from (§5c)
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0 \
             "" "" "" "" "" /home/natasha/multisensory-steering/cfgs/dsrl.yaml
```

- `eval.sh` activates `policy/pi05/.venv` and calls `script/eval_policy.py` with
  `policy/pi05/deploy_policy.yml`.
- `deploy_policy.py` → `pi_model.py::PI0` loads the trained policy from
  `policy/pi05/checkpoints/<train_config_name>/<model_name>/<checkpoint_id>` and runs
  inference. `checkpoint_id` (default 15000) and `pi0_step` (how many of the 50 chunk
  steps are executed per call, default 10) come from `deploy_policy.yml`.
- Camera → model mapping (in `deploy_policy.py::encode_obs` / `pi_model.py`):
  `head_camera → cam_high`, `left_camera → cam_left_wrist`, `right_camera → cam_right_wrist`.
- Results land in `eval_result/<task_name>/<policy_name>/<task_config>/<ckpt_setting>/<timestamp>/`.
  Alongside `_result.txt` / `_episode_results.csv`, each run snapshots `deploy_policy.yml` and the
  `critic_config_path` file it includes into that dir (`script/eval_policy.py::snapshot_config`).
  The copies keep their comments but carry the values **actually used** — `eval.sh`'s positional
  args are written in, so `task_name`, `seed`, `guidance_scale` etc. read as resolved rather than
  as the `null`/default in the source yml.
- `use_step_reward` (`deploy_policy.yml`, or `eval.sh`'s 10th arg) picks **which reward the run
  scores itself with** — what `online_critic.commit()` is fed, what the W&B reward curves and the
  `reward` column of `_episode_results.csv` measure. `true` (default) is 1.0 on the control step
  that first reaches success plus the task's own shaped progress term (`step_reward()`, a delta —
  `envs/lift_pot.py`, `envs/open_microwave.py`, `envs/put_object_cabinet.py`); `false` is sparse:
  1.0 on success and 0.0 everywhere else, even for a task that defines shaping (`step_reward()`
  is then never called at all, so its own delta state never advances). Unlike the guidance knobs
  this applies to the plain baseline too. `script/eval_policy.py::control_step_reward` is the one
  place it acts, so a guided run's TD targets and the printout agree by construction; the
  equivalent for a collected dataset's `reward` column is §7a, which is always shaped.

#### Crash recovery

An eval run is hours long and can end early two ways: it can crash, and it can wedge.

> **The wedge, for reference.** The renderer hangs: `SapienRenderCameraInternal::waitForRender`
> → `libnvidia-eglcore` → `poll()`, a render fence that never signals, with the GPU completely
> idle and every thread asleep. It surfaces as a run that neither progresses nor dies. Diagnose
> with `sudo py-spy dump --pid <pid> --native` (needs root: `ptrace_scope=1`). `_base_task.py`
> uses the `rt` shader with the **`optix`** denoiser (`script/test_render.py` uses `oidn` for
> the same block), which is the first thing to vary if it recurs. Nothing detects this
> automatically — a hung run has to be noticed and killed by hand.

Either way the episodes already finished are safe. Every one is committed to the run dir in
strict order — its row
appended to `_episode_results.csv`, the critic checkpointed, then `resume_state.json` written
atomically. The json is the commit marker, so an interruption rewinds to the last episode that
completed all three; a csv row one ahead of it is trimmed on resume.

`resume: true` (`deploy_policy.yml`) then continues the newest interrupted run for this
task/policy/config/ckpt **in its existing directory** rather than opening a new timestamped one.
`resume: "<run dir>"` names one specific run instead — needed when a second eval is writing into
the same directory, since it rewrites its own `resume_state.json` every episode and so stays the
"newest" one however long ago the run you meant stopped.

| Restored | From | Why it cannot be recomputed |
|---|---|---|
| `now_seed`, `now_id`, `suc_test_seed_list` | `resume_state.json` | which seeds the expert check rejected is not a function of the episode index |
| global numpy RNG state | `resume_state.json` | the episode's instruction is drawn with `np.random.choice` |
| `test_num`, `suc`, `chunk_count` | `resume_state.json` | counters the guidance ramp and MA windows key off |
| episode rows + MA windows | `_episode_results.csv` | reloaded so the averages continue across the break |
| critic params, target, **Adam state, LR schedule position** | `online_value_critic.pkl` | see below |
| guidance ramp position | `critic_ramp_baseline` in `resume_state.json` | see below |
| best-checkpoint bar | `best_success_rate_ma` in `resume_state.json` | otherwise the resumed run's first episode overwrites a better `online_value_critic_best.pkl` (§5a) |
| W&B run | `wandb_run_id` → `resume="allow"` | keeps the critic curves one continuous series |

The optimizer half needed a change in `multisensory_steering`: `OnlineValueCritic.save` now
records `opt_state` and `lr_step`, and restores them when `restore_optimizer` is set — which
`eval_policy.py` sets only on a resume. A plain warm start from `critic_ckpt` still gets a fresh
optimizer, which is the right default for pointing an offline critic at a new task but was
silently wrong for resuming: the run came back at the *bottom of warmup* (lr 0) and climbed the
schedule again. The state is checked against the optimizer actually built (`clip_grad` and
`freeze_encoder` change its tree) and dropped with a message rather than crashing if it no
longer fits; checkpoints written before this load exactly as they did.

The **guidance ramp** had the same shape of bug one level up. A resume points `critic_ckpt` at
the run's own `online_value_critic.pkl`, and `PI0._init_critic` re-bases the ramp at whatever
update count a checkpoint restores (§5a — deliberate for an offline critic, so a warm start eases
in like a fresh one). Applied to a run's own checkpoint that means the ramp restarts: a run that
had reached full `guidance_scale` comes back at **0** and climbs the whole
`guidance_ramp_updates` again. So `resume_state.json` now carries `critic_ramp_baseline` — the
count *that* run's ramp was measured from (0 when its critic was trained from scratch here) — and
`eval_policy.py` hands it back through `deploy_policy.py` as `PI0(critic_ramp_baseline=...)`,
which uses it instead of re-deriving one. The startup banner prints where the ramp resumes. It is
not a user-facing config key: nothing but the resume path sets it, and it is applied only when
the critic checkpoint is actually reloaded (against a critic starting at 0 updates it would pin
guidance at 0 instead). State files written before 2026-08-08 have no such key: a run whose
critic trained from scratch is reconstructed exactly (its baseline was 0), one that warm-started
from an offline `critic_ckpt` cannot be, and keeps the old restart-the-ramp behavior with a
printed note.

**Not** restored: the replay buffer (gigabytes of SigLIP features, §5a), so a resumed critic
keeps its weights and optimizer but refills the buffer from empty and runs no TD update until
`start_training` transitions are back in it.

`_episode_results.csv` gained explicit `episode` and `seed` columns and lost its unnamed index.

**A run started before this landed can still be resumed**, because its job log already recorded
per episode everything `resume_state.json` holds: the `Success rate: <suc>/<test_num> … current
seed: <n>` line is the episode counter, the success count and the seed, and the seeds that reach
that print are exactly the ones the expert check accepted. `script/resume_state_from_log.py
<slurm_log> <run_dir>` parses them back into a `resume_state.json` (and reconstructs
`_episode_results.csv`), after which `resume` continues the run normally. Run it once the job has
actually exited, and only for a run with **no critic** — a critic's weights are not in any log, so
the reconstructed `chunk_count`/`critic_updates`/`critic_ramp_baseline` are zeros, true only when
`guidance_scale` was 0 and `best_of_n` 1. Two of the columns are inexact and neither is load-bearing:
`num_steps` is the last step the episode printed, and `reward` is recovered by inverting the printed
moving average (`r_n = S_n - S_{n-1} + r_{n-window}`), which accumulates the print's 3-decimal
rounding. The numpy RNG state is the one field genuinely not in the log, and it does not matter:
`_init_task_env_` reseeds numpy from the episode's own seed at every `setup_demo`, so nothing
downstream depends on the state the loop carried in. (The instruction is drawn by
`generate_episode_descriptions` from the `random` module, which nothing seeds and
`resume_state.json` does not carry — so instructions already differ across any resume.)

Setting `debug: true` in the **task config** turns on extra per-episode diagnostics
(`script/eval_policy.py::visualize_debug_obs`), all written into `debug_vis/episode<N>/`
under that same result dir:

| Output | File (under `debug_vis/episode<N>/`) | Notes |
|---|---|---|
| Per-link wrench histograms | `wrench_hist_episode<N>.png` | one histogram per component (Fx/Fy/Fz/Tx/Ty/Tz), every gripper link overlaid (aloha: `fl_link7`, `fl_link8`, `fr_link7`, `fr_link8`) |
| Rollout + wrench GIF | `wrench_episode<N>.gif` | head camera on the left with the **world** axes drawn as labelled x/y/z arrows, projected into the camera and anchored at each arm's TCP (the axes the components are resolved in, at the point they act); one trace column per gripper link with a step cursor on the right. One GIF frame per policy call; the traces behind the cursor carry every primitive step, `pi0_step` of them per call |
| Raw series | `wrench_episode<N>.npz` | `step` (the control step each row belongs to, one per primitive step), `control_step` / `rows_per_call` (the policy calls the rows were drained at, and how many each drained), `components`, `links`, plus one `(num_rows, 6)` array per link name |
| Critic Q trace | `q_episode<N>.png` | guided runs only (see below) |
| Rollout + Q GIF | `q_episode<N>.gif` | head camera on the left, the Q and reward traces with a step cursor on the right |
| Raw series | `q_episode<N>.npz` | `step`, `q` `(num_samples, num_qs)`, `reward`, `guidance_scale`, `return_to_go`, plus the scalars `return_mean` / `return_std` / `gamma_h` |
| Retrieval phase plot | `retrieval_episode<N>.png` | retrieval runs only (§5b): retrieved demo frame + match similarity vs rollout step |
| Rollout + retrieved demos GIF | `retrieval_episode<N>.gif` | head camera on the left, the top-`debug_top_k` demo head frames it matched on the right |
| Raw series | `retrieval_episode<N>.npz` | `step`, `bank_index`, `distance`, `demo_episode`, `demo_frame`, `bank_frames` |
| Depth / segmentation tiles, point clouds | `obs_step<S>.png` / `pcd_step<S>.ply` | only for the modalities the task config's `data_type` enables |

Both GIFs animate the same rollout, so `envs/utils/debug_vis.py::RolloutFrameLog` downscales
and holds the head-camera frames once for all of them, keyed by sim step. That module holds
both recorders (`TCPWrenchRecorder`, `QValueRecorder`) and the plotting/GIF plumbing they
share; `eval_policy.py` keeps `visualize_debug_obs` and the code that constructs, feeds and
flushes them.

The wrench is computed by `envs/utils/wrench.py` (shared with the rollout-dataset collector, §7a).
It is the **net contact wrench on the end-effector links** (wrist link + gripper
fingers + any `fix_gripper_name` links), summed from `scene.get_contacts()` impulses divided
by the sim timestep, with torque taken about the TCP origin. Both vectors are resolved in the
**world** frame (N and N·m) — only the moment arm is TCP-relative, so a trace stays comparable
across steps as the gripper rotates. It is contact-only: an arm moving through free space reads exactly
zero — this is not a joint-torque estimate. Every consumer sees it at the same
**primitive-step** rate (below), the debug plots included: the recorder drains the `pi0_step`
rows the previous chunk left and pairs them with that call's head-camera frame, so the x axis is
exact `take_action_cnt` and the frames, the world-axis overlay and the GIF cursor sit on it.
Under a task config with `data_type.wrench` off there is no log to drain and the recorder falls
back to a single instantaneous reading per policy call.

It is recorded at **two granularities at once**, from the one contact query
(`wrench_vectors`): one `(6,)` per gripper **link**, and one per **arm** that is exactly the sum
of that arm's links. The links are what keep a finger squeezing against its opposite — equal and
opposite forces that cancel in the arm total — visible; the arm totals are the coarser
two-signal version. `_base_task._log_step_wrench` logs both after **every `scene.step()`** and
everything downstream just names the keys it wants, so the demo HDF5's `wrench/<key>` groups
(§3.4), the rollout dataset (§7a) and a critic (§5a) all carry or can name either family. The
link labels are the embodiment's URDF link names, so **which** `wrench.<link>` keys exist
follows the robot: aloha agilex gives four (`fl_link7`, `fl_link8`, `fr_link7`, `fr_link8`),
plus `left` and `right`. `ee_link_labels` is the link → arm grouping; `compute_tcp_wrench` /
`tcp_wrench_vector` are the arm half on their own, `link_wrench_vector` the link half.

**The log holds one row per primitive step**, in all three control loops that run a recorded
trajectory, so a demo HDF5, a rollout dataset and the critic's live view hold the same kind of
trace. The contact query itself still runs after **every** `scene.step()` — 1/250 s of simulated
time — but what a row records is their **time average**:

- In the expert loops (`take_dense_action`, `together_move_to_pose`) one iteration *is* one
  `scene.step()`, so the two rates coincide and a saved frame still carries exactly `save_freq`
  rows, one per physics step. Nothing changed on the demo path.
- A single `take_action` instead runs a whole TOPP-interpolated trajectory — measured on this
  machine at **34–196** physics steps depending on the size of the joint delta (median 94 for a
  0.02 rad chunk step, 45 at 0.005, 146 at 0.05) — and `_close_step_wrench` commits **one** row
  for it: the total contact impulse the step delivered divided by the time it took
  (`sum(J_i) / (n · dt)`, which with a fixed timestep is exactly the mean of the per-step
  wrenches). Dividing by the elapsed time rather than by `dt` is what makes rows comparable —
  an un-normalized sum would read as a bigger force for nothing but a longer move. So a
  `pi0_step: 10` chunk drains exactly **10** rows.

That is why a brief contact reads as a fraction of its peak: 10 of a step's 40 physics steps at
5 N is a row of 1.25 N. The peak is not recoverable from a row — this is an average force, not
an envelope.

The drain is stacked to a **fixed** `wrench_trace_len` (`deploy_policy.yml` /
`collect_dataset.yml`, **`pi0_step`** — the drivers derive it from `pi0_step` when the key is
absent), NaN-padded when a chunk was cut short by success or `step_lim`. It has to be fixed —
the critic's obs shape and the dataset's `Array2D` column width are both settled before the
first row arrives — and it is an architecture key in all but name: a critic warm-started from
`critic_ckpt`, and any rollout dataset it was pretrained on, must have been made with the same
value. `_base_task` holds the log in a `maxlen` deque of that size, so an overlong drain has
already dropped its **oldest** rows (the ones furthest from the observation the trace is paired
with), and a run that never drains — a baseline eval has no critic, and
`critic_obs_modalities` returns before draining — cannot grow it without bound.

Cost is one contact query per physics step: `wrench_vectors` measured at **223 µs** here at 35
contacts, so ~21 ms per control step, about a quarter again on top of the ~81 ms
`sample_actions` itself takes. (Almost all of that is the Python loop over contact points —
`scene.get_contacts()` alone is 6 µs.) Turn `data_type.wrench` off if that is not worth it.

The debug plots draw the **link** half only — an arm total overlaid on its own links is just
their sum drawn twice. `analysis/plot_wrench_hist.py` plots the same histograms straight off a
collected rollout dataset, for when you want the distribution over a whole run rather than per
episode, and takes `--family links|arms|all` for the same reason.

> Changed on 2026-08-29. A rollout row's wrench trace is now `(pi0_step, 6)` — one averaged row
> per primitive step — where it used to be `(wrench_trace_len, 6)` at one sample per physics step
> (1024, mostly NaN padding). `wrench_trace_len` is unchanged as a key and still means "rows per
> drain"; only its correct value moved, from 1024 to `pi0_step`. It remains an architecture key,
> so a critic checkpoint or rollout dataset made before this cannot be mixed with one made after:
> the shapes differ, and so does what a row means. The demo-collection path (§3.4) is untouched.
>
> Changed on 2026-08-28. Datasets collected before it carry only `observation.wrench.{left,right}`.
> Those keep working unchanged: the arm columns and the `wrench.left` / `wrench.right` modalities
> mean exactly what they did, so an existing critic checkpoint still warm-starts as long as its
> `encoder_modalities` names them (it is an architecture key — a checkpoint trained on the arm
> totals cannot be pointed at the link modalities without retraining, and vice versa). What
> changed for such a checkpoint is the **encoder**: `wrench.*` now defaults to a 2-D `map_cnn`
> over the trace's (T, 6) time × component grid rather than a temporal `conv1d`, so an older one
> has to pin `encoder: conv1d` in its modality mapping to keep loading.

### 6.2 Critic gradient guidance (`guidance_scale` is the on/off switch)

> **Status on Rorqual: both dependencies are staged and installed.**
> `multisensory_steering` is checked out at
> `/lustre09/project/6028519/natashay/multisensory-steering` **and editable-installed
> into `policy/pi05/.venv`** (`pip install -e … --no-deps`) — cloning alone is not enough, since
> `pi_model.py` imports it at module scope and a *baseline* eval fails without it (§8). The
> **QMFM repo** it imports `ReplayBuffer` from (by explicit path, `$QMFM_ROOT/utils/datasets.py`)
> is at `/home/natashay/links/projects/def-florian7/natashay/QMFM`. `eval.sh` exports that as the
> `QMFM_ROOT` default — override the env var to point elsewhere. It also forces
> `WANDB_MODE=offline`: compute nodes have no internet, and the guided path opens a W&B run
> per eval, so an online `wandb.init()` times out (90 s) and can take the job down. Sync the
> offline runs from a login node afterwards (`cluster/wandb_sync.sh`). To turn guidance on,
> stage a critic checkpoint and set `guidance_scale` (or pass it as `eval.sh`'s 7th arg).
>
> Note `critic_config_path` is read **unconditionally** by `parse_args_and_config`, even
> at `guidance_scale: 0.0` — so if that path doesn't exist, *baseline* eval crashes too.
> It points at the `multisensory-steering` checkout above; repoint it when porting.

The **Q outputs need a critic**, so they appear only when `debug: true` meets a nonzero
`guidance_scale` or a `best_of_n > 1` (§5a); a baseline run prints `critic Q logging OFF` and
writes none. Each row
is one control step: `pi_model.py::PI0.get_action` scores the chunk it is about to execute with
the critic that produced it — the same normalized `(50, 14)` chunk the guidance climbed and/or
best-of-N selected, against the same observation, via `OnlineValueCritic.q_values` — and
`debug_vis.py::QValueRecorder` pairs
it with the reward that chunk earned and the guidance scale in force at the time. Unlike the
wrench it is recorded *after* the control step (the value does not exist until the chunk has
been drawn), against the step index of the observation it was drawn from, so it lines up with
the same frame. The extra critic forward per chunk is why `PI0.record_q_values` is off unless
the driver turns it on.

Both plots draw the **realized discounted return-to-go** (at the critic's own `discount **
horizon`) against Q, in return units — Q is un-normalized by the checkpoint's
`return_mean`/`return_std` first, which is the identity for a critic trained online from
scratch. Q tracking that curve is a calibrated critic; a flat Q means it is not distinguishing
the states it is steering through, a persistent gap means it over- or under-values them, and an
ensemble range that stays wide means the members disagree about states the guidance follows
anyway. The `.npz` keeps `q` **raw** (as the guidance sees it) plus the constants to convert.

### 5a. Critic-conditioned sampling (`guidance_scale` and `best_of_n`)

There is **one** eval script and **one** config. Two independent keys in `deploy_policy.yml`
decide whether a QMFM critic acts on `Pi0.sample_actions`, and **either one** on its own builds
the critic (and, by default, TD-trains it online). (A third way is to name the *other* critic
family — `critic_type: dsrl`, §5c — which steers by choosing the sampler's noise and is
mutually exclusive with both keys below.)

| Key | Off | On |
|---|---|---|
| `guidance_scale` (7th positional arg) | `0.0` | an ensemble QMFM `Value` critic steers each denoising step by gradient guidance, ramping `0 → guidance_scale` over `guidance_ramp_updates` TD updates (`0` jumps to target after the first update) |
| `best_of_n` (11th arg) | `1` | `n` candidate chunks are drawn per control step and the highest-Q one is executed |

Both off is the plain pi0.5 baseline — no critic is built, no replay collection, no TD updates,
no W&B.

#### Best-of-N selection

`best_of_n: n` draws `n` chunks from independent noise in a **single** `sample_actions` call and
executes the one with the highest ensemble-mean Q — the same aggregation the guidance ascends, so
with both switched on the selection agrees with what the steering was trying to do instead of
pulling against it. The winner is the chunk that goes into the replay buffer, the `_episode_results.csv`
reward and the debug Q log: everything downstream is about the action that actually ran, and the
losing candidates are discarded inside the sampler.

The two knobs compose but are not the same thing. Guidance moves a *single* sample toward higher Q
and can walk it off the policy's own distribution if the critic is wrong there; best-of-N only ever
returns something the frozen pi0.5 sampler drew on its own, so a bad critic costs it nothing beyond
the wasted compute — with an untrained critic it degrades to picking a candidate at random, which is
exactly the baseline. That is why guidance needs `guidance_ramp_updates` and best-of-N needs no ramp.

Cost is `n` denoising loops, not `n` policy calls: the candidates ride along as extra batch elements,
replicated **after** the SigLIP tower and the prefix pass, so those still run once per control step
and only the loop and the prefix KV cache scale with `n` (the cache is ~15 MB/candidate at this
prefix length). `guidance_scale: 0` with `best_of_n > 1` is meaningfully cheaper per candidate than
the guided path, which pays two extra forward passes per denoising step for the value gradient —
`pi_model.py` passes `guidance_scale=None` in that case so the gradient branch is compiled out
rather than multiplied by a constant zero. `best_of_n` is a static jit arg: changing it recompiles.

Per-episode W&B (`script/eval_policy.py::BestOfNRecorder`) reports `bestofn/q_gain` — the winner's Q
minus the mean over its candidates, i.e. what the selection bought over executing an arbitrary one of
them — and `bestofn/q_spread`, the range it chose over. A `q_gain` near zero means the critic cannot
separate the chunks the sampler draws and the extra `n`-fold denoising is buying nothing.


How much of the critic online TD is then allowed to move is decided by two keys in the
**critic config** (`critic_config_path`, i.e. `cfgs/qmfm.yaml` — not `deploy_policy.yml`, though
that file and the CLI can override them like any other critic key). Both only mean anything once
a critic is running:

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
so with none it would steer on the gradients of a random network (or rank best-of-N candidates by
one), and `pi_model.py` raises at startup rather than running it. The replay buffer is allocated lazily on the first stash, so a
frozen run never pays its memory either, and `save_critic` is skipped — the critic is
byte-identical to the checkpoint the snapshotted config names. `eval.sh`'s 9th positional arg
overrides `train_online` for a one-off run.

Under `save_critic`, that critic is written to `online_value_critic.pkl` in the run's result dir
**after every episode**, not once at the end — a 100-episode guided eval is many hours, and
losing all of its TD training to a SLURM time limit is the expensive failure. There is one file
per run, overwritten each time, so nothing accumulates; to resume, point the next run's
`critic_ckpt` at it (the checkpoint carries `num_updates`, so the guidance ramp picks up where
it left off — `PI0.scheduled_guidance_scale` still counts only *this* run's updates, §6.2).
`script/eval_policy.py::save_critic_atomically` pickles into a sibling `.tmp` and renames it
into position, so a kill mid-write leaves the previous complete checkpoint rather than a
truncated one — writing in place at this rate would otherwise make the interrupt this is meant
to survive destroy the checkpoint too. There is **no** separate end-of-run save: a run that
finishes got its last write from its last episode, so `main` only prints where the file is.

It is additionally written **every `critic_save_every_updates` TD updates** (default **200**;
`0` turns the mid-episode writes off), because an episode runs for up to `step_lim` control
steps and so can be hundreds of updates long — an interrupt inside one would otherwise discard
every update since the last episode boundary. The cadence counts the critic's own lifetime
`num_updates`, so it is in updates regardless of `train_freq`. This is the one file that can be
*ahead* of `resume_state.json` rather than behind it: a resume then replays the interrupted
episode's seed against a critic that already saw part of that episode, which duplicates a little
training data and loses none. Only the state file's `critic_updates` field goes stale, and
nothing reads it back — the guidance ramp is measured from `critic_ramp_baseline` against the
checkpoint's own counter, which is right precisely because those updates did happen.

**Two files, not one.** `online_value_critic.pkl` is the run's *state* — whatever the last
episode left, which is what a resume must pick up — but online TD on a few thousand correlated
transitions is not monotone, so the last episode is generally not the run's best. Alongside it,
`online_value_critic_best.pkl` holds the critic as of the episode with the highest
`success_rate_ma` (the `wandb_ma_window`-episode moving average, the same number the csv and the
W&B curve carry). That is the one to point a later `critic_ckpt` at when you want to *use* the
critic — evaluate it frozen, warm-start another run — and the latest is the one to point at when
you want to *continue* this run. Never resume from the best file: it would rewind the optimizer
to an episode `_episode_results.csv` and `resume_state.json` already count as done.

Both are written the same way, and the best one is a copy of the latest (which was pickled from
the same object a moment earlier) rather than a second pickle, through the same temp-and-rename.
Only the latest moves on the mid-episode cadence above — `success_rate_ma` is an episode-level
number, so there is nothing to rank a mid-episode critic against.
Two details of "best": while the moving-average window is still **filling** the best file just
tracks the latest — a mean over one episode makes a single early success read as a success rate
of 1.0 that no honest 20-episode average could beat, which would freeze "best" at episode 1 —
and once it is full only a **strict** improvement moves it, so a plateau keeps the earliest
critic to reach it. The bar itself rides in `resume_state.json`
(`best_success_rate_ma` / `best_success_rate_ma_episode`) so a resumed run keeps comparing
against the pre-interruption best instead of replacing it with its own first episode; a state
file written before 2026-08-29, or reconstructed by `resume_state_from_log.py`, has no such key
and the bar starts over. This applies to whichever critic family is running — `eval_policy.py`
only ever calls `online_critic.save`, so the DSRL critic (§5c) is checkpointed identically.

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

`guidance_scale` and `best_of_n` default to whatever `deploy_policy.yml` says; the positional args
only override them (pass `0` and `1` to force the baseline). The critic's own hyperparameters are **not** in
`deploy_policy.yml` — it carries `critic_config_path`, and `parse_args_and_config` merges that
file in underneath, so precedence is **CLI > deploy_policy.yml > critic_config_path**. The
critic implementation and its config both come from the `multisensory_steering` package
(editable install from `/lustre09/project/6028519/natashay/multisensory-steering`, config at
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
| `wrench.<link>` (aloha: `fl_link7`, `fl_link8`, `fr_link7`, `fr_link8`) | per-primitive-step contact wrench per end-effector link, logged by the env | `(wrench_trace_len, 6)` each |
| `wrench.<arm>` (`left`, `right`) | the same reading summed over that arm's links | `(wrench_trace_len, 6)` each |
| `action_proposals` | task config `data_type.action_proposals` (§5b) | `(top_k, 50, 14)` |
| `noise_proposals` | task config `data_type.noise_proposals` (§5b) | `(top_k, 50, 14)` |

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
  that has it off — or naming a link this embodiment does not have — fails at startup.
- **The proposal modalities are retrieved, not sensed.** `action_proposals` /
  `noise_proposals` come from a bank of demonstrations rather than from the sim (§5b), so they
  are the two `data_type` flags nothing in `get_obs` produces — `script/eval_policy.py` forwards
  them to `deploy_policy.py::get_model` and `pi_model.py` fills them in per control step. They
  are also the only modalities a rollout dataset does **not** carry, so a critic that uses them
  cannot (yet) be pretrained offline.
- **Modalities are an architecture key.** They go into the checkpoint, and a warm start rebuilds
  the same encoder stack; changing the list makes an existing critic checkpoint refuse to load
  (loudly, leaf by leaf). Offline pretraining takes the same names — `multisensory_steering`'s
  `cfgs/train_offline.yaml` maps them to dataset columns under `dataset.modalities` — and a
  checkpoint only warm-starts a run whose list matches. Each modality is also stored twice per
  transition (obs and next_obs) — each `siglip.*` view is ~1.15 MB/transition (so all three come
  to ~3.5 MB), a depth camera ~0.6 MB, so `buffer_size` needs revisiting when adding one.

### 5b. Demo retrieval (`action_proposals` / `noise_proposals`)

Two more critic modalities, and the only ones that do not come from this episode: the actions a
**successful demonstration** took from a state that looks like the one the robot is in now, and
the noise seeds that would make the sampler reproduce them here.

```yaml
# task_config/<config>.yml — the switches, both default false
data_type:
  action_proposals: true    # the retrieved demo chunks
  noise_proposals: true     # the seeds that map to them (pays for the flow inversion)
```
```yaml
# policy/pi05/deploy_policy.yml — where they come from
demo_retrieval:
  repo_id: NatashaYang/robotwin_demo_clean_50_lerobot
  num_demos: 1        # demonstrations drawn per eval episode
  top_k: 1            # neighbours retrieved per control step
  bank_size: 512      # the bank is a jit argument, so it is padded to a fixed size
  num_steps: 10       # MUST equal the sampler's denoising steps
  num_inner_steps: 10
  signals: null       # what the distance ranks by; null = [siglip, state]
```

Once per **run** (on the first `PI0.set_language`): `num_demos` episodes of the task under
evaluation are drawn at random from the demo dataset and every frame of them is encoded into a
bank, which is then **fixed for the whole eval** — it is a critic input, so re-drawing it
between episodes would evaluate a value learned against one set of demonstrations on a different
set, and two episodes run from the same seed would not be comparable. One draw also makes the
choice reproducible from `seed` alone and survives a resume, since the bank is rebuilt
identically rather than restored. `DemoRetriever.select_bank` still draws explicitly for a
caller that wants a new one. A bank row is a
**pooled** SigLIP embedding per camera view (`Pi0.embed_observation_maps`, the image tower alone,
averaged over patches and left at its natural magnitude — nothing here is unit-normalized), the
normalized action chunk that was the policy's training target at that frame, the pose, and the
frame's own uint8 model inputs. All of it goes through
the policy's **own input transform**, so a bank row is in exactly the space the sampler works in
(delta-encoded against the demo's own state, normalized, padded to `action_dim`, then narrowed
back to the embodiment's 14 for the critic).

Per control **step** (`Pi0.propose_from_demos`): the live observation is embedded by the same
tower — free, since the prefix pass runs it anyway — scored against every bank row, and the
`top_k` **nearest** rows contribute their chunk as `action_proposals`. With
`noise_proposals` on, those chunks are additionally run through `Pi0.invert_actions` **under the
live observation**, giving the noise that would make *this* observation's sampler produce a
demo-like chunk. The inversion shares the prefix the retrieval already computed.

**What the critic does with the shortlist.** A critic whose proposal encoder is
`attn_proposals` (`multisensory_steering`'s default) treats the `top_k` rows as a *candidate
pool* rather than an answer: it **cross-attends** them, with the live observation as its query
and the candidates' own observations as the keys, so which demonstration the value rests on is
learned by the TD loss instead of fixed by the distance. A key is built by the critic's *own*
encoders, which means a demo frame's un-pooled SigLIP **patch map** — 0.6 MB per view even at
fp16. Those are **not** kept per bank row (that was ~0.9 GB of device memory at `bank_size: 512`
over three cameras, resident for the whole run and growing with the bank):
`DemoRetriever.critic_keys` re-runs the image tower on the `top_k` retrieved rows each control
step, which is why the bank keeps their frames. The cost moves accordingly — one extra tower
pass at batch `top_k` per control step, and a stored transition now carries its candidates'
maps (~1.2 MB/transition per key view, obs and next_obs together) instead of a `(top_k,)` row
index. `bank_size` is then free in device memory, and `top_k` / the critic's `key_modalities`
are what the replay buffer's size keys off. A critic that pools the set
(`encoder: action_proposals`) asks for none of this.

**A key can be any modality the demo dataset carries**, not just the SigLIP maps and the pose.
`DemoRetriever` reads the dataset's own schema (`LeRobotEpisodeReader.sensor_columns`) and maps
each column to the modality name the *live* observation uses, so a retrieved demo frame can
serve its own `depth.head`, `pointcloud` or raw `images.<cam>` under exactly the name the critic
encodes the query with. Which ones exist follows the converter: the rgb-only pipeline writes
three camera views and nothing else, the multimodal one adds `depth.<cam>`, `pointcloud`, the
`wrench.*` columns and a fourth `front` camera (which this fork's `get_obs` has no counterpart
for, so it stays `images.front` rather than being guessed into `images.third_view`).

| modality | where a demo row's copy comes from |
|---|---|
| `siglip.<view>` | re-encoded per control step by this run's own image tower |
| `state` | the policy's input transform, model space, narrowed to the embodiment's dims |
| `wrench.<key>` | windowed out of the dataset's per-frame rows at `wrench_trace_len` |
| `depth.<cam>`, `pointcloud`, `images.<cam>`, … | the dataset column, cast to the dtype the sim hands the critic |

The last row is **opt-in**, via `demo_retrieval.sensor_modalities` in `deploy_policy.yml`
(`null` keeps none — the behaviour before the key existed; `true` keeps everything the dataset
has). Opt-in because unlike a SigLIP map these cannot be re-encoded from something smaller: a
depth map *is* the key, so it has to be resident for every bank row, and the cost scales with
`num_demos` × episode length rather than with `top_k` — ~0.23 MB a camera frame, ~0.15 MB a
depth map, ~24 KB a point cloud. The bank banner prints the total, and a name the dataset does
not have is refused when the retriever is built rather than an hour into the rollouts.

**Shape agreement is the thing to check.** A demo row has to be the same array the sim hands the
critic online, and only the dtype is normalized here. The rgb-only demo datasets store 480×640
frames against the sim's 240×320, so their `images.<cam>` is *not* a usable key; the multimodal
ones store 240×320 and are. `pointcloud` matches only when the task config's
`pcd_down_sample_num` equals the dataset's (1024 for the multimodal converter). Nothing on this
side can check it — the online shape is not known until the first observation — so a mismatch
surfaces in the critic's encoder.

**What "nearest" means.** Not a cosine but the **average of several independent relative L2
distances**, one per signal, smallest wins (`signals`, `SIMILARITY_SIGNALS`):

| signal | vector | terms |
|---|---|---|
| `siglip` | mean-pooled patch features per camera view | one per view in `views` (3 by default) |
| `state` | the robot's pose in the model's **normalized** space, `(32,)` padded | 1 |

Each term is `‖query − row‖ / ‖query‖` — the distance as a fraction of the magnitude of the
*current observation's* own vector for that signal. That query-relative scaling is what makes
terms of very different size averageable without normalizing the magnitude away: measured on
this machine, raw distances run ~13 for `siglip.right_wrist` against ~1.6 for `state`, which
would make the pose about 7% of a plain mean; divided by the query's magnitude (~24 and ~2.5
respectively) every term lands at O(0.1–1). Nothing is unit-normalized, so unlike a cosine this
is sensitive to magnitude — two poses pointing the same way but ten times apart in size are
identical to a cosine and far apart here.

So the default is a mean of **four** distances, and every term counts the same — `state` is a
quarter of the score, not a tiebreaker. `signals: [siglip]` is vision only (`siglip` is
required; nothing else identifies the scene). The pose is read from the transformed
observation, i.e. the exact vector the sampler is conditioned on, and both sides go through the
same transform, so a pose distance compares like with like; the zero padding past the
embodiment's dims contributes nothing to it. The denominator is floored, so a query vector that
is somehow zero reads as a plain unscaled distance rather than an infinity that swallows the
average.

There is no `wrench` signal, and now for a different reason than "no demo dataset carries one".
A **multimodal** demo dataset does carry it (`observation.wrench.<key>`, one row per frame — see
below); what stops it being a *distance* term is NaN: a frame near the start of an episode has
fewer rows behind it than the trace is wide, and an L2 over NaN is NaN. It reaches the critic as
a modality instead — a bank row's own `(wrench_trace_len, 6)` trace, served by
`DemoRetriever.critic_keys` alongside the row's SigLIP maps and pose, so a cross-attending
critic keys on it even though the shortlist was not ranked by it — the same route every other
recorded sensor now takes (above). The stock rgb-only pipeline (`process_data.py` →
`convert_aloha_data_to_lerobot_robotwin.py`) still drops the wrench, so which of the two a
`repo_id` came from decides whether the modality exists; naming one a dataset does not have
raises at startup, listing what it does have.

**How a demonstration's wrench becomes a `(wrench_trace_len, 6)` trace.** The dataset stores one
`(6,)` row per **frame** — that frame's physics steps averaged, the same reduction
`_close_step_wrench` makes online (§6.1) — because a demo frame *is* one primitive step. The
trace a control step observes is then the window of those rows the previous chunk covered:
`t - wrench_trace_len + 1 … t`, ending at the row's own frame, NaN-padded at the tail when the
episode has fewer rows behind it (frame 0 carries a single sample, exactly as an episode's first
drain does online). `demo_retrieval.wrench_traces` does the windowing, for the retrieval bank and
the co-training rows (§5d) alike, at the run's own `wrench_trace_len` — which is why that key is
the one thing the width follows, not the demo's `save_freq` and not `pi0_step` separately.
`script/demo_wrench_per_control_step.py` collapses a dataset written the old way, one
`(save_freq, 6)` physics-step trace per frame, into the per-frame rows; the reader averages an
uncollapsed one on the way in as well, so both layouts reach a critic as the same array.

**What control mode a proposal is in.** pi0.5's native aloha mode, not absolute pose: the
action is a **joint-space** target (6 arm joints + 1 gripper per arm), and
`LeRobotAlohaDataConfig`'s `DeltaActions(make_bool_mask(6, -1, 6, -1))` makes the twelve
arm-joint dims a **delta from the pose the chunk is conditioned on**, leaving the two gripper
dims absolute widths. A bank row is therefore the demo's *motion away from the demo's own pose
at that frame*, and `AbsoluteActions` adds the live state back on the way out — the same
round trip the sampler's own chunk makes, so a proposal and the chunk the critic is scoring are
the same kind of object term for term. There is no end-effector pose anywhere in this path.

`DemoRetriever` does not take that on trust. It probes the chain at startup — push the same
actions through under two different states, see which dims move — and reports the answer in the
banner. If the chain delta-encodes nothing (`use_delta_joint_actions=False`, so the bank would
hold another episode's absolute joint targets), it applies pi0.5's **own**
`DeltaActions(NATIVE_DELTA_MASK)` before normalization instead, which is pinned by test to
reproduce the delta chain's bank exactly rather than inventing a second control mode.

One thing the delta does *not* absorb: its origin. A proposal is relative to the **demo's** pose
at the retrieved frame, while the sampler's chunk is relative to the **rollout's current** pose.
Measured over a `beat_block_hammer` rollout retrieved against another episode, the gap between
those two origins is 0.038 rad mean per joint (‖gap‖₂ 0.24 rad mean, 1.25 max) against a mean
chunk delta of 0.142 rad — about 27%. So a proposal says "make the motion the demo made", not
"go where the demo went", and the two differ by roughly a quarter of the motion's own size. That
is the intended reading (it is what keeps the proposal in the sampler's space), but it is the
thing to revisit if proposals ever look systematically offset.

Which demonstrations count as "the same task" is decided by RoboTwin **task**, not by
instruction string: one task expands into hundreds of instructions and the LeRobot dataset's own
`task_index` indexes instructions, so
`openpi/training/episode_selection.py::assign_episode_tasks` (shared with `episodes_per_task`
training subsets) maps each demo episode back to its task via the templates in
`description/task_instruction/` plus episode-index contiguity. A demo whose instruction reads
nothing like the eval episode's is still a valid demo of the same environment.

The reader (`openpi/policies/demo_retrieval.py::LeRobotEpisodeReader`) parses the LeRobot layout
directly rather than going through `lerobot`, because the installed lerobot refuses a dataset
written in a newer codebase version than its own — both the v2.1 (one parquet per episode) and
v3.0 (episodes share files) layouts on this machine are read.

Cost. Measured on this machine (RTX 5090, `pi05_base_aloha_lora_clean_50x25` at step 15000,
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`, `num_steps=10`, `num_inner_steps=10`):

| | per control step | peak GPU |
|---|---|---|
| baseline `sample_actions` | 81 ms | 6.50 GiB |
| `+ action_proposals` | +25 ms | 6.54 GiB |
| `+ noise_proposals`, `top_k=1` | 284 ms | 6.54 GiB |
| `+ noise_proposals`, `top_k=4` | 453 ms | 6.54 GiB |
| `+ noise_proposals`, `top_k=8` | 563 ms | 6.54 GiB |

**Memory is not the constraint** — retrieval adds ~40 MiB on top of the policy, flat in both
`top_k` and `bank_size` (the bank itself is ~10 MB, and the inversion reuses the prefix rather
than widening it much). The 0.4 memory fraction gives JAX ~13 GiB of the card's 32 GB, so there
is ~6.5 GiB of headroom left inside the pool for the critic and its replay buffer.

**Time is.** `noise_proposals` is 3.5–7x the per-step policy cost, because the inversion is
`num_steps × num_inner_steps` = 100 sequential action-expert passes. At a 170-control-step
episode that is roughly +80 s per episode. `num_inner_steps` is the lever: the fixed point is
already at ~1e-4 by 4 iterations (`pi0_invert_test.py::test_invert_actions_converges_with_more_inner_steps`),
so `num_inner_steps: 4` cuts the inversion to about 110 ms if proposal-grade noise is good
enough. `action_proposals` alone costs only the extra tower + prefix pass and is nearly free by
comparison.

**Does mean pooling actually retrieve corresponding frames?** Measured, not assumed. Query every
frame of one `beat_block_hammer` demo against a bank built from a *different* demo of the same
task, and check whether the retrieved frame's position tracks the query's:

| pooling | rank corr (Spearman) | mean phase error |
|---|---|---|
| mean-pool (what this does) | +0.999 | 1.5% of the episode |
| max-pool | +0.999 | 1.3% |
| full 256x1152 patch map | +0.999 | 1.5% |

(Those were measured under a cosine, before the switch to relative L2. Re-measured against the
same pair of episodes, raw-L2 retrieval tracks phase identically — rho +0.9989 and 2.4% phase
error against the cosine's +0.9989 and 2.6% — while agreeing with the cosine's top-1 pick on
87% of frames, so it is a genuinely different metric that is just as good at finding
corresponding frames.)

All three tie, so pooling to 1152 dims costs nothing over keeping the whole patch map — which is
the expensive alternative (256x the bytes per bank row). Two controls back this up: querying an
episode against *itself* is exact (rho 1.000, error 0), and querying a **different task's**
episode collapses (rho +0.25, error 0.50 over all three views) — so the embedding is carrying
task content, not just a clock. Head-camera-only is the exception: it keeps rho +0.84 even on
the wrong task, because the head view of an arm sweeping out and back looks similar whatever the
task. That is an argument for leaving `views: null` (all three) rather than narrowing to the
head.

Two caveats the numbers do not cover. Neighbouring ranks are separated by very little (under a
cosine, ~0.001 between rank 1 and rank 3 out of ~0.98), so the score is a *ranking* signal and a
poor confidence measure — do not threshold on it. And retrieval is phase-dominant: matched frames reliably show the same stage of
the task, but the object can be somewhere else in the scene, which is exactly what the GIF below
exists to make visible.

**Watching it during a rollout.** With `debug: true` in the task config, a retrieval run writes
`retrieval_episode<N>.{png,gif,npz}` next to the wrench and Q outputs. The png plots the
retrieved demo frame against the rollout step — a retrieval that is tracking task phase draws a
roughly diagonal line, a flat one means every step matched the same demo frame, a scattered one
means it is not tracking phase at all — with the per-rank similarity underneath. The gif puts
the rollout's head camera beside the demo head frames it matched, captioned with episode, frame
and distance, which is the only output that shows *why* a match is wrong. The distance panel
reads the opposite way round from a similarity: lower is nearer. `debug_top_k` (default
3) sets how many ranks are drawn and is independent of `top_k`: the similarity over the whole
bank comes back anyway, so showing neighbours the critic was not given is free, and the ones it
was are marked with a `*`.

Building the bank is 5 s (1 demo) to 18 s (4 demos) at `frame_stride: 1`, paid once for the
whole run rather than per episode. Encoded episodes are additionally cached by index, so an
explicit re-draw landing on the same demonstration costs nothing; the cache tops out at a couple
of MB per demo episode in host RAM.

Nothing is built unless a critic exists **and** its `encoder_modalities` actually names one of
the two: a run that turns the flags on without the critic side prints that it is turning
retrieval back off rather than paying for output nobody reads.

`Pi0.invert_actions` is the general method underneath (`pi0.py`, tested in
`pi0_invert_test.py`): given a clean chunk it recovers the exact noise `sample_actions` would
denoise into it, by solving each explicit-Euler step's implicit inverse with a fixed-point
iteration. Exact, not DDIM-approximate — replaying the recovered noise reproduces the chunk to
~4e-7 in float32. It inverts the plain sampler only, not the guided or best-of-N paths.

### 5c. DSRL — steering by choosing the noise (`critic_type: dsrl`)

The second critic family, and a different idea from §5a's. QMFM scores **action chunks** and
pushes the denoising around with a value gradient. DSRL
([Diffusion Steering RL](https://diffusion-steering.github.io/), local checkout `~/dsrl_pi0`,
`$DSRL_ROOT`) leaves the frozen policy completely alone and moves the RL problem into the
sampler's **latent noise**: the agent's action is the noise chunk `w` pi0.5 denoises from, the
environment's action is whatever the frozen policy turns `w` into, and SAC runs on `(s, w)`.

It is selected by pointing `critic_config_path` at the other config — the family is that file's
own `critic_type`, and there is no scale to set:

```yaml
# policy/pi05/deploy_policy.yml
critic_config_path: /home/natasha/multisensory-steering/cfgs/dsrl.yaml   # `critic_type: dsrl`
```
```bash
# or per run, as eval.sh's 12th positional arg (args 7 and 11 must stay 0 / 1)
bash eval.sh <task> <config> <train_config> <model> 0 0 "" "" "" "" "" \
             /home/natasha/multisensory-steering/cfgs/dsrl.yaml
```

Naming the family **is** the switch. `guidance_scale != 0` or `best_of_n > 1` alongside it
raises at startup: a Q over latents cannot score an action chunk, so there is nothing for either
of them to guide or rank with. `--overrides critic_config_path ...` is resolved *before* the
include is merged (`parse_args_and_config`), so swapping the file swaps the family and its
defaults together rather than only the recorded path.

**The noise space.** One row of it is `action_dim` wide — pi0.5's padded **32**, not the
embodiment's 14. That is the one place a latent and an action differ in width: the trailing dims
of an *action* are dead (they normalize to constant zero for aloha and §5a strips them), but the
trailing dims of a *noise* are not — all 32 go through `action_in_proj` and shape the 14 that
come out. `noise_horizon` (default **1**, dsrl_pi0's) is how many rows SAC actually chooses;
`pi0.py::hold_noise` repeats the last one out to the sampler's 50 denoising rows. That is what
keeps the policy 32-dimensional instead of 1600-dimensional, and it is why the critic config
carries `noise_action_dim` separately from `state_dim`. The actor is tanh-squashed to
`action_magnitude` (default 1.0), so the reachable latents are a strict subset of the standard
normal pi0.5 was trained to denoise — ~32% of that mass sits outside [-1, 1].

**Where the actor runs.** Inside `Pi0.sample_actions`, after the prefix pass, through the
`noise_apply` / `actor_params` hooks — the same static-function / traced-params split
`critic_apply` / `critic_params` use, so online updates need no recompile. Deliberately there
rather than on the host: the actor then conditions on *exactly* what the QMFM critic conditions
on, SigLIP maps included, which is the one thing that does not exist before the sampler has been
called. The chunk it chose comes back in the aux dict as `critic_noise`, because nothing outside
the sampler can recover it, and that is what `stash` stores as the transition's action. Tests:
`policy/pi05/src/openpi/models/pi0_dsrl_test.py`.

**What it observes** is the same registry as §5a's `encoder_modalities`, with the same names and
the same startup error for one the run does not produce. Two families get dsrl_pi0's own
treatment: `images.<cam>` through jaxrl2's pixel tower (the `/255` inside the encoder, no other
normalization) and `state` straight into the head. When those two are *all* that is configured
the networks become jaxrl2's own module objects — same parameter paths, same init keys, same
random-crop and color-jitter augmentation — and the run is dsrl_pi0's. Naming anything else (a
SigLIP map, depth, wrench) switches to `multisensory_steering`'s multimodal encoder stack and
keeps the pixel tower for the camera views.

Two costs are worth knowing before turning cameras on:

- **The pixel bottleneck is large at RoboTwin's resolution.** jaxrl2's `encoder_type='small'`
  tower is four 3x3 VALID convs at stride 2,1,1,1, so a 240x320 frame lands at 113x153x32 and
  the `Dense(latent_dim)` after it is ~27M parameters *per network* (measured: 28.2M critic,
  27.7M actor at one camera). jaxrl2 runs the same shape on LIBERO's 256x256 frames, so this is
  faithful rather than a mistake — but `latent_dim` is the lever if it is too much.
- **Raw frames are ~0.45 MB/transition each** (obs and next_obs, uint8), so all three cameras at
  `buffer_size: 5000` is 6.9 GB of host RAM. `cfgs/dsrl.yaml` enables only `images.head`.

**Warmup.** `noise_warmup_chunks` (default 0) spends that many control steps on the base
policy's own Gaussian latent before the actor starts choosing, so the buffer opens with
transitions from the distribution pi0.5 was trained to denoise rather than from an untrained
tanh actor; dsrl_pi0 spends its whole first trajectory this way. It is a rollout-loop knob, not a
critic hyperparameter, so it is read by `deploy_policy.py` and kept out of the critic checkpoint.
The switchover costs one sampler recompile — passing a `noise=` argument at all is what changes.

**What does not carry over from §5a:** `offline_mix` (it raises for `critic_type: dsrl` rather
than mixing something else in — though the reason is now only half true: rollout datasets
collected since 2026-08-28 do carry the sampler's latent as `action.noise` (§7a), so the
transitions exist; what is missing is the loader-side support for reading them, and a latent
recorded under a `hold`/`lowrank` parameterization would still have to be projected back into
whatever family the agent acts in), the guidance ramp (nothing to ramp), and best-of-N. `train_online: false`,
`freeze_encoder`, `critic_ckpt`, `restore_optimizer`, `save_critic`, the resume machinery,
`use_step_reward` and the W&B/debug outputs all work exactly as they do for the QMFM critic —
`script/eval_policy.py` keys off `model.online_critic` and never learns which family it got.
The per-update W&B log now forwards whatever keys the update reported, so a DSRL run's
`critic/actor_loss`, `critic/entropy`, `critic/temperature` and `critic/policy_std_mean` appear
alongside the shared TD curves. `debug: true` still writes `q_episode<N>.*` — the Q plotted there
is over the latent the actor chose, against the same realized return-to-go.

### 5d. Co-training the critic on the fine-tuning demonstrations (`offline_mix`, `kind: demo`)

`offline_mix` (`cfgs/qmfm.yaml`) draws `frac` of every TD batch from a fixed dataset instead of
from the live replay buffer, which otherwise holds only what the policy being steered just did —
small, correlated, and on a task the policy fails at, quite possibly containing no success at
all. It normally points at a **rollout** dataset (§7). It can now point at the **supervised
fine-tuning set** instead: the LeRobot dataset of expert demonstrations the policy was trained
on, e.g. `NatashaYang/robotwin_demo_clean_50_lerobot`.

```yaml
# whatever `offline_mix.config` names (default cfgs/train_offline.yaml)
dataset:
  kind: demo                                        # `rollout` is the default
  repo_id: NatashaYang/robotwin_demo_clean_50_lerobot
  task: null                                        # null = the task being evaluated
  reward_col: terminal_reward                       # the only one a demonstration can have
```
```yaml
# or without editing that file, from cfgs/qmfm.yaml
offline_mix:
  enabled: true
  set: ["dataset.kind=demo",
        "dataset.repo_id=NatashaYang/robotwin_demo_clean_50_lerobot",
        "dataset.reward_col=terminal_reward"]
```

**Two episodes of the same task, not the same instruction.** An SFT dataset covers every task
the policy was fine-tuned on at once (2500 episodes over 50 tasks here), and a demonstration of
another task is a different MDP wearing the same observation shapes, so the first thing that
happens is the filter: `episode_selection.assign_episode_tasks` maps each episode back to its
RoboTwin task through the instruction templates in `description/task_instruction/` — the same
mapping §5b's bank uses, and not the dataset's own `task_index`, which indexes instructions.
`dataset.task` pins a different task deliberately (does a related task's demonstrations
transfer?); blank means the one under evaluation. `split` / `num_episodes` / `episode_seed` then
choose among the matching episodes exactly as they do for a rollout dataset.

**A demo dataset has none of the critic's columns**, and does not need them: every one of them is
*derived from the frame by the policy*. So the rows are built at startup by the run's own pi0.5 —
`DemoRetriever.cotrain_rows`, through the same image tower and the same input transform §5b's
bank goes through — and `pi_model.py::_attach_demo_cotrain` hands them to the critic. Which is
why the load is deferred: `OnlineValueCritic` cannot do it itself, so it records
`pending_demo_cotrain` and refuses to build a batch until the policy has attached one.

| | a demo row | how |
|---|---|---|
| `siglip.{head,left_wrist,right_wrist}` | ✔ `(16, 16, 1152)` fp16 | this run's own tower pass |
| `state` | ✔ normalized, embodiment dims | this run's own input transform |
| action | ✔ the `(50, 14)` normalized delta chunk | the policy's training target at that frame |
| `wrench.<key>` | ✔ `(wrench_trace_len, 6)`, **multimodal demo datasets only** | windowed out of the dataset's own per-frame rows (§5b) |
| `depth.<cam>`, `pointcloud`, raw `images.<cam>`, … | ✔ **if the dataset has the column and the run kept it** | the column itself, cast to the dtype the sim hands the critic (§5b) |
| privileged task state | ✘ | nothing persists it through the demo pipeline |

The ✔ on the third row is conditional twice over, and both conditions are checked at startup
rather than an hour into the rollouts. First, the **dataset** has to carry the column: an
rgb-only demo dataset has three camera views and nothing else, the multimodal converter's has
depth, point clouds, the wrench and a fourth camera. Second, the **run** has to have asked to
keep it, in `demo_retrieval.sensor_modalities` — the co-training rows come from the same
retriever the bank does, so one key controls both. A critic naming something neither offers is
refused, listing what a demonstration does carry (`DemoRetriever.cotrain_rows`, and
`offline_replay.check_demo_modalities` hoists the check to startup).

Two dates worth knowing. The wrench moved off the ✘ list on 2026-08-30: `cfgs/qmfm.yaml`'s
default `encoder_modalities` names the `wrench.*` modalities, and co-training on a multimodal
demo dataset serves them instead of forcing them off — which keys exist follows the dataset
(`robotwin_demo_clean_multimodal_50x10_lerobot` has the four link columns only; the randomized
one also has the `left` / `right` arm totals). Everything else recorded followed on 2026-09-01,
by reading the schema instead of a fixed list. `images.<cam>` had been excluded for a subtler
reason than "not recorded" — the rgb-only converter writes 480x640 frames against the sim's
240x320, so they are not the same array — and that is still true of *those* datasets; the
multimodal ones write 240x320 and work.

Rows are the frames at `frame_index % horizon == 0` — the control steps the online critic takes,
so a demo row and a live transition are the same distance apart in time and the same discount
applies to both. Only those frames are decoded (`LeRobotEpisodeReader.read_episode(frames=...)`;
states and actions still come back whole, since a chunk reaches 50 steps past its frame). At
`pi0_step: 50` a ~220-frame demonstration is ~5 rows, so 50 episodes of a task cost ~250 rows,
~0.4 GB resident over three SigLIP views.

**Two things to weigh.** The reward: nothing recorded a `step_reward()` while a demonstration was
collected — it is a delta against its own previous call and exists only while an episode runs —
so a demo row's reward is the sparse success it earns by definition, 1.0 on its last control
step. Against a run scoring itself with the shaped reward (`use_step_reward: true`, §6.1) the two
halves of the batch are then **not the same reward function**; `use_step_reward: false` makes
them agree exactly. And the demonstrations came from the expert motion planner, not from pi0.5,
so their SARSA next-actions are further off-policy than a rollout dataset's — the offline half is
a stabilizer and a source of what success looks like, not an estimate of *this* policy's value.
Lower `frac` rather than changing how the batch is drawn.

`train_offline.py` refuses a `kind: demo` config outright: it has no policy to encode a frame
with, so demonstrations can be co-trained on from inside an eval run and not fitted offline.

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
  - `collect_critic_obs: true` — also record the policy's **model-space** view of each step,
    including `action.noise`, the flow-matching latent each chunk was denoised from.
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
| `action.noise` | the flow-matching latent that chunk was denoised from, **padded** dims | `(50, 32)` |
| `siglip.{head,left_wrist,right_wrist}` | per-camera SigLIP patch features, fp16 | `(256, 1152)` each |
| `observation.wrench.<link>` | world-frame contact wrench per end-effector link, one row per primitive step (that step's physics steps averaged) | `(wrench_trace_len, 6)` each, i.e. `(pi0_step, 6)` |
| `observation.wrench.<arm>` | the same, summed over that arm's links | `(wrench_trace_len, 6)` each |
| `reward` | reward earned by this row's own chunk | scalar |

Each raw column and its `.model` counterpart have the same width and hold the same quantity in
different spaces: the raw ones have been unnormalized by the output transform, the `.model` ones
have not. Neither carries the model's internal zero padding to `action_dim=32` — it is stripped
before the tensors leave the sampler, so `state_dim` is `14`, not `32`. `action.noise` is the
one exception, and deliberately so (see below). Only `pi0_step` of the
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

`action.noise` is the third thing that call returns (`sample_noise`): the noise chunk the action
expert integrated from to produce this row's `action.model`. It is the action of a **noise-space**
agent's MDP (§5c) rather than of the robot's, so it is what an offline `Q(s, w)` would be trained
on — `action_col: action.noise` — and it is meaningless to a QMFM critic, which scores chunks.
Three things make it unlike every other column:

- **It is `(50, 32)`, not `(50, 14)`.** The padded width is not padding here. The trailing dims of
  an *action* normalize to constant zero for aloha and are stripped, but all 32 dims of a *noise*
  go through `action_in_proj` and shape the 14 that come out, so a narrowed seed no longer
  reproduces its chunk.
- **It is a draw, not a function of the observation.** Everything else in the row could be
  recomputed from a stored frame by a later pass; the seed is what made this chunk *this* sample
  rather than another one, and once the sampler has returned, only `Pi0.invert_actions` (§5b) can
  recover it — 100 action-expert passes a row, and only for the unguided sampler. That is why it
  has no switch of its own: it rides along with `collect_critic_obs` at 6.4 KB/row, against the
  576 KB a single SigLIP view costs.
- **Rows collected under a critic are not seeds of what ran.** Collection runs the plain sampler,
  so the column is exactly the Gaussian the chunk came from. A guided (§5a) sampler moves the
  chunk off that seed's own trajectory, and best-of-N reports the winning candidate's — neither
  applies to `collect_dataset.py`, which never builds a critic, but both matter if the aux dict is
  read anywhere else.

`policy/pi05/src/openpi/models/pi0_sample_noise_test.py` pins the property the column rests on:
feeding `sample_noise` back in as `noise=` reproduces the chunk it came with.

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

`observation.wrench.*` is the **same quantity** §6.1's debug GIF plots draw — net contact
wrench, `[Fx, Fy, Fz, Tx, Ty, Tz]` in the world frame with torque about that arm's TCP (the same
reference point for all of an arm's links, so they stay comparable with each other and with
their sum) — computed by the shared `envs/utils/wrench.py::wrench_vectors`, so the eval,
demo-collection and rollout-collection paths cannot drift apart. There is one column per
end-effector link (aloha agilex: `observation.wrench.{fl_link7,fl_link8,fr_link7,fr_link8}`)
**and** one per arm (`observation.wrench.{left,right}`, each the sum of that arm's links), so a
critic can be trained offline on whichever granularity it observes online. The dataset (like the
critic's own view and, since 2026-08-28, like the debug plots) carries **one row per primitive
step** — the contact wrench averaged over the whole TOPP trajectory that step ran, §6.1 — so the
column is a `(wrench_trace_len, 6)` trace rather than a single vector: exactly `pi0_step` rows,
NaN-padded only when a chunk was cut short. That is 240 B/row per column at `pi0_step: 10`, so
the six together are ~1.4 KB/row — negligible beside the 576 KB a single SigLIP view costs.
Datasets collected before 2026-08-29 hold `(1024, 6)` at one sample per physics step instead,
and those before 2026-08-28 have only the two arm columns, at one sample per control step.

Which trace matters: it is the one the **previous** chunk produced — the primitive steps between
the previous row's observation and this one. The env commits a row at the end of each
`take_action` (`_base_task.py::_close_step_wrench`, so the reading belongs to the action that
just executed),
and the rollout loop drains the log with `pop_step_wrench()` **before** running the chunk, at
the same point in the loop the guided eval path drains it (§6.2). The column is therefore an
observation — something the policy could have conditioned on — not the outcome of the row's own
action, and offline training pairs it with the row's state/action as-is.

> Datasets collected before 2026-07-30 drained the log *after* the chunk, so their
> `observation.wrench.*` is the trace of the row's **own** chunk. To train on one, shift the
> column a row later within each episode — `multisensory_steering`'s `dataset.modalities`
> takes `{column: ..., shift: 1}` for exactly this.

`reward` is the **same quantity the online critic is fed** during a guided eval: 1.0 on the
control step that first reaches success, and otherwise the task's own `step_reward()` — the
shaped, *delta*-valued progress term (`envs/lift_pot.py`, `envs/open_microwave.py`,
`envs/put_object_cabinet.py`); tasks that define none leave the column sparse (0 everywhere but
the successful step). Both drivers call `eval_policy.py::control_step_reward` at the same point
in their loop so the two cannot drift — `eval_policy.py` passes the result to
`online_critic.commit()`, `collect_dataset.py` stores it. Unlike the wrench, it belongs to the
row's **own** chunk (it is the outcome of this row's action, not an observation preceding it),
which is exactly what an offline `(s, a, r, s')` needs. It cannot be recomputed after the fact:
`step_reward()` is a difference against its own previous call, so it only exists while the
episode runs. Train on it with `reward_col: reward` in `multisensory_steering`'s
`cfgs/train_offline.yaml` (the alternative, `terminal_reward`, derives the sparse signal from
`success` and ignores the shaping). Datasets collected before 2026-08-07 have no such column.

`envs/utils/wrench.py::stack_step_wrench` does the stacking, shared with the critic's online
view of the same modality and blind to which family a key belongs to. A trace can be short — an episode's first row has no previous chunk
and carries a single sample of the contact state at that instant, and a chunk cut off by success
or `step_lim` contributes only the steps it ran — so the tail is padded with **NaN**, not zeros,
since zero is a meaningful reading (the arm touching nothing). `wrench_trace_len` is what fixes
the width; the same value must be set on the eval side for a critic pretrained on these columns
to load against them.

`wrench` is a `data_type` like the others, but it is the one the env cannot pick up from the
flag itself: contacts are a scene query, not part of `get_obs`. So each driver reads
`data_type.wrench` and passes `record_step_wrench` into the env — `collect_data.py`,
`collect_dataset.py` and `eval_policy.py` all do, and nothing logs a wrench with the flag off.
Turning it off drops these columns from the dataset entirely (and makes a `wrench.*` critic
modality unavailable, §6.2). What differs between the drivers is only *where* the log is drained
— see §3.4 for the expert-demo path.


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
- **pytorch3d is needed only for point clouds — and on Rorqual it is installed in the
  conda env (0.7.8) but *not* in `policy/pi05/.venv`.** `fps` in `envs/camera/camera.py`
  (point-cloud downsampling) imports a compiled `_C.so` pinned to a torch ABI, and the two
  envs run different torch (conda RoboTwin = 2.4.1, `policy/pi05/.venv` = 2.7.0), so if you
  do need it, **both** need their own build — collection runs in the first,
  eval/rollout-collection in the second.
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
- **curobo must be built into BOTH envs — eval will not even import without it.**
  `envs/robot/robot.py` does an unconditional `from .planner import CuroboPlanner`, and
  `envs/robot/planner.py` wraps the curobo import in a `try:` — so a missing curobo is **not**
  a graceful fallback to mplib. The `except` prints "Something wrong happened when importing
  CuroboPlanner", leaves the class undefined, and the very next import raises
  `ImportError: cannot import name 'CuroboPlanner'`, killing the run before the sim loads.
  Collection runs in the conda env (py3.10 / torch 2.4.1) but **eval and rollout collection
  run in `policy/pi05/.venv`** (py3.11 / torch 2.7.0), so that venv needs its own build:
  ```bash
  cd envs/curobo
  module load cuda/12.6; export CUDA_HOME=$EBROOTCUDA
  export TORCH_CUDA_ARCH_LIST=9.0 FORCE_CUDA=1
  ../../policy/pi05/.venv/bin/python -m pip install -e . \
      --no-build-isolation --no-deps --force-reinstall
  ```
  The two builds **coexist in the same source tree** — the extensions carry the interpreter
  tag (`geom_cu.cpython-310-*.so` vs `cpython-311-*.so`), so building the second does not
  disturb the first. Beware that the `rm -f src/curobo/curobolib/*.so` in the §1.2 clean-rebuild
  deletes *both*; narrow the glob to one tag if you only mean to rebuild one env.
  Building against torch `cu128` with the `cuda/12.6` module is fine — same CUDA major, so
  torch's `cpp_extension` warns rather than raising.
- **Switching GPU generation means rebuilding curobo.** Its CUDA extensions are compiled
  ahead of time for whatever `TORCH_CUDA_ARCH_LIST` said at build time; an `sm_89` build
  fails on an H100 at kernel launch ("no kernel image is available"), and an `sm_90` one
  fails the same way on an L40S. Check what you have
  with `cuobjdump --list-elf envs/curobo/src/curobo/curobolib/*.so | grep -o 'sm_[0-9]*'`,
  and clean-rebuild per §1.2. **Build it on a compute node, not the login node** — the
  login node's per-user memory cap kills `nvcc` mid-file, and the failure prints a bare
  `FAILED:` line with *no* diagnostic, which reads like a source error but isn't. The
  build needs no internet (`--no-build-isolation --no-deps`), so a compute node is fine.
- **The critic path needs two out-of-repo checkouts — and staging them on disk is not
  enough.** `guidance_scale != 0` needs the `multisensory_steering` package and the QMFM repo
  it imports `ReplayBuffer` from via `$QMFM_ROOT`. On Rorqual they are at
  `/lustre09/project/6028519/natashay/multisensory-steering` and
  `/home/natashay/links/projects/def-florian7/natashay/QMFM`, and `policy/pi05/eval.sh` exports
  the `QMFM_ROOT` default. On a new cluster re-point that export and `critic_config_path`.
  Two traps:
  - **`multisensory_steering` is `import`ed, so it must be installed into `policy/pi05/.venv`**,
    not merely cloned. Editable, and **`--no-deps`** — its pyproject lists `jax` unpinned and
    warns that the venv's jax is CUDA-specific and must not be upgraded (all four deps are
    already present):
    ```bash
    policy/pi05/.venv/bin/python -m pip install -e <multisensory-steering> --no-deps
    ```
  - **`pi_model.py` imports it at module scope** (`from multisensory_steering import
    load_critic`), even though the name is only used inside `_init_critic` on the guided path.
    So a *baseline* eval at `guidance_scale: 0.0` also fails with `ModuleNotFoundError`
    without it. Same shape as the `critic_config_path` trap in §6.2 — the guidance switch does
    not gate the guidance imports.
  See §6.2 before turning guidance on.
- **`uv` may not exist on a new cluster.** §5.1's `uv sync` / `uv run` instructions assume it
  (on Rorqual it is at `~/.local/bin/uv`); install it from https://astral.sh/uv/install.sh on a
  login node if `uv: command not found`. `policy/pi05/.venv` also ships a working `pip`, so for
  installing *into* the venv `policy/pi05/.venv/bin/python -m pip ...` works without uv.
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
