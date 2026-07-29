# CLAUDE.md — RoboTwin + π0.5 (pi05) Reproduction Guide

This document records how **this specific machine** is set up so the full pipeline
(data collection → processing → π0.5 fine-tuning → inference/eval → rollout-dataset
collection) can be reproduced on another machine.

It captures the local customizations that are **not** in the upstream RoboTwin /
openpi docs. For general RoboTwin usage see the
[official docs](https://robotwin-platform.github.io/doc/).

---

## 0. Hardware & driver assumptions

- **GPU:** NVIDIA GeForce RTX 5090 (Blackwell, `sm_120`).
- The 5090 requires **CUDA 12.8** for `curobo` (the motion planner used during data
  collection / eval). This is why there is a dedicated conda env `cuda128` and the
  collection/eval scripts prepend it to `PATH` (see below).
- Multiple CUDA toolkits live side by side as conda envs: `cuda124`, `cuda128`.
- Repo root on this machine: `/home/natasha/RoboTwin`. Several scripts and the
  training config contain **hard-coded absolute paths** to this location — grep for
  `/home/natasha` when porting and fix them (notably
  `policy/pi05/src/openpi/training/config.py` and the `PATH=.../cuda128` lines in the
  collection/eval shell scripts).

---

## 1. Two separate environments

This repo uses **two independent Python environments** — do not mix them.

| Env | Manager | Python | Used for |
|-----|---------|--------|----------|
| `RoboTwin` (conda) | conda/pip | 3.10 | Simulation, data **collection**, data **processing**, eval driver (`script/*.py`) |
| `policy/pi05/.venv` | **uv** | 3.11 | π0.5 (openpi): LeRobot conversion, norm stats, **training**, policy server |

The `cuda128` conda env only supplies the CUDA 12.8 toolkit/libraries on `PATH` for
curobo; it is not a Python workspace of its own.

### 1a. RoboTwin simulation env

Follow the [RoboTwin install doc](https://robotwin-platform.github.io/doc/usage/robotwin-install.html),
which runs `script/_install.sh`. Key steps that script performs:

```bash
# inside the RoboTwin conda env
pip install -r script/requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
# patches sapien/wrapper/urdf_loader.py (utf-8) and mplib/planner.py (collision check)
# installs curobo v0.7.8:
cd envs && git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo && pip install -e . --no-build-isolation
pip install warp-lang==1.12.0 setuptools==69.5.1
```

Then download embodiment/task assets from HuggingFace (see `INSTALLATION.md`).

### 1b. π0.5 (pi05) env

`policy/pi05` is an [openpi](https://github.com/Physical-Intelligence/openpi) tree
managed by **uv** (`.python-version` = 3.11, deps pinned in `pyproject.toml` /
`uv.lock`; jax[cuda12]==0.5.3, torch==2.7.0, flax==0.10.2, lerobot, etc.).

```bash
cd policy/pi05
uv sync            # creates .venv from uv.lock
# activated implicitly by `uv run ...`; scripts also `source .venv/bin/activate`
```

π0.5 base checkpoint is pulled by the `weight_loader` in the train config (originally
`s3://openpi-assets/checkpoints/pi05_base/params` — see §4 for the local override).

---

## 2. Data collection (RoboTwin sim → HDF5)

Run from repo root in the **RoboTwin conda env**.

```bash
# single task
bash collect_data.sh <task_name> <task_config> <gpu_id>
# example
bash collect_data.sh beat_block_hammer demo_randomized 0

# all tasks defined under description/task_instruction/*.json, one config
bash collect_all_data.sh <task_config> <gpu_id>
```

- `collect_data.sh` runs `script/collect_data.py`, which first searches for random
  seeds that the expert solver can complete, then replays them to record trajectories.
- Output: `data/<task_name>/<task_config>/` (per-episode HDF5 + camera streams). The
  `.cache` dir is removed afterward.
- Task configs live in `task_config/*.yml` (create new ones with
  `bash task_config/create_task_config.sh <name>`). Common settings on this machine:
  `demo_clean` and `demo_randomized`.

---

## 3. Data processing (HDF5 → LeRobot dataset)

All in `policy/pi05`. Two stages: repack to openpi-flavored HDF5, then convert to a
LeRobot dataset.

### 3a. Repack

```bash
cd policy/pi05
bash process_data_pi05.sh <task_name> <setting> <expert_data_num>
# = python scripts/process_data.py <task_name> <setting> <expert_data_num>
# example
bash process_data_pi05.sh beat_block_hammer demo_clean 50
```

- Reads `../../data/<task_name>/<setting>/` (i.e. the collection output).
- Writes `processed_data/<task_name>-<setting>-<expert_data_num>/episode_*` — each
  episode is an HDF5 with `observations/qpos`, `left/right_arm_dim`, and JPEG-encoded
  `cam_high`, `cam_left_wrist`, `cam_right_wrist`.

### 3b. Convert to LeRobot format

```bash
cd policy/pi05
bash generate.sh <data_dir> <repo_id>
# = uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
#       --raw_dir <data_dir> --repo_id <repo_id>
# example
bash generate.sh processed_data/beat_block_hammer-demo_clean-50 NatashaYang/robotwin_lerobot_dataset
```

- Writes the LeRobot dataset under `$HF_LEROBOT_HOME/<repo_id>` (an existing dir at
  that repo_id is deleted and rebuilt).
- The `repo_id` you use here must match `repo_id=` in the train config (§4).

---

## 4. Fine-tuning π0.5

Config lives in `policy/pi05/src/openpi/training/config.py`. The active config for
this setup is **`pi05_base_aloha_lora`** (a LoRA fine-tune of pi05_base):

- `model = Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")`
- `data.repo_id = "NatashaYang/robotwin_lerobot_dataset"` — **must match §3b**
- `prompt_from_task=True` (prompt derived from the task)
- `batch_size=32` (total, not per-GPU), `num_train_steps=30000`, `fsdp_devices=1`,
  `ema_decay=None`
- **Local customizations currently in the working tree (uncommitted):**
  - `weight_loader` points at a **local checkpoint** instead of the S3 base:
    `/home/natasha/RoboTwin/policy/pi05/checkpoints/pi05_base_aloha_lora/Pi05RoboTwinSubsetLoraFT/19000/params`
  - `resume=True` — resumes an existing run rather than starting fresh.
  - `finetune.sh` had `--overwrite` **removed** (so a run is resumed, not wiped).

  > Porting note: on a fresh machine that has no local checkpoint, set
  > `weight_loader` back to `CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi05_base/params")`,
  > set `resume=False`, and add `--overwrite` for the first run.

### 4a. Compute normalization stats (required before first train)

```bash
cd policy/pi05
uv run scripts/compute_norm_stats.py pi05_base_aloha_lora
```

### 4b. Train

```bash
cd policy/pi05
bash finetune.sh <train_config_name> <model_name> <gpu_id>
# example (matches the local checkpoint layout)
bash finetune.sh pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0
```

- `finetune.sh` runs:
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp-name=<model_name>`
- Checkpoints: `policy/pi05/checkpoints/<train_config_name>/<model_name>/<step>/`
  (e.g. `.../pi05_base_aloha_lora/Pi05RoboTwinSubsetLoraFT/19000/`). `params` lives
  inside each step dir.
- Logs go to Weights & Biases (`policy/pi05/wandb/`).

---

## 5. Inference / evaluation (policy in the sim)

Run from `policy/pi05` in the RoboTwin conda env; the script `source`s `.venv` and
`cd`s to repo root itself. Uses the **cuda128** toolkit on PATH and caps GPU memory
(`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`) so it fits alongside the sim on a 24G budget.

```bash
cd policy/pi05
bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> \
             [guidance_scale] [guidance_ramp_updates]
# baseline (no critic guidance)
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0
# online critic-guided
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0 0.3 256
```

- Driver: `script/eval_policy.py` with `policy/pi05/deploy_policy.yml`.
- `deploy_policy.py` → `pi_model.py::PI0` loads the trained policy from
  `policy/pi05/checkpoints/<train_config_name>/<model_name>/<checkpoint_id>` and runs
  inference. `checkpoint_id` (default 30000) and `pi0_step` (action chunk length, 50)
  come from `deploy_policy.yml`.
- Camera → model mapping (in `deploy_policy.py::encode_obs` / `pi_model.py`):
  `head_camera → cam_high`, `left_camera → cam_left_wrist`, `right_camera → cam_right_wrist`.
- Results land in `eval_result/<task_name>/<policy_name>/<task_config>/<ckpt_setting>/<timestamp>/`.
  Alongside `_result.txt` / `_episode_results.csv`, each run snapshots `deploy_policy.yml` and the
  `critic_config_path` file it includes into that dir (`script/eval_policy.py::snapshot_config`).
  The copies keep their comments but carry the values **actually used** — `eval.sh`'s positional
  args are written in, so `task_name`, `seed`, `guidance_scale` etc. read as resolved rather than
  as the `null`/default in the source yml.

Setting `debug: true` in the **task config** turns on extra per-episode diagnostics
(`script/eval_policy.py::visualize_debug_obs`), all written into `debug_vis/episode<N>/`
under that same result dir:

| Output | File (under `debug_vis/episode<N>/`) | Notes |
|---|---|---|
| TCP wrench histograms | `wrench_hist_episode<N>.png` | one histogram per component (Fx/Fy/Fz/Tx/Ty/Tz), left and right arm overlaid |
| Rollout + wrench GIF | `wrench_episode<N>.gif` | head camera on the left with the **world** axes drawn as labelled x/y/z arrows, projected into the camera and anchored at each arm's TCP (the axes the components are resolved in, at the point they act); the wrench traces with a step cursor on the right |
| Raw series | `wrench_episode<N>.npz` | `step`, `left`, `right` — `(num_samples, 6)` each |
| Depth / segmentation tiles, point clouds | `obs_step<S>.png` / `pcd_step<S>.ply` | only for the modalities the task config's `data_type` enables |

The wrench is computed by `envs/utils/wrench.py` (shared with the rollout-dataset collector, §6a).
It is the **net contact wrench on the end-effector links** (wrist link + gripper
fingers + any `fix_gripper_name` links), summed from `scene.get_contacts()` impulses divided
by the sim timestep, with torque taken about the TCP origin. Both vectors are resolved in the
**world** frame (N and N·m) — only the moment arm is TCP-relative, so a trace stays comparable
across steps as the gripper rotates. It is contact-only: an arm moving through free space reads exactly
zero — this is not a joint-torque estimate. One sample is taken per policy call (so
`pi0_step` sim frames apart), paired with that call's head-camera frame.

### 5a. Critic gradient guidance (`guidance_scale` is the on/off switch)

There is **one** eval script and **one** config. `guidance_scale` in `deploy_policy.yml`
(overridable as `eval.sh`'s 7th positional arg) decides whether a critic steers the frozen
pi0.5 flow sampler in `Pi0.sample_actions`:

| `guidance_scale` | Behavior |
|---|---|
| `0.0` (default) | plain pi0.5 baseline — no critic is built, no replay collection, no TD updates, no W&B |
| nonzero | an ensemble QMFM `Value` critic is trained **online** by TD during the rollouts; guidance ramps `0 → guidance_scale` over `guidance_ramp_updates` TD updates (`0` jumps to target after the first update) |

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
(editable install at `/home/natasha/multisensory-steering`, config at `cfgs/qmfm.yaml`), which
imports QMFM's `ReplayBuffer` from `$QMFM_ROOT` (default `/home/natasha/QMFM`, exported by
`eval.sh`). Only the guided path logs to W&B, collects replay transitions, and honors
`save_critic` / `critic_ckpt` / the TD hyperparameters; `script/eval_policy.py` keys all of it
off whether the policy object exposes an `online_critic`.

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
them apart. See §6 for collecting the right columns.

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

The names are the §6a dataset columns minus their `observation.` prefix, so a critic trained
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
  been executed. In a rollout dataset the same array sits one row *earlier* (there it is the
  trace the row's own chunk produced), so offline training on `observation.wrench.*` must pair
  it with the *next* row's state/action. The env's per-step logging is switched by the task
  config's `data_type.wrench` (§6a) — a critic configured for `wrench.*` against a config that
  has it off fails at startup. The guided path drains the log before the chunk runs, collection
  drains it after, so the two never compete.
- **Modalities are an architecture key.** They go into the checkpoint, and a warm start rebuilds
  the same encoder stack; changing the list makes an existing critic checkpoint refuse to load
  (loudly, leaf by leaf). Offline pretraining takes the same names — `multisensory_steering`'s
  `cfgs/train_offline.yaml` maps them to dataset columns under `dataset.modalities` — and a
  checkpoint only warm-starts a run whose list matches. Each modality is also stored twice per
  transition (obs and next_obs) — each `siglip.*` view is ~1.15 MB/transition (so all three come
  to ~3.5 MB), a depth camera ~0.6 MB, so `buffer_size` needs revisiting when adding one.

For remote / server-based inference see `policy/pi05/docs/remote_inference.md`
(`scripts/serve_policy.py`).

---

## 6. Collecting a rollout dataset (policy rollouts → HF Hub)

This is the pipeline added in commit `1aa5b89`. It rolls out a trained policy in the
sim and builds a HuggingFace dataset of the resulting trajectories.

```bash
cd policy/pi05
bash collect_dataset.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id>
```

- Driver: `script/collect_dataset.py` with `policy/pi05/collect_dataset.yml`.
- Behavior is controlled by `collect_dataset.yml`:
  - `num_episodes: 100` — successful, expert-checked rollouts to collect.
  - `expert_check: true` — only roll out on seeds the expert can solve.
  - `output_dir: ./rollout_datasets` — local `save_to_disk` location.
  - `push_to_hub: true`, `hub_repo_id: NatashaYang/robotwin_pi05_rollouts_dataset`.
  - `checkpoint_id: 30000`, `pi0_step: 50`, `instruction_type: unseen`.
  - `collect_critic_obs: true` — also record the policy's **model-space** view of each step.
  - `collect_siglip: true` — within that, also record the SigLIP patch features of **every**
    camera the policy sees (`siglip.head`, `siglip.left_wrist`, `siglip.right_wrist`). Set
    `false` to keep the dataset small, or list the views you want (`[head, left_wrist]`).
  - `resume: true` — see below.

Everything **beyond** rgb + qpos is decided by the **task config**, not by `collect_dataset.yml`:
whatever its `data_type` block turns on reaches `envs/_base_task.py::get_obs`, and
`extra_obs_columns` records all of it (§6b). `demo_clean` therefore yields only the columns in
§6a's table, while `demo_clean_privileged` roughly doubles the bytes per row.

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

### 6a. Columns, and which ones a steering critic needs

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
critic intended to steer inside the sampler (§5a) **must** be trained on these; point
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

`observation.wrench.left` / `.right` are the **same quantity** §5's debug GIF plots — net contact
wrench on that arm's end-effector links, `[Fx, Fy, Fz, Tx, Ty, Tz]` in the world frame, torque
about the TCP — computed by the shared `envs/utils/wrench.py::tcp_wrench_vector` so the eval and
collection paths cannot drift apart. The rate differs: `eval_policy.py` samples once per policy
call, while collection samples after **every** primitive step, so a row carries the whole
`(pi0_step, 6)` trace of the chunk it executed rather than a single vector. The env does the
logging (`_base_task.py::_log_step_wrench`); the rollout loop drains it with `pop_step_wrench()`
once per inference call, and `envs/utils/wrench.py::stack_step_wrench` does the stacking, shared
with the critic's online view of the same modality. An episode's last chunk stops early —
`take_action` is a no-op once the task succeeds or `step_lim` is hit — and the unexecuted steps
are padded with **NaN**, not zeros, since zero is a meaningful reading (the arm touching
nothing). At 480 B/row they are the cheapest column here.

`wrench` is a `data_type` like the others, but it is the one the env cannot pick up from the
flag itself: contacts are a scene query, not part of `get_obs`. So each driver reads
`data_type.wrench` and passes `record_step_wrench` into the env — `collect_data.py`,
`collect_dataset.py` and `eval_policy.py` all do, and nothing logs a wrench with the flag off.
Turning it off drops these columns from the dataset entirely (and makes a `wrench.*` critic
modality unavailable, §5a).

### 6b. Extra data types (privileged task configs)

`script/collect_dataset.py::extra_obs_columns` records everything else the observation carries,
so the dataset follows the task config's `data_type` block automatically. With
`demo_clean_privileged` (depth / pointcloud / third_view / mesh + actor segmentation all `true`)
a row gains, per camera `<cam>` ∈ `head` / `left_wrist` / `right_wrist`:

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
  SigLIP view adds ~0.56 MB, so all three make it ~2.2 MB) vs **~2.2 MB/row**
  for `demo_clean_privileged`. Depth is the bulk of the difference — the segmentation and
  third-view columns are PNG-compressed. There is no per-column switch here: to collect less,
  use a task config with fewer `data_type` flags.

---

## 7. End-to-end quick reference

```bash
# 1. collect sim demos            (RoboTwin conda env, repo root)
bash collect_data.sh beat_block_hammer demo_clean 0

# 2. repack + convert to LeRobot  (policy/pi05, uv)
cd policy/pi05
bash process_data_pi05.sh beat_block_hammer demo_clean 50
bash generate.sh processed_data/beat_block_hammer-demo_clean-50 NatashaYang/robotwin_lerobot_dataset

# 3. norm stats + fine-tune       (policy/pi05, uv)
uv run scripts/compute_norm_stats.py pi05_base_aloha_lora
bash finetune.sh pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0

# 4. evaluate in sim
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0

# 5. (optional) collect a rollout dataset and push to HF
bash collect_dataset.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0
```

---

## 8. Machine-specific gotchas (fix these when porting)

- **Hard-coded paths:** `/home/natasha/...` appears in `config.py` (`weight_loader`)
  and the `PATH=.../miniconda3/envs/cuda128/bin` lines in `collect_dataset.sh` /
  `eval.sh`. Update to the new machine's layout.
- **CUDA 12.8 requirement** is specific to the RTX 5090 + curobo. On an older GPU you
  may not need the `cuda128` env; drop the `PATH` prepend.
- **pytorch3d must be rebuilt whenever torch changes — in BOTH envs.** pytorch3d ships
  a compiled `_C.so` (used by `fps` in `envs/camera/camera.py` for point-cloud
  downsampling) that is linked against a specific torch ABI. After torch was upgraded
  to `2.7.0+cu128` (CXX11 ABI) for the 5090, the old build broke with
  `ImportError: ... _C...so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs`.
  Worse, `camera.py` swallows this in a bare `except:` and prints only
  `fps error: missing pytorch3d`, so the same message appears whether pytorch3d is
  ABI-broken *or* simply not installed. The eval pipeline (`eval.sh`) runs in
  `policy/pi05/.venv` (py3.11), while collection runs in the RoboTwin conda env
  (py3.10) — **pytorch3d must be built into both**, or the env you happen to run in
  will fail. Rebuild against the *current* torch with the CUDA 12.8 toolkit and the
  5090's `sm_120` arch (pin the target interpreter by absolute path so the `cuda128`
  env's own python does not shadow it):

  ```bash
  export CUDA_HOME=/home/natasha/miniconda3/envs/cuda128
  export PATH="$CUDA_HOME/bin:$PATH"      # cuda128 nvcc (12.8); RoboTwin's default nvcc is 12.1 and lacks sm_120
  export TORCH_CUDA_ARCH_LIST="12.0"      # Blackwell / RTX 5090
  # RoboTwin conda env (collection):
  /home/natasha/miniconda3/envs/RoboTwin/bin/python -m pip install --no-build-isolation --no-deps \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  # pi05 .venv (eval / policy):
  /home/natasha/RoboTwin/policy/pi05/.venv/bin/python -m pip install --no-build-isolation --no-deps \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  ```
- **`repo_id` must be consistent** between `generate.sh` (§3b) and the train config's
  `data.repo_id` (§4), or training loads the wrong / no dataset.
- **HF Hub auth:** rollout-dataset push (§6) and any dataset/checkpoint pull need
  `huggingface-cli login` (token for the `NatashaYang/...` repos).
- **W&B:** training logs to Weights & Biases; run `wandb login` or set
  `WANDB_MODE=offline`.
- **First run vs. resume:** the committed working tree is configured to **resume** an
  existing LoRA run from a local checkpoint. For a clean first fine-tune on a new
  machine, revert `weight_loader` to the S3 base, set `resume=False`, and pass
  `--overwrite` (see §4).
