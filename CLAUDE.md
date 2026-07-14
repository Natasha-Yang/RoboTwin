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
bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id>
# example
bash eval.sh beat_block_hammer demo_clean pi05_base_aloha_lora Pi05RoboTwinSubsetLoraFT 0 0
```

- Driver: `script/eval_policy.py` with `policy/pi05/deploy_policy.yml`.
- `deploy_policy.py` → `pi_model.py::PI0` loads the trained policy from
  `policy/pi05/checkpoints/<train_config_name>/<model_name>/<checkpoint_id>` and runs
  inference. `checkpoint_id` (default 30000) and `pi0_step` (action chunk length, 50)
  come from `deploy_policy.yml`.
- Camera → model mapping (in `deploy_policy.py::encode_obs` / `pi_model.py`):
  `head_camera → cam_high`, `left_camera → cam_left_wrist`, `right_camera → cam_right_wrist`.

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
