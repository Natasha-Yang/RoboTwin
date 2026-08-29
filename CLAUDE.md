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
- `use_step_reward` (`deploy_policy.yml`, or `eval.sh`'s 10th arg) picks **which reward the run
  scores itself with** — what `online_critic.commit()` is fed, what the W&B reward curves and the
  `reward` column of `_episode_results.csv` measure. `true` (default) is 1.0 on the control step
  that first reaches success plus the task's own shaped progress term (`step_reward()`, a delta —
  `envs/lift_pot.py`, `envs/open_microwave.py`, `envs/put_object_cabinet.py`); `false` is sparse:
  1.0 on success and 0.0 everywhere else, even for a task that defines shaping (`step_reward()`
  is then never called at all, so its own delta state never advances). Unlike the guidance knobs
  this applies to the plain baseline too. `script/eval_policy.py::control_step_reward` is the one
  place it acts, so a guided run's TD targets and the printout agree by construction; the
  equivalent for a collected dataset's `reward` column is §6a, which is always shaped.

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
task/policy/config/ckpt **in its existing directory** rather than opening a new timestamped one:

| Restored | From | Why it cannot be recomputed |
|---|---|---|
| `now_seed`, `now_id`, `suc_test_seed_list` | `resume_state.json` | which seeds the expert check rejected is not a function of the episode index |
| global numpy RNG state | `resume_state.json` | the episode's instruction is drawn with `np.random.choice` |
| `test_num`, `suc`, `chunk_count` | `resume_state.json` | counters the guidance ramp and MA windows key off |
| episode rows + MA windows | `_episode_results.csv` | reloaded so the averages continue across the break |
| critic params, target, **Adam state, LR schedule position** | `online_value_critic.pkl` | see below |
| guidance ramp position | `critic_ramp_baseline` in `resume_state.json` | see below |
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

Setting `debug: true` in the **task config** turns on extra per-episode diagnostics
(`script/eval_policy.py::visualize_debug_obs`), all written into `debug_vis/episode<N>/`
under that same result dir:

| Output | File (under `debug_vis/episode<N>/`) | Notes |
|---|---|---|
| Per-link wrench histograms | `wrench_hist_episode<N>.png` | one histogram per component (Fx/Fy/Fz/Tx/Ty/Tz), every gripper link overlaid (aloha: `fl_link7`, `fl_link8`, `fr_link7`, `fr_link8`) |
| Rollout + wrench GIF | `wrench_episode<N>.gif` | head camera on the left with the **world** axes drawn as labelled x/y/z arrows, projected into the camera and anchored at each arm's TCP (the axes the components are resolved in, at the point they act); one trace column per gripper link with a step cursor on the right |
| Raw series | `wrench_episode<N>.npz` | `step`, `components`, `links`, plus one `(num_samples, 6)` array per link name |
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

The wrench is computed by `envs/utils/wrench.py` (shared with the rollout-dataset collector, §6a).
It is the **net contact wrench on the end-effector links** (wrist link + gripper
fingers + any `fix_gripper_name` links), summed from `scene.get_contacts()` impulses divided
by the sim timestep, with torque taken about the TCP origin. Both vectors are resolved in the
**world** frame (N and N·m) — only the moment arm is TCP-relative, so a trace stays comparable
across steps as the gripper rotates. It is contact-only: an arm moving through free space reads exactly
zero — this is not a joint-torque estimate. One sample is taken per policy call (so
`pi0_step` sim frames apart), paired with that call's head-camera frame.

It is recorded at **two granularities at once**, from the one contact query
(`wrench_vectors`): one `(6,)` per gripper **link**, and one per **arm** that is exactly the sum
of that arm's links. The links are what keep a finger squeezing against its opposite — equal and
opposite forces that cancel in the arm total — visible; the arm totals are the coarser
two-signal version. `_base_task._log_step_wrench` logs both once per primitive step and
everything downstream just names the keys it wants, so the rollout dataset carries both column
families (§6a) and a critic can condition on either or both (§5a). The link labels are the
embodiment's URDF link names, so **which** `wrench.<link>` keys exist follows the robot: aloha
agilex gives four (`fl_link7`, `fl_link8`, `fr_link7`, `fr_link8`), plus `left` and `right`.
`ee_link_labels` is the link → arm grouping; `compute_tcp_wrench` / `tcp_wrench_vector` are the
arm half on their own. The debug plots draw the **link** half only — an arm total overlaid on
its own links is just their sum drawn twice — and `analysis/plot_wrench_hist.py` takes
`--family links|arms|all` for the same reason.

> Changed on 2026-08-28. Datasets collected before it carry only `observation.wrench.{left,right}`.
> Those keep working unchanged: the arm columns and the `wrench.left` / `wrench.right` modalities
> mean exactly what they did, so an existing critic checkpoint still warm-starts as long as its
> `encoder_modalities` names them (it is an architecture key — a checkpoint trained on the arm
> totals cannot be pointed at the link modalities without retraining, and vice versa). What
> changed for such a checkpoint is the **encoder**: `wrench.*` now defaults to a 2-D `map_cnn`
> over the trace's (T, 6) time × component grid rather than a temporal `conv1d`, so an older one
> has to pin `encoder: conv1d` in its modality mapping to keep loading.

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
| `train_online: true` | ✔ | as above — transitions are stashed into the replay buffer after every control step, TD updates run, `save_critic` writes the result |
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
(editable install at `/home/natasha/multisensory-steering`, config at `cfgs/qmfm.yaml`), which
imports QMFM's `ReplayBuffer` from `$QMFM_ROOT` (default `/home/natasha/QMFM`, exported by
`eval.sh`). Only a run with a critic logs to W&B, collects replay transitions, and honors
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
| `wrench.<link>` (aloha: `fl_link7`, `fl_link8`, `fr_link7`, `fr_link8`) | per-step contact wrench per end-effector link, logged by the env | `(pi0_step, 6)` each |
| `wrench.<arm>` (`left`, `right`) | the same reading summed over that arm's links | `(pi0_step, 6)` each |
| `action_proposals` | task config `data_type.action_proposals` (§5b) | `(top_k, 50, 14)` |
| `noise_proposals` | task config `data_type.noise_proposals` (§5b) | `(top_k, 50, 14)` |

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
  been executed. Rollout collection drains the log at the same point in its loop, so a dataset's
  `observation.wrench.*` holds exactly this against exactly this row's state and action, and a
  critic trained offline on it needs no realignment. (Datasets collected before 2026-07-30 store
  the *following* chunk's trace instead — see §6a.) The env's per-step logging is switched by the
  task config's `data_type.wrench` (§6a); a critic configured for `wrench.*` against a config
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

There is no `wrench` signal, though it would be the natural third: **no demo dataset carries
one.** `collect_data.py` computes the contact wrench live and never writes it to the HDF5, and
neither `process_data.py` nor the LeRobot converter carries it downstream — so it exists only in
rollout datasets (§6a) and in the live observation, never in a demonstration. Adding it would
mean persisting the wrench through the demo pipeline and re-collecting.

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
collected since 2026-08-28 do carry the sampler's latent as `action.noise` (§6a), so the
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
all. It normally points at a **rollout** dataset (§6). It can now point at the **supervised
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
| `wrench.*`, `depth.*`, `pointcloud`, privileged poses | ✘ | nothing persists them through the demo pipeline |
| `images.<cam>` | ✘ | a demo frame is 480x640, the sim's is 240x320 — not the same array |

A critic naming one of the ✘ modalities is refused **at startup**, listing them, rather than an
hour into the rollouts (`offline_replay.check_demo_modalities`). Note that `cfgs/qmfm.yaml`'s
default `encoder_modalities` lists the `wrench.*` modalities at whichever granularity, so
turning this on means turning those off.

Rows are the frames at `frame_index % horizon == 0` — the control steps the online critic takes,
so a demo row and a live transition are the same distance apart in time and the same discount
applies to both. Only those frames are decoded (`LeRobotEpisodeReader.read_episode(frames=...)`;
states and actions still come back whole, since a chunk reaches 50 steps past its frame). At
`pi0_step: 50` a ~220-frame demonstration is ~5 rows, so 50 episodes of a task cost ~250 rows,
~0.4 GB resident over three SigLIP views.

**Two things to weigh.** The reward: nothing recorded a `step_reward()` while a demonstration was
collected — it is a delta against its own previous call and exists only while an episode runs —
so a demo row's reward is the sparse success it earns by definition, 1.0 on its last control
step. Against a run scoring itself with the shaped reward (`use_step_reward: true`, §5) the two
halves of the batch are then **not the same reward function**; `use_step_reward: false` makes
them agree exactly. And the demonstrations came from the expert motion planner, not from pi0.5,
so their SARSA next-actions are further off-policy than a rollout dataset's — the offline half is
a stabilizer and a source of what success looks like, not an estimate of *this* policy's value.
Lower `frac` rather than changing how the batch is drawn.

`train_offline.py` refuses a `kind: demo` config outright: it has no policy to encode a frame
with, so demonstrations can be co-trained on from inside an eval run and not fitted offline.

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
  - `hub_private: true` — create the hub repo private. This is the **default when the key is
    absent**, so a run cannot publish a dataset by omission; set it `false` to deliberately
    create a public repo. It is honored only when the push *creates* the repo — an existing
    repo keeps whatever visibility it already has, so this cannot retroactively hide (or
    expose) a dataset that has been pushed before. To change an existing repo, use
    `HfApi().update_repo_settings(repo_id, repo_type="dataset", private=...)`.
  - `checkpoint_id: 30000`, `pi0_step: 50`, `instruction_type: unseen`.
  - `collect_critic_obs: true` — also record the policy's **model-space** view of each step,
    including `action.noise`, the flow-matching latent each chunk was denoised from.
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
| `action.noise` | the flow-matching latent that chunk was denoised from, **padded** dims | `(50, 32)` |
| `siglip.{head,left_wrist,right_wrist}` | per-camera SigLIP patch features, fp16 | `(256, 1152)` each |
| `observation.wrench.<link>` | world-frame contact wrench per end-effector link, one row per executed step | `(pi0_step, 6)` each |
| `observation.wrench.<arm>` | the same, summed over that arm's links | `(pi0_step, 6)` each |
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
critic intended to steer inside the sampler (§5a) **must** be trained on these; point
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

`observation.wrench.*` is the **same quantity** §5's debug GIF plots — net contact wrench,
`[Fx, Fy, Fz, Tx, Ty, Tz]` in the world frame with torque about that arm's TCP — computed by the
shared `envs/utils/wrench.py::wrench_vectors` so the eval and collection paths cannot drift
apart. There is one column per end-effector link (aloha agilex:
`observation.wrench.{fl_link7,fl_link8,fr_link7,fr_link8}`) **and** one per arm
(`observation.wrench.{left,right}`, each the sum of that arm's links), so a critic can be trained
offline on whichever granularity it observes online; at 480 B/row apiece the six together are
under 3 KB/row, cheaper than choosing. The rate differs from the debug plots': `eval_policy.py`
samples once per policy call for those, while the dataset (and the critic's own view) samples
after **every** primitive step, so a row carries a whole `(pi0_step, 6)` trace rather than a
single vector. Datasets collected before 2026-08-28 have only the two arm columns.

Which trace matters: it is the one the **previous** chunk produced — the steps between the
previous row's observation and this one. The env logs a sample after each `take_action`
(`_base_task.py::_log_step_wrench`, so the reading belongs to the action that just executed),
and the rollout loop drains the log with `pop_step_wrench()` **before** running the chunk, at
the same point in the loop the guided eval path drains it (§5a). The column is therefore an
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
since zero is a meaningful reading (the arm touching nothing). At 480 B/row apiece they are
still the cheapest columns here — six of them for aloha come to under 3 KB/row, against the
576 KB a single SigLIP view costs.

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
