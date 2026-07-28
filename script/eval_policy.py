"""Policy evaluation driver (expert feasibility check, seed loop, per-chunk rollout).

If the policy exposes a ``model.online_critic`` (pi05 builds one only when
``guidance_scale != 0``), the rollout additionally collects a chunk-level transition after
every control step and runs TD updates on that critic, which steers the frozen pi0.5 flow
sampler; the critic persists and keeps learning across episodes for the whole eval run, and
progress is logged to W&B. With no critic this is the plain baseline rollout. Configure via
``policy/<policy_name>/deploy_policy.yml``.
"""

import sys
import os
import re
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from envs.utils.wrench import WRENCH_COMPONENTS, tcp_wrench_vector

import numpy as np
from pathlib import Path
from collections import deque
import traceback
import pandas as pd

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = Path(__file__).resolve()
parent_directory = current_file_path.parent


# ===== Online critic logging =====
# Only used when the policy exposes a trained-online critic (pi05 with guidance_scale != 0);
# the plain baseline never reaches any of this, so wandb stays an optional dependency.


def init_wandb(usr_args, save_dir, current_time):
    import wandb

    if not usr_args.get("wandb_enabled", True):
        wandb.init(mode="disabled")
        return None

    run_name = usr_args.get("wandb_run_name")
    if run_name is None:
        run_name = (
            f"{usr_args['task_name']}-{usr_args['policy_name']}-"
            f"{usr_args.get('model_name', 'model')}-seed{usr_args['seed']}-{current_time}"
        )

    init_kwargs = {
        "project": usr_args.get("wandb_project", "qmfm"),
        "name": run_name,
        "config": {**usr_args, "eval_save_dir": str(save_dir)},
        "dir": str(save_dir),
        "tags": ["pi05", "online-critic", "eval"],
        "entity": "natashayang04-university-of-toronto",
    }
    wandb_mode = usr_args.get("wandb_mode", None)
    if wandb_mode is not None:
        init_kwargs["mode"] = wandb_mode

    run = wandb.init(**init_kwargs)
    wandb.define_metric("critic/update")
    wandb.define_metric("critic/*", step_metric="critic/update")
    wandb.define_metric("rollout/*", step_metric="critic/update")
    wandb.define_metric("eval/episode")
    wandb.define_metric("eval/*", step_metric="eval/episode")
    return run


def _uses_online_critic(model):
    """Whether this policy intends to run a trained-online critic.

    Distinct from `model.online_critic`, which stays None until the critic is actually built
    -- pi05 defers that to the first observation, since the critic's action width comes from
    the embodiment's joint vector.
    """
    return bool(getattr(model, "uses_online_critic", getattr(model, "online_critic", None) is not None))


def _scalar(value):
    return float(np.asarray(value))


def _window_mean(values):
    return float(np.mean(values)) if values else 0.0


def log_critic_update(wandb_run, online_critic, model, info, chunk_count, episode_idx, action_count):
    if wandb_run is None or info is None:
        return
    buf = online_critic.buffer.size if online_critic.buffer is not None else 0
    wandb_run.log({
        "critic/update": int(online_critic.num_updates),
        "critic/loss": _scalar(info["critic_loss"]),
        "critic/q_mean": _scalar(info["q_mean"]),
        "critic/q_max": _scalar(info["q_max"]),
        "critic/q_min": _scalar(info["q_min"]),
        "critic/target_q_mean": _scalar(info["target_q_mean"]),
        "critic/reward_mean": _scalar(info["reward_mean"]),
        "critic/buffer_size": int(buf),
        "critic/guidance_scale": float(model.scheduled_guidance_scale()),
        "critic/guidance_scale_target": float(model.guidance_scale_target),
        "rollout/chunk_count": int(chunk_count),
        "rollout/episode": int(episode_idx),
        "rollout/action_count": int(action_count),
    })


def log_episode(
    wandb_run,
    online_critic,
    model,
    episode_idx,
    success,
    num_steps,
    episode_reward,
    success_rate,
    success_rate_ma,
    reward_ma,
    ma_window,
):
    if wandb_run is None:
        return
    metrics = {
        "eval/episode": int(episode_idx),
        "eval/success": float(success),
        "eval/num_steps": int(num_steps),
        "eval/reward": float(episode_reward),
        "eval/success_rate": float(success_rate),
        "eval/success_rate_ma": float(success_rate_ma),
        "eval/reward_ma": float(reward_ma),
        "eval/ma_window": int(ma_window),
    }
    if online_critic is not None:
        buf = online_critic.buffer.size if online_critic.buffer is not None else 0
        metrics.update({
            "critic/update": int(online_critic.num_updates),
            "critic/buffer_size": int(buf),
            "critic/guidance_scale": float(model.scheduled_guidance_scale()),
            "critic/guidance_scale_target": float(model.guidance_scale_target),
        })
    wandb_run.log(metrics)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


# ===== TCP wrench debug logging =====
# Also gated on `debug: true` in the task config. Unlike the image/point-cloud dump above,
# this needs the live env (contacts are a scene query, not part of the observation), so the
# recorder takes TASK_ENV and is driven from the same visualize_debug_obs call site.
# The wrench itself is computed in `envs/utils/wrench.py`, shared with the per-step logging
# `script/collect_dataset.py` records into the rollout dataset.

WRENCH_GIF_FRAME_WIDTH = 320  # rollout frames are downscaled to this before being kept in RAM
WRENCH_GIF_FPS = 5
WRENCH_GIF_MAX_FRAMES = 200  # long episodes are subsampled; the traces still cover every sample
WRENCH_AXIS_LENGTH = 0.08  # metres; length of the world frame arrows drawn on the rollout
WRENCH_LABEL_OFFSET = 7  # points past the arrow tip, along the arrow, to place its label
# One colour per axis, shared by the trace lines and the arrows drawn on the rollout, so the
# x/y/z arrow and its Fx/Tx trace read as the same thing.
WRENCH_AXIS_COLORS = ("tab:blue", "tab:orange", "tab:green")  # x, y, z


class TCPWrenchRecorder:
    """Per-episode TCP wrench log -> component histograms + a rollout/wrench GIF.

    One sample is taken per policy call (the rate `visualize_debug_obs` is called at, i.e.
    every `pi0_step` sim frames), paired with the head-camera frame from the same
    observation. ``flush`` writes three files into the episode's own debug dir, alongside
    the image/point-cloud dumps `visualize_debug_obs` puts there:
    ``<debug_save_dir>/episode<N>/`` gets ``wrench_hist_episode<N>.png``,
    ``wrench_episode<N>.gif`` and ``wrench_episode<N>.npz``.
    """

    def __init__(self, debug_save_dir):
        self.debug_save_dir = Path(debug_save_dir)
        self._reset()

    def episode_dir(self, episode_idx):
        return self.debug_save_dir / f"episode{episode_idx}"

    def _reset(self):
        self.steps = []
        self.wrench = {arm: [] for arm in ("left", "right")}
        self.frames = []
        self.world_axes = []

    def record(self, task_env, observation, step_idx):
        try:
            wrench = tcp_wrench_vector(task_env)
        except Exception as e:
            print(f"[debug] TCP wrench sampling failed: {e}")
            return
        self.steps.append(step_idx)
        for arm, vector in wrench.items():
            self.wrench[arm].append(vector)

        rgb = observation.get("observation", {}).get("head_camera", {}).get("rgb", None)
        if rgb is not None:
            from PIL import Image

            img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
            scale = 1.0
            if img.width > WRENCH_GIF_FRAME_WIDTH:  # keep the kept-in-RAM rollout small
                scale = WRENCH_GIF_FRAME_WIDTH / img.width
                img = img.resize((WRENCH_GIF_FRAME_WIDTH, max(1, round(img.height * scale))), Image.BILINEAR)
            self.frames.append(np.asarray(img, dtype=np.uint8))
            self.world_axes.append(self._project_world_axes(task_env, observation, scale))

    @staticmethod
    def _project_world_axes(task_env, observation, scale):
        """The world axes, anchored at each TCP, as head-camera pixels at the frame's scale.

        The wrench is resolved in world axes, so those are what the rollout should show; they
        are anchored at each arm's TCP because that is the point the wrench acts on (and it
        keeps the triad in frame, unlike the world origin). Both triads therefore point the
        same way and only their origins differ. Returns ``{arm: (origin_uv, tips_uv(3, 2))}``,
        skipping an arm whose TCP is behind the camera; anything merely outside the image is
        clipped when it is drawn.
        """
        cam = observation.get("observation", {}).get("head_camera", {})
        if "intrinsic_cv" not in cam or "extrinsic_cv" not in cam:
            return {}  # camera matrices missing: draw no axes rather than guess
        K = np.asarray(cam["intrinsic_cv"], dtype=np.float64)
        ext = np.asarray(cam["extrinsic_cv"], dtype=np.float64)[:3]  # world -> camera (OpenCV)

        out = {}
        for arm_tag in ("left", "right"):
            origin = np.asarray(getattr(task_env.robot, f"get_{arm_tag}_tcp_pose")(), dtype=np.float64)[:3]
            # (4, 3): origin, then the world x/y/z unit axes stepped out from it
            pts_world = np.vstack([origin, origin + WRENCH_AXIS_LENGTH * np.eye(3)])
            pts_cam = pts_world @ ext[:, :3].T + ext[:, 3]
            if np.any(pts_cam[:, 2] <= 1e-6):  # at or behind the image plane: not projectable
                continue
            uv = (pts_cam @ K.T)[:, :2] / pts_cam[:, 2:3] * scale
            out[arm_tag] = (uv[0], uv[1:])
        return out

    def flush(self, episode_idx):
        """Render this episode's outputs and start a fresh episode. No-op with no samples."""
        if not self.steps:
            self._reset()
            return
        out_dir = self.episode_dir(episode_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        steps = np.asarray(self.steps)
        series = {arm: np.asarray(vals) for arm, vals in self.wrench.items()}
        try:
            self._save_histograms(out_dir, episode_idx, series)
            self._save_gif(out_dir, episode_idx, steps, series)
            np.savez_compressed(
                out_dir / f"wrench_episode{episode_idx}.npz",
                step=steps,
                components=np.array(WRENCH_COMPONENTS),
                **{arm: vals for arm, vals in series.items()},
            )
            print(f"\033[93m[debug] wrench log written to {out_dir}/wrench_*\033[0m")
        except Exception as e:
            print(f"[debug] TCP wrench output failed: {e}")
        self._reset()

    def _save_histograms(self, out_dir, episode_idx, series):
        """One histogram per wrench component, both arms overlaid."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        for i, name in enumerate(WRENCH_COMPONENTS):
            ax = axes[i // 3, i % 3]
            for arm, color in (("left", "tab:blue"), ("right", "tab:orange")):
                vals = series[arm][:, i]
                ax.hist(vals, bins=40, alpha=0.55, color=color,
                        label=f"{arm}: {vals.mean():+.3g} ± {vals.std():.3g}")
            ax.set_xlabel(f"{name} [{'N' if i < 3 else 'N·m'}]")
            ax.set_ylabel("policy calls")
            # Most of an episode is free space, i.e. an exact zero; log counts keep the
            # contact tail readable next to that spike.
            ax.set_yscale("log")
            ax.legend(fontsize="small")
        fig.suptitle(f"episode {episode_idx} — TCP wrench distribution, world frame "
                     f"({len(series['left'])} samples)")
        fig.tight_layout()
        fig.savefig(out_dir / f"wrench_hist_episode{episode_idx}.png", dpi=100)
        plt.close(fig)

    def _draw_world_axes(self, ax, frame_idx):
        """Overlay the world frame on the rollout as labelled x/y/z arrows, one triad per TCP.

        These are the axes the force and torque traces are resolved in: the Fx trace is the
        contact force along this arrow, Tx the moment about it (taken about the TCP the triad
        sits on).
        """
        import matplotlib.patheffects as pe

        if frame_idx >= len(self.world_axes):
            return
        for arm_tag, (origin, tips) in self.world_axes[frame_idx].items():
            for tip, label, color in zip(tips, "xyz", WRENCH_AXIS_COLORS):
                ax.annotate("", xy=tip, xytext=origin, annotation_clip=True,
                            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6,
                                            shrinkA=0, shrinkB=0))
                # Offset the label along its own arrow rather than a fixed direction: a world
                # axis pointing near the camera projects short, and two such arrows can end up
                # close together, so a fixed offset lets one arm's label drift onto its
                # neighbour's arrow and read as swapped. (dy flips: image y grows downward,
                # offset-point y grows upward.)
                d = np.asarray(tip, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
                norm = float(np.linalg.norm(d)) or 1.0
                ax.annotate(f"{arm_tag[0]}{label}", xy=tip,
                            xytext=WRENCH_LABEL_OFFSET * d / norm * (1, -1), textcoords="offset points",
                            ha="center", va="center",
                            color=color, fontsize="x-small", fontweight="bold", annotation_clip=True,
                            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])

    def _save_gif(self, out_dir, episode_idx, steps, series):
        """Rollout on the left, the wrench traces with a step cursor on the right."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        if not self.frames:
            return
        n = min(len(self.frames), len(steps))
        # Fixed limits across frames so only the cursor moves.
        lims = {}
        for row, sl in (("force", slice(0, 3)), ("torque", slice(3, 6))):
            vals = np.concatenate([series[arm][:n, sl].ravel() for arm in ("left", "right")])
            span = max(float(np.abs(vals).max()), 1e-6) * 1.1
            lims[row] = (-span, span)

        gif_frames = []
        for k in range(0, n, max(1, -(-n // WRENCH_GIF_MAX_FRAMES))):
            fig = plt.figure(figsize=(12, 5.5))
            gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1, 1])
            ax_img = fig.add_subplot(gs[:, 0])
            ax_img.imshow(self.frames[k])
            ax_img.axis("off")
            ax_img.set_title(f"rollout — step {steps[k]}")
            self._draw_world_axes(ax_img, k)
            for r, (row, sl) in enumerate((("force", slice(0, 3)), ("torque", slice(3, 6)))):
                for c, arm in enumerate(("left", "right")):
                    ax = fig.add_subplot(gs[r, c + 1])
                    for j, comp in enumerate(WRENCH_COMPONENTS[sl]):
                        ax.plot(steps[:n], series[arm][:n, sl][:, j], linewidth=1.0,
                                color=WRENCH_AXIS_COLORS[j], label=comp)
                    ax.axvline(steps[k], color="k", linewidth=1.2)
                    ax.set_xlim(steps[0], max(steps[n - 1], steps[0] + 1))
                    ax.set_ylim(*lims[row])
                    if r == 0:  # units live on the y axis, so the title only names the arm
                        ax.set_title(f"{arm} arm TCP (world frame)", fontsize="small")
                    ax.set_xlabel("sim step", fontsize="x-small")
                    ax.set_ylabel(f"{row} [{'N' if row == 'force' else 'N·m'}]", fontsize="x-small")
                    ax.tick_params(labelsize="x-small")
                    ax.legend(fontsize="xx-small", ncol=3, loc="upper right")
            fig.tight_layout()
            fig.canvas.draw()
            gif_frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
            plt.close(fig)

        gif_frames[0].save(
            out_dir / f"wrench_episode{episode_idx}.gif",
            save_all=True,
            append_images=gif_frames[1:],
            duration=int(1000 / WRENCH_GIF_FPS),
            loop=0,
        )


def visualize_debug_obs(observation, step_idx=0, save_dir=None, show=True, task_env=None, wrench_recorder=None):
    """Visualize per-camera images (rgb / depth / segmentation) and the point cloud.

    When a ``wrench_recorder`` is passed it also samples the end-effector contact wrench
    from ``task_env`` at this step (see TCPWrenchRecorder); that part needs neither depth
    nor point clouds, so it works under any task config that sets `debug: true`.

    Enabled by `debug: true` in the task config. The observation layout follows
    ``_base_task.get_obs`` (see envs/_base_task.py); each entry is present only when
    the matching ``data_type`` flag is enabled:
      - observation["observation"][camera_name]["rgb"]:                H x W x 3 uint8
      - observation["observation"][camera_name]["depth"]:              H x W float64 depth in mm
      - observation["observation"][camera_name]["mesh_segmentation"]:  H x W x 3 uint8 (palette-colored)
      - observation["observation"][camera_name]["actor_segmentation"]: H x W x 3 uint8 (palette-colored)
      - observation["pointcloud"]:                                     N x 6 (xyz + rgb in [0, 1])

    Whichever image modalities are present get tiled into one figure (rows = modality,
    columns = camera). When ``show`` is True the figure/point-cloud open interactively
    (blocking, so close each window to step forward). PNG/PLY copies are always written
    to ``save_dir`` when it is provided, which keeps debug output usable headlessly.
    """
    if wrench_recorder is not None and task_env is not None:
        wrench_recorder.record(task_env, observation, step_idx)

    import matplotlib
    if not show:
        matplotlib.use("Agg")  # no display: render to file only
    import matplotlib.pyplot as plt

    obs = observation.get("observation", {})
    # Per-camera image modalities to tile, in display order (rendered when present).
    image_modalities = ["rgb", "depth", "mesh_segmentation", "actor_segmentation"]
    cam_names = [name for name, data in obs.items()
                 if isinstance(data, dict) and any(m in data for m in image_modalities)]
    # Rows = only the modalities that at least one camera actually carries.
    rows = [m for m in image_modalities if any(m in obs[name] for name in cam_names)]

    if save_dir is not None and (rows or observation.get("pointcloud", [])):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ---- Image modalities (rgb / depth / mesh & actor segmentation) ----
    if cam_names and rows:
        n, r = len(cam_names), len(rows)
        fig, axes = plt.subplots(r, n, figsize=(4 * n, 3.5 * r), squeeze=False)
        for ci, name in enumerate(cam_names):
            cam = obs[name]
            for ri, mod in enumerate(rows):
                ax = axes[ri, ci]
                ax.axis("off")
                if mod not in cam:
                    continue
                if mod == "depth":
                    # Depth stored in mm; convert to meters and mask invalid (0) pixels.
                    depth_m = np.asarray(cam["depth"], dtype=np.float32) / 1000.0
                    masked = np.ma.masked_where(depth_m <= 0, depth_m)
                    im = ax.imshow(masked, cmap="turbo")
                    ax.set_title(f"{name} depth [m]")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    # rgb / mesh_segmentation / actor_segmentation are H x W x 3 uint8.
                    ax.imshow(cam[mod])
                    ax.set_title(f"{name} {mod}")
        fig.suptitle(f"step {step_idx}")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(save_dir / f"obs_step{step_idx:04d}.png", dpi=100)
        if show:
            plt.show()
        plt.close(fig)

    # ---- Point cloud ----
    pcd = np.asarray(observation.get("pointcloud", []))
    if pcd.ndim == 2 and pcd.shape[0] > 0:
        try:
            import open3d as o3d

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
            if pcd.shape[1] >= 6:
                cloud.colors = o3d.utility.Vector3dVector(np.clip(pcd[:, 3:6], 0.0, 1.0))
            if save_dir is not None:
                # open3d's bindings take a str filename, not a PathLike.
                o3d.io.write_point_cloud(str(Path(save_dir) / f"pcd_step{step_idx:04d}.ply"), cloud)
            if show:
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                o3d.visualization.draw_geometries([cloud, frame], window_name=f"pointcloud step {step_idx}")
        except Exception as e:
            print(f"[debug] point cloud visualization failed: {e}")

def get_camera_config(camera_type):
    camera_config_path = parent_directory.parent / "task_config" / "_camera_config.yml"

    assert camera_config_path.is_file(), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = Path(robot_file) / "config.yml"
    with robot_config_file.open("r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def snapshot_config(src_path, values, dst_dir):
    """Copy a yml config into `dst_dir` with the values actually used written in.

    Comments and layout are preserved; a line is rewritten only when `values` resolves that
    top-level key differently (a CLI override, or -- in the critic config -- a key the deploy
    config shadows). Keys whose value opens a block on the next line are skipped: nothing
    overrides those.
    """
    src_path = Path(src_path)
    with src_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    original = yaml.safe_load("".join(lines)) or {}

    for i, line in enumerate(lines):
        match = re.match(r"^([A-Za-z_][\w.-]*):[ \t]+(\S[^\n]*?)[ \t]*$", line)
        if match is None:  # comment, blank, indented, or a key with a block value
            continue
        key, raw_value = match.group(1), match.group(2)
        if key not in values or values[key] == original.get(key):
            continue
        # Keep any trailing `# ...` comment, unless the value itself is quoted (a `#` could
        # then be part of the string rather than the start of a comment).
        comment = ""
        if not raw_value.startswith(("'", '"')):
            hash_at = raw_value.find(" #")
            if hash_at != -1:
                comment = "  " + raw_value[hash_at:].lstrip()
        lines[i] = yaml.safe_dump({key: values[key]}, sort_keys=False).strip() + comment + "\n"

    header = f"# Snapshot of {src_path.resolve()} as used by this eval run.\n"
    with (Path(dst_dir) / src_path.name).open("w", encoding="utf-8") as f:
        f.writelines([header] + lines)


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with (Path("./task_config") / f"{task_config}.yml").open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"

    with embodiment_config_path.open("r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with (Path(CONFIGS_PATH) / "_camera_config.yml").open("r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)
    args["eval_save_dir"] = str(save_dir)

    # Snapshot the deploy config (and the critic config it includes) next to the results, with
    # the CLI overrides written in, so a run's settings stay readable -- and accurate -- after
    # the ymls are edited.
    for config_path in (usr_args.get("_config_path"), usr_args.get("critic_config_path")):
        if config_path and Path(config_path).is_file():
            snapshot_config(config_path, usr_args, save_dir)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    # Online-critic knobs (see deploy_policy.yml) forwarded into the eval loop; inert unless
    # the policy actually built a critic.
    args["train_freq"] = usr_args.get("train_freq", 1)
    args["wandb_ma_window"] = usr_args.get("wandb_ma_window", 20)

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = usr_args.get("test_num", 100)
    topk = 1

    model = get_model(usr_args)
    # The policy decides whether guidance is on (pi05: guidance_scale != 0). The critic object
    # itself may not exist until the first observation (its shape depends on the embodiment),
    # so W&B keys off the policy's declared intent rather than off `model.online_critic`.
    wandb_run = init_wandb(usr_args, save_dir, current_time) if _uses_online_critic(model) else None

    st_seed, suc_num, episode_results = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   wandb_run=wandb_run)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = save_dir / "_result.txt"
    with file_path.open("w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    episode_file_path = save_dir / "_episode_results.csv"
    # write num_steps and success to csv file
    episode_results_df = pd.DataFrame(episode_results)
    episode_results_df.to_csv(episode_file_path)

    # Persist the online-trained critic alongside the eval results.
    online_critic = getattr(model, "online_critic", None)
    if online_critic is not None and usr_args.get("save_critic", False):
        critic_path = save_dir / "online_value_critic.pkl"
        online_critic.save(critic_path)
        print(f"saved online critic to {critic_path}")

    print(f"Data has been saved to {file_path}")
    if wandb_run is not None:
        wandb_run.finish()
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                wandb_run=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    episode_results = {"num_steps": [], "success": [], "reward": [], "success_rate_ma": [], "reward_ma": []}

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    # ===== Online QMFM Value critic (trained across the whole eval run) =====
    # The critic object is read fresh via `getattr(model, "online_critic", None)` at each use
    # site rather than captured here: pi05 builds it lazily on the first observation, because
    # its action width comes from the embodiment. It stays None for the plain baseline, which
    # makes every critic block below inert.
    train_freq = int(args.get("train_freq", 1))
    ma_window = max(1, int(args.get("wandb_ma_window", 20)))
    success_window = deque(maxlen=ma_window)
    reward_window = deque(maxlen=ma_window)
    chunk_count = 0
    last_info = None

    # Debug visualization of depth maps / point clouds (see visualize_debug_obs).
    debug = args.get("debug", False)
    debug_show = bool(os.environ.get("DISPLAY"))  # only pop up windows when a display exists
    save_dir = Path(args.get("eval_save_dir", "eval_result"))
    debug_save_dir = save_dir / "debug_vis" if debug else None
    # Both share `debug_vis/episode<N>/`: the recorder appends the episode dir itself, since it
    # only learns the episode index at flush time.
    wrench_recorder = TCPWrenchRecorder(debug_save_dir) if debug else None
    if debug:
        print(f"\033[93m[debug] depth/point-cloud visualization ON "
              f"(interactive={debug_show}, saving to {debug_save_dir})\033[0m")
        print(f"\033[93m[debug] TCP wrench logging ON (saving to {debug_save_dir}/episode<N>/)\033[0m")

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        expert_success = False
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                expert_success = TASK_ENV.plan_success and TASK_ENV.check_success()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", e)
                print(stack_trace)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or expert_success:
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        prev_success = False
        episode_reward = 0.0
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            if debug:
                visualize_debug_obs(
                    observation,
                    step_idx=TASK_ENV.take_action_cnt,
                    save_dir = (
                        debug_save_dir / f"episode{TASK_ENV.test_num}"
                        if debug_save_dir else None
                    ),
                    show=debug_show,
                    task_env=TASK_ENV,
                    wrench_recorder=wrench_recorder,
                )
            eval_func(TASK_ENV, model, observation)
            success_now = bool(TASK_ENV.eval_success)
            reward = 1.0 if (success_now and not prev_success) else 0.0
            episode_reward += reward

            # Online critic: close the chunk transition (SARSA), then run a TD update.
            online_critic = getattr(model, "online_critic", None)
            if online_critic is not None:
                done = success_now or (TASK_ENV.take_action_cnt >= TASK_ENV.step_lim)
                online_critic.commit(reward, done)
                chunk_count += 1
                if chunk_count % train_freq == 0:
                    info = online_critic.train_step()
                    if info is not None:
                        last_info = info
                        log_critic_update(
                            wandb_run,
                            online_critic,
                            model,
                            info,
                            chunk_count,
                            TASK_ENV.test_num,
                            TASK_ENV.take_action_cnt,
                        )

            prev_success = success_now
            if success_now:
                succ = True
                break
        episode_results["num_steps"].append(TASK_ENV.take_action_cnt)
        episode_results["success"].append(succ)
        episode_results["reward"].append(episode_reward)
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()
        if wrench_recorder is not None:
            wrench_recorder.flush(TASK_ENV.test_num)

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        # Online-critic diagnostics.
        online_critic = getattr(model, "online_critic", None)
        if online_critic is not None:
            buf = online_critic.buffer.size if online_critic.buffer is not None else 0
            guidance = model.scheduled_guidance_scale()
            if last_info is not None:
                print(f"\033[96m[critic]\033[0m buffer={buf} updates={online_critic.num_updates} "
                      f"guidance={guidance:.4g}/{model.guidance_scale_target:.4g} "
                      f"loss={float(last_info['critic_loss']):.4f} "
                      f"q_mean={float(last_info['q_mean']):.3f} "
                      f"target_q={float(last_info['target_q_mean']):.3f} "
                      f"reward_mean={float(last_info['reward_mean']):.3f}")
            else:
                print(f"\033[96m[critic]\033[0m buffer={buf} "
                      f"guidance={guidance:.4g}/{model.guidance_scale_target:.4g} "
                      f"(warming up, no update yet)")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1
        success_window.append(float(succ))
        reward_window.append(float(episode_reward))
        success_rate = TASK_ENV.suc / TASK_ENV.test_num
        success_rate_ma = _window_mean(success_window)
        reward_ma = _window_mean(reward_window)
        episode_results["success_rate_ma"].append(success_rate_ma)
        episode_results["reward_ma"].append(reward_ma)
        log_episode(
            wandb_run,
            online_critic,
            model,
            TASK_ENV.test_num,
            succ,
            TASK_ENV.take_action_cnt,
            episode_reward,
            success_rate,
            success_rate_ma,
            reward_ma,
            ma_window,
        )

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(success_rate*100, 1)}%\033[0m "
            f"(MA{ma_window}: \033[95m{round(success_rate_ma*100, 1)}%\033[0m, reward={reward_ma:.3f}), "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, episode_results


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Policy-specific hyperparameters may be factored out into a file that ships with the
    # implementation they configure (e.g. the critic config in the multisensory_steering
    # repo, so it stays in sync with the critic that reads it). Merge it in *underneath* the
    # deploy config: precedence is CLI overrides > deploy config > included file.
    include_path = config.get("critic_config_path")
    if include_path:
        with Path(include_path).open("r", encoding="utf-8") as f:
            included = yaml.safe_load(f) or {}
        config = {**included, **config}

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    # Kept so `main` can copy the deploy config into the eval_result dir.
    config["_config_path"] = args.config

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
