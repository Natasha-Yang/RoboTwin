"""Convert RoboTwin demo HDF5 into a LeRobot dataset that KEEPS the extra modalities.

The stock path (scripts/process_data.py -> convert_aloha_data_to_lerobot_robotwin.py)
reads only /joint_action and /observation/<cam>/rgb, so depth, point cloud, wrench,
endpose and the camera matrices are dropped before the converter ever runs. This
reads the collected episodes directly and carries all of it through.

Column names follow CLAUDE.md 7a/7b so a critic trained offline on a rollout dataset
lines up with one trained here. The three RGB cameras keep their pi0.5 names
(cam_high / cam_left_wrist / cam_right_wrist) so the same dataset still feeds the
pi0.5 repack transform, which simply ignores every other column.

Example:
    .venv/bin/python examples/aloha_real/convert_robotwin_multimodal_to_lerobot.py \
        --data-dir ../../data --task-config demo_clean_multimodal --episodes-per-task 10 \
        --repo-id NatashaYang/robotwin_multimodal_full_50x10_lerobot
"""

import dataclasses
import json
import os
from pathlib import Path
import shutil
from typing import Literal

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import numpy as np
import tqdm
import tyro

# sim camera name -> the suffix used in the dataset columns (CLAUDE.md 7b)
CAMERA_SUFFIX = {
    "head_camera": "head",
    "left_camera": "left_wrist",
    "right_camera": "right_wrist",
    "front_camera": "front",
}
# ... and the pi0.5-facing names for the three cameras the policy actually sees
RGB_NAME = {"head": "cam_high", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}
CAM_MATRICES = ("intrinsic_cv", "extrinsic_cv", "cam2world_gl")

MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


def episode_files(data_dir: Path, task_config: str, episodes_per_task: int) -> list[tuple[str, Path]]:
    """(task_name, episode hdf5) for every task that has this config, in a stable order."""
    out = []
    for task_dir in sorted(p for p in data_dir.iterdir() if (p / task_config / "data").is_dir()):
        for i in range(episodes_per_task):
            ep = task_dir / task_config / "data" / f"episode{i}.hdf5"
            if ep.exists():
                out.append((task_dir.name, ep))
    return out


def probe_schema(path: Path) -> dict:
    """Read one episode's layout so the feature dict follows the task config's data_type."""
    with h5py.File(path, "r") as f:
        obs = f["observation"]
        cams = [c for c in obs if c in CAMERA_SUFFIX]
        schema = {
            "cameras": sorted(cams, key=lambda c: list(CAMERA_SUFFIX).index(c)),
            "has_depth": {c: "depth" in obs[c] for c in cams},
            "has_matrices": {c: all(m in obs[c] for m in CAM_MATRICES) for c in cams},
            "depth_shape": None,
            "pointcloud_shape": tuple(f["pointcloud"].shape[1:]) if "pointcloud" in f else None,
            "wrench_links": sorted(f["wrench"].keys()) if "wrench" in f else [],
            "wrench_shape": None,
            "has_endpose": "endpose" in f,
        }
        for c in cams:
            if schema["has_depth"][c]:
                schema["depth_shape"] = tuple(obs[c]["depth"].shape[1:])
                break
        if schema["wrench_links"]:
            schema["wrench_shape"] = tuple(f["wrench"][schema["wrench_links"][0]].shape[1:])
        schema["rgb_shape"] = cv2.imdecode(
            np.frombuffer(obs[cams[0]]["rgb"][0], np.uint8), cv2.IMREAD_COLOR
        ).shape
    return schema


def build_features(schema: dict, depth_dtype: str) -> dict:
    h, w, _ = schema["rgb_shape"]
    features: dict = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (14,), "names": [MOTORS]},
    }
    for cam in schema["cameras"]:
        sfx = CAMERA_SUFFIX[cam]
        key = RGB_NAME.get(sfx, sfx)
        features[f"observation.images.{key}"] = {
            "dtype": "image", "shape": (3, h, w), "names": ["channels", "height", "width"],
        }
        if schema["has_depth"][cam]:
            features[f"observation.depth.{sfx}"] = {
                "dtype": depth_dtype, "shape": schema["depth_shape"], "names": ["height", "width"],
            }
        if schema["has_matrices"][cam]:
            for m in CAM_MATRICES:
                features[f"observation.camera.{sfx}.{m}"] = {
                    "dtype": "float32", "shape": {"intrinsic_cv": (3, 3), "extrinsic_cv": (3, 4),
                                                  "cam2world_gl": (4, 4)}[m], "names": None,
                }
    if schema["pointcloud_shape"]:
        features["observation.pointcloud"] = {
            "dtype": "float32", "shape": schema["pointcloud_shape"], "names": ["points", "xyzrgb"],
        }
    for link in schema["wrench_links"]:
        features[f"observation.wrench.{link}"] = {
            "dtype": "float32", "shape": schema["wrench_shape"], "names": ["step", "FxFyFzTxTyTz"],
        }
    if schema["has_endpose"]:
        for side in ("left", "right"):
            features[f"observation.endpose.{side}_endpose"] = {
                "dtype": "float32", "shape": (7,), "names": ["xyz_quat"]}
            features[f"observation.endpose.{side}_gripper"] = {
                "dtype": "float32", "shape": (1,), "names": None}
    return features


def load_episode(path: Path, schema: dict, depth_dtype: str) -> dict:
    """Every column of one episode, already trimmed to the state/action alignment.

    Matches the stock pipeline: frame i pairs state qpos[i] with action qpos[i+1], so an
    N-frame episode yields N-1 rows and every other modality is sliced to [:-1] to stay on
    the observation side of that pair.
    """
    with h5py.File(path, "r") as f:
        ja = f["joint_action"]
        qpos = np.concatenate(
            [ja["left_arm"][:], ja["left_gripper"][:][:, None],
             ja["right_arm"][:], ja["right_gripper"][:][:, None]], axis=1
        ).astype(np.float32)
        n = qpos.shape[0] - 1
        if n <= 0:
            raise ValueError(f"{path}: needs >= 2 frames, got {qpos.shape[0]}")

        cols: dict = {"observation.state": qpos[:-1], "action": qpos[1:]}
        obs = f["observation"]
        for cam in schema["cameras"]:
            sfx = CAMERA_SUFFIX[cam]
            raw = obs[cam]["rgb"][:]
            # cv2.imdecode round-trips whatever cv2.imencode wrote in pkl2hdf5.py, so the
            # channel order matches the stock converter's; do not "fix" it here.
            cols[f"observation.images.{RGB_NAME.get(sfx, sfx)}"] = np.stack(
                [cv2.imdecode(np.frombuffer(raw[i], np.uint8), cv2.IMREAD_COLOR) for i in range(n)]
            )
            if schema["has_depth"][cam]:
                d = obs[cam]["depth"][:n]
                if depth_dtype == "uint16":
                    # depth is float32 millimetres; store 1 mm integers. Non-finite and
                    # out-of-range samples clamp to 0, which the sim already uses for
                    # "no return", and 65535 mm is far beyond any scene extent here.
                    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                    d = np.clip(np.rint(d), 0, 65535).astype(np.uint16)
                else:
                    d = d.astype(np.float32)
                cols[f"observation.depth.{sfx}"] = d
            if schema["has_matrices"][cam]:
                for m in CAM_MATRICES:
                    cols[f"observation.camera.{sfx}.{m}"] = obs[cam][m][:n].astype(np.float32)
        if schema["pointcloud_shape"]:
            cols["observation.pointcloud"] = f["pointcloud"][:n].astype(np.float32)
        for link in schema["wrench_links"]:
            cols[f"observation.wrench.{link}"] = f["wrench"][link][:n].astype(np.float32)
        if schema["has_endpose"]:
            ep = f["endpose"]
            for side in ("left", "right"):
                cols[f"observation.endpose.{side}_endpose"] = ep[f"{side}_endpose"][:n].astype(np.float32)
                cols[f"observation.endpose.{side}_gripper"] = (
                    ep[f"{side}_gripper"][:n].astype(np.float32).reshape(n, 1))
    return cols


def main(
    repo_id: str,
    data_dir: Path = Path("../../data"),
    task_config: str = "demo_clean_multimodal",
    episodes_per_task: int = 10,
    depth_dtype: Literal["uint16", "float32"] = "uint16",
    instruction_type: str = "seen",
    seed: int = 42,
    limit_tasks: int | None = None,
    dataset_config: DatasetConfig = DatasetConfig(),
):
    data_dir = data_dir.resolve()
    eps = episode_files(data_dir, task_config, episodes_per_task)
    if not eps:
        raise ValueError(f"no episodes under {data_dir}/*/{task_config}/data")
    if limit_tasks is not None:
        keep = sorted({t for t, _ in eps})[:limit_tasks]
        eps = [(t, p) for t, p in eps if t in keep]
    tasks = sorted({t for t, _ in eps})
    print(f"{len(eps)} episodes across {len(tasks)} tasks ({task_config}, depth={depth_dtype})")

    schema = probe_schema(eps[0][1])
    print(f"  cameras   : {schema['cameras']}")
    print(f"  rgb       : {schema['rgb_shape']}   depth: {schema['depth_shape']}")
    print(f"  pointcloud: {schema['pointcloud_shape']}   wrench: {schema['wrench_links']} {schema['wrench_shape']}")

    features = build_features(schema, depth_dtype)
    print(f"  {len(features)} columns")

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type="aloha",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

    rng = np.random.default_rng(seed)
    for task_name, ep_path in tqdm.tqdm(eps):
        instr_path = ep_path.parent.parent / "instructions" / f"{ep_path.stem}.json"
        with open(instr_path) as fh:
            instructions = json.load(fh)[instruction_type]
        instruction = str(rng.choice(instructions))

        cols = load_episode(ep_path, schema, depth_dtype)
        num_frames = cols["observation.state"].shape[0]
        for i in range(num_frames):
            frame = {k: v[i] for k, v in cols.items()}
            frame["task"] = instruction
            dataset.add_frame(frame)
        dataset.save_episode()

    print(f"done: {HF_LEROBOT_HOME / repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
