"""The sensor modalities a critic can condition on, pulled out of a RoboTwin observation.

`_base_task.get_obs` returns whatever the task config's `data_type` block turns on, nested by
camera. This flattens the numeric parts of it into one flat ``{name: ndarray}`` dict with stable
names, so a critic can be pointed at a modality by name and the same names mean the same thing
in the online (eval) and offline (rollout-dataset) paths:

    ``images.head``     ``images.left_wrist``  ``images.right_wrist``  (H, W, 3) uint8
    ``images.third_view``                                              (H, W, 3) uint8
    ``depth.head``      ``depth.left_wrist``   ``depth.right_wrist``   (H, W) float32, mm
    ``pointcloud``      (N, 6) float32, world-frame xyz + rgb
    ``wrench.<link>``   ``wrench.<arm>``       (num_steps, 6) float32, world-frame contact
                                               wrench at both granularities, see below

The names are the rollout dataset's column names minus their ``observation.`` prefix (see
`script/collect_dataset.py::extra_obs_columns`), which is what lets a critic trained offline on
those columns be warm-started into the online sampler.

Note that the rgb views overlap with what the policy already gives the critic: ``siglip.<view>``
is that same camera as seen by pi0.5's own image tower, for every view the policy has (head plus
both wrists). Taking one here as raw pixels is a second, much cheaper to train and much weaker,
encoding of a frame the critic can already see through the policy's encoder.

Not included yet: the segmentation / endpose / camera-matrix entries, which rollout datasets
record but no critic encoder consumes. Adding one is a line here plus an encoder in the critic.
"""

import numpy as np

from .wrench import stack_step_wrench

# Cameras keep their sim names in the observation dict (`head_camera`, `left_camera`, ...);
# modality names -- like the dataset's columns -- use the short suffixes the rgb columns use.
CAMERA_SUFFIX = {"head_camera": "head", "left_camera": "left_wrist", "right_camera": "right_wrist"}


def camera_suffix(name):
    if name in CAMERA_SUFFIX:
        return CAMERA_SUFFIX[name]
    return name[:-len("_camera")] if name.endswith("_camera") else name


def obs_modalities(observation, step_wrench=(), num_steps=0):
    """Flatten one observation into ``{modality_name: float32 ndarray}``.

    Driven off what the observation actually holds rather than off the `data_type` flags, so a
    task config with fewer of them simply yields fewer keys and the caller decides what to do
    about a modality its critic wanted. The wrench is the exception, as it is a scene query
    rather than part of the observation: pass the samples `_base_task.pop_step_wrench` collected
    over the last chunk as `step_wrench`, and the fixed trace length (`wrench_trace_len`) as
    `num_steps`. That is not `pi0_step` -- the env samples every physics step, and one control
    step runs a whole TOPP trajectory of them -- and it fixes the critic's obs shape, so it has
    to match whatever a warm-start checkpoint or an offline rollout dataset was made with.

    Arrays are handed over raw, in their natural dtype -- rgb as uint8, depth in millimetres,
    point clouds in world metres plus 0-255 rgb, wrench in N / N*m -- and carry NaN where a
    chunk ended early. Scaling and NaN handling belong to the critic's per-modality encoders,
    which are the only thing that knows what range its network wants; keeping the raw dtype
    here also keeps a camera frame at a quarter of the bytes through the replay buffer.
    """
    mods = {}
    for cam_name, cam_obs in observation.get("observation", {}).items():
        suffix = camera_suffix(cam_name)
        if "rgb" in cam_obs:
            mods[f"images.{suffix}"] = np.asarray(cam_obs["rgb"], dtype=np.uint8)
        if "depth" in cam_obs:
            mods[f"depth.{suffix}"] = np.asarray(cam_obs["depth"], dtype=np.float32)

    if "third_view_rgb" in observation:
        mods["images.third_view"] = np.asarray(observation["third_view_rgb"], dtype=np.uint8)

    pointcloud = observation.get("pointcloud", [])
    if len(pointcloud) > 0:
        mods["pointcloud"] = np.asarray(pointcloud, dtype=np.float32)

    # Both granularities, since the env logs both (`envs/utils/wrench.py::wrench_vectors`): one
    # trace per end-effector link -- aloha's `wrench.fl_link7`, `wrench.fl_link8`,
    # `wrench.fr_link7`, `wrench.fr_link8`, whose labels are the embodiment's URDF link names,
    # so which of them exist follows the robot -- and one per arm, `wrench.left` /
    # `wrench.right`, each the sum of that arm's links. A critic names whichever it wants;
    # naming both is legal and just gives its encoder the sum twice over.
    for link, samples in stack_step_wrench(step_wrench, num_steps).items():
        mods[f"wrench.{link}"] = samples

    return mods
