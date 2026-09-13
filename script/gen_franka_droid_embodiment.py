#!/usr/bin/env python3
"""Generate the `franka-panda-droid` embodiment: one working arm, centred, second arm parked.

RoboTwin has no single-arm code path. `Robot` builds 121 `self.left_*` and 122 `self.right_*`
attributes unconditionally, `take_action` always slices the action as
`[left_arm, left_gripper, right_arm, right_gripper]`, `get_obs` always emits both halves, and
the drivers' embodiment parser accepts only a 1- or 3-element `embodiment` list. So a
single-arm robot like the Panda is used by instantiating *two* of them
(`[franka-panda, franka-panda, <dis>]`) and ignoring one.

pi05-DROID drives one arm, so the second is dead weight -- and worse, it is a second robot in
the exterior camera's frame, which nothing in DROID's training distribution accounts for. This
embodiment fixes both without touching any shared code:

  right arm  DRIVEN. Base at x = -0.07, directly under `open_microwave`'s spawn
             (xlim [-0.12, -0.02]). The stock `[franka-panda, franka-panda, 0.6]` puts an arm
             at x = +/-0.30, a 0.23 m lateral offset that makes it reach diagonally across.
  left arm   PARKED. Base at x = +6.07, six metres out on the exterior camera's side and
             behind it: out of every camera (checked in `verify`), far beyond any collision or
             planning interference with the driven arm.

Which arm is driven is a policy-side choice (`arm:` in policy/pi05_droid/deploy_policy.yml) and
must match DRIVEN_ARM here -- `verify` checks. `open_microwave` picks whichever arm's base is
nearer the microwave, so its scripted expert, and eval's feasibility gate, use the driven arm.

Both positions come from `robot_pose`, whose two entries are read independently -- left takes
`robot_pose[0]`, right takes `robot_pose[1]` (`envs/robot/robot.py:63` and `:90`). The task
config therefore passes `embodiment_dis: 0.0`, since `dis` shifts the two symmetrically about
the config's own x and would drag the working arm back off-centre.

Holding the parked arm still is the policy adapter's job, not this file's:
`policy/pi05_droid/deploy_policy.py` latches its joint vector at episode start and re-commands
that same vector every step. Note that padding with ZEROS would not do it -- `take_action`
treats the vector as absolute joint targets and TOPP-interpolates from the current pose, so
zeros command the arm to joint angles [0]*7 and swing it there.

THE WRIST CAMERAS
-----------------
Shipped, the Panda's D435 mount looks straight down the approach axis from 45 mm off the finger
centerline, so the gripper sits ~33 deg below the optical axis -- outside the D435's 18.5 deg
vertical half-angle, and never in frame. DROID's wrist camera is a ZED Mini (30 deg half-angle),
which leaves a 2.7 deg gap; pitching the mount 10 deg toward the fingers closes it. Both are
applied here (URDF pitch in this directory's own panda.urdf, `wrist_camera_type: ZED_Mini` in
this embodiment's task config) and the gripper renders at the bottom of the wrist frame. The
render `near` plane (0.1 m) is NOT a blocker despite clipping the finger link origins: rendering
the same pose at near=0.1 and near=0.005 differs by 5 pixels out of 57,600.

EVERYTHING THIS WRITES IS GITIGNORED (`assets/`, `task_config/`), which is why this is a
checked-in generator rather than committed files. Re-run it after `script/_download_assets.sh`.
It is idempotent.
"""
import argparse
import os
import re
import shutil
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "assets", "embodiments", "franka-panda")
EMBODIMENT = "franka-panda-droid"
DST_DIR = os.path.join(REPO_ROOT, "assets", "embodiments", EMBODIMENT)
EMBODIMENT_REGISTRY = os.path.join(REPO_ROOT, "task_config", "_embodiment_config.yml")
TASK_CONFIG = os.path.join(REPO_ROOT, "task_config", "demo_clean_franka_panda_droid.yml")
SRC_TASK_CONFIG = os.path.join(REPO_ROOT, "task_config", "demo_clean.yml")

# The wrist camera DROID actually records with. Scoping both wrist changes to this embodiment
# is free: the tilt lives in THIS directory's own copy of panda.urdf, and `wrist_camera_type`
# comes from this embodiment's own task config -- neither touches franka-panda or any other run.
WRIST_CAMERA_TYPE = "ZED_Mini"

# The episode video should show the view the policy actually conditions on. pi05_droid reads
# `exterior_camera`, so a head-camera video would show a viewpoint that had no part in
# producing the actions. `envs/_base_task.py` takes this key (default `head_camera`, i.e.
# unchanged for every other run) and `script/eval_policy.py` sizes the ffmpeg pipe from the
# same camera's type.
EVAL_VIDEO_CAMERA = "exterior_camera"

# Pitch of the wrist camera toward the finger centerline, about panda_hand's +Y.
#
# The mount sits 45 mm to the side of the fingers, so shipped (0 deg, straight down the approach
# axis) the fingertips are 32.7 deg below the optical axis. The stock D435 has an 18.5 deg
# vertical half-angle, so no tilt small enough to keep the workspace in view could reach them --
# which is why this was abandoned earlier. The ZED Mini's 30 deg half-angle changes that: the
# gap is now 2.7 deg, and 10 deg closes it with margin, landing the fingertips at 22.7 deg
# vertical / 22.0 deg horizontal, i.e. low in frame with the workspace above -- DROID's wrist
# composition.
#
# `camera_joint`'s rpy is expressed in camera_base coordinates, so this is
# R1^-1 @ R_y(-10deg) @ R1 @ R2_shipped as sxyz euler angles. Position is left as shipped.
WRIST_TILT_DEG = 10.0
WRIST_CAMERA_RPY = (1.57079632679, 0.0, 0.1745329252)

QUAT = [0.707, 0, 0, 0.707]
DRIVEN_ARM = "right"                  # must match `arm:` in policy/pi05_droid/deploy_policy.yml
DRIVEN_BASE = [-0.07, -0.65, 0.75]    # centred on open_microwave's spawn
PARKED_BASE = [6.07, -0.65, 0.75]     # behind the exterior camera; see verify()
# robot_pose[0] is the left entity, robot_pose[1] the right (envs/robot/robot.py:63 / :90).
LEFT_BASE, RIGHT_BASE = ((PARKED_BASE, DRIVEN_BASE) if DRIVEN_ARM == "right"
                         else (DRIVEN_BASE, PARKED_BASE))

# The exterior camera stands in for DROID's `exterior_image_1_left`, and is the ZED 2 DROID
# records it with. It sits to the robot's RIGHT (+x; the robot faces +y) and looks straight
# across the workspace toward -x, which also puts the parked left arm at x=+6.07 behind it.
#
# Constraints on where it can go:
#   * the back wall is a box at p=[0,1,1.5], half_size=[3,0.6,1.5]
#     (`envs/_base_task.py::create_table_and_wall`), so y must stay < 0.4;
#   * a framing-only objective lands BEHIND the robot, looking over its shoulder, so the view
#     direction is constrained (EXTERIOR_MAX_SIDE_ANGLE_DEG) and verify() enforces it;
#   * it has to be close enough that the manipulated object reads at a useful size. That is why
#     the arm's base is NOT a required landmark: fitting the base, 0.83 m behind the microwave,
#     is what held earlier poses back at ~18% of frame height for the microwave.
#
# Candidates are SOLVED against EXTERIOR_LANDMARKS (distance to the microwave fixed, microwave
# pixel size maximised, parked arm excluded) and then CHOSEN by rendering them across the
# expert rollout: the door swings open and can occlude a close camera, which the landmark math
# does not model. Chosen from that render: a front-right 3/4 view, 50 deg around from the side
# axis, 0.37 m from the microwave and 0.18 m above it, aimed at (-0.07, 0.0, 0.88). It shows the
# door face closed and the opened door plus gripper after the expert; 0.34 m and nearer push the
# forearm off the frame edge. Microwave reads 48 px tall of 180.
EXTERIOR_CAMERA = {
    "name": "exterior_camera",
    "type": "ZED_2",
    "position": [0.1678, -0.1084, 1.03],
    "forward": [-0.7892, 0.3598, -0.4977],
    "left": [-0.4149, -0.9099, 0.0],
}

# Horizontal view direction must be within this many degrees of the x axis, i.e. looking across
# the robot's +y facing rather than along it.
EXTERIOR_MAX_SIDE_ANGLE_DEG = 30.0

# What the exterior view has to contain, in world coordinates: the driven forearm, the gripper
# where it meets the object, and the microwave. `verify` projects these and fails if any falls
# outside the frame.
EXTERIOR_LANDMARKS = {
    "forearm": [-0.07, -0.20, 1.02],
    "gripper at object": [-0.07, 0.05, 0.95],
    "microwave centre": [-0.07, 0.175, 0.80],
    "microwave top": [-0.07, 0.175, 0.95],
}

# Every camera the parked arm has to stay out of: (name, position, forward, left, fovy, w, h).
# The observer is the one behind `data_type.third_view`, hardcoded in camera.py.
def cameras_to_clear(cam_types):
    """Every camera the parked arm must stay out of, with its real frustum.

    head_camera's pose is the embodiment's own; the observer's is hardcoded in
    `envs/camera/camera.py` (fovy 93, 320x240) and is not configurable. The exterior entry is
    built from EXTERIOR_CAMERA["type"] so a camera swap re-checks against the new, wider FOV --
    which matters, because a wider exterior view is exactly what could pull the parked arm back
    into shot.
    """
    ext = cam_types[EXTERIOR_CAMERA["type"]]
    head = cam_types["D435"]
    return [
        ("head_camera", [-0.032, -0.45, 1.35], [0, 0.6, -0.8], [-1, 0, 0],
         head["fovy"], head["w"], head["h"]),
        ("exterior_camera", EXTERIOR_CAMERA["position"], EXTERIOR_CAMERA["forward"],
         EXTERIOR_CAMERA["left"], ext["fovy"], ext["w"], ext["h"]),
        ("observer (third_view)", [0.0, 0.23, 1.33], [0, -1, -1.02], [1, 0, 0], 93, 320, 240),
    ]


def _in_frame(cam_pos, cam_fwd, cam_left, fovy_deg, width, height, point):
    """Whether `point` falls inside that camera's frustum. Returns (in_frame, detail)."""
    cam_pos = np.asarray(cam_pos, float)
    fwd = np.asarray(cam_fwd, float); fwd /= np.linalg.norm(fwd)
    left = np.asarray(cam_left, float); left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    d = np.asarray(point, float) - cam_pos
    depth = d @ fwd
    if depth <= 0:
        return False, "behind the camera"
    fovy = np.deg2rad(fovy_deg)
    fovx = 2 * np.arctan(np.tan(fovy / 2) * width / height)
    up_ratio = abs(d @ up) / depth / np.tan(fovy / 2)
    left_ratio = abs(d @ left) / depth / np.tan(fovx / 2)
    inside = up_ratio <= 1.0 and left_ratio <= 1.0
    return inside, f"depth {depth:.2f} m, frame ratios u={up_ratio:.2f} l={left_ratio:.2f}"


def build_config(src_config):
    """The new embodiment config: source values, with both bases and the exterior camera set."""
    cfg = dict(src_config)
    cfg["robot_pose"] = [list(LEFT_BASE) + list(QUAT), list(RIGHT_BASE) + list(QUAT)]
    cameras = [c for c in (cfg.get("static_camera_list") or [])
               if c.get("name") != EXTERIOR_CAMERA["name"]]
    cameras.append(dict(EXTERIOR_CAMERA))
    cfg["static_camera_list"] = cameras
    return cfg


HEADER = f"""# GENERATED by script/gen_franka_droid_embodiment.py -- do not edit by hand.
#
# `franka-panda`, with the two base poses placed independently: the {DRIVEN_ARM.upper()} arm (the
# one a single-arm policy drives) centred on open_microwave's spawn at x={DRIVEN_BASE[0]}, and the
# other arm parked at x={PARKED_BASE[0]}, out of every camera and far past any collision or
# planning interference. Use it with `embodiment_dis: 0.0`, since dis shifts both symmetrically.
#
# Also carries `exterior_camera` (ZED 2, on the robot's right side). The wrist cameras are
# pitched 10 deg toward the fingers and run as ZED Minis via the task config.
"""


def generate(dry_run=False):
    if not os.path.isdir(SRC_DIR):
        sys.exit(f"missing {SRC_DIR} -- run `bash script/_download_assets.sh` first")

    changed = False

    # -- 1. the embodiment directory ------------------------------------------------------
    if not os.path.isdir(DST_DIR):
        print(f"  copying {os.path.basename(SRC_DIR)} -> {EMBODIMENT}")
        if not dry_run:
            shutil.copytree(SRC_DIR, DST_DIR)
        changed = True
    else:
        print(f"  {EMBODIMENT}/ already exists (reusing; config and curobo paths rewritten)")

    if not dry_run and os.path.isdir(DST_DIR):
        # -- 2. config.yml ----------------------------------------------------------------
        with open(os.path.join(SRC_DIR, "config.yml"), "r", encoding="utf-8") as handle:
            src_cfg = yaml.safe_load(handle)
        with open(os.path.join(DST_DIR, "config.yml"), "w", encoding="utf-8") as handle:
            handle.write(HEADER + "\n")
            yaml.safe_dump(build_config(src_cfg), handle, sort_keys=False, default_flow_style=False)
        print(f"  config.yml: left base {LEFT_BASE}, right base {RIGHT_BASE}, "
              f"+{EXTERIOR_CAMERA['name']}")

        # -- 3. curobo paths --------------------------------------------------------------
        # curobo.yml carries absolute paths and curobo_tmp.yml a ${ASSETS_PATH} template;
        # both name the source embodiment directory and must be repointed at this one.
        for name in ("curobo.yml", "curobo_tmp.yml"):
            path = os.path.join(DST_DIR, name)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            fixed = text.replace("embodiments/franka-panda/", f"embodiments/{EMBODIMENT}/")
            if fixed != text:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(fixed)
                print(f"  {name}: repointed at {EMBODIMENT}/")
        # -- 3b. wrist camera tilt (this embodiment's URDF copy only) ----------------------
        urdf = os.path.join(DST_DIR, "panda.urdf")
        with open(urdf, "r", encoding="utf-8") as handle:
            text = handle.read()
        pattern = re.compile(
            r'(<joint name="camera_joint" type="fixed">.*?<origin rpy=")([^"]*)(")', re.DOTALL)
        match = pattern.search(text)
        if match is None:
            sys.exit(f"could not find camera_joint's <origin> in {urdf}")
        rpy = " ".join(f"{v:.11f}" for v in WRIST_CAMERA_RPY)
        if match.group(2).strip() != rpy:
            with open(urdf, "w", encoding="utf-8") as handle:
                handle.write(pattern.sub(lambda m: m.group(1) + rpy + m.group(3), text, count=1))
        print(f"  panda.urdf: wrist camera pitched {WRIST_TILT_DEG:.0f} deg toward the fingers")
        changed = True

    # -- 4. register it -------------------------------------------------------------------
    with open(EMBODIMENT_REGISTRY, "r", encoding="utf-8") as handle:
        registry_text = handle.read()
    if EMBODIMENT not in yaml.safe_load(registry_text):
        print(f"  registering {EMBODIMENT} in {os.path.basename(EMBODIMENT_REGISTRY)}")
        if not dry_run:
            with open(EMBODIMENT_REGISTRY, "a", encoding="utf-8") as handle:
                handle.write(f'\n{EMBODIMENT}:\n  file_path: "./assets/embodiments/'
                             f'{EMBODIMENT}/"\n')
        changed = True
    else:
        print(f"  {EMBODIMENT} already registered")

    # -- 5. a task config that uses it ----------------------------------------------------
    if not os.path.exists(TASK_CONFIG):
        print(f"  writing {os.path.basename(TASK_CONFIG)}")
        if not dry_run:
            with open(SRC_TASK_CONFIG, "r", encoding="utf-8") as handle:
                task_text = handle.read()
            # dis MUST be 0.0: it shifts both arms symmetrically about the config's own x, so
            # anything else drags the working arm back off-centre.
            task_text = task_text.replace(
                "embodiment: [aloha-agilex]",
                f"embodiment: [{EMBODIMENT}, {EMBODIMENT}, 0.0]")
            with open(TASK_CONFIG, "w", encoding="utf-8") as handle:
                handle.write(task_text)
        changed = True
    else:
        print(f"  {os.path.basename(TASK_CONFIG)} already exists (reusing)")

    # The wrist camera type is set every run, existing file or not -- it pairs with the tilt
    # above and the two only make sense together.
    if not dry_run and os.path.exists(TASK_CONFIG):
        with open(TASK_CONFIG, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        base = os.path.basename(TASK_CONFIG)

        # wrist_camera_type: replace the existing key in place (it is nested under `camera:`).
        for i, line in enumerate(lines):
            if line.strip().startswith("wrist_camera_type:"):
                indent = line[: len(line) - len(line.lstrip())]
                want = f"{indent}wrist_camera_type: {WRIST_CAMERA_TYPE}\n"
                if line != want:
                    lines[i] = want
                    changed = True
                print(f"  {base}: wrist_camera_type = {WRIST_CAMERA_TYPE}")
                break
        else:
            sys.exit(f"no wrist_camera_type key in {TASK_CONFIG}")

        # eval_video_camera: a top-level key the stock configs do not carry, so add it if absent.
        want = f"eval_video_camera: {EVAL_VIDEO_CAMERA}\n"
        for i, line in enumerate(lines):
            if line.startswith("eval_video_camera:"):
                if line != want:
                    lines[i] = want
                    changed = True
                break
        else:
            for i, line in enumerate(lines):
                if line.startswith("eval_video_log:"):
                    lines.insert(i + 1, want)
                    break
            else:
                lines.append("\n" + want)
            changed = True
        print(f"  {base}: eval_video_camera = {EVAL_VIDEO_CAMERA}")

        with open(TASK_CONFIG, "w", encoding="utf-8") as handle:
            handle.writelines(lines)

    return changed


def verify():
    ok = True
    with open(os.path.join(DST_DIR, "config.yml"), "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    poses = cfg.get("robot_pose") or []
    if len(poses) != 2:
        print(f"  FAIL: robot_pose has {len(poses)} entries, need 2 (left, right)")
        return False
    left, right = np.asarray(poses[0][:3], float), np.asarray(poses[1][:3], float)
    driven, parked = (right, left) if DRIVEN_ARM == "right" else (left, right)
    other_arm = "left" if DRIVEN_ARM == "right" else "right"
    print(f"  {DRIVEN_ARM} base (driven)   : {driven}")
    print(f"  {other_arm} base (parked)    : {parked}")

    # The policy has to drive the arm this embodiment centres, or it drives the parked one.
    adapter_yml = os.path.join(REPO_ROOT, "policy", "pi05_droid", "deploy_policy.yml")
    if os.path.exists(adapter_yml):
        with open(adapter_yml, "r", encoding="utf-8") as handle:
            adapter_arm = (yaml.safe_load(handle) or {}).get("arm")
        print(f"  adapter drives       : {adapter_arm}")
        if adapter_arm != DRIVEN_ARM:
            print(f"  FAIL: policy/pi05_droid/deploy_policy.yml drives {adapter_arm!r}, but this "
                  f"embodiment centres the {DRIVEN_ARM} arm"); ok = False

    # Centred on open_microwave's spawn, xlim [-0.12, -0.02] -> centre -0.07.
    offset = abs(driven[0] - (-0.07))
    print(f"  driven offset from microwave centre : {offset * 1000:.0f} mm")
    if offset > 0.02:
        print("  FAIL: driven arm is not centred on the object"); ok = False

    separation = float(np.linalg.norm(left - right))
    print(f"  arm separation       : {separation:.2f} m")
    # A Panda reaches 0.855 m; two of them cannot interact beyond ~1.8 m even fully extended.
    if separation < 2.0:
        print("  FAIL: parked arm is close enough to collide or interfere"); ok = False

    all_cam_types = yaml.safe_load(open(os.path.join(REPO_ROOT, "task_config",
                                                     "_camera_config.yml")).read())
    for name, pos, fwd, cleft, fovy, width, height in cameras_to_clear(all_cam_types):
        inside, detail = _in_frame(pos, fwd, cleft, fovy, width, height, parked)
        print(f"  parked arm vs {name:22s}: {'IN FRAME' if inside else 'out'} ({detail})")
        if inside:
            print(f"  FAIL: parked arm is visible to {name}"); ok = False

    # The exterior view must actually contain the arm and the object. Specs come from
    # _camera_config.yml by type, so changing EXTERIOR_CAMERA["type"] re-checks against the
    # right frustum instead of a stale hardcoded D435.
    cam_types = yaml.safe_load(open(os.path.join(REPO_ROOT, "task_config",
                                                 "_camera_config.yml")).read())
    ext_type = EXTERIOR_CAMERA["type"]
    if ext_type not in cam_types:
        print(f"  FAIL: exterior camera type {ext_type} is not in _camera_config.yml")
        return False
    ext_spec = cam_types[ext_type]
    width, height = ext_spec["w"], ext_spec["h"]
    fovy = np.deg2rad(ext_spec["fovy"])
    fovx = 2 * np.arctan(np.tan(fovy / 2) * width / height)
    print(f"  exterior camera type : {ext_type} "
          f"({ext_spec['fovy']} deg fovy, {width}x{height}, "
          f"{np.degrees(fovx):.0f} deg horizontal)")
    cam_pos = np.asarray(EXTERIOR_CAMERA["position"], float)
    fwd = np.asarray(EXTERIOR_CAMERA["forward"], float); fwd /= np.linalg.norm(fwd)
    cleft = np.asarray(EXTERIOR_CAMERA["left"], float); cleft /= np.linalg.norm(cleft)
    cup = np.cross(fwd, cleft)
    horiz = np.array([fwd[0], fwd[1]]); horiz /= np.linalg.norm(horiz)
    side_angle = np.degrees(np.arccos(min(1.0, abs(horiz[0]))))
    print(f"  exterior view angle  : {side_angle:.1f} deg from a pure side view "
          f"(max {EXTERIOR_MAX_SIDE_ANGLE_DEG:.0f}), camera on the "
          f"{'-x' if cam_pos[0] < DRIVEN_BASE[0] else '+x'} side")
    if side_angle > EXTERIOR_MAX_SIDE_ANGLE_DEG:
        print("  FAIL: exterior camera is not looking across from the side"); ok = False
    if cam_pos[1] < DRIVEN_BASE[1]:
        print("  FAIL: exterior camera sits behind the working arm's base"); ok = False
    # The robot faces +y, so its right is +x: the camera must be there, looking back toward -x.
    if cam_pos[0] <= DRIVEN_BASE[0] or fwd[0] >= 0:
        print("  FAIL: exterior camera is not on the robot's right side looking across"); ok = False
    for label, point in EXTERIOR_LANDMARKS.items():
        d = np.asarray(point, float) - cam_pos
        depth = d @ fwd
        if depth <= 0:
            print(f"  FAIL: {label} is behind the exterior camera"); ok = False; continue
        x = width / 2 - (d @ cleft) / depth / np.tan(fovx / 2) * (width / 2)
        y = height / 2 - (d @ cup) / depth / np.tan(fovy / 2) * (height / 2)
        inside = 0 <= x < width and 0 <= y < height
        print(f"  exterior sees {label:18s}: pixel ({x:6.1f},{y:6.1f}) "
              f"{'ok' if inside else 'OUT OF FRAME'}")
        if not inside:
            print(f"  FAIL: {label} is outside the exterior view"); ok = False

    # -- wrist camera: the tilt, the type, and whether the gripper actually renders --------
    import transforms3d as t3d
    with open(os.path.join(DST_DIR, "panda.urdf"), "r", encoding="utf-8") as handle:
        urdf_text = handle.read()

    def joint_origin(name):
        m = re.search(rf'<joint name="{name}" type="fixed">.*?<origin rpy="([^"]*)" xyz="([^"]*)"',
                      urdf_text, re.DOTALL)
        return ([float(v) for v in m.group(1).split()], [float(v) for v in m.group(2).split()])

    rpy1, xyz1 = joint_origin("hand_to_camera_mount")
    rpy2, xyz2 = joint_origin("camera_joint")
    R1 = t3d.euler.euler2mat(*rpy1, "sxyz")
    Rw = R1 @ t3d.euler.euler2mat(*rpy2, "sxyz")
    cam_hand = np.asarray(xyz1) + R1 @ np.asarray(xyz2)
    wfwd, wleft, wup = Rw[:, 0], Rw[:, 1], Rw[:, 2]
    pitch = np.degrees(np.arctan2(-wfwd[0], wfwd[2]))
    print(f"  wrist camera pitch   : {pitch:.1f} deg toward the fingers "
          f"(want {WRIST_TILT_DEG:.0f})")
    if abs(pitch - WRIST_TILT_DEG) > 0.5:
        print("  FAIL: wrist tilt not applied"); ok = False

    task_cfg = yaml.safe_load(open(TASK_CONFIG, "r", encoding="utf-8").read())
    got_type = (task_cfg.get("camera") or {}).get("wrist_camera_type")
    print(f"  wrist camera type    : {got_type}")
    if got_type != WRIST_CAMERA_TYPE:
        print(f"  FAIL: wrist_camera_type is {got_type}, want {WRIST_CAMERA_TYPE}"); ok = False

    cam_types = yaml.safe_load(open(os.path.join(REPO_ROOT, "task_config",
                                                 "_camera_config.yml")).read())
    if WRIST_CAMERA_TYPE not in cam_types:
        print(f"  FAIL: {WRIST_CAMERA_TYPE} is not in _camera_config.yml"); ok = False
    else:
        spec = cam_types[WRIST_CAMERA_TYPE]
        wfovy = np.deg2rad(spec["fovy"])
        wfovx = 2 * np.arctan(np.tan(wfovy / 2) * spec["w"] / spec["h"])
        # Fingertips in panda_hand coordinates; they open to about y=+/-0.03 at grasp width.
        near = 0.1                      # envs/camera/camera.py::load_camera
        clipped = False
        for label, P in (("near fingertip", (0.0, 0.031, 0.112)),
                         ("far fingertip", (0.0, -0.020, 0.112))):
            d = np.asarray(P) - cam_hand
            depth = d @ wfwd
            av = np.degrees(np.arctan2(abs(d @ wup), depth))
            ah = np.degrees(np.arctan2(abs(d @ wleft), depth))
            in_fov = av <= np.degrees(wfovy / 2) and ah <= np.degrees(wfovx / 2)
            if depth < near:
                clipped = True
            print(f"  {label:18s}: depth {depth*100:4.1f} cm, vert {av:4.1f} deg "
                  f"(half {np.degrees(wfovy/2):.1f}), horiz {ah:4.1f} deg "
                  f"(half {np.degrees(wfovx/2):.1f}) -> FOV {'in' if in_fov else 'OUT'}")
            if not in_fov:
                print(f"  FAIL: {label} outside the wrist FOV"); ok = False
        if clipped:
            # NOT a blocker, despite how it looks. Measured: rendering this exact pose at
            # near=0.1 and near=0.005 differs by 5 pixels out of 57,600. The near plane clips
            # the finger link ORIGINS, which sit ~2.4 cm out, but the finger bodies extend
            # forward past it and render normally -- they appear at the bottom of the frame
            # straddling whatever the gripper is holding. So `envs/camera/camera.py` needs no
            # change; an earlier reading of this as fatal was wrong.
            print(f"  note: finger link origins are inside the near plane ({near} m), but the "
                  f"finger bodies render past it -- measured, not assumed (near 0.1 vs 0.005 "
                  f"differs by 5 px of 57600).")

    got_video = task_cfg.get("eval_video_camera")
    print(f"  eval video camera    : {got_video}")
    if got_video != EVAL_VIDEO_CAMERA:
        print(f"  FAIL: eval_video_camera is {got_video}, want {EVAL_VIDEO_CAMERA}"); ok = False
    elif got_video not in [c.get("name") for c in (cfg.get("static_camera_list") or [])]:
        print(f"  FAIL: eval_video_camera {got_video} is not a camera this embodiment has")
        ok = False

    cams = [c.get("name") for c in (cfg.get("static_camera_list") or [])]
    print(f"  static_camera_list   : {cams}")
    if EXTERIOR_CAMERA["name"] not in cams:
        print("  FAIL: exterior camera missing"); ok = False

    for name in ("curobo.yml", "panda.urdf", "config.yml"):
        path = os.path.join(DST_DIR, name)
        if not os.path.exists(path):
            print(f"  FAIL: {name} missing from {EMBODIMENT}/"); ok = False
    with open(os.path.join(DST_DIR, "curobo.yml"), "r", encoding="utf-8") as handle:
        curobo = handle.read()
    if "embodiments/franka-panda/" in curobo:
        print("  FAIL: curobo.yml still points at the source embodiment"); ok = False

    registry = yaml.safe_load(open(EMBODIMENT_REGISTRY, "r", encoding="utf-8").read())
    if EMBODIMENT not in registry:
        print(f"  FAIL: {EMBODIMENT} not registered"); ok = False
    else:
        print(f"  registered as        : {registry[EMBODIMENT]['file_path']}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"generating {EMBODIMENT}")
    changed = generate(dry_run=args.dry_run)
    print("done" if changed else "nothing to do (already generated)")

    if args.verify:
        print("\nverifying:")
        if args.dry_run:
            print("  (--dry-run: verifying what is on disk)")
        sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()
