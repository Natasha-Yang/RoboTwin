#!/usr/bin/env python3
"""Prepare the assets the dumbbell-rack / bookcase / knife-cover / battery-slot tasks need.

`assets/*` is gitignored (.gitignore:18), so nothing under it can be committed. Anything a
task depends on therefore has to be reproducible from a checked-in script -- the same status
`script/gen_peg_socket_asset.py` has for the peg-insertion family. This script does two
different jobs:

**Annotate** four assets that ship with the asset pack but are unusable as they stand. 24 of
the ~121 shipped assets carry only `{stable, center, extents}` in their `model_data*.json`:
no `scale`, no contact points, no functional points. That is not a soft failure --
`create_actor` swallows the missing key in a bare `except` (envs/utils/create_actor.py:531),
leaves `scale` at its (1,1,1) default and sets `model_data=None`, so the asset loads at its
raw ~1.9 m size and every `get_contact_point` / `get_functional_point` call returns None.
Three of the four tasks here need exactly such assets, so this script writes their
annotations. It also retunes two already-annotated assets (`034_knife`, `043_book`) that are
annotated correctly but not for these tasks.

**Generate** the two receptacles nothing in the asset library provides: a knife cover and a
battery slot. Both are blind slots with a specified clearance, built the same way
`gen_peg_socket_asset.py` builds its sockets -- as a union of convex pieces exported as one
named submesh each, so `add_multiple_convex_collisions_from_file` gives one hull per piece
and the bore stays open. A shipped asset cannot substitute: a survey of every asset's
interior voids found only hollow containers (mugs, cups, shoes, bottles), never a bore.

All annotation is written **in place** and is idempotent. No task in `envs/` referenced any
of these assets before this change (`grep -rl <name> envs/*.py` was empty for all six), so
rewriting their `model_data` breaks nothing. Re-running `script/_download_assets.sh` restores
the shipped files and reverts the annotation -- re-run this script after doing so.

Usage:
    python script/gen_task_assets.py [--verify] [--clean]

`--verify` loads every asset it touched in a headless SAPIEN scene and asserts the geometry
survived. That check is not optional paranoia: SAPIEN's actor builder swallows collision-shape
cook failures in a bare `except RuntimeError: continue`, so a mesh that fails to cook yields an
actor with zero collision shapes and no error message -- the knife would pass straight through
its cover and the task would report success.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

# --- assets that ship unannotated, or annotated for a different purpose ----------------------

RACK_MODEL = "013_dumbbell-rack"
RACK_ID = 0
RACK_SCALE = 0.13
# The cradle, in the rack's own raw units. Measured, not derived: a dumbbell was dropped onto
# the rack over a 132-pose grid (both orientations, the full width and depth, two drop
# heights) and exactly six poses came to rest on it -- all six with the bar PERPENDICULAR to
# the rack's length, straddling its two rails, at this one depth offset. Every pose with the
# bar along the rack, and every other depth, ends with the dumbbell on the floor. So this is
# the only placement the asset actually affords, and RACK_ALIGN_DEPTH below is what makes the
# expert produce it.
RACK_CRADLE_Y = 0.177
RACK_CRADLE_Z = 1.123

BOOKCASE_MODEL = "014_bookcase"
BOOKCASE_ID = 3
BOOKCASE_SCALE = 0.10

BOOK_MODEL = "043_book"
BOOK_ID = 0
BOOK_SCALE = 0.6  # ships at 1.0 -- see book_model_data for why it has to come down
BOOK_GRASP_Z = 0.090  # local z of the added top-edge contact points; see book_model_data

DUMBBELL_MODEL = "052_dumbbell"
DUMBBELL_ID = 1

KNIFE_MODEL = "034_knife"
KNIFE_ID = 0
KNIFE_SCALE = 0.13  # ships at 0.31 -> a 458 mm knife, longer than the workspace is wide. Even
#                     at 0.20 (296 mm) the knife plus its cover needs ~400 mm of table along
#                     one axis, which leaves no room for the approach; 0.13 gives a 192 mm
#                     knife with a 96 mm blade and fits comfortably.

BATTERY_MODEL = "061_battery"
BATTERY_ID = 1  # the fatter standing cell: base3 is 4.3x as long as it is wide and topples
BATTERY_SCALE = 0.05  # -> 22.7 mm dia x 96.8 mm
BATTERY_GRASP_FRAC = 0.75  # contact band, as a fraction of the cell's length from its base.
#                            High enough that the fingers stay clear of the slot mouth at
#                            full insertion depth -- see envs/put_battery_slot.py.

# --- assets generated from scratch ----------------------------------------------------------

COVER_MODEL = "124_knife-cover"
COVER_CLEARANCE = 0.002  # per side, on both cross-section axes. Tighter than the peg
#                          family's 3 mm because the blade is only ~6 mm thick.
COVER_LEAD_IN = 0.005  # 45 deg chamfer at the mouth
COVER_WALL = 0.010
COVER_FLOOR = 0.010
COVER_DEPTH = 0.065  # slot depth below the mouth

SLOT_MODEL = "125_battery-slot"
SLOT_CLEARANCE = 0.003
SLOT_LEAD_IN = 0.005
SLOT_SECTORS = 32  # must be a multiple of 4 -- see _wall_sector
SLOT_WALL = 0.012
SLOT_FLOOR = 0.010
SLOT_DEPTH = 0.040

COVER_COLOR = (0.20, 0.23, 0.28, 1.0)  # dark charcoal, so it reads against the steel blade
SLOT_COLOR = (0.30, 0.33, 0.38, 1.0)

ROOT = Path(__file__).resolve().parent.parent


# --- shared helpers -------------------------------------------------------------------------


def _paint(mesh: trimesh.Trimesh, color) -> trimesh.Trimesh:
    """Give the mesh a real glTF PBR material rather than per-face vertex colors.

    `ColorVisuals` exports to a glb as a COLOR_0 vertex attribute and NO material, and
    SAPIEN's glb loader takes its base color from the material -- so a mesh painted that way
    renders in SAPIEN's default gray whatever color trimesh was told.
    """
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=color, metallicFactor=0.1, roughnessFactor=0.6))
    return mesh


def _upright(mesh_vertices: np.ndarray) -> np.ndarray:
    """Local (x, y, z) -> the frame the asset stands in, (X, Y=depth, Z=up).

    The objaverse-derived assets are modelled +Y up and every task spawns them with
    `qpos=[0.707, 0.707, 0, 0]`, a 90 deg rotation about X that maps (x, y, z) -> (x, -z, y).
    Measuring in that frame is the only way the numbers below read as the object you see.
    """
    v = mesh_vertices
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)


def _to_local(X: float, Y: float, Z: float) -> list[float]:
    """The inverse of `_upright` for a single point: upright (X, Y, Z) -> local (x, y, z)."""
    return [X, Z, -Y]


def collision_parts(model: str, model_id: int, upright: bool = True) -> list[trimesh.Trimesh]:
    scene = trimesh.load(ROOT / "assets" / "objects" / model / "collision" / f"base{model_id}.glb",
                         force="scene")
    parts = []
    for geom in scene.geometry.values():
        part = geom.copy()
        if upright:
            part.vertices = _upright(geom.vertices)
        parts.append(part)
    return parts


def _bounds(parts: list[trimesh.Trimesh]) -> np.ndarray:
    allv = np.vstack([p.vertices for p in parts])
    return np.array([allv.min(axis=0), allv.max(axis=0)])


def _load_model_data(model: str, model_id: int) -> dict:
    with open(ROOT / "assets" / "objects" / model / f"model_data{model_id}.json") as f:
        return json.load(f)


def _write_model_data(model: str, model_id: int, data: dict) -> None:
    path = ROOT / "assets" / "objects" / model / f"model_data{model_id}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def _base_fields(model: str, model_id: int, scale: float) -> dict:
    """The keys every model_data carries, preserving whatever the shipped file already had."""
    old = _load_model_data(model, model_id)
    return {
        "center": old.get("center", [0.0, 0.0, 0.0]),
        "extents": old.get("extents", [0.0, 0.0, 0.0]),
        "scale": [scale, scale, scale],
        "transform_matrix": np.eye(4).tolist(),
        "target_pose": [],
        "contact_points_pose": [],
        "functional_matrix": [],
        "orientation_point": [],
        "contact_points_group": [],
        "contact_points_mask": [],
        "target_point_discription": [],
        "contact_points_discription": [],
        "functional_point_discription": [],
        "orientation_point_discription": [],
        "stable": True,
    }


def _solid_runs(parts, axis_pts):
    """[(lo, hi), ...] of the occupied intervals along a sampled ray."""
    occ = np.zeros(len(axis_pts), dtype=bool)
    for part in parts:
        try:
            occ |= part.contains(axis_pts)
        except Exception:
            pass
    runs, start = [], None
    for k, v in enumerate(occ):
        if v and start is None:
            start = k
        if not v and start is not None:
            runs.append((start, k - 1))
            start = None
    if start is not None:
        runs.append((start, len(occ) - 1))
    return runs, occ


# --- 013_dumbbell-rack: annotate the top rail ----------------------------------------------


def rack_model_data() -> tuple[dict, dict]:
    """Scale + one functional point on the centre of the rack's top rail.

    Shaped like `074_displaystand`, the in-tree "place an object on top of me" receptacle:
    `target_pose` empty, one `functional_matrix` whose +Z is the local +Y that the
    `qpos=[0.707, 0.707, 0, 0]` spawn turns into world up. `place_object_stand` places against
    exactly that, so `place_actor(..., constrain="free")` behaves here as it does there.
    """
    parts = collision_parts(RACK_MODEL, RACK_ID)
    box = _bounds(parts)

    rail_top, rail_y = RACK_CRADLE_Z, RACK_CRADLE_Y

    data = _base_fields(RACK_MODEL, RACK_ID, RACK_SCALE)
    tx, ty, tz = _to_local(0.0, rail_y, rail_top)
    # columns x=(1,0,0), y=(0,0,-1), z=(0,1,0): +Z is the local +Y the spawn turns into
    # world up. Byte-identical in form to 074_displaystand's own functional frame.
    data["functional_matrix"] = [[
        [1.0, 0.0, 0.0, tx],
        [0.0, 0.0, 1.0, ty],
        [0.0, -1.0, 0.0, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]]
    data["functional_point_discription"] = [
        "Point0: the cradle between the rack's rails; the axis points up out of it. A "
        "dumbbell seats here with its bar across the rails, not along them."
    ]
    stats = {
        "rail_top_raw": rail_top,
        "rail_y_raw": rail_y,
        "size_mm": (box[1] - box[0]) * RACK_SCALE * 1000,
        "rail_height_mm": rail_top * RACK_SCALE * 1000,
        "bottom_raw": box[0][2],
    }
    return data, stats


# --- 014_bookcase: annotate the centre bay --------------------------------------------------


def bookcase_model_data() -> tuple[dict, dict]:
    """Scale + one functional point at the floor of the centre bay, axis pointing DOWN it.

    `base3` is a four-post rack on a base slab, giving three open bays. The posts run the full
    depth; the base slab is interrupted by an arch through the middle of the depth, so a book
    standing in a bay bridges the arch and rests on the slab either side of it. Nothing else in
    the asset library is an open bookcase -- every other `014_bookcase` variant is modelled
    already full of books, a solid blob after convex decomposition.

    The functional frame's +Z points DOWN into the bay, because `get_place_pose` computes
    `target - grasp_bias - pre_dis * (R(target) @ [0, 0, 1])` (envs/_base_task.py:1384-1395):
    a positive `pre_dis` then holds the book above the mouth and a negative `dis` drives it in.
    Its +X is the bay's *depth* direction, which is the axis the book's own +X (its 161 mm
    width) has to be aligned onto -- the book has to go in edge-on, not face-on.
    """
    parts = collision_parts(BOOKCASE_MODEL, BOOKCASE_ID)
    box = _bounds(parts)

    # The posts, read off a horizontal scan clear of the arch.
    xs = np.linspace(box[0][0], box[1][0], 400)
    probe_y = 0.6 * box[0][1]  # well behind the arch, where the base slab is continuous
    runs, _ = _solid_runs(
        parts, np.stack([xs, np.full_like(xs, probe_y), np.full_like(xs, 0.5 * box[1][2])], axis=1))
    if len(runs) < 4:
        raise RuntimeError(f"{BOOKCASE_MODEL}: expected 4 posts, found {len(runs)}")
    # The centre bay is the gap between the two middle posts.
    mid = len(runs) // 2
    bay_lo, bay_hi = xs[runs[mid - 1][1]], xs[runs[mid][0]]
    bay_x = 0.5 * (bay_lo + bay_hi)
    bay_width = bay_hi - bay_lo

    # The bay floor: the top of the base slab, measured inside the bay but clear of the arch.
    zs = np.linspace(box[0][2], box[1][2], 400)
    franks, _ = _solid_runs(
        parts, np.stack([np.full_like(zs, bay_x), np.full_like(zs, probe_y), zs], axis=1))
    if not franks:
        raise RuntimeError(f"{BOOKCASE_MODEL}: no base slab under the centre bay")
    floor_z = zs[franks[0][1]]
    post_top = box[1][2]

    data = _base_fields(BOOKCASE_MODEL, BOOKCASE_ID, BOOKCASE_SCALE)
    tx, ty, tz = _to_local(bay_x, 0.0, floor_z)
    # columns: x = the bay's depth direction (upright +Y = local -z), y, z = down the bay.
    data["functional_matrix"] = [[
        [0.0, 1.0, 0.0, tx],
        [0.0, 0.0, -1.0, ty],
        [-1.0, 0.0, 0.0, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]]
    data["functional_point_discription"] = [
        "Point0: the floor of the bookcase's centre bay; the axis points down into the bay."
    ]
    stats = {
        "bay_width_mm": bay_width * BOOKCASE_SCALE * 1000,
        "bay_x_raw": bay_x,
        "floor_mm": floor_z * BOOKCASE_SCALE * 1000,
        "clear_height_mm": (post_top - floor_z) * BOOKCASE_SCALE * 1000,
        "size_mm": (box[1] - box[0]) * BOOKCASE_SCALE * 1000,
        "bottom_raw": box[0][2],
        "posts": len(runs),
    }
    return data, stats


# --- 043_book: add a top-edge grasp ---------------------------------------------------------


def book_model_data() -> tuple[dict, dict]:
    """Rescale the book, add a top-down grasp of its top edge, and mark its bottom edge.

    Two things the shipped annotation cannot do for `put_book_bookcase`:

    * Its contact points sit at **mid-height** and approach along the book's **width**. Both
      are wrong here. Mid-height would put the fingers inside the bay alongside the book at
      full depth, where they hit the posts; and the sideways approach is unreachable (see the
      comment on the new points below). Points 2 and 3 replace it with a top-down grasp near
      the top edge, which fixes both at once. Nothing is removed -- `contact_point_id=[0, 1]`
      still selects the shipped grasp.
    * It has **no functional point at all**, so `place_actor` has no way to know which end of
      the book goes in. One is added at the centre of the bottom edge.
    """
    old = _load_model_data(BOOK_MODEL, BOOK_ID)
    data = dict(old)
    data["scale"] = [BOOK_SCALE, BOOK_SCALE, BOOK_SCALE]
    parts = collision_parts(BOOK_MODEL, BOOK_ID, upright=False)
    box = _bounds(parts)
    base = [np.array(m, dtype=float) for m in old["contact_points_pose"]]
    if len(base) < 2:
        raise RuntimeError(f"{BOOK_MODEL}: expected 2 shipped contact points, found {len(base)}")
    # Only ever the first two: this script is idempotent, so re-running it must not keep
    # appending copies of the copies it added last time.
    base = base[:2]

    # Points 2 and 3: a TOP-DOWN grasp of the book's top edge, fingers spanning its 32 mm
    # thickness. The two shipped points approach along the book's WIDTH, from beyond its left
    # or right edge -- and `get_grasp_pose` stands the end-effector 0.12 m back from the
    # contact plus `pre_grasp_dis` (_base_task.py:1200), so at the book's spawn range that
    # pre-pose lands near |x| = 0.45 m, outside the arm's reach. Grasping from directly above
    # keeps the whole approach over the table. Measured: the side grasp failed to plan on
    # every seed tried; this is the fix.
    #
    # Frame: approach is -contact_y, so contact_y = +Z puts the gripper coming straight down;
    # the fingers separate along -contact_z, so contact_z = +Y spans the thickness.
    # columns x=(-1,0,0), y=(0,0,1), z=(0,1,0), det +1.
    mid_thickness = 0.5 * (box[0][1] + box[1][1])
    top = []
    for sign in (+1.0, -1.0):
        top.append([
            [-sign, 0.0, 0.0, 0.0],
            [0.0, 0.0, sign, mid_thickness],
            [0.0, 1.0, 0.0, BOOK_GRASP_Z],
            [0.0, 0.0, 0.0, 1.0],
        ])
    data["contact_points_pose"] = [m.tolist() for m in base] + top
    data["contact_points_group"] = [[0, 1], [2, 3]]
    data["contact_points_mask"] = [True, True]
    data["contact_points_discription"] = (list(old.get("contact_points_discription", []))[:2] + [
        "Point2: the top edge of the book, grasped from directly above.",
        "Point3: the top edge of the book, grasped from directly above, gripper reversed.",
    ])
    scale = np.array(data["scale"], dtype=float)

    # A functional point at the centre of the bottom edge, +Z pointing DOWN out of it: the
    # direction the book travels as it goes into a bay. Same convention as the generated
    # sockets -- `get_place_pose` subtracts `pre_dis * (R(target) @ [0,0,1])`, so a positive
    # `pre_dis` holds the book above the bay and a negative `dis` drives it in. The book ships
    # with no functional point at all, and `place_actor` needs one to know which end goes in.
    # columns x=(1,0,0) the book's width, y=(0,-1,0), z=(0,0,-1).
    data["functional_matrix"] = [[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.5 * (box[0][1] + box[1][1])],
        [0.0, 0.0, -1.0, box[0][2]],
        [0.0, 0.0, 0.0, 1.0],
    ]]
    data["functional_point_discription"] = [
        "Point0: the centre of the book's bottom edge; the axis points down out of it."
    ]

    stats = {
        "size_mm": (box[1] - box[0]) * scale * 1000,
        "grasp_above_bottom_mm": (BOOK_GRASP_Z - box[0][2]) * scale[2] * 1000,
        "n_contact": len(data["contact_points_pose"]),
        "thickness_mm": (box[1][1] - box[0][1]) * scale[1] * 1000,
    }
    return data, stats


# --- 034_knife: rescale, and mark the blade tip ---------------------------------------------


def _blade_section(parts, y_tip: float, depth_raw: float):
    """Bounding box of the blade over the length that goes into the cover, in local x/z."""
    v = np.vstack([p.vertices for p in parts])
    sel = v[(v[:, 1] >= y_tip - depth_raw) & (v[:, 1] <= y_tip)]
    if len(sel) < 8:
        raise RuntimeError("034_knife: could not sample the blade cross-section")
    return sel[:, 0].min(), sel[:, 0].max(), sel[:, 2].min(), sel[:, 2].max()


def knife_model_data() -> tuple[dict, dict]:
    """Rescale to something the arm can swing, and add a functional point at the blade tip.

    The shipped `scale` of 0.31 gives a 458 mm knife -- longer than the reachable workspace is
    wide. The shipped functional point 0 sits on the *spine* pointing up, which is the right
    frame for cutting and the wrong one for sheathing, so point 1 is added at the tip with its
    +Z along the blade: the direction the knife travels as it goes into the cover.
    """
    old = _load_model_data(KNIFE_MODEL, KNIFE_ID)
    parts = collision_parts(KNIFE_MODEL, KNIFE_ID, upright=False)
    box = _bounds(parts)
    y_tip = box[1][1]
    depth_raw = COVER_DEPTH / KNIFE_SCALE
    x0, x1, z0, z1 = _blade_section(parts, y_tip, depth_raw)
    cx, cz = 0.5 * (x0 + x1), 0.5 * (z0 + z1)

    data = dict(old)
    data["scale"] = [KNIFE_SCALE, KNIFE_SCALE, KNIFE_SCALE]
    # Only the shipped point 0 (the spine) is kept as the base: this script is idempotent,
    # so re-running it must not keep appending another tip point each time.
    fps = [np.array(m, dtype=float).tolist() for m in old.get("functional_matrix", [])][:1]
    # columns x=(1,0,0), y=(0,0,-1), z=(0,1,0): +Z runs out of the tip along the blade.
    fps.append([
        [1.0, 0.0, 0.0, cx],
        [0.0, 0.0, 1.0, y_tip],
        [0.0, -1.0, 0.0, cz],
        [0.0, 0.0, 0.0, 1.0],
    ])
    data["functional_matrix"] = fps
    desc = list(old.get("functional_point_discription", []))[:1] or [""]
    desc.append("Point1: the tip of the blade; the axis runs out of the tip along the blade.")
    data["functional_point_discription"] = desc

    stats = {
        "size_mm": (box[1] - box[0]) * KNIFE_SCALE * 1000,
        "blade_thickness_mm": (x1 - x0) * KNIFE_SCALE * 1000,
        "blade_height_mm": (z1 - z0) * KNIFE_SCALE * 1000,
        "section": (x0, x1, z0, z1),
        "cx": cx, "cz": cz, "y_tip": y_tip,
    }
    return data, stats


# --- 061_battery: annotate the cell ----------------------------------------------------------


def battery_model_data() -> tuple[dict, dict]:
    """Scale, four contact points around the cell's waist, and a functional point at its base.

    `base3` is a single convex cylinder whose long axis is the local +Y that the standard
    spawn turns into world up. The contact translations sit on the **centreline**, not on the
    surface: that is the convention `create_box("long")` uses and the one `122_peg-round`
    copies, and it is what puts the gripper's closing axis across the diameter rather than
    tangent to it.
    """
    old = _load_model_data(BATTERY_MODEL, BATTERY_ID)
    parts = collision_parts(BATTERY_MODEL, BATTERY_ID, upright=False)
    box = _bounds(parts)
    y_lo, y_hi = box[0][1], box[1][1]
    radius = 0.25 * ((box[1][0] - box[0][0]) + (box[1][2] - box[0][2]))
    waist = y_lo + BATTERY_GRASP_FRAC * (y_hi - y_lo)

    data = _base_fields(BATTERY_MODEL, BATTERY_ID, BATTERY_SCALE)
    contacts = []
    for phi in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        c, s = float(np.cos(phi)), float(np.sin(phi))
        # columns x=(0,1,0) the long axis, y=(-s,0,c), z=(c,0,s) radial outward.
        contacts.append([
            [0.0, -s, c, 0.0],
            [1.0, 0.0, 0.0, waist],
            [0.0, c, s, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    data["contact_points_pose"] = contacts
    data["contact_points_group"] = [[0, 1, 2, 3]]
    data["contact_points_mask"] = [True]
    data["contact_points_discription"] = [
        f"Point{i}: the upper body of the battery, approached from {lbl}."
        for i, lbl in enumerate(("the front", "the right", "the back", "the left"))
    ]
    # columns x=(1,0,0), y=(0,0,1), z=(0,-1,0): +Z runs out of the negative terminal.
    data["functional_matrix"] = [[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, y_lo],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]]
    data["functional_point_discription"] = [
        "Point0: the base of the battery; the axis points out of it, down into the slot."
    ]
    stats = {
        "diameter_mm": 2 * radius * BATTERY_SCALE * 1000,
        "grasp_above_base_mm": (waist - y_lo) * BATTERY_SCALE * 1000,
        "length_mm": (y_hi - y_lo) * BATTERY_SCALE * 1000,
        "radius_raw": radius,
        "y_lo": y_lo,
        "pieces": len(parts),
    }
    return data, stats


# --- generated receptacles ------------------------------------------------------------------
#
# Both are built the way gen_peg_socket_asset.py builds its sockets: as a list of convex
# pieces exported one-per-submesh, so `add_multiple_convex_collisions_from_file` gives one
# hull each and the slot stays open. A single nonconvex mesh is not an option -- `convex=False`
# routes to `add_nonconvex_collision_from_file`, which no task in this repo uses, and a mesh
# that fails to cook leaves the actor with zero collision shapes and prints nothing.


def _rect_wall(inner: tuple[float, float], outer: tuple[float, float], lead_in: float,
               floor_z: float, height: float, axis: int, sign: int) -> trimesh.Trimesh:
    """One wall of a rectangular blind slot, as the hull of its pentagon cross-section."""
    a, b = inner[axis], inner[1 - axis]
    A, B = outer[axis], outer[1 - axis]
    pentagon = [
        (a, floor_z),
        (A, floor_z),
        (A, height),
        (a + lead_in, height),
        (a, height - lead_in),
    ]
    points = []
    for u, z in pentagon:
        for w in (-B, B):
            xy = [0.0, 0.0]
            xy[axis] = sign * u
            xy[1 - axis] = w
            points.append([xy[0], xy[1], z])
    return trimesh.PointCloud(np.array(points)).convex_hull


def _floor(outer: tuple[float, float], floor_z: float) -> trimesh.Trimesh:
    return trimesh.creation.box(
        extents=(2 * outer[0], 2 * outer[1], floor_z),
        transform=trimesh.transformations.translation_matrix((0.0, 0.0, floor_z / 2.0)))


def build_cover_pieces(inner: tuple[float, float], outer: tuple[float, float],
                       floor_z: float, height: float) -> list[trimesh.Trimesh]:
    pieces = [_floor(outer, floor_z)]
    for axis in (0, 1):
        for sign in (+1, -1):
            pieces.append(_rect_wall(inner, outer, COVER_LEAD_IN, floor_z, height, axis, sign))
    return pieces


def _square_edge_radius(theta: float, half: float) -> float:
    """Distance from the centre to the bounding square's edge along `theta`."""
    c, s = abs(np.cos(theta)), abs(np.sin(theta))
    return half / max(c, s)


def _wall_sector(bore_r: float, mouth_r: float, outer_half: float, floor_z: float,
                 height: float, index: int) -> trimesh.Trimesh:
    """One angular wedge of a round bore's wall.

    An annulus is not convex, so the wall is cut into `SLOT_SECTORS` wedges. Each wedge's
    inner face is a chord rather than an arc, so the collision bore is really a regular
    polygon; the radii are scaled by 1/cos(pi/K) by the caller so that polygon *circumscribes*
    the nominal circle and the narrowest point of the bore is exactly the nominal clearance.
    Boundaries start at 45 degrees so the outer square's corners land on them and each wedge
    meets exactly one square edge, which keeps the hull equal to the wedge.
    """
    step = 2.0 * np.pi / SLOT_SECTORS
    points = []
    for theta in (np.pi / 4.0 + index * step, np.pi / 4.0 + (index + 1) * step):
        d = np.array([np.cos(theta), np.sin(theta)])
        outer = _square_edge_radius(theta, outer_half)
        for u, z in ((bore_r, floor_z), (outer, floor_z), (outer, height),
                     (mouth_r, height), (bore_r, height - (mouth_r - bore_r))):
            points.append([d[0] * u, d[1] * u, z])
    return trimesh.PointCloud(np.array(points)).convex_hull


def build_slot_pieces(bore_r: float, outer_half: float, floor_z: float,
                      height: float) -> list[trimesh.Trimesh]:
    correction = 1.0 / np.cos(np.pi / SLOT_SECTORS)
    return ([_floor((outer_half, outer_half), floor_z)] +
            [_wall_sector(bore_r * correction, (bore_r + SLOT_LEAD_IN) * correction,
                          outer_half, floor_z, height, i) for i in range(SLOT_SECTORS)])


def receptacle_model_data(outer: tuple[float, float], height: float, floor_z: float,
                          what: str) -> dict:
    """`model_data0.json` for a generated blind-slot receptacle, z-up.

    The functional frame's +Z points DOWN the slot, for the same reason the bookcase's does:
    `get_place_pose` subtracts `pre_dis * (R(target) @ [0, 0, 1])`, so a positive `pre_dis`
    stands the held object above the mouth and a negative `dis` drives it in.
    """
    return {
        "center": [0.0, 0.0, height / 2.0],
        "extents": [2 * outer[0], 2 * outer[1], height],
        "scale": [1.0, 1.0, 1.0],
        "transform_matrix": np.eye(4).tolist(),
        "target_pose": [],
        "contact_points_pose": [],
        "contact_points_group": [[0]],
        "contact_points_mask": [True],
        "functional_matrix": [[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, height],
            [0.0, 0.0, 0.0, 1.0],
        ]],
        "functional_point_discription": [f"Point0: the mouth of the {what}; the axis points in."],
        "target_point_discription": [],
        "contact_points_discription": [],
        "orientation_point": [],
        "orientation_point_discription": [],
        "stable": True,
    }


def _export(modeldir: Path, model_id: int, pieces: list[trimesh.Trimesh], color, data: dict):
    (modeldir / "visual").mkdir(parents=True, exist_ok=True)
    (modeldir / "collision").mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for i, piece in enumerate(pieces):
        scene.add_geometry(piece, node_name=f"part{i}", geom_name=f"part{i}")
    scene.export(modeldir / "collision" / f"base{model_id}.glb")
    merged = trimesh.boolean.union(pieces)
    if isinstance(merged, trimesh.Scene):
        merged = merged.to_mesh()
    _paint(merged, color).export(modeldir / "visual" / f"base{model_id}.glb")
    with open(modeldir / f"model_data{model_id}.json", "w") as f:
        json.dump(data, f, indent=4)


# --- orchestration ---------------------------------------------------------------------------


def write_all(root: Path) -> dict:
    """Annotate the four shipped assets and generate the two receptacles. Idempotent."""
    out = {}

    rack, rack_stats = rack_model_data()
    _write_model_data(RACK_MODEL, RACK_ID, rack)
    print(f"  {RACK_MODEL}/base{RACK_ID}   scale {RACK_SCALE}  "
          f"{np.round(rack_stats['size_mm'], 0)} mm  rail at {rack_stats['rail_height_mm']:.0f} mm")
    out["rack"] = rack_stats

    case, case_stats = bookcase_model_data()
    _write_model_data(BOOKCASE_MODEL, BOOKCASE_ID, case)
    print(f"  {BOOKCASE_MODEL}/base{BOOKCASE_ID}      scale {BOOKCASE_SCALE}  "
          f"{np.round(case_stats['size_mm'], 0)} mm  {case_stats['posts']} posts  "
          f"centre bay {case_stats['bay_width_mm']:.1f} mm wide, "
          f"clear {case_stats['clear_height_mm']:.0f} mm")
    out["bookcase"] = case_stats

    book, book_stats = book_model_data()
    _write_model_data(BOOK_MODEL, BOOK_ID, book)
    print(f"  {BOOK_MODEL}/base{BOOK_ID}          {np.round(book_stats['size_mm'], 0)} mm  "
          f"{book_stats['n_contact']} contact points, top grasp "
          f"{book_stats['grasp_above_bottom_mm']:.0f} mm above the bottom")
    out["book"] = book_stats

    knife, knife_stats = knife_model_data()
    _write_model_data(KNIFE_MODEL, KNIFE_ID, knife)
    print(f"  {KNIFE_MODEL}/base{KNIFE_ID}         scale {KNIFE_SCALE}  "
          f"{np.round(knife_stats['size_mm'], 0)} mm  blade section "
          f"{knife_stats['blade_thickness_mm']:.1f} x {knife_stats['blade_height_mm']:.1f} mm")
    out["knife"] = knife_stats

    battery, batt_stats = battery_model_data()
    _write_model_data(BATTERY_MODEL, BATTERY_ID, battery)
    print(f"  {BATTERY_MODEL}/base{BATTERY_ID}       scale {BATTERY_SCALE}  "
          f"dia {batt_stats['diameter_mm']:.1f} mm x {batt_stats['length_mm']:.0f} mm  "
          f"grasp {batt_stats['grasp_above_base_mm']:.0f} mm up  "
          f"({batt_stats['pieces']} convex piece)")
    out["battery"] = batt_stats

    # The cover is sized from the knife's measured blade, not from a hardcoded number, so a
    # change to KNIFE_SCALE cannot silently leave the two disagreeing.
    x0, x1, z0, z1 = knife_stats["section"]
    a = 0.5 * (x1 - x0) * KNIFE_SCALE + COVER_CLEARANCE
    b = 0.5 * (z1 - z0) * KNIFE_SCALE + COVER_CLEARANCE
    inner = (a, b)
    outer = (a + COVER_LEAD_IN + COVER_WALL, b + COVER_LEAD_IN + COVER_WALL)
    height = COVER_FLOOR + COVER_DEPTH
    pieces = build_cover_pieces(inner, outer, COVER_FLOOR, height)
    _export(root / "assets" / "objects" / COVER_MODEL, 0, pieces, COVER_COLOR,
            receptacle_model_data(outer, height, COVER_FLOOR, "knife cover's slot"))
    print(f"  {COVER_MODEL}     slot {2*a*1000:.1f} x {2*b*1000:.1f} mm  "
          f"clearance {COVER_CLEARANCE*1000:.1f} mm/side  depth {COVER_DEPTH*1000:.0f} mm  "
          f"outer {2*outer[0]*1000:.0f} x {2*outer[1]*1000:.0f} x {height*1000:.0f} mm  "
          f"({len(pieces)} convex pieces)")
    out["cover"] = {"inner": inner, "outer": outer, "height": height}

    bore_r = batt_stats["radius_raw"] * BATTERY_SCALE + SLOT_CLEARANCE
    slot_outer = bore_r + SLOT_LEAD_IN + SLOT_WALL
    slot_height = SLOT_FLOOR + SLOT_DEPTH
    pieces = build_slot_pieces(bore_r, slot_outer, SLOT_FLOOR, slot_height)
    _export(root / "assets" / "objects" / SLOT_MODEL, 0, pieces, SLOT_COLOR,
            receptacle_model_data((slot_outer, slot_outer), slot_height, SLOT_FLOOR,
                                  "battery slot"))
    print(f"  {SLOT_MODEL}     bore dia {2*bore_r*1000:.1f} mm  "
          f"clearance {SLOT_CLEARANCE*1000:.1f} mm/side  depth {SLOT_DEPTH*1000:.0f} mm  "
          f"outer {2*slot_outer*1000:.0f} mm sq x {slot_height*1000:.0f} mm  "
          f"({len(pieces)} convex pieces)")
    out["slot"] = {"bore_r": bore_r, "outer": slot_outer, "height": slot_height}

    return out


def verify(root: Path, stats: dict) -> bool:
    """Build every touched asset in a headless SAPIEN scene and assert the geometry survived."""
    sys.path.insert(0, str(root))
    import os
    os.chdir(root)
    import sapien
    from envs.utils.create_actor import create_actor

    scene = sapien.Scene()
    ok = True

    def build(model, model_id, is_static=True):
        actor = create_actor(scene=scene, pose=sapien.Pose([0, 0, 0]), modelname=model,
                             convex=True, is_static=is_static, model_id=model_id)
        if actor is None:
            raise RuntimeError(f"{model}/base{model_id}: create_actor returned None")
        n = sum(len(c.get_collision_shapes())
                for c in actor.actor.get_components()
                if isinstance(c, (sapien.physx.PhysxRigidStaticComponent,
                                  sapien.physx.PhysxRigidDynamicComponent)))
        return actor, n

    checks = [
        (RACK_MODEL, RACK_ID, "functional", 0),
        (BOOKCASE_MODEL, BOOKCASE_ID, "functional", 0),
        (BOOK_MODEL, BOOK_ID, "contact", 2),
        (BOOK_MODEL, BOOK_ID, "functional", 0),
        (KNIFE_MODEL, KNIFE_ID, "functional", 1),
        (BATTERY_MODEL, BATTERY_ID, "contact", 0),
        (COVER_MODEL, 0, "functional", 0),
        (SLOT_MODEL, 0, "functional", 0),
    ]
    for model, model_id, kind, idx in checks:
        try:
            actor, n = build(model, model_id)
            if n == 0:
                print(f"  FAIL {model}/base{model_id}: zero collision shapes")
                ok = False
                continue
            pt = actor.get_point(kind, idx, "pose")
            if pt is None:
                print(f"  FAIL {model}/base{model_id}: {kind} point {idx} is None")
                ok = False
                continue
            print(f"  ok   {model}/base{model_id}: {n} collision shapes, "
                  f"{kind}{idx} at {np.round(pt.p, 4)}")
        except Exception as exc:
            print(f"  FAIL {model}/base{model_id}: {exc}")
            ok = False

    ok &= _verify_cover(stats)
    ok &= _verify_slot(stats)
    return ok


def _verify_cover(stats) -> bool:
    """The slot must be clear at the nominal cross-section and solid a wall's depth beyond."""
    a, b = stats["cover"]["inner"]
    A, B = stats["cover"]["outer"]
    parts = collision_parts(COVER_MODEL, 0, upright=False)
    z = COVER_FLOOR + 0.5 * COVER_DEPTH
    inside = np.array([[sa * a * 0.98, sb * b * 0.98, z] for sa in (-1, 1) for sb in (-1, 1)])
    outside = np.array([[sa * (A - 0.001), 0.0, z] for sa in (-1, 1)] +
                       [[0.0, sb * (B - 0.001), z] for sb in (-1, 1)])
    occ_in = np.zeros(len(inside), bool)
    occ_out = np.zeros(len(outside), bool)
    for p in parts:
        occ_in |= p.contains(inside)
        occ_out |= p.contains(outside)
    if occ_in.any():
        print(f"  FAIL {COVER_MODEL}: slot is obstructed at the nominal cross-section")
        return False
    if not occ_out.all():
        print(f"  FAIL {COVER_MODEL}: wall is missing outside the slot")
        return False
    print(f"  ok   {COVER_MODEL}: slot clear at nominal, wall solid beyond")
    return True


def _verify_slot(stats) -> bool:
    """Same check for the round bore, sampled at azimuths that miss the wedge boundaries.

    61 is coprime with SLOT_SECTORS and the offset is not a multiple of the sector angle, so
    no sample lands exactly on a shared face -- `contains` is a ray cast and is genuinely
    ambiguous there.
    """
    bore_r = stats["slot"]["bore_r"]
    parts = collision_parts(SLOT_MODEL, 0, upright=False)
    z = SLOT_FLOOR + 0.5 * SLOT_DEPTH
    thetas = np.linspace(0.0, 2 * np.pi, 61, endpoint=False) + 0.01
    inside = np.stack([bore_r * np.cos(thetas), bore_r * np.sin(thetas), np.full_like(thetas, z)], 1)
    r_out = bore_r + 0.003
    outside = np.stack([r_out * np.cos(thetas), r_out * np.sin(thetas),
                        np.full_like(thetas, z)], 1)
    occ_in = np.zeros(len(thetas), bool)
    occ_out = np.zeros(len(thetas), bool)
    for p in parts:
        occ_in |= p.contains(inside)
        occ_out |= p.contains(outside)
    if occ_in.any():
        print(f"  FAIL {SLOT_MODEL}: bore obstructed at the nominal radius "
              f"({occ_in.sum()}/{len(thetas)} azimuths)")
        return False
    if not occ_out.all():
        print(f"  FAIL {SLOT_MODEL}: wall missing 3 mm outside the bore "
              f"({(~occ_out).sum()}/{len(thetas)} azimuths)")
        return False
    print(f"  ok   {SLOT_MODEL}: bore clear at nominal radius at all "
          f"{len(thetas)} azimuths, wall solid 3 mm beyond")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="build every touched asset in a headless SAPIEN scene and check it")
    parser.add_argument("--clean", action="store_true",
                        help="remove the two generated receptacles before writing")
    args = parser.parse_args()

    if args.clean:
        for model in (COVER_MODEL, SLOT_MODEL):
            d = ROOT / "assets" / "objects" / model
            if d.exists():
                shutil.rmtree(d)
                print(f"removed {d}")

    print("writing assets:")
    stats = write_all(ROOT)

    if args.verify:
        print("\nverifying:")
        if not verify(ROOT, stats):
            print("\nVERIFY FAILED")
            return 1
        print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
