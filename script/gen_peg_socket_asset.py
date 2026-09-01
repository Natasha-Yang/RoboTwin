"""Generate the peg-insertion socket asset (`assets/objects/121_peg-socket`).

`assets/*` is gitignored, so this script is the checked-in source of the mesh: run it
once per clone / cluster, the same way `script/_download_assets.sh` is run.

    python script/gen_peg_socket_asset.py              # write the asset
    python script/gen_peg_socket_asset.py --verify     # write it, then load it in SAPIEN

Three model ids are written -- the difficulty ladder consumed by
`insert_peg_socket_{loose,med,tight}`:

    id  variant  clearance/side  45deg lead-in  approx capture radius
    0   loose        6.0 mm         14 mm            19 mm
    1   med          3.0 mm          6 mm             8 mm
    2   tight        1.5 mm          2 mm             3 mm

BOTH numbers scale, and the lead-in is the one that matters. An earlier revision varied only
the bore and held the chamfer mouth fixed at 35 mm: all three rungs then had the SAME ~15 mm
lateral capture, a released peg self-centred on the chamfer regardless of bore, and the three
tasks produced byte-identical expert results. Clearance alone only bites once the peg is
already aligned, which is the easy half of the problem. The chamfer is what sets the error
budget an imprecise agent actually has.

Geometry (metres, origin at the bottom face centre so the default `zlim=[0.741]` rests it
on the table, matching every in-tree asset):

    outer          0.10 x 0.10 x 0.05
    floor          z in [0, 0.012]                      -- blind bore, so the peg seats on
                                                           the socket rather than the table
    straight bore  z in [0.012, 0.05 - L], half-width b
    chamfer        the top L, opening from b to b + L   -- 45 deg, so the lead-in L is both
                                                           its height and its lateral reach

The collision mesh is FIVE CONVEX PIECES (floor + four wall slabs), loaded with
`convex=True` -> `add_multiple_convex_collisions_from_file`, one hull per submesh. It is
deliberately not a single non-convex mesh: SAPIEN's actor builder swallows shape-cook
failures in a bare `except RuntimeError: continue`, so a mesh that fails to cook yields an
actor with zero collision shapes and NO error -- the peg would pass straight through the
socket and the task would "succeed" nonsensically. `--verify` exists to catch exactly that.

Each wall slab's cross-section in the (u, z) plane is the pentagon

    (b, 0.012) - (0.05, 0.012) - (0.05, 0.05) - (b + L, 0.05) - (b, 0.05 - L)

which is convex for any b + L < 0.05, extruded across the full 0.10 width. A point is inside
the bore iff |x| < f(z) and |y| < f(z), i.e. iff it is outside all four slabs -- so the
union of the four reproduces the tapered hole exactly.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

MODELNAME = "121_peg-socket"

OUTER_HALF = 0.05  # 0.10 x 0.10 footprint
HEIGHT = 0.05
FLOOR_Z = 0.012  # blind bore: the peg seats here, 0.038 below the mouth
PEG_HALF = 0.020  # must match the peg in envs/_peg_insertion_base.py

# model_id -> (label, clearance per side, 45deg chamfer lead-in)
VARIANTS = {
    0: ("loose", 0.0060, 0.014),
    1: ("med", 0.0030, 0.006),
    2: ("tight", 0.0015, 0.002),
}


def bore_half_width(clearance: float) -> float:
    return PEG_HALF + clearance


def _wall_slab(bore_half: float, lead_in: float, axis: int, sign: int) -> trimesh.Trimesh:
    """One convex wall slab, as the hull of its pentagon cross-section swept across the part.

    `axis` 0 -> the slab bounded in x, 1 -> bounded in y; `sign` which side.
    """
    pentagon = [
        (bore_half, FLOOR_Z),
        (OUTER_HALF, FLOOR_Z),
        (OUTER_HALF, HEIGHT),
        (bore_half + lead_in, HEIGHT),
        (bore_half, HEIGHT - lead_in),
    ]
    points = []
    for u, z in pentagon:
        for w in (-OUTER_HALF, OUTER_HALF):
            xy = [0.0, 0.0]
            xy[axis] = sign * u
            xy[1 - axis] = w
            points.append([xy[0], xy[1], z])
    return trimesh.PointCloud(np.array(points)).convex_hull


def _floor_slab() -> trimesh.Trimesh:
    return trimesh.creation.box(
        extents=(2 * OUTER_HALF, 2 * OUTER_HALF, FLOOR_Z),
        transform=trimesh.transformations.translation_matrix((0.0, 0.0, FLOOR_Z / 2.0)),
    )


def build_pieces(bore_half: float, lead_in: float) -> list[trimesh.Trimesh]:
    """The five convex pieces the collision mesh is made of."""
    pieces = [_floor_slab()]
    for axis in (0, 1):
        for sign in (+1, -1):
            pieces.append(_wall_slab(bore_half, lead_in, axis, sign))
    return pieces


def build_visual(pieces: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """A single watertight solid, so the overlapping slab faces do not z-fight."""
    merged = trimesh.boolean.union(pieces)
    if isinstance(merged, trimesh.Scene):
        merged = merged.to_mesh()
    merged.visual = trimesh.visual.ColorVisuals(
        merged, face_colors=np.tile([120, 125, 135, 255], (len(merged.faces), 1)))
    return merged


def model_data(bore_half: float, clearance: float, lead_in: float) -> dict:
    """`model_data<id>.json`, shaped like the in-tree static receptacle 040_rack.

    `scale` is REQUIRED: create_actor swallows a JSON load error into `model_data=None`,
    which then fails much later and opaquely on the first get_point call.

    The functional frame is the crux of the task. Its +Z must point DOWN THE BORE, because
    `get_place_pose` computes `target - grasp_bias - pre_dis * (R(target) @ [0,0,1])`
    (envs/_base_task.py:1384-1395) -- so a positive `pre_dis` stands the peg above the mouth
    and a negative `dis` drives it in. Columns below are x=[1,0,0], y=[0,-1,0], z=[0,0,-1]
    (det +1), translated to the top face centre.
    """
    identity = np.eye(4).tolist()
    return {
        "center": [0.0, 0.0, HEIGHT / 2.0],
        "extents": [2 * OUTER_HALF, 2 * OUTER_HALF, HEIGHT],
        "scale": [1.0, 1.0, 1.0],
        "transform_matrix": identity,
        "target_pose": [identity],
        "contact_points_pose": [],
        "contact_points_group": [[0]],
        "contact_points_mask": [True],
        "functional_matrix": [[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, HEIGHT],
            [0.0, 0.0, 0.0, 1.0],
        ]],
        "target_point_discription": ["The center point on the bottom of the socket."],
        "contact_points_discription": [],
        "functional_point_discription":
        ["Point0: The mouth of the socket bore, the axis direction points into the hole."],
        "orientation_point_discription": [""],
        "stable": True,
        # Not read by the sim -- recorded so a dataset can be traced back to its geometry.
        "bore_half_width": bore_half,
        "clearance_per_side": clearance,
        "lead_in": lead_in,
        "bore_depth": HEIGHT - FLOOR_Z,
    }


def write_asset(root: Path) -> Path:
    modeldir = root / "assets" / "objects" / MODELNAME
    (modeldir / "visual").mkdir(parents=True, exist_ok=True)
    (modeldir / "collision").mkdir(parents=True, exist_ok=True)

    for model_id, (label, clearance, lead_in) in VARIANTS.items():
        bore_half = bore_half_width(clearance)
        pieces = build_pieces(bore_half, lead_in)

        # One named node per convex piece: add_multiple_convex_collisions_from_file makes
        # one hull per submesh, which is what keeps the bore open.
        scene = trimesh.Scene()
        for i, piece in enumerate(pieces):
            scene.add_geometry(piece, node_name=f"part{i}", geom_name=f"part{i}")
        scene.export(modeldir / "collision" / f"base{model_id}.glb")

        build_visual(pieces).export(modeldir / "visual" / f"base{model_id}.glb")

        with open(modeldir / f"model_data{model_id}.json", "w") as f:
            json.dump(model_data(bore_half, clearance, lead_in), f, indent=4)

        print(f"  base{model_id} ({label:5s}) bore {2*bore_half*1000:.1f} mm  "
              f"clearance {clearance*1000:.1f} mm/side  lead-in {lead_in*1000:.0f} mm  "
              f"({len(pieces)} convex pieces)")

    return modeldir


def verify(root: Path) -> bool:
    """Load each variant in a headless SAPIEN scene and assert the geometry survived.

    The failure this is here for is silent: a collision shape that fails to cook leaves the
    actor with zero shapes and prints nothing.
    """
    import sapien

    # create_actor resolves `assets/objects` relative to the CWD and lives in the repo root
    # package, so make both work regardless of where this script was invoked from.
    sys.path.insert(0, str(root))
    import os
    os.chdir(root)

    ok = True
    for model_id, (label, clearance, lead_in) in VARIANTS.items():
        bore_half = bore_half_width(clearance)
        scene = sapien.Scene()

        from envs.utils.create_actor import create_actor

        actor = create_actor(scene, sapien.Pose([0, 0, 0]), MODELNAME, convex=True,
                             is_static=True, model_id=model_id)
        if actor is None:
            print(f"  base{model_id} FAIL: create_actor returned None")
            ok = False
            continue

        shapes = actor.actor.find_component_by_type(
            sapien.physx.PhysxRigidStaticComponent).get_collision_shapes()
        n_expected = len(build_pieces(bore_half, lead_in))
        if len(shapes) != n_expected:
            print(f"  base{model_id} FAIL: {len(shapes)} collision shapes, expected {n_expected}")
            ok = False
            continue

        # The functional frame must sit at the mouth with +Z pointing down the bore.
        fp = np.array(actor.get_functional_point(0, "matrix"))
        if not np.allclose(fp[:3, 3], [0, 0, HEIGHT], atol=1e-9):
            print(f"  base{model_id} FAIL: functional point at {fp[:3, 3]}, expected [0,0,{HEIGHT}]")
            ok = False
            continue
        if not np.allclose(fp[:3, 2], [0, 0, -1], atol=1e-9):
            print(f"  base{model_id} FAIL: functional axis {fp[:3, 2]}, expected [0,0,-1] (into the bore)")
            ok = False
            continue

        # The bore must actually be empty, and the walls must actually be there.
        pieces = build_pieces(bore_half, lead_in)
        mid_bore_z = (FLOOR_Z + HEIGHT - lead_in) / 2.0
        inside = [0.0, 0.0, mid_bore_z]  # mid-bore, must be free
        wall = [bore_half + 0.005, 0.0, mid_bore_z]  # in the wall, must be solid
        occupied = lambda p: any(bool(m.contains([p])[0]) for m in pieces)
        if occupied(inside):
            print(f"  base{model_id} FAIL: bore centre is solid -- the hole closed up")
            ok = False
            continue
        if not occupied(wall):
            print(f"  base{model_id} FAIL: wall sample is empty -- the socket has no wall")
            ok = False
            continue

        print(f"  base{model_id} ({label:5s}) ok: {len(shapes)} collision shapes, bore open, "
              f"fp at the mouth pointing in")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="after writing, load each variant in SAPIEN and check the geometry")
    parser.add_argument("--clean", action="store_true", help="remove the asset dir first")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.clean:
        shutil.rmtree(root / "assets" / "objects" / MODELNAME, ignore_errors=True)

    print(f"writing assets/objects/{MODELNAME} ...")
    modeldir = write_asset(root)
    print(f"wrote {modeldir}")

    if args.verify:
        print("verifying ...")
        if not verify(root):
            print("VERIFY FAILED")
            return 1
        print("verify ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
