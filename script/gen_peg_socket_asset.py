"""Generate the peg-insertion socket asset (`assets/objects/121_peg-socket`).

`assets/*` is gitignored, so this script is the checked-in source of the mesh: run it
once per clone / cluster, the same way `script/_download_assets.sh` is run.

    python script/gen_peg_socket_asset.py              # write the asset
    python script/gen_peg_socket_asset.py --verify     # write it, then load it in SAPIEN

TWO FAMILIES are written, sharing the same envelope and the same clearance ladder and
differing only in the cross-section of the fit:

    121_peg-socket        square bore   <- insert_peg_socket_{loose,med,tight}, peg is a
                                           create_box primitive (no asset needed)
    122_peg-round         round peg     \
    123_peg-socket-round  round bore    /  <- insert_peg_socket_round_med

Three model ids are written per socket -- the difficulty ladder consumed by
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

THE ROUND FAMILY reuses all of that -- same 0.10 x 0.10 x 0.05 outer block, same 0.012 floor,
same 0.038 bore depth, same clearance/lead-in per rung -- so the two sockets differ in nothing
but the shape of the hole. Two things change:

  * the peg becomes an asset. A box gets contact and functional points from `create_box` for
    free; a cylinder has no such primitive (`create_cylinder` returns a bare Entity with no
    Actor data, and its long axis is X, not Z), so `122_peg-round` is a mesh with a
    hand-written `model_data0.json` whose points MIRROR `create_box(boxtype="long")` exactly:
    the same two functional points at +-0.060, and the same 8 contact points -- 4 per band, at
    the same azimuths, at the same +-0.7 * half-length. Ids [0,1,2,3] therefore mean the same
    upper band on both pegs, so the two tasks grasp identically and any difference between
    them is attributable to the insertion rather than to the grasp.

  * an annulus is not convex, so the socket wall is cut into ROUND_SECTORS angular wedges
    instead of four slabs. Each wedge is the convex hull of the same pentagon profile swept
    between two boundary rays, so its inner face is a CHORD rather than an arc -- the bore is
    really a regular ROUND_SECTORS-gon. The radii are scaled by 1/cos(pi/K) so the polygon
    CIRCUMSCRIBES the nominal circle: the tightest point of the bore is then exactly the
    nominal clearance and the faceting only ever adds room, at most
    b * (1/cos(pi/K) - 1) = 0.05 mm at K=32 -- 3% of the tight rung's clearance and far below
    the ~1.3 mm placement error the expert actually has.

    The sector boundaries are placed at 45 + j*(90/m) degrees with K = 4m, so the outer
    square's four corners fall ON boundaries. Each wedge then meets exactly one square edge
    and its outer face is a single plane, which is what keeps the hull equal to the wedge
    instead of cutting a corner off.
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

# --- the round family --------------------------------------------------------------------
ROUND_PEG_MODELNAME = "122_peg-round"
ROUND_SOCKET_MODELNAME = "123_peg-socket-round"

PEG_RADIUS = 0.020  # same 40 mm width as the square peg, so it inscribes in it
PEG_HALF_LEN = 0.060  # identical to PEG_HALF_SIZE[2]: same 120 mm standing peg
PEG_SECTIONS = 64  # the peg is one convex hull, so sections are nearly free
ROUND_SECTORS = 32  # must be a multiple of 4 -- see the module docstring
CONTACT_BAND = 0.7  # create_box("long") puts its contact points at +-0.7 * half-length

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


SOCKET_COLOR = (0.47, 0.49, 0.53, 1.0)  # gray
PEG_COLOR = (0.85, 0.35, 0.1, 1.0)  # orange -- the same base_color create_box gives the square peg


def _paint(mesh: trimesh.Trimesh, color) -> trimesh.Trimesh:
    """Give the mesh a real glTF PBR material rather than per-face vertex colors.

    This matters: `ColorVisuals` exports to a glb as a COLOR_0 vertex attribute and NO
    material, and SAPIEN's glb loader takes its base color from the material -- so a mesh
    painted that way renders in SAPIEN's default gray whatever color trimesh was told.
    `create_box` sets `base_color` straight on a `RenderMaterial`, so this is what makes an
    asset peg match the primitive one.
    """
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=color, metallicFactor=0.1, roughnessFactor=0.6))
    return mesh


def build_visual(pieces: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """A single watertight solid, so the overlapping slab faces do not z-fight."""
    merged = trimesh.boolean.union(pieces)
    if isinstance(merged, trimesh.Scene):
        merged = merged.to_mesh()
    return _paint(merged, SOCKET_COLOR)


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


# --- the round family ---------------------------------------------------------------------


def bore_radius(clearance: float) -> float:
    return PEG_RADIUS + clearance


def _round_peg_mesh() -> trimesh.Trimesh:
    """The peg: one convex piece, long axis +Z, origin at the centre (as `create_box`)."""
    mesh = trimesh.creation.cylinder(radius=PEG_RADIUS, height=2 * PEG_HALF_LEN,
                                     sections=PEG_SECTIONS)
    return _paint(mesh, PEG_COLOR)


def _square_edge_radius(theta: float) -> float:
    """Distance from the axis to the outer square along direction `theta`."""
    d = np.array([np.cos(theta), np.sin(theta)])
    # The binding edge is the one this direction actually leaves through.
    return OUTER_HALF / np.max(np.abs(d))


def _wall_sector(bore_r: float, mouth_r: float, index: int) -> trimesh.Trimesh:
    """One convex wall wedge, the pentagon profile swept between two boundary rays.

    The profile is the same one the square slabs use, in (radius, z):

        (b, 0.012) - (outer, 0.012) - (outer, 0.05) - (b + L, 0.05) - (b, 0.05 - L)

    swept between the two rays bounding this sector. Boundaries start at 45 deg so the
    square's corners land on them and `outer` is a single plane per wedge.
    """
    step = 2.0 * np.pi / ROUND_SECTORS
    points = []
    for theta in (np.pi / 4.0 + index * step, np.pi / 4.0 + (index + 1) * step):
        d = np.array([np.cos(theta), np.sin(theta)])
        outer = _square_edge_radius(theta)
        for u, z in ((bore_r, FLOOR_Z), (outer, FLOOR_Z), (outer, HEIGHT),
                     (mouth_r, HEIGHT), (bore_r, HEIGHT - (mouth_r - bore_r))):
            points.append([d[0] * u, d[1] * u, z])
    return trimesh.PointCloud(np.array(points)).convex_hull


def build_round_pieces(clearance: float, lead_in: float) -> list[trimesh.Trimesh]:
    """Floor + ROUND_SECTORS wall wedges.

    Radii are scaled by 1/cos(pi/K) so the K-gon CIRCUMSCRIBES the nominal circle -- the
    narrowest point of the bore is then exactly `clearance` per side and the faceting can only
    ever add room. Scaling the mouth by the same factor keeps the chamfer at 45 deg.
    """
    correction = 1.0 / np.cos(np.pi / ROUND_SECTORS)
    bore_r = bore_radius(clearance) * correction
    mouth_r = (bore_radius(clearance) + lead_in) * correction
    return [_floor_slab()] + [_wall_sector(bore_r, mouth_r, i) for i in range(ROUND_SECTORS)]


def round_peg_model_data() -> dict:
    """`model_data0.json` for the peg, mirroring `create_box(boxtype="long")` point for point.

    `scale` is [1,1,1] and the mesh is written at true size, so every translation below is in
    metres -- unlike `create_box`, whose scale IS its half-size and whose point translations
    are therefore unit-normalised.

    Contact frames follow the box's convention exactly: the frame's +X is the peg's own long
    axis and its +Z is the outward surface normal at azimuth phi, with the translation on the
    CENTRELINE at +-0.7 * half-length rather than on the surface. Four azimuths per band in
    the box's own order (front, right, left, back), so contact_point_id=[0,1,2,3] selects the
    same upper band on either peg. A cylinder would happily take eight, but matching the box
    keeps the grasp phase of the two tasks comparable.
    """

    def contact_frame(phi: float, z: float) -> list:
        c, s = float(np.cos(phi)), float(np.sin(phi))
        return [[0.0, s, c, 0.0], [0.0, -c, s, 0.0], [1.0, 0.0, 0.0, z], [0.0, 0.0, 0.0, 1.0]]

    azimuths = [0.0, -np.pi / 2, np.pi / 2, np.pi]  # front, right, left, back
    band = CONTACT_BAND * PEG_HALF_LEN
    contacts = ([contact_frame(phi, band) for phi in azimuths]
                + [contact_frame(phi, -band) for phi in azimuths])

    # +Z out of the end cap, as create_box's own functional points do: fp0 = the tip.
    def cap(z: float) -> list:
        return [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, z],
                [0.0, 0.0, 0.0, 1.0]]

    return {
        "center": [0.0, 0.0, 0.0],
        "extents": [2 * PEG_RADIUS, 2 * PEG_RADIUS, 2 * PEG_HALF_LEN],
        "scale": [1.0, 1.0, 1.0],
        "transform_matrix": np.eye(4).tolist(),
        "target_pose": [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, PEG_HALF_LEN],
                         [0.0, 0.0, 0.0, 1.0]]],
        "contact_points_pose": contacts,
        "contact_points_group": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "contact_points_mask": [True, True],
        "functional_matrix": [cap(-PEG_HALF_LEN), cap(PEG_HALF_LEN)],
        "target_point_discription": ["The center point on the top of the peg."],
        "contact_points_discription": [],
        "functional_point_discription": [
            "Point0: The tip of the peg, the axis direction points out of the tip.",
            "Point1: The top of the peg, the axis direction points out of the top.",
        ],
        "orientation_point_discription": [""],
        "stable": True,
        "peg_radius": PEG_RADIUS,
        "peg_half_length": PEG_HALF_LEN,
    }


def round_socket_model_data(clearance: float, lead_in: float) -> dict:
    """Same block, same functional frame as the square socket -- only the bore differs."""
    data = model_data(bore_radius(clearance), clearance, lead_in)
    data.pop("bore_half_width")
    data["bore_radius"] = bore_radius(clearance)
    data["sectors"] = ROUND_SECTORS
    data["functional_point_discription"] = [
        "Point0: The mouth of the round socket bore, the axis direction points into the hole."
    ]
    return data


def write_round_assets(root: Path) -> tuple[Path, Path]:
    pegdir = root / "assets" / "objects" / ROUND_PEG_MODELNAME
    (pegdir / "visual").mkdir(parents=True, exist_ok=True)
    (pegdir / "collision").mkdir(parents=True, exist_ok=True)

    peg = _round_peg_mesh()
    peg.export(pegdir / "visual" / "base0.glb")
    peg.export(pegdir / "collision" / "base0.glb")
    with open(pegdir / "model_data0.json", "w") as f:
        json.dump(round_peg_model_data(), f, indent=4)
    print(f"  {ROUND_PEG_MODELNAME}/base0  dia {2*PEG_RADIUS*1000:.0f} mm x "
          f"{2*PEG_HALF_LEN*1000:.0f} mm long  ({PEG_SECTIONS}-gon, 1 convex piece)")

    socketdir = root / "assets" / "objects" / ROUND_SOCKET_MODELNAME
    (socketdir / "visual").mkdir(parents=True, exist_ok=True)
    (socketdir / "collision").mkdir(parents=True, exist_ok=True)

    for model_id, (label, clearance, lead_in) in VARIANTS.items():
        pieces = build_round_pieces(clearance, lead_in)

        scene = trimesh.Scene()
        for i, piece in enumerate(pieces):
            scene.add_geometry(piece, node_name=f"part{i}", geom_name=f"part{i}")
        scene.export(socketdir / "collision" / f"base{model_id}.glb")

        build_visual(pieces).export(socketdir / "visual" / f"base{model_id}.glb")

        with open(socketdir / f"model_data{model_id}.json", "w") as f:
            json.dump(round_socket_model_data(clearance, lead_in), f, indent=4)

        print(f"  {ROUND_SOCKET_MODELNAME}/base{model_id} ({label:5s}) bore dia "
              f"{2*bore_radius(clearance)*1000:.1f} mm  clearance {clearance*1000:.1f} mm/side  "
              f"lead-in {lead_in*1000:.0f} mm  ({len(pieces)} convex pieces)")

    return pegdir, socketdir


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


def verify_round(root: Path) -> bool:
    """Same silent-failure check for the round family, plus the one thing faceting could break.

    An annulus cut into wedges is only a round bore if the wedges' chord faces stay OUTSIDE
    the nominal circle. So the fit itself is asserted: every azimuth must be free at the
    nominal bore radius (the peg's own surface) and solid a few mm beyond it.
    """
    import os

    import sapien

    sys.path.insert(0, str(root))
    os.chdir(root)
    from envs.utils.create_actor import create_actor

    ok = True
    # 61 is coprime with ROUND_SECTORS and the offset is not a multiple of the sector angle,
    # so no sample lands exactly on a wedge boundary -- `contains` is a ray cast and is
    # genuinely ambiguous on the shared face between two pieces.
    azimuths = np.linspace(0.0, 2 * np.pi, 61, endpoint=False) + 0.01

    # -- the peg ---------------------------------------------------------------------------
    scene = sapien.Scene()
    peg = create_actor(scene, sapien.Pose([0, 0, 0]), ROUND_PEG_MODELNAME, convex=True,
                       is_static=False, model_id=0)
    if peg is None:
        print(f"  {ROUND_PEG_MODELNAME} FAIL: create_actor returned None")
        return False
    shapes = peg.actor.find_component_by_type(
        sapien.physx.PhysxRigidDynamicComponent).get_collision_shapes()
    if len(shapes) != 1:
        print(f"  {ROUND_PEG_MODELNAME} FAIL: {len(shapes)} collision shapes, expected 1")
        ok = False
    fp = np.array(peg.get_functional_point(0, "matrix"))
    if not np.allclose(fp[:3, 3], [0, 0, -PEG_HALF_LEN], atol=1e-9):
        print(f"  {ROUND_PEG_MODELNAME} FAIL: fp0 at {fp[:3, 3]}, expected the tip")
        ok = False
    if not np.allclose(fp[:3, 2], [0, 0, -1], atol=1e-9):
        print(f"  {ROUND_PEG_MODELNAME} FAIL: fp0 axis {fp[:3, 2]}, expected [0,0,-1]")
        ok = False
    # The grasp band must sit where create_box("long") puts it, or contact_point_id=[0,1,2,3]
    # would not mean the same band on the two pegs.
    cp = np.array(peg.get_contact_point(0, "matrix"))
    if not np.allclose(cp[:3, 3], [0, 0, CONTACT_BAND * PEG_HALF_LEN], atol=1e-9):
        print(f"  {ROUND_PEG_MODELNAME} FAIL: contact 0 at {cp[:3, 3]}, expected the upper band")
        ok = False
    if ok:
        print(f"  {ROUND_PEG_MODELNAME}/base0 ok: 1 convex shape, tip and upper band in place")

    # -- the sockets -----------------------------------------------------------------------
    for model_id, (label, clearance, lead_in) in VARIANTS.items():
        scene = sapien.Scene()
        actor = create_actor(scene, sapien.Pose([0, 0, 0]), ROUND_SOCKET_MODELNAME, convex=True,
                             is_static=True, model_id=model_id)
        if actor is None:
            print(f"  base{model_id} FAIL: create_actor returned None")
            ok = False
            continue

        shapes = actor.actor.find_component_by_type(
            sapien.physx.PhysxRigidStaticComponent).get_collision_shapes()
        pieces = build_round_pieces(clearance, lead_in)
        if len(shapes) != len(pieces):
            print(f"  base{model_id} FAIL: {len(shapes)} collision shapes, expected {len(pieces)}")
            ok = False
            continue

        fp = np.array(actor.get_functional_point(0, "matrix"))
        if not (np.allclose(fp[:3, 3], [0, 0, HEIGHT], atol=1e-9)
                and np.allclose(fp[:3, 2], [0, 0, -1], atol=1e-9)):
            print(f"  base{model_id} FAIL: functional frame is not the mouth pointing in")
            ok = False
            continue

        r = bore_radius(clearance)
        mid_bore_z = (FLOOR_Z + HEIGHT - lead_in) / 2.0
        ring = lambda rad: np.stack(
            [rad * np.cos(azimuths), rad * np.sin(azimuths),
             np.full(azimuths.shape, mid_bore_z)], axis=1)
        occupied = lambda pts: np.any([m.contains(pts) for m in pieces], axis=0)

        if occupied(ring(r - 1e-5)).any():
            bad = int(occupied(ring(r - 1e-5)).sum())
            print(f"  base{model_id} FAIL: bore is under-sized at {bad}/{len(azimuths)} azimuths "
                  f"-- the wedge chords cut inside the nominal circle")
            ok = False
            continue
        if not occupied(ring(r + 0.003)).all():
            print(f"  base{model_id} FAIL: a wall sample 3 mm outside the bore is empty")
            ok = False
            continue

        facet = r * (1.0 / np.cos(np.pi / ROUND_SECTORS) - 1.0)
        print(f"  base{model_id} ({label:5s}) ok: {len(shapes)} convex shapes, bore clear at the "
              f"nominal {r*1000:.1f} mm radius all round (+{facet*1000:.2f} mm faceting), "
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
        for name in (MODELNAME, ROUND_PEG_MODELNAME, ROUND_SOCKET_MODELNAME):
            shutil.rmtree(root / "assets" / "objects" / name, ignore_errors=True)

    print(f"writing assets/objects/{MODELNAME} (square) ...")
    modeldir = write_asset(root)
    print(f"wrote {modeldir}")

    print(f"writing assets/objects/{{{ROUND_PEG_MODELNAME},{ROUND_SOCKET_MODELNAME}}} (round) ...")
    pegdir, socketdir = write_round_assets(root)
    print(f"wrote {pegdir} and {socketdir}")

    if args.verify:
        print("verifying ...")
        if not (verify(root) and verify_round(root)):
            print("VERIFY FAILED")
            return 1
        print("verify ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
