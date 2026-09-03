"""Shared implementation for the peg-insertion tasks.

Not a task itself (leading underscore, like `_base_task.py`) -- the registered tasks are thin
subclasses that differ only in class attributes. Task discovery is
`importlib.import_module(f"envs.{task_name}")`, so only the concrete files are reachable.

    insert_peg_socket_{loose,med,tight}   square peg (40 x 40 mm) in a square bore
    insert_peg_socket_round_med           round peg (40 mm dia)   in a round bore

The two families share this file, the same 120 mm peg length, the same 0.10 x 0.10 x 0.05
socket block, the same 0.038 m bore depth and the same clearance ladder, so the cross-section
of the fit is the only thing that differs between them. Three attributes carry it:
`socket_modelname` / `socket_model_id` pick the bore, `peg_modelname` picks the peg (None =
the square `create_box` primitive), and `place_constrain` picks how yaw is handled -- see
`_place_kwargs`.

The square peg is a `create_box` primitive: a box Actor comes with working contact and
functional points for free (create_actor.py:159-196), and `boxtype="long"` gives 8 side
contact points in two bands along the long axis -- ids [0,1,2,3] are the upper band, which is
the one to grasp so the fingers stay clear of the socket rim at full insertion depth. There is
no equivalent primitive for a cylinder (`create_cylinder` returns a bare Entity with no Actor
data, and its long axis is X), so the round peg is the generated asset `122_peg-round`, whose
`model_data0.json` reproduces those same points at the same offsets -- GRASP_BAND therefore
means the same band on either peg and the grasp phase of the two families is identical.

The socket is the generated mesh asset `121_peg-socket` (script/gen_peg_socket_asset.py),
whose functional point 0 sits at the mouth of the bore with its +Z pointing INTO the hole.
That axis convention is what makes the insertion expressible: `get_place_pose` computes
`target - grasp_bias - pre_dis * (R(target) @ [0,0,1])` (_base_task.py:1384-1395), so a
positive `pre_dis` stands the peg above the mouth and a NEGATIVE `dis` drives it in. Same
convention as envs/hanging_mug.py, which threads a mug onto a rack with `dis=-0.05`.
"""

from pathlib import Path

import numpy as np

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import *
from .utils import *

SOCKET_MODELNAME = "121_peg-socket"
ROUND_PEG_MODELNAME = "122_peg-round"
ROUND_SOCKET_MODELNAME = "123_peg-socket-round"

# Peg: 40 x 40 x 120 mm, standing. envs/handover_block.py stands a larger 60 x 60 x 200 mm
# box successfully, so this is the conservative end of a shipped precedent. The round peg is
# 40 mm in diameter and the same 120 mm long -- it inscribes in the square one, so both have
# the same width across the fit and the same grasp span.
PEG_HALF_SIZE = (0.020, 0.020, 0.060)

TABLE_Z = 0.741  # rand_pose's own default; preprocess() adds table_z_bias on top

# Socket geometry, mirrored from script/gen_peg_socket_asset.py. Kept as constants rather
# than read back from model_data<id>.json so the thresholds below read as numbers.
SOCKET_HEIGHT = 0.05
SOCKET_FLOOR_Z = 0.012
BORE_DEPTH = SOCKET_HEIGHT - SOCKET_FLOOR_Z  # 0.038 m from the mouth to the seat

# --- insertion ---------------------------------------------------------------------------
PRE_GRASP_DIS = 0.07  # verbatim envs/handover_block.py:60
GRASP_BAND = [0, 1, 2, 3]  # the UPPER contact band, 0.102 m above the tip
LIFT_Z = 0.12
PRE_INSERT_DIS = 0.07  # tip 70 mm above the mouth, clear of the socket + chamfer
NEAR_INSERT_DIS = 0.015  # a second waypoint just above the mouth, so the final constrained
#                          descent is 45 mm rather than 100 mm -- the planner does not track
#                          a constrained straight line exactly, and the drift it accumulates
#                          is what the tight variants have no room for
INSERT_DIS = -0.030  # tip 30 mm below the mouth; the seat is at 38 mm, so it is
#                      released 8 mm short and free-falls the rest, self-centring.
#                      Commanding it all the way down just widens the jamming window
#                      against a stiff position-controlled arm.
RETRACT_DIS = 0.06

# --- success -----------------------------------------------------------------------------
SUCCESS_DEPTH = 0.030  # of a 0.038 m bore. Unachievable outside the hole (a peg resting on
#                        the rim reads 0), so this threshold carries the check.
SUCCESS_LATERAL = 0.010  # deliberately loose -- depth is the real test, and a tight lateral
#                          bound would make phase-2 replay flip (collect_data.py:252)
SUCCESS_UPRIGHT = 0.98

# --- reward ------------------------------------------------------------------------------
# `step_reward` is a DELTA (see below), so these constants only fix gates and scale.
GRASP_BONUS = 0.1  # paid once, on the first correct grasp
GRASP_MIN_AXIAL = PEG_HALF_SIZE[2]  # = half the peg's length, so the grasp must be on the
#                                     UPPER half -- GRASP_BAND sits 0.102 m above the tip,
#                                     and a grasp lower down puts the fingers inside the
#                                     socket footprint at full insertion depth.
GRASP_UPRIGHT = 0.9  # a toppled peg is not a correct grasp
ALIGN_RADIUS = SUCCESS_LATERAL  # tip this close to the bore axis == actually in the bore.
#                                 Same bound check_success uses, so the depth term only ever
#                                 accrues in states on the success path. It is far inside the
#                                 socket's own 0.10 x 0.10 footprint, so a peg standing on the
#                                 table beside the socket can never clear it.
UPRIGHT_WEIGHT = 0.2  # uprightness is a cosine, not a length; this puts recovering from a
#                       30 deg tilt (~0.13 of cosine) on the scale of the 0.038 m of depth.
DELTA_CLIP = 0.1  # per-term, per-call, as in envs/lift_pot.py / envs/put_object_cabinet.py

_OBJECTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "objects"
_MISSING = [
    name for name in (SOCKET_MODELNAME, ROUND_PEG_MODELNAME, ROUND_SOCKET_MODELNAME)
    if not (_OBJECTS_DIR / name).exists()
]
if _MISSING:
    # create_actor only prints "is not exist model file!" and returns None, and the seed
    # search in collect_data.py is uncapped -- so a missing asset would spin forever instead
    # of failing. Raise at import, which is outside class_decorator's try/except. One script
    # writes all of them, so any one missing means the same fix.
    raise FileNotFoundError(
        f"{', '.join(_MISSING)} missing from assets/objects. `assets/` is gitignored, so these "
        f"meshes are generated rather than downloaded:\n"
        f"    python script/gen_peg_socket_asset.py --verify")


class _PegInsertionBase(Base_Task):
    """Grasp a standing peg and insert it into a socket's blind bore, single-arm."""

    # -- what a concrete task varies ------------------------------------------------------
    socket_modelname: str = SOCKET_MODELNAME
    socket_model_id: int = None  # the rung: 0 loose, 1 med, 2 tight
    peg_modelname: str = None  # None -> the square `create_box` primitive
    peg_description: str = "peg"  # the `{B}` slot; an "<asset>/base<id>" path needs a file
    #                               under description/objects_description/
    place_constrain: str = "align"  # "align" (square) or "free" (a solid of revolution)

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # Peg on one side of the table, socket on the same side so one arm reaches both.
        # |x| > 0.08 keeps the peg off the midline, where neither arm has a clean approach.
        for _ in range(100):
            peg_pose = rand_pose(
                xlim=[-0.26, 0.26],
                ylim=[-0.08, 0.02],
                zlim=[TABLE_Z + PEG_HALF_SIZE[2]],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.785],  # yaw only; roll/pitch is what would topple it
            )
            if abs(peg_pose.p[0]) > 0.08:
                break

        side = 1.0 if peg_pose.p[0] > 0 else -1.0
        for _ in range(100):
            socket_pose = rand_pose(
                xlim=[0.08, 0.26] if side > 0 else [-0.26, -0.08],
                ylim=[0.12, 0.20],
                zlim=[TABLE_Z],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.785],
            )
            # The gripper sits 0.12 m from the grasp point (_base_task.py:1145); closer than
            # this and the wrist is inside the socket footprint while grasping the peg.
            if np.linalg.norm(socket_pose.p[:2] - peg_pose.p[:2]) > 0.15:
                break

        self.peg = self._create_peg(peg_pose)
        self.socket = create_actor(
            scene=self,
            pose=socket_pose,
            modelname=self.socket_modelname,
            convex=True,  # -> one convex hull per submesh, which is what keeps the bore open
            is_static=True,
            model_id=self.socket_model_id,
        )

        self.add_prohibit_area(self.peg, padding=0.1)
        self.add_prohibit_area(self.socket, padding=0.1)

        # Fixed in load_actors, not in play_once, so check_success and step_reward do not
        # depend on the expert having run -- during a policy rollout it has not.
        self.arm_tag = ArmTag("left" if peg_pose.p[0] < 0 else "right")

        # Reward state, reset per episode because load_actors runs on every setup_demo.
        lateral, depth = self._insertion_state()
        self.last_lateral = lateral
        self.last_depth = self._bore_depth(lateral, depth)
        self.last_upright = None  # None == "not in the bore", see step_reward
        self.grasp_rewarded = False

    def _create_peg(self, pose):
        """The square peg is a primitive; the round one is a mesh asset.

        Named "peg" either way -- `create_actor` would otherwise name it after the asset, and
        the name is what `get_gripper_actor_contact_position` and the segmentation columns
        key on.
        """
        if self.peg_modelname is None:
            return create_box(
                scene=self,
                pose=pose,
                half_size=PEG_HALF_SIZE,
                color=(0.85, 0.35, 0.1),
                name="peg",
                boxtype="long",
            )
        peg = create_actor(
            scene=self,
            pose=pose,
            modelname=self.peg_modelname,
            convex=True,  # one submesh -> one hull, and a cylinder IS convex
            is_static=False,
            model_id=0,
        )
        peg.set_name("peg")
        return peg

    # -- geometry -------------------------------------------------------------------------

    def _insertion_state(self):
        """(lateral offset from the bore axis, depth below the mouth) of the peg tip, metres.

        Depth is positive once the tip is inside; the seat is at BORE_DEPTH.
        """
        tip = self.peg.get_functional_point(0, "pose").p  # fp0 = the box's bottom face
        socket = np.array(self.socket.get_functional_point(0, "matrix"))
        mouth, into = socket[:3, 3], socket[:3, 2]  # +Z points down the bore
        offset = np.array(tip) - mouth
        depth = float(offset @ into)
        lateral = float(np.linalg.norm(offset - depth * into))
        return lateral, depth

    def _bore_depth(self, lateral, depth):
        """Depth into the bore, or 0 when the peg is not in it.

        The alignment gate is the point of this helper. `depth` on its own is the signed
        distance below the MOUTH PLANE, which extends across the whole table -- a peg simply
        standing on the table next to the socket is 50 mm below that plane and reads a full
        bore's worth of depth. Only a tip within ALIGN_RADIUS of the bore axis is actually in
        the hole.
        """
        if lateral >= ALIGN_RADIUS:
            return 0.0
        return float(np.clip(depth, 0.0, BORE_DEPTH))

    def _peg_grasped(self):
        """True while the arm holds the peg correctly: closed on its upper half, still upright.

        Evidence is the gripper<->peg contact set rather than the TCP pose, since a closed
        gripper *near* the peg is not a closed gripper *on* it. The gripper test is the
        commanded width (`robot.set_gripper` is what `take_action` drives, so this is the
        policy's own gripper dim) and is deliberately the loose `not open_half` bound: the
        expert commands 0.0, but a policy holding the peg need only be closed enough to hold.
        """
        open_half = (self.is_left_gripper_open_half()
                     if self.arm_tag == "left" else self.is_right_gripper_open_half())
        if open_half:
            return False
        contacts = self.get_gripper_actor_contact_position(self.peg.get_name())
        if not contacts:
            return False
        if get_face_prod(self.peg.get_pose().q, [0, 0, 1], [0, 0, 1]) < GRASP_UPRIGHT:
            return False
        # Contact height measured along the peg's OWN long axis, so a tilted peg is judged by
        # where on it the fingers are rather than by world height.
        tip = np.array(self.peg.get_functional_point(0, "pose").p)
        axis = self.peg.get_pose().to_transformation_matrix()[:3, 2]
        axial = (np.array(contacts) - tip) @ axis
        return bool(float(np.mean(axial)) > GRASP_MIN_AXIAL)

    def _bore_align_axis(self):
        """The square bore's four symmetry axes, for `constrain="align"`.

        Leaving `align_axis=None` would use the single vector `R(target) @ [1,0,0]`
        (transforms.py:499-500), forcing the peg's +X onto the socket's +X. For a peg with a
        4-fold-symmetric cross-section that is up to a 90 deg wrist swing for a geometric
        no-op -- and near 180 deg `get_align_matrix` hits its `||v1 x v2|| < 1e-6` branch and
        silently returns identity (transforms.py:397-399), i.e. no yaw correction at all.
        Offering all four caps the correction at 45 deg and removes that discontinuity.

        Must be a list of lists: transforms.py:501-504 reshapes `(-1,3).T` on the list branch
        but `(3,-1)` on the ndarray branch, which would silently transpose the meaning.
        """
        socket = np.array(self.socket.get_functional_point(0, "matrix"))
        ax, ay = socket[:3, 0], socket[:3, 1]
        return [ax.tolist(), ay.tolist(), (-ax).tolist(), (-ay).tolist()]

    def _place_kwargs(self):
        """How `get_place_pose` is asked to orient the peg over the mouth.

        `constrain="align"` (square) snaps the peg's +X onto the nearest of the axes
        `_bore_align_axis` offers. `constrain="free"` imposes NO yaw at all: it only rotates
        the peg's functional +Z onto the bore axis, by the minimal rotation, leaving the yaw
        the grasp happened to produce. That is the right answer for a solid of revolution --
        every yaw seats identically, so any imposed one is a wrist swing for a geometric
        no-op. It is the same argument that made the square peg offer four axes instead of
        one, taken to its limit, and it also removes the `get_align_matrix` near-antiparallel
        discontinuity entirely rather than merely keeping the correction away from it.
        """
        kwargs = dict(functional_point_id=0, pre_dis_axis="fp", constrain=self.place_constrain)
        if self.place_constrain == "align":
            kwargs["align_axis"] = self._bore_align_axis()
        return kwargs

    # -- expert ---------------------------------------------------------------------------

    def insert_peg(self, arm_tag: ArmTag):
        """The descent, built by hand because `place_actor` cannot constrain it.

        `place_actor` emits two bare `Action(arm, "move", ...)` with no `constraint_pose`
        (_base_task.py:1432-1435), and the planner is a trajectory optimizer whose collision
        world knows nothing about the peg or the socket -- so over a 100 mm descent the path
        can bow by more than the clearance and the orientation is free to drift. `move()`
        does forward `constraint_pose` (_base_task.py:1035-1052), and `grasp_actor` already
        uses [1,1,1,0,0,0] for its own final approach, so use the same here: orientation
        held, position free.
        """
        if not self.plan_success:
            return False

        if self.need_plan:
            target = self.socket.get_functional_point(0, "pose")
            kwargs = self._place_kwargs()
            pre_pose = self.get_place_pose(self.peg, arm_tag, target, pre_dis=PRE_INSERT_DIS, **kwargs)
            near_pose = self.get_place_pose(self.peg, arm_tag, target, pre_dis=NEAR_INSERT_DIS, **kwargs)
            insert_pose = self.get_place_pose(self.peg, arm_tag, target, pre_dis=INSERT_DIS, **kwargs)
        else:
            # Replay consumes the cached joint path by index and ignores the pose, but it
            # must not be None. Mirrors place_actor's own need_plan branch.
            pre_pose = near_pose = insert_pose = [0, 0, 0, 0, 0, 0, 0]

        return self.move((
            arm_tag,
            [
                Action(arm_tag, "move", target_pose=pre_pose),
                Action(arm_tag, "move", target_pose=near_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
                Action(arm_tag, "move", target_pose=insert_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
                Action(arm_tag, "open", target_gripper_pos=1.0),
            ],
        ))

    def play_once(self):
        arm_tag = self.arm_tag

        # Grasp the upper band of the standing peg, so the fingers stay ~0.07 m above the
        # socket rim once it is fully inserted.
        self.move(
            self.grasp_actor(
                self.peg,
                arm_tag=arm_tag,
                pre_grasp_dis=PRE_GRASP_DIS,
                grasp_dis=0.0,
                contact_point_id=GRASP_BAND,
            ))
        self.move(self.move_by_displacement(arm_tag, z=LIFT_Z))

        self.insert_peg(arm_tag)

        # "arm" retracts along the end-effector's own approach direction, i.e. straight back
        # out of the (side) grasp rather than up through the peg.
        self.move(self.move_by_displacement(arm_tag, z=RETRACT_DIS, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))
        self.delay(4)  # let the released peg settle before check_success

        self.info["info"] = {
            "{A}": f"{self.socket_modelname}/base{self.socket_model_id}",
            "{B}": self.peg_description,
            "{a}": str(arm_tag),
        }
        return self.info

    # -- outcome --------------------------------------------------------------------------

    def check_success(self):
        lateral, depth = self._insertion_state()
        upright = get_face_prod(self.peg.get_pose().q, [0, 0, 1], [0, 0, 1]) > SUCCESS_UPRIGHT
        gripper_open = (self.is_left_gripper_open()
                        if self.arm_tag == "left" else self.is_right_gripper_open())
        return bool(depth > SUCCESS_DEPTH and lateral < SUCCESS_LATERAL and upright and gripper_open)

    def step_reward(self):
        """Shaped progress, as a DELTA since the last call (script/eval_policy.py:152-175).

        Four terms, summed rather than switched on a phase, so there is no sticky state to
        keep consistent beyond the one-shot grasp flag:

        1. **grasp** -- a one-time GRASP_BONUS the first time the peg is held correctly
           (`_peg_grasped`). Not a delta: it is an event, and paying it once means picking the
           peg up and putting it down cannot be farmed.
        2. **approach** -- closing on the bore AXIS: the lateral offset only, with height
           deliberately left out. So lifting the peg is worth exactly 0 rather than reading as
           moving away from the mouth, and carrying it over the socket is the whole of this
           term. The cost is a dead zone: the descent from the pre-insert waypoint down to the
           mouth plane earns nothing, since term 3 does not switch on until the tip is inside.
        3. **depth** -- descending into the bore, and ONLY while the tip is inside it
           (`_bore_depth`). Without that gate `depth` is measured against the mouth *plane*,
           which spans the table, so the peg standing anywhere on the table scores a full
           bore of depth -- which made the shaping actively penalise lifting the peg and pay
           for setting it back down again.
        4. **uprightness** -- only once the peg is in the socket, where straightening it is
           what turns a jammed peg into a seated one. Outside the bore `last_upright` is
           reset to None, so re-entering measures its delta from the value on entry instead
           of paying the whole cosine as a one-step jump.

        Every delta is symmetric (undoing progress refunds it), so nothing ratchets.
        """
        lateral, depth = self._insertion_state()
        reward = 0.0

        # if not self.grasp_rewarded and self._peg_grasped():
        #     self.grasp_rewarded = True
        #     reward += GRASP_BONUS

        reward += float(np.clip(self.last_lateral - lateral, -DELTA_CLIP, DELTA_CLIP))
        self.last_lateral = lateral

        bore_depth = self._bore_depth(lateral, depth)
        reward += float(np.clip(bore_depth - self.last_depth, -DELTA_CLIP, DELTA_CLIP))
        self.last_depth = bore_depth

        if bore_depth > 0.0:
            upright = get_face_prod(self.peg.get_pose().q, [0, 0, 1], [0, 0, 1])
            if self.last_upright is not None:
                reward += UPRIGHT_WEIGHT * float(
                    np.clip(upright - self.last_upright, -DELTA_CLIP, DELTA_CLIP))
            self.last_upright = upright
        else:
            self.last_upright = None

        return float(reward)
