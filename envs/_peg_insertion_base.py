"""Shared implementation for the `insert_peg_socket_{loose,med,tight}` difficulty ladder.

Not a task itself (leading underscore, like `_base_task.py`) -- the three registered tasks
are thin subclasses that differ only in which socket variant they load. Task discovery is
`importlib.import_module(f"envs.{task_name}")`, so only the concrete files are reachable.

The peg is a `create_box` primitive: a box Actor comes with working contact and functional
points for free (create_actor.py:159-196), and `boxtype="long"` gives 8 side contact points
in two bands along the long axis -- ids [0,1,2,3] are the upper band, which is the one to
grasp so the fingers stay clear of the socket rim at full insertion depth.

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

# Peg: 40 x 40 x 120 mm, standing. envs/handover_block.py stands a larger 60 x 60 x 200 mm
# box successfully, so this is the conservative end of a shipped precedent.
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

_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "objects" / SOCKET_MODELNAME
if not _ASSET_DIR.exists():
    # create_actor only prints "is not exist model file!" and returns None, and the seed
    # search in collect_data.py is uncapped -- so a missing asset would spin forever instead
    # of failing. Raise at import, which is outside class_decorator's try/except.
    raise FileNotFoundError(
        f"{SOCKET_MODELNAME} is missing from assets/objects. `assets/` is gitignored, so the "
        f"socket mesh is generated rather than downloaded:\n"
        f"    python script/gen_peg_socket_asset.py --verify")


class _PegInsertionBase(Base_Task):
    """Grasp a standing peg and insert it into a socket's blind bore, single-arm."""

    socket_model_id: int = None  # set by each concrete task

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

        self.peg = create_box(
            scene=self,
            pose=peg_pose,
            half_size=PEG_HALF_SIZE,
            color=(0.85, 0.35, 0.1),
            name="peg",
            boxtype="long",
        )
        self.socket = create_actor(
            scene=self,
            pose=socket_pose,
            modelname=SOCKET_MODELNAME,
            convex=True,  # -> one convex hull per submesh, which is what keeps the bore open
            is_static=True,
            model_id=self.socket_model_id,
        )

        self.add_prohibit_area(self.peg, padding=0.1)
        self.add_prohibit_area(self.socket, padding=0.1)

        # Fixed in load_actors, not in play_once, so check_success and step_reward do not
        # depend on the expert having run -- during a policy rollout it has not.
        self.arm_tag = ArmTag("left" if peg_pose.p[0] < 0 else "right")

        lateral, depth = self._insertion_state()
        self.last_lateral = lateral
        self.last_depth = depth

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
            kwargs = dict(
                functional_point_id=0,
                constrain="align",
                align_axis=self._bore_align_axis(),
                pre_dis_axis="fp",
            )
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
            "{A}": f"{SOCKET_MODELNAME}/base{self.socket_model_id}",
            "{B}": "peg",
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

        Two terms summed rather than a phase switch, so there is no extra sticky state to
        keep consistent: closing on the bore axis, then descending into it. Depth is clipped
        at 0 below the mouth so hovering above the socket contributes nothing.
        """
        lateral, depth = self._insertion_state()
        depth = float(np.clip(depth, 0.0, BORE_DEPTH))
        reward = (np.clip(self.last_lateral - lateral, -0.1, 0.1)
                  + np.clip(depth - self.last_depth, -0.1, 0.1))
        self.last_lateral, self.last_depth = lateral, depth
        return float(reward)
