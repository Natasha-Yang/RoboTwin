"""Put a battery into a slot.

A clearance fit: a 22.7 mm cell into a 28.7 mm blind bore, 3.0 mm per side. Structurally the
same task as `envs/insert_peg_socket_round_med.py` -- a solid of revolution into a round bore
-- and it uses the same idioms for the same reasons: a hand-built two-stage constrained
descent (see `insert_battery`) and `constrain="free"` yaw (see `_place_kwargs`).

Two assets, both produced by `script/gen_task_assets.py`, which must be run once per clone
(`assets/*` is gitignored, so nothing under it can be committed):

* `061_battery/base3` ships **unannotated** -- only `{stable, center, extents}`, no `scale`
  and no points, which `create_actor` swallows in a bare `except` (create_actor.py:531),
  leaving the mesh at its raw 1.9 m size and every `get_contact_point` returning None. It is
  a single convex cylinder whose long axis is the local +Y that the standard spawn turns into
  world up, so it is annotated with four contact points around its body and a functional
  point at its base.
* `125_battery-slot` is **generated**. Nothing in the asset library is a slot: a survey of
  every shipped asset's interior voids found ~50 with a 20-90 mm hole and every one is a
  hollow container -- mugs, cups, bottles, shoes -- opening at a face and flaring, never a
  parallel-walled blind bore. It is built as 33 convex pieces (a floor plus 32 angular wall
  wedges) because an annulus is not convex; see `gen_task_assets.build_slot_pieces`.

The contact points sit at 75% of the cell's length rather than at its waist, and that is
load-bearing: the bore is 40 mm deep, so a cell grasped at its middle (48 mm up) would put
the fingertips 8 mm *below* the slot mouth at full depth. At 73 mm up they stay 33 mm clear.
"""

import numpy as np
import sapien

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import *
from .utils import *
from ._task_assets import require_assets

BATTERY_MODEL = "061_battery"
BATTERY_ID = 1
SLOT_MODEL = "125_battery-slot"

TABLE_Z = 0.741
BATTERY_BOTTOM = 0.00085  # the mesh dips this far below its own origin, at scale 0.05

# Slot geometry, mirrored from script/gen_task_assets.py. Kept as constants rather than read
# back from model_data0.json so the thresholds below read as numbers.
BORE_DEPTH = 0.040
SLOT_HEIGHT = 0.050

PRE_GRASP_DIS = 0.07
LIFT_Z = 0.10
PRE_INSERT_DIS = 0.07  # base 70 mm above the mouth, clear of the slot and its chamfer
NEAR_INSERT_DIS = 0.015  # a second waypoint just above the mouth, so the final constrained
#                          descent is short -- the planner does not track a constrained
#                          straight line exactly, and the drift it accumulates is what a
#                          3 mm clearance has no room for
INSERT_DIS = -0.028  # base 28 mm below the mouth; the floor is at 40 mm, so it is released
#                      12 mm short and free-falls the rest, self-centring on the chamfer.
RETRACT_DIS = 0.06

SUCCESS_DEPTH = 0.030  # of a 0.040 m bore. Unachievable outside it -- a cell standing on the
#                        table beside the slot reads 0 -- so this threshold carries the check.
SUCCESS_LATERAL = 0.010
SUCCESS_UPRIGHT = 0.95

ALIGN_RADIUS = SUCCESS_LATERAL
DELTA_CLIP = 0.1


require_assets((BATTERY_MODEL, BATTERY_ID), (SLOT_MODEL, 0))


class put_battery_slot(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # Battery and slot on the same side, so one arm reaches both. |x| > 0.08 keeps the
        # cell off the midline, where neither arm has a clean approach.
        for _ in range(100):
            battery_pose = rand_pose(
                # Narrower than the peg family's +-0.26: the cell is grasped from the side,
                # and `get_grasp_pose` stands the end-effector 0.12 m back from the contact
                # plus `pre_grasp_dis` (_base_task.py:1200). Past about +-0.22 that pre-pose
                # can land outside the arm's reach for the azimuth the chooser picks.
                xlim=[-0.22, 0.22],
                ylim=[-0.05, 0.05],
                zlim=[TABLE_Z + BATTERY_BOTTOM],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, 0.785, 0],  # yaw only; roll/pitch is what would topple it
            )
            if abs(battery_pose.p[0]) > 0.08:
                break

        side = 1.0 if battery_pose.p[0] > 0 else -1.0
        for _ in range(100):
            slot_pose = rand_pose(
                xlim=[0.10, 0.26] if side > 0 else [-0.26, -0.10],
                # POSITIVE y, matching envs/_peg_insertion_base.py's socket. The insertion is
                # the same constrained descent the peg family runs, and it plans there; with
                # the slot near the robot instead the arm is folded and that final leg failed
                # on seed after seed with the cell already aligned to 0.2 mm above the mouth.
                # (Note this is the opposite of what the *place-on* tasks want -- see the
                # reach trap in CLAUDE.md 3.6.)
                ylim=[0.12, 0.20],
                zlim=[TABLE_Z],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.785],
            )
            # The gripper sits 0.12 m from the grasp point (_base_task.py:1145); closer than
            # this and the wrist is inside the slot footprint while grasping the cell.
            if np.linalg.norm(slot_pose.p[:2] - battery_pose.p[:2]) > 0.14:
                break

        self.battery = create_actor(
            scene=self,
            pose=battery_pose,
            modelname=BATTERY_MODEL,
            convex=True,
            model_id=BATTERY_ID,
        )
        self.battery.set_name("battery")
        # Static: a 0.01 kg slot would be shoved across the table by the insertion itself.
        self.slot = create_actor(
            scene=self,
            pose=slot_pose,
            modelname=SLOT_MODEL,
            convex=True,  # -> one convex hull per submesh, which is what keeps the bore open
            is_static=True,
            model_id=0,
        )

        self.add_prohibit_area(self.battery, padding=0.10)
        self.add_prohibit_area(self.slot, padding=0.10)

        # Fixed here rather than in play_once, so check_success and step_reward do not depend
        # on the expert having run -- during a policy rollout it has not.
        self.arm_tag = ArmTag("left" if battery_pose.p[0] < 0 else "right")

        lateral, depth = self._insertion_state()
        self.last_lateral = lateral
        self.last_depth = self._bore_depth(lateral, depth)

    # -- geometry --------------------------------------------------------------------------

    def _insertion_state(self):
        """(lateral offset from the bore axis, depth below the mouth) of the cell's base."""
        base = self.battery.get_functional_point(0, "pose").p
        slot = np.array(self.slot.get_functional_point(0, "matrix"))
        mouth, into = slot[:3, 3], slot[:3, 2]  # +Z points down the bore
        offset = np.array(base) - mouth
        depth = float(offset @ into)
        lateral = float(np.linalg.norm(offset - depth * into))
        return lateral, depth

    def _bore_depth(self, lateral, depth):
        """Depth into the bore, or 0 when the cell is not in it.

        The alignment gate is the point of this helper. `depth` on its own is the signed
        distance below the MOUTH PLANE, which extends across the whole table -- a cell simply
        standing on the table next to the slot is 50 mm below that plane and reads a full
        bore's worth of depth.
        """
        if lateral >= ALIGN_RADIUS:
            return 0.0
        return float(np.clip(depth, 0.0, BORE_DEPTH))

    # -- expert ----------------------------------------------------------------------------

    def insert_battery(self, arm_tag: ArmTag):
        """The descent, built by hand because `place_actor` cannot constrain it.

        `place_actor` emits bare `Action(arm, "move", ...)` with no `constraint_pose`
        (_base_task.py:1432-1435), and the planner is a trajectory optimizer whose collision
        world knows nothing about the cell or the slot -- so the path can bow by more than
        the 3 mm clearance and the orientation is free to drift. `move()` does forward
        `constraint_pose` (_base_task.py:1035-1052), and `grasp_actor` already uses
        [1,1,1,0,0,0] for its own final approach, so use the same: orientation held, position
        free.

        `constrain="free"` imposes NO yaw: it only rotates the cell's functional +Z onto the
        bore axis, by the minimal rotation. That is the right answer for a solid of
        revolution -- every yaw seats identically, so any imposed one is a wrist swing for a
        geometric no-op -- and it avoids `get_align_matrix`'s near-antiparallel branch, where
        it silently returns identity (transforms.py:397-399).
        """
        if not self.plan_success:
            return False

        if self.need_plan:
            target = self.slot.get_functional_point(0, "pose")
            kwargs = dict(functional_point_id=0, pre_dis_axis="fp", constrain="free")
            pre_pose = self.get_place_pose(self.battery, arm_tag, target, pre_dis=PRE_INSERT_DIS, **kwargs)
            near_pose = self.get_place_pose(self.battery, arm_tag, target, pre_dis=NEAR_INSERT_DIS, **kwargs)
            insert_pose = self.get_place_pose(self.battery, arm_tag, target, pre_dis=INSERT_DIS, **kwargs)
        else:
            # Replay consumes the cached joint path by index and ignores the pose, but it
            # must not be None. Mirrors place_actor's own need_plan branch.
            pre_pose = near_pose = insert_pose = [0, 0, 0, 0, 0, 0, 0]


        if any(p is None for p in (pre_pose, near_pose, insert_pose)):
            # get_place_pose returns None when a required point is unavailable. Passing that
            # into an Action trips a bare assert deep in `move`; failing the plan here instead
            # lets the seed be skipped like any other planning failure.
            self.plan_success = False
            return False

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

        self.move(
            self.grasp_actor(
                self.battery,
                arm_tag=arm_tag,
                pre_grasp_dis=PRE_GRASP_DIS,
                grasp_dis=0.0,
            ))
        self.move(self.move_by_displacement(arm_tag, z=LIFT_Z))

        self.insert_battery(arm_tag)

        # "arm" retracts along the end-effector's own approach direction, i.e. straight back
        # out of the (side) grasp rather than up through the cell.
        self.move(self.move_by_displacement(arm_tag, z=RETRACT_DIS, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))
        self.delay(4)  # let the released cell settle before check_success

        self.info["info"] = {
            "{A}": f"{SLOT_MODEL}/base0",
            "{B}": f"{BATTERY_MODEL}/base{BATTERY_ID}",
            "{a}": str(arm_tag),
        }
        return self.info

    # -- outcome ---------------------------------------------------------------------------

    def check_success(self):
        lateral, depth = self._insertion_state()
        # The cell's long axis is its local +Y, not +Z: `061_battery` is one of the
        # objaverse-derived assets modelled +Y up, which is why every task spawns them with
        # `qpos=[0.707, 0.707, 0, 0]`. Testing [0,0,1] here would read ~0 for a perfectly
        # upright cell and reject every successful insertion.
        upright = get_face_prod(self.battery.get_pose().q, [0, 1, 0], [0, 0, 1]) > SUCCESS_UPRIGHT
        gripper_open = (self.is_left_gripper_open()
                        if self.arm_tag == "left" else self.is_right_gripper_open())
        return bool(depth > SUCCESS_DEPTH and lateral < SUCCESS_LATERAL
                    and upright and gripper_open)

    def step_reward(self):
        """Shaped progress, as a DELTA since the last call (script/eval_policy.py:152-175).

        Two clipped deltas summed: closing on the bore axis, and descending into it. Both are
        symmetric, so undoing progress refunds it and nothing ratchets. Approach is the
        lateral offset only -- height is deliberately left out, so lifting the cell is worth
        exactly 0 rather than reading as moving away from the mouth.
        """
        lateral, depth = self._insertion_state()
        bore = self._bore_depth(lateral, depth)
        reward = float(np.clip(self.last_lateral - lateral, -DELTA_CLIP, DELTA_CLIP))
        reward += float(np.clip(bore - self.last_depth, -DELTA_CLIP, DELTA_CLIP))
        self.last_lateral, self.last_depth = lateral, bore
        return reward
