"""Insert a knife into its cover.

STATUS: DOES NOT CURRENTLY YIELD DEMOS -- 0/6 seeds. Its instruction file is parked as
`description/task_instruction/insert_knife_cover.json.disabled` **on purpose**:
`collect_all_data.sh` globs that directory and `collect_data.py`'s seed loop is UNCAPPED, so
leaving it in place would make a bulk collection run spin forever rather than fail. Rename it
back once the task yields.

The assets are verified and the spawn is now sane (the knife used to be ejected 42 m -- see
KNIFE_QPOS below), but the expert still fails: most seeds fail to plan, and two of six still
throw on a None grasp pose, which means the knife is still being ejected on those. The most
likely cause is the last one left unfixed: the knife is 192 mm long and the separation test
below only compares actor ORIGINS at 0.15 m, so a knife whose blade points at the cover can
still spawn intersecting it. A length-aware separation test, and a wider gap, are the next
things to try. This is the hardest of the four geometrically -- a long thin object going into
a receptacle that has to lie on its side -- and it is the one that needs more work.

A clearance fit: a 6.3 x 43.5 mm blade into a 10.3 x 47.5 mm slot, 2.0 mm per side on both
cross-section axes. Same family as `envs/_peg_insertion_base.py` and `envs/put_battery_slot.py`,
and it uses the same hand-built two-stage constrained descent for the same reason.

Two assets, both produced by `script/gen_task_assets.py`, which must be run once per clone
(`assets/*` is gitignored, so nothing under it can be committed):

* `034_knife/base0` ships annotated, but at `scale` 0.31 -- a 458 mm knife, longer than the
  reachable workspace is wide. It is rescaled to 0.13 (a 192 mm knife with a 96 mm blade) and
  given a **second** functional point at the blade tip, with its +Z running out of the tip
  along the blade. The shipped point 0 sits on the spine pointing up, which is the right frame
  for cutting and the wrong one for sheathing; it is left untouched.
* `124_knife-cover` is **generated**, and sized from the knife's own measured blade rather
  than from a hardcoded number, so changing the knife's scale cannot silently leave the two
  disagreeing. Nothing in the asset library is a sheath -- a survey of every shipped asset's
  interior voids found only hollow containers.

**The cover lies on its side, and that is the whole design.** A knife rests flat on a table.
An upright cover with its mouth up would need the wrist to carry the knife through a ~90 deg
reorientation to point the blade down, which is a large swing to plan and the thing the peg
task deliberately avoided by standing its peg up in the first place. Laying the cover down
instead -- slot axis horizontal, mouth facing the knife -- keeps the knife flat from grasp to
release and makes the insertion a horizontal slide, which is also how a knife is actually
sheathed on a worktop. Concretely the cover is spawned with a -90 deg rotation about Y, which
maps its local +X (the slot's narrow 15.7 mm direction) to world +Z, its local +Y (the wide
72.9 mm direction) to world +Y, and the slot axis to world +X.

The knife is spawned with `qpos=[0.5, -0.5, -0.5, -0.5]`, the 120 deg rotation about (1,1,1)
that cycles its local x -> world Z (blade thickness vertical, so it lies flat), y -> world X
(its length points at the cover) and z -> world Y (the blade's height horizontal, matching the
slot's wide direction). Note the sign: the *conjugate* `[0.5, 0.5, 0.5, 0.5]` cycles the other
way, x -> Y and y -> Z, which stands the 192 mm knife on end. Spawned at a height meant for a
6 mm half-thickness it then intersects the table by ~90 mm and PhysX ejects it -- measured, it
ended up 42 m away, and `check_stable` does not catch this because it tests only quaternion
drift, not position.

Treat the spawn ranges and the insertion distances as a first estimate; they have not been
tuned against a working expert. Re-check with
`python script/collect_data.py insert_knife_cover demo_smoke` once the spawn overlap above is
fixed.
"""

import numpy as np
import sapien

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import *
from .utils import *
from ._task_assets import require_assets

KNIFE_MODEL = "034_knife"
KNIFE_ID = 0
COVER_MODEL = "124_knife-cover"

TABLE_Z = 0.741
KNIFE_HALF_THICK = 0.0064  # the knife's local x half-extent at scale 0.13, vertical when flat
COVER_HALF_NARROW = 0.020  # the cover's local x half-extent, vertical when laid on its side

# Rotations, as quaternions. See the docstring for what each mapping is for.
KNIFE_QPOS = [0.5, -0.5, -0.5, -0.5]
COVER_QPOS = [0.70710678, 0.0, -0.70710678, 0.0]

BLADE_TIP_FP = 1  # the added functional point; 0 is the shipped spine point
PRE_GRASP_DIS = 0.08
LIFT_Z = 0.06

PRE_INSERT_DIS = 0.08  # tip 80 mm out from the mouth, clear of the cover
NEAR_INSERT_DIS = 0.010
INSERT_DIS = -0.045  # tip 45 mm into a 65 mm slot; released short so it is not driven
#                      against the blind end by a stiff position-controlled arm
RETRACT_DIS = 0.06

SUCCESS_DEPTH = 0.030  # of a 0.065 m slot. Unachievable outside it.
SUCCESS_LATERAL = 0.010
DELTA_CLIP = 0.1


require_assets((KNIFE_MODEL, KNIFE_ID), (COVER_MODEL, 0))


class insert_knife_cover(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # The knife lies flat with its length along world X, pointing at the cover; the cover
        # lies on its side with its mouth facing back along -X. Both on the same side of the
        # table so one arm reaches both.
        for _ in range(100):
            knife_pose = rand_pose(
                xlim=[-0.20, 0.20],
                ylim=[-0.04, 0.08],
                zlim=[TABLE_Z + KNIFE_HALF_THICK],
                qpos=KNIFE_QPOS,
                rotate_rand=True,
                rotate_lim=[0.35, 0, 0],  # yaw: local x is the axis that maps to world up
            )
            if abs(knife_pose.p[0]) > 0.08:
                break

        side = 1.0 if knife_pose.p[0] > 0 else -1.0
        for _ in range(100):
            cover_pose = rand_pose(
                xlim=[0.10, 0.22] if side > 0 else [-0.22, -0.10],
                ylim=[-0.16, -0.09],
                zlim=[TABLE_Z + COVER_HALF_NARROW],
                qpos=COVER_QPOS,
                rotate_rand=False,
            )
            if np.linalg.norm(cover_pose.p[:2] - knife_pose.p[:2]) > 0.15:
                break

        self.knife = create_actor(
            scene=self,
            pose=knife_pose,
            modelname=KNIFE_MODEL,
            convex=True,
            model_id=KNIFE_ID,
        )
        self.knife.set_name("knife")
        # Static: a 0.01 kg cover would be pushed across the table by the insertion itself.
        self.cover = create_actor(
            scene=self,
            pose=cover_pose,
            modelname=COVER_MODEL,
            convex=True,  # -> one convex hull per submesh, which is what keeps the slot open
            is_static=True,
            model_id=0,
        )

        self.add_prohibit_area(self.knife, padding=0.10)
        self.add_prohibit_area(self.cover, padding=0.10)

        self.arm_tag = ArmTag("left" if knife_pose.p[0] < 0 else "right")

        lateral, depth = self._insertion_state()
        self.last_lateral = lateral
        self.last_depth = max(0.0, min(depth, SUCCESS_DEPTH))

    # -- geometry --------------------------------------------------------------------------

    def _insertion_state(self):
        """(lateral offset from the slot axis, depth past the mouth) of the blade tip."""
        tip = self.knife.get_functional_point(BLADE_TIP_FP, "pose").p
        cover = np.array(self.cover.get_functional_point(0, "matrix"))
        mouth, into = cover[:3, 3], cover[:3, 2]  # +Z points into the slot
        offset = np.array(tip) - mouth
        depth = float(offset @ into)
        lateral = float(np.linalg.norm(offset - depth * into))
        return lateral, depth

    def _slot_align_axis(self):
        """The slot's NARROW direction and its opposite, for `constrain="align"`.

        The blade is 9.7 mm thick and 66.9 mm tall, so unlike the round battery the roll about
        the insertion axis matters: the blade's flat faces have to line up with the slot's
        15.7 mm dimension. The knife's functional frame 1 has its +X along the blade's
        thickness and the cover's frame has its +X across the slot's narrow direction, so
        aligning those is exactly the constraint. Both signs are offered because a blade is
        2-fold symmetric about its own axis for this purpose, which caps the correction at
        90 deg and keeps `get_align_matrix` away from its `||v1 x v2|| < 1e-6` branch, where
        it silently returns identity (transforms.py:397-399).

        Must be a list of lists: transforms.py:501-504 reshapes `(-1,3).T` on the list branch
        but `(3,-1)` on the ndarray branch, which would silently transpose the meaning.
        """
        cover = np.array(self.cover.get_functional_point(0, "matrix"))
        narrow = cover[:3, 0]
        return [narrow.tolist(), (-narrow).tolist()]

    # -- expert ----------------------------------------------------------------------------

    def insert_knife(self, arm_tag: ArmTag):
        """The slide, built by hand because `place_actor` cannot constrain it.

        `place_actor` emits bare `Action(arm, "move", ...)` with no `constraint_pose`
        (_base_task.py:1432-1435), and the planner is a trajectory optimizer whose collision
        world knows nothing about the knife or the cover -- so the path can bow by more than
        the 3 mm clearance and the orientation is free to drift. `move()` does forward
        `constraint_pose` (_base_task.py:1035-1052), so use [1,1,1,0,0,0] here: orientation
        held, position free.
        """
        if not self.plan_success:
            return False

        if self.need_plan:
            target = self.cover.get_functional_point(0, "pose")
            kwargs = dict(functional_point_id=BLADE_TIP_FP, pre_dis_axis="fp",
                          constrain="align", align_axis=self._slot_align_axis())
            pre_pose = self.get_place_pose(self.knife, arm_tag, target, pre_dis=PRE_INSERT_DIS, **kwargs)
            near_pose = self.get_place_pose(self.knife, arm_tag, target, pre_dis=NEAR_INSERT_DIS, **kwargs)
            insert_pose = self.get_place_pose(self.knife, arm_tag, target, pre_dis=INSERT_DIS, **kwargs)
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
                self.knife,
                arm_tag=arm_tag,
                pre_grasp_dis=PRE_GRASP_DIS,
                grasp_dis=0.0,
            ))
        self.move(self.move_by_displacement(arm_tag, z=LIFT_Z))

        self.insert_knife(arm_tag)

        self.move(self.move_by_displacement(arm_tag, z=RETRACT_DIS, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))
        self.delay(4)

        self.info["info"] = {
            "{A}": f"{COVER_MODEL}/base0",
            "{B}": f"{KNIFE_MODEL}/base{KNIFE_ID}",
            "{a}": str(arm_tag),
        }
        return self.info

    # -- outcome ---------------------------------------------------------------------------

    def check_success(self):
        lateral, depth = self._insertion_state()
        gripper_open = (self.is_left_gripper_open()
                        if self.arm_tag == "left" else self.is_right_gripper_open())
        return bool(depth > SUCCESS_DEPTH and lateral < SUCCESS_LATERAL and gripper_open)

    def step_reward(self):
        """Shaped progress, as a DELTA since the last call (script/eval_policy.py:152-175).

        Two clipped deltas summed: closing on the slot axis, and advancing into it. Depth only
        counts while the tip is roughly on the axis, for the same reason the peg task gates
        its own depth term -- the mouth PLANE extends across the table, so without the gate a
        knife lying anywhere past it already reads as fully inserted.
        """
        lateral, depth = self._insertion_state()
        gated = 0.0 if lateral >= SUCCESS_LATERAL else float(np.clip(depth, 0.0, SUCCESS_DEPTH))
        reward = float(np.clip(self.last_lateral - lateral, -DELTA_CLIP, DELTA_CLIP))
        reward += float(np.clip(gated - self.last_depth, -DELTA_CLIP, DELTA_CLIP))
        self.last_lateral, self.last_depth = lateral, gated
        return reward
