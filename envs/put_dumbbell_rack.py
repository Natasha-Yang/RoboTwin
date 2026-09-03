"""Put a dumbbell on a dumbbell rack.

STATUS: DOES NOT CURRENTLY YIELD DEMOS -- 0/4 seeds. Its instruction file is parked as
`description/task_instruction/put_dumbbell_rack.json.disabled` **on purpose**:
`collect_all_data.sh` globs that directory and `collect_data.py`'s seed loop is UNCAPPED, so
leaving it in place would make a bulk collection run spin forever rather than fail. Rename it
back once the task yields. The blocker is the asset, not the plumbing -- the measurement is in
`_cradle_align_axis` and below.

Why it is hard, measured rather than guessed. A dumbbell was dropped onto the rack over a
132-pose grid -- both orientations, the full width and depth, two drop heights -- and exactly
**6** poses came to rest on it, all six the same one: bar perpendicular to the rack, straddling
its two rails, at a single depth offset. Every pose with the bar along the rack, and every
other depth, ended with the dumbbell on the floor. `013_dumbbell-rack`'s tiers are narrow
inclined ridges rather than shelves (`base1`'s top rail is a 9 mm knife edge at this scale), so
the target is a line, not a surface. The expert reaches the cradle -- the dumbbell has been
observed at the cradle height mid-episode -- but does not leave it there: releasing flush
interpenetrates the rack and ejects the dumbbell off the table, and releasing above it lands
off the line. What is left to try is a deeper cradle (a different variant, or scaling the rack
up so the ridge spacing exceeds the weight diameter), or a success criterion that does not
require the dumbbell to stay put.


A place-on task, the same shape as `envs/place_object_stand.py`: grasp the object, lift it,
and set it down on a receptacle's functional point with `constrain="free"`. What is new is
the receptacle. `013_dumbbell-rack` ships in the asset pack but **unannotated** -- its
`model_data*.json` carries only `{stable, center, extents}`, with no `scale` and no points at
all, which `create_actor` swallows in a bare `except` (create_actor.py:531), leaving the mesh
at its raw 1.9 m size and every `get_functional_point` call returning None. Its annotation is
written by `script/gen_task_assets.py`, which must be run once per clone (`assets/*` is
gitignored, so nothing under it can be committed).

`base1` is the two-tier rack; the functional point sits on the centre of its **top** rail with
its axis pointing up, exactly as `074_displaystand`'s does, which is what lets the placement
here be the one `place_object_stand` already proves out.

Only the bar-shaped dumbbell variants are used. `052_dumbbell` has seven, and three of them
(0, 2, 3) stand the dumbbell on end, where the shipped contact point is a top-down grasp of
an end cap rather than of the bar -- a different task. Variants 1/4/5/6 all lie with the bar
along the model's local +X and carry the same waist contact points, so the grasp is identical
across them and the variation is only in the weights' size and shape.
"""

import numpy as np
import sapien

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import *
from .utils import *
from ._task_assets import require_assets

DUMBBELL_MODEL = "052_dumbbell"
RACK_MODEL = "013_dumbbell-rack"
RACK_ID = 0

# The bar-shaped variants: long axis along local +X, waist contact points on the bar.
BAR_VARIANTS = [1, 4, 5, 6]

TABLE_Z = 0.741
LIFT_Z = 0.08
PRE_PLACE_DIS = 0.07

SUCCESS_XY = 0.050  # the cradle, not merely "near the rack"
SUCCESS_ABOVE_RAIL = -0.02  # the dumbbell's centre may sit slightly below the rail top


require_assets((DUMBBELL_MODEL, BAR_VARIANTS[0]), (RACK_MODEL, RACK_ID))


class put_dumbbell_rack(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # Dumbbell and rack on the same side, so one arm reaches both. |x| > 0.08 keeps the
        # dumbbell off the midline, where neither arm has a clean approach.
        for _ in range(100):
            dumbbell_pose = rand_pose(
                xlim=[-0.26, 0.26],
                ylim=[-0.05, 0.05],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 6, 0],  # yaw only; the bar must stay horizontal
            )
            if abs(dumbbell_pose.p[0]) > 0.08:
                break

        side = 1.0 if dumbbell_pose.p[0] > 0 else -1.0
        for _ in range(100):
            rack_pose = rand_pose(
                xlim=[0.10, 0.26] if side > 0 else [-0.26, -0.10],
                ylim=[-0.18, -0.12],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 9, 0],
            )
            # The gripper sits 0.12 m from the grasp point (_base_task.py:1145); closer than
            # this and the wrist is inside the rack footprint while grasping the dumbbell.
            if np.linalg.norm(rack_pose.p[:2] - dumbbell_pose.p[:2]) > 0.14:
                break

        self.dumbbell_id = int(np.random.choice(BAR_VARIANTS))
        self.dumbbell = create_actor(
            scene=self,
            pose=dumbbell_pose,
            modelname=DUMBBELL_MODEL,
            convex=True,
            model_id=self.dumbbell_id,
        )
        self.dumbbell.set_mass(0.02)

        # Static: a 0.01 kg rack would be shoved across the table by the placement itself.
        self.rack = create_actor(
            scene=self,
            pose=rack_pose,
            modelname=RACK_MODEL,
            convex=True,
            is_static=True,
            model_id=RACK_ID,
        )

        self.add_prohibit_area(self.dumbbell, padding=0.10)
        self.add_prohibit_area(self.rack, padding=0.10)

        # Fixed here rather than in play_once, so check_success does not depend on the expert
        # having run -- during a policy rollout it has not.
        self.arm_tag = ArmTag("left" if dumbbell_pose.p[0] < 0 else "right")

    def play_once(self):
        arm_tag = self.arm_tag

        self.move(self.grasp_actor(self.dumbbell, arm_tag=arm_tag, pre_grasp_dis=0.10))
        self.move(self.move_by_displacement(arm_tag, z=LIFT_Z))

        self.move(
            self.place_actor(
                self.dumbbell,
                arm_tag=arm_tag,
                target_pose=self.rack.get_functional_point(0),
                constrain="align",
                align_axis=self._cradle_align_axis(),
                pre_dis=PRE_PLACE_DIS,
                dis=0.015,  # release 15 mm above the cradle and let it drop in: placed
                #             flush it interpenetrates the rack and is ejected off the table.
                #             The drop sweep rested it from both 5 mm and 30 mm up.
            ))

        # Settle BEFORE retracting, not only after. The dumbbell is released a few mm above
        # the cradle and has to drop into it; retracting through that drop is what used to
        # sweep it off the rack, and it lands on the floor rather than merely off-target.
        self.delay(4)
        self.move(self.move_by_displacement(arm_tag, z=0.10))  # straight up, clear of the rack
        self.move(self.back_to_origin(arm_tag))
        self.delay(4)  # let it settle again before check_success

        self.info["info"] = {
            "{A}": f"{DUMBBELL_MODEL}/base{self.dumbbell_id}",
            "{B}": f"{RACK_MODEL}/base{RACK_ID}",
            "{a}": str(arm_tag),
        }
        return self.info

    def _cradle_align_axis(self):
        """The rack's DEPTH direction and its opposite, for `constrain="align"`.

        This is the whole task. A dumbbell only stays on this rack with its bar across the
        two rails, not along them -- dropped over a 132-pose grid, the six poses that came to
        rest were all of that one orientation and every pose with the bar along the rack ended
        on the floor. The dumbbell's own +X is its long axis, so aligning it to the rack's
        depth is what produces the straddle. Both signs are offered because a dumbbell is
        2-fold symmetric end to end, which caps the correction at 90 deg and keeps
        `get_align_matrix` away from its `||v1 x v2|| < 1e-6` branch, where it silently
        returns identity (transforms.py:397-399).

        Must be a list of lists: transforms.py:501-504 reshapes `(-1,3).T` on the list branch
        but `(3,-1)` on the ndarray branch, which would silently transpose the meaning.
        """
        rack = np.array(self.rack.get_functional_point(0, "matrix"))
        depth = rack[:3, 1]
        return [depth.tolist(), (-depth).tolist()]

    def _rail_state(self):
        """Lateral offset from the rail centre, and height above the rail top."""
        rail = self.rack.get_functional_point(0, "matrix")
        centre, up = rail[:3, 3], rail[:3, 2]
        d = self.dumbbell.get_pose().p - centre
        height = float(d @ up)
        lateral = float(np.linalg.norm(d - height * up))
        return lateral, height

    def check_success(self):
        lateral, height = self._rail_state()
        gripper_open = (self.is_left_gripper_open()
                        if self.arm_tag == "left" else self.is_right_gripper_open())
        return bool(lateral < SUCCESS_XY and height > SUCCESS_ABOVE_RAIL
                    and self.check_actors_contact(self.dumbbell.get_name(), self.rack.get_name())
                    and gripper_open)
