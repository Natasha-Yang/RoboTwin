"""Put a book into a bookcase.

A clearance fit, not a place-on: the book goes down into a bay 6 mm/side wider than it is
thick, so the bay walls guide it and contact force is informative. That puts this task in the
same family as `envs/_peg_insertion_base.py`, and it uses the same two-stage constrained
descent for the same reason -- see `insert_book`.

Both assets need `script/gen_task_assets.py` run once per clone (`assets/*` is gitignored, so
nothing under it can be committed):

* `014_bookcase/base3` ships **unannotated** -- only `{stable, center, extents}`, no `scale`
  and no points, which `create_actor` swallows in a bare `except` (create_actor.py:531),
  leaving the mesh at its raw 1.9 m size and every `get_functional_point` returning None.
  `base3` is the one variant that is an open rack: four posts on a base slab giving three
  bays. The others are modelled already **full of books** and decompose to a solid blob.
* `043_book/base0` ships annotated but at scale 1.0 (a 215 mm book) with both contact points
  at mid-height, approached from beyond its left or right edge. It is rescaled to 0.6 and
  given a **top-down** grasp of its top edge (points 2 and 3) plus a functional point at the
  centre of its bottom edge -- the end that goes in. The top-down grasp is not cosmetic: the
  shipped side grasp approaches along the book's width, and `get_grasp_pose` stands the
  end-effector 0.12 m back from the contact plus `pre_grasp_dis` (_base_task.py:1200), so at
  the book's spawn range that pre-pose lands near |x| = 0.45 m, outside the arm's reach. The
  side grasp failed to plan on **every** seed tried; with the top-down grasp the task yields.

The sizing is driven by three constraints that pull against each other, since the bay's width
and the posts' height scale together:

    bay 43.9 mm wide vs book 31.9 mm thick    -> 6.0 mm clearance per side
    posts 110 mm above the bay floor          -> the book's bottom must clear them
    book grasped 118 mm above its own bottom  -> the fingers must clear them too

The last is why the book is released 45 mm above the bay floor rather than driven onto it:
at that point 65 mm of book is already between the posts, so the drop is guided, and the
gripper stays 45 mm clear of the posts instead of 5 mm. `envs/_peg_insertion_base.py` releases
its peg early for the same reason.
"""

import numpy as np
import sapien

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import *
from .utils import *
from ._task_assets import require_assets

BOOK_MODEL = "043_book"
BOOK_ID = 0
BOOKCASE_MODEL = "014_bookcase"
BOOKCASE_ID = 3

TABLE_Z = 0.741
BOOK_HALF_H = 0.0645  # 129 mm tall at scale 0.6; the spawn has to stand it on the table
BOOKCASE_BOTTOM = 0.0017  # the mesh dips this far below its own origin, at scale 0.10

GRASP_BAND = [2, 3]  # the added top-down contact points; see the module docstring
PRE_GRASP_DIS = 0.09
LIFT_Z = 0.10

# Heights of the book's bottom above the bay floor, in metres. The functional frame's +Z
# points DOWN the bay, so a POSITIVE pre_dis holds the book above the floor.
PRE_INSERT_DIS = 0.14  # 30 mm above the posts
NEAR_INSERT_DIS = 0.075
RELEASE_DIS = 0.045
RETRACT_DIS = 0.08

SUCCESS_SEATED = 0.025  # the book's bottom is within this of the bay floor
SUCCESS_LATERAL = 0.020  # the bay is 43.9 mm wide, so this cannot be met outside it
SUCCESS_UPRIGHT = 0.90


require_assets((BOOK_MODEL, BOOK_ID), (BOOKCASE_MODEL, BOOKCASE_ID))


class put_book_bookcase(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # The book stands on the table: local +Z is its 129 mm height, so the identity qpos
        # already stands it up and the spawn only has to lift it by its own half-height.
        # Yaw only -- roll or pitch is what would topple it.
        for _ in range(100):
            book_pose = rand_pose(
                xlim=[-0.26, 0.26],
                ylim=[-0.10, 0.02],
                zlim=[TABLE_Z + BOOK_HALF_H],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.6],
            )
            if abs(book_pose.p[0]) > 0.09:
                break

        side = 1.0 if book_pose.p[0] > 0 else -1.0
        for _ in range(100):
            bookcase_pose = rand_pose(
                xlim=[0.10, 0.26] if side > 0 else [-0.26, -0.10],
                # NEGATIVE y, i.e. nearer the robot, the same side the shipped receptacle
                # tasks put theirs on (place_object_stand uses [-0.15, -0.1]). At +0.10..+0.18
                # the planner failed on 7 of 8 seeds trying to carry the book out over the
                # bookcase; the goal pose is reachable in isolation, the path is not.
                ylim=[-0.18, -0.11],
                zlim=[TABLE_Z + BOOKCASE_BOTTOM],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 9, 0],
            )
            # The gripper sits 0.12 m from the grasp point (_base_task.py:1145); closer than
            # this and the wrist is inside the bookcase footprint while grasping the book.
            if np.linalg.norm(bookcase_pose.p[:2] - book_pose.p[:2]) > 0.15:
                break

        self.book = create_actor(
            scene=self,
            pose=book_pose,
            modelname=BOOK_MODEL,
            convex=True,
            model_id=BOOK_ID,
        )
        # Static: a 0.01 kg bookcase would be shoved across the table by the insertion itself.
        self.bookcase = create_actor(
            scene=self,
            pose=bookcase_pose,
            modelname=BOOKCASE_MODEL,
            convex=True,  # -> one convex hull per submesh, which is what keeps the bays open
            is_static=True,
            model_id=BOOKCASE_ID,
        )

        self.add_prohibit_area(self.book, padding=0.10)
        self.add_prohibit_area(self.bookcase, padding=0.10)

        # Fixed here rather than in play_once, so check_success and step_reward do not depend
        # on the expert having run -- during a policy rollout it has not.
        self.arm_tag = ArmTag("left" if book_pose.p[0] < 0 else "right")

        seated, lateral = self._bay_state()
        self.last_seated = seated
        self.last_lateral = lateral

    # -- geometry --------------------------------------------------------------------------

    def _bay_state(self):
        """(height of the book's bottom above the bay floor, lateral offset from the bay axis)."""
        bay = np.array(self.bookcase.get_functional_point(0, "matrix"))
        floor, down = bay[:3, 3], bay[:3, 2]
        d = self.book.get_functional_point(0, "pose").p - floor
        seated = -float(d @ down)  # +Z points down the bay, so negate to get height
        lateral = float(np.linalg.norm(d + seated * down))
        return seated, lateral

    def _bay_align_axis(self):
        """The bay's depth direction and its opposite, for `constrain="align"`.

        The book's own +X is its 97 mm width, which has to end up along the bay's depth -- it
        goes in edge-on, not face-on. Leaving `align_axis=None` would force the book's +X onto
        the bay's +X (transforms.py:499-500), which is the right axis but only one of the two
        that work: a book is 2-fold symmetric about its vertical, so spine-left and
        spine-right seat identically. Offering both caps the correction at 90 deg and keeps
        `get_align_matrix` away from its `||v1 x v2|| < 1e-6` branch, where it silently
        returns identity (transforms.py:397-399) and applies no yaw correction at all.

        Must be a list of lists: transforms.py:501-504 reshapes `(-1,3).T` on the list branch
        but `(3,-1)` on the ndarray branch, which would silently transpose the meaning.
        """
        bay = np.array(self.bookcase.get_functional_point(0, "matrix"))
        ax = bay[:3, 0]
        return [ax.tolist(), (-ax).tolist()]

    # -- expert ----------------------------------------------------------------------------

    def insert_book(self, arm_tag: ArmTag):
        """The descent, built by hand because `place_actor` cannot constrain it.

        `place_actor` emits bare `Action(arm, "move", ...)` with no `constraint_pose`
        (_base_task.py:1432-1435), and the planner is a trajectory optimizer whose collision
        world knows nothing about the book or the bookcase -- so over a 100 mm descent the
        path can bow by more than the 6 mm clearance and the orientation is free to drift.
        `move()` does forward `constraint_pose` (_base_task.py:1035-1052), and `grasp_actor`
        already uses [1,1,1,0,0,0] for its own final approach, so use the same here:
        orientation held, position free.
        """
        if not self.plan_success:
            return False

        if self.need_plan:
            target = self.bookcase.get_functional_point(0, "pose")
            kwargs = dict(functional_point_id=0, pre_dis_axis="fp", constrain="align",
                          align_axis=self._bay_align_axis())
            pre_pose = self.get_place_pose(self.book, arm_tag, target, pre_dis=PRE_INSERT_DIS, **kwargs)
            near_pose = self.get_place_pose(self.book, arm_tag, target, pre_dis=NEAR_INSERT_DIS, **kwargs)
            release_pose = self.get_place_pose(self.book, arm_tag, target, pre_dis=RELEASE_DIS, **kwargs)
        else:
            # Replay consumes the cached joint path by index and ignores the pose, but it
            # must not be None. Mirrors place_actor's own need_plan branch.
            pre_pose = near_pose = release_pose = [0, 0, 0, 0, 0, 0, 0]


        if any(p is None for p in (pre_pose, near_pose, release_pose)):
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
                Action(arm_tag, "move", target_pose=release_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
                Action(arm_tag, "open", target_gripper_pos=1.0),
            ],
        ))

    def play_once(self):
        arm_tag = self.arm_tag

        self.move(
            self.grasp_actor(
                self.book,
                arm_tag=arm_tag,
                pre_grasp_dis=PRE_GRASP_DIS,
                grasp_dis=0.0,
                contact_point_id=GRASP_BAND,
            ))
        self.move(self.move_by_displacement(arm_tag, z=LIFT_Z))

        self.insert_book(arm_tag)

        # "arm" retracts along the end-effector's own approach direction, i.e. straight back
        # out of the (side) grasp rather than up through the book.
        self.move(self.move_by_displacement(arm_tag, z=RETRACT_DIS, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))
        self.delay(4)  # let the dropped book settle before check_success

        self.info["info"] = {
            "{A}": f"{BOOKCASE_MODEL}/base{BOOKCASE_ID}",
            "{B}": f"{BOOK_MODEL}/base{BOOK_ID}",
            "{a}": str(arm_tag),
        }
        return self.info

    # -- outcome ---------------------------------------------------------------------------

    def check_success(self):
        seated, lateral = self._bay_state()
        upright = get_face_prod(self.book.get_pose().q, [0, 0, 1], [0, 0, 1]) > SUCCESS_UPRIGHT
        gripper_open = (self.is_left_gripper_open()
                        if self.arm_tag == "left" else self.is_right_gripper_open())
        return bool(seated < SUCCESS_SEATED and lateral < SUCCESS_LATERAL
                    and upright and gripper_open)

    def step_reward(self):
        """Shaped progress, as a DELTA since the last call (script/eval_policy.py:152-175).

        Two clipped deltas, summed: closing on the bay axis, and descending into the bay. Both
        are symmetric, so undoing progress refunds it and nothing ratchets. Descent only
        counts once the book is roughly over the bay (`lateral` inside the bay's own
        half-width), for the same reason the peg task gates its depth term: otherwise a book
        standing anywhere on the table already reads as fully seated, which makes the shaping
        penalise lifting it and pay for setting it back down.
        """
        seated, lateral = self._bay_state()
        reward = float(np.clip(self.last_lateral - lateral, -0.1, 0.1))
        if lateral < SUCCESS_LATERAL:
            reward += float(np.clip(self.last_seated - seated, -0.1, 0.1))
            self.last_seated = seated
        else:
            self.last_seated = max(seated, self.last_seated)
        self.last_lateral = lateral
        return reward
