"""Self-test for the resumable scripted experts, with no GPU and no simulator.

Run it after changing any task's `scripted_stages` / `resume_stage`:

    python script/test_resume_stages.py

Two properties, and the first is the one that matters. Splitting `play_once` into stages is
only safe if it is a pure regrouping -- data collection replays these demos, so a dropped,
reordered or subtly re-argumented `move` would silently change the dataset. `test_stages_*`
drives both the pre-change body (kept verbatim below) and the real `scripted_stages()` against
a symbolic stand-in for the scene, and diffs the resulting call traces.

`test_resume_stage_*` then walks every branch of each task's `resume_stage` and checks it
lands inside its own stage list. An index past the end silently skips the rest of the demo;
one too low sends the expert back to re-grasp something the other arm is already holding,
which is the failure the whole mechanism exists to avoid.
"""

import itertools
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from envs.utils.action import ArmTag

LEFT, RIGHT = ArmTag("left"), ArmTag("right")


# ===================================== stage equivalence =====================================

def fmt(value):
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    return repr(value)


class Sym:
    """A symbolic scene object: every attribute read and call records how it was reached."""

    def __init__(self, path):
        object.__setattr__(self, "path", path)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Sym(f"{self.path}.{name}")

    def __call__(self, *args, **kwargs):
        inner = [fmt(a) for a in args] + [f"{k}={fmt(v)}" for k, v in sorted(kwargs.items())]
        return Sym(f"{self.path}({', '.join(inner)})")

    def __getitem__(self, key):
        return Sym(f"{self.path}[{key!r}]")

    def __repr__(self):
        return self.path


class Trace:
    """A stand-in `self`: the primitives record what they were passed, the rest is symbolic."""

    RECORDED = ("move", "delay", "insert_peg")
    PRIMITIVES = ("grasp_actor", "place_actor", "move_by_displacement", "open_gripper",
                  "close_gripper", "back_to_origin", "get_place_pose")

    def __init__(self, **attrs):
        self.calls = []
        for key, value in attrs.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name in self.RECORDED:
            def record(*args, **kwargs):
                inner = ([fmt(a) for a in args]
                         + [f"{k}={fmt(v)}" for k, v in sorted(kwargs.items())])
                self.calls.append(f"{name}({', '.join(inner)})")
            return record
        if name in self.PRIMITIVES:
            return Sym(name)
        return Sym(f"self.{name}")


# The arms are passed in rather than re-derived: which arm is chosen is fixed by the spawn
# limits and is not what this compares. The sequence of primitives is.

def old_handover_block(self, grasp_arm_tag, place_arm_tag):
    self.move(
        self.grasp_actor(self.box, arm_tag=grasp_arm_tag, pre_grasp_dis=0.07, grasp_dis=0.0,
                         contact_point_id=[0, 1, 2, 3]))
    self.move(self.move_by_displacement(grasp_arm_tag, z=0.1))
    self.move(
        self.place_actor(self.box, target_pose=self.block_middle_pose, arm_tag=grasp_arm_tag,
                         functional_point_id=0, pre_dis=0, dis=0, is_open=False,
                         constrain="free"))
    self.move(
        self.grasp_actor(self.box, arm_tag=place_arm_tag, pre_grasp_dis=0.07, grasp_dis=0.0,
                         contact_point_id=[4, 5, 6, 7]))
    self.move(self.open_gripper(grasp_arm_tag))
    self.move(self.move_by_displacement(grasp_arm_tag, z=0.1, move_axis="arm"))
    self.move(
        self.back_to_origin(grasp_arm_tag),
        self.place_actor(self.box, target_pose=self.target_box.get_functional_point(1, "pose"),
                         arm_tag=place_arm_tag, functional_point_id=0, pre_dis=0.05, dis=0.,
                         constrain="align", pre_dis_axis="fp"),
    )


def old_handover_mic(self, grasp_arm_tag, handover_arm_tag):
    self.move(
        self.grasp_actor(self.microphone, arm_tag=grasp_arm_tag,
                         contact_point_id=[1, 9, 10, 11, 12, 13, 14, 15], pre_grasp_dis=0.1))
    self.move(
        self.move_by_displacement(
            grasp_arm_tag, z=0.12,
            quat=(GRASP_DIRECTION_DIC["front_right"]
                  if grasp_arm_tag == "left" else GRASP_DIRECTION_DIC["front_left"]),
            move_axis="arm"))
    self.move(
        self.place_actor(self.microphone, arm_tag=grasp_arm_tag,
                         target_pose=self.handover_middle_pose, functional_point_id=0,
                         pre_dis=0.0, dis=0.0, is_open=False, constrain="free"))
    self.move(
        self.grasp_actor(self.microphone, arm_tag=handover_arm_tag,
                         contact_point_id=[0, 2, 3, 4, 5, 6, 7, 8], pre_grasp_dis=0.1))
    self.move(self.open_gripper(grasp_arm_tag))
    self.move(
        self.move_by_displacement(grasp_arm_tag, z=0.07, move_axis="arm"),
        self.move_by_displacement(handover_arm_tag,
                                  x=0.05 if handover_arm_tag == "right" else -0.05),
    )


def old_turn_switch(self, arm_tag, _unused):
    self.move(self.close_gripper(arm_tag=arm_tag, pos=0))
    self.move(self.grasp_actor(self.switch, arm_tag=arm_tag, pre_grasp_dis=0.04))


def old_put_object_cabinet(self, arm_tag, _unused):
    self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))
    self.move(self.grasp_actor(self.cabinet, arm_tag=arm_tag.opposite, pre_grasp_dis=0.05))
    for _ in range(4):
        self.move(self.move_by_displacement(arm_tag=arm_tag.opposite, y=-0.04))
    self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))
    target_pose = self.cabinet.get_functional_point(0)
    self.move(self.place_actor(self.object, arm_tag=arm_tag, target_pose=target_pose,
                               pre_dis=0.13, dis=0.1))


def old_lift_pot(self, left_arm_tag, right_arm_tag):
    self.move(
        self.close_gripper(left_arm_tag, pos=0.5),
        self.close_gripper(right_arm_tag, pos=0.5),
    )
    self.move(
        self.grasp_actor(self.pot, left_arm_tag, pre_grasp_dis=0.035, contact_point_id=0),
        self.grasp_actor(self.pot, right_arm_tag, pre_grasp_dis=0.035, contact_point_id=1),
    )
    self.move(
        self.move_by_displacement(left_arm_tag, z=0.88 - self.pot.get_pose().p[2]),
        self.move_by_displacement(right_arm_tag, z=0.88 - self.pot.get_pose().p[2]),
    )


def old_peg_insertion(self, arm_tag, _unused):
    from envs._peg_insertion_base import GRASP_BAND, LIFT_Z, PRE_GRASP_DIS, RETRACT_DIS

    self.move(
        self.grasp_actor(self.peg, arm_tag=arm_tag, pre_grasp_dis=PRE_GRASP_DIS, grasp_dis=0.0,
                         contact_point_id=GRASP_BAND))
    self.move(self.move_by_displacement(arm_tag, z=LIFT_Z))
    self.insert_peg(arm_tag)
    self.move(self.move_by_displacement(arm_tag, z=RETRACT_DIS, move_axis="arm"))
    self.move(self.back_to_origin(arm_tag))
    self.delay(4)


class Pot:
    """A pot at a fixed height. Stage 2 computes `0.88 - height`, so this cannot be symbolic."""

    def __init__(self, height):
        self.height = height

    def get_pose(self):
        return type("P", (), {"p": [0.0, 0.0, self.height]})()

    def get_contact_point(self, index):
        return [0.0, 0.1 if index else -0.1, self.height, 1, 0, 0, 0]

    def __repr__(self):
        return "self.pot"


def task_class(module_name, class_name):
    import importlib

    return getattr(importlib.import_module(module_name), class_name)


CASES = [
    ("handover_block", old_handover_block, ("envs.handover_block", "handover_block"),
     {"grasp_arm_tag": LEFT, "place_arm_tag": RIGHT, "box": Sym("self.box"),
      "target_box": Sym("self.target_box"), "block_middle_pose": [0, 0.0, 0.9, 0, 1, 0, 0]},
     (LEFT, RIGHT)),
    ("handover_mic", old_handover_mic, ("envs.handover_mic", "handover_mic"),
     {"grasp_arm_tag": RIGHT, "handover_arm_tag": LEFT, "microphone": Sym("self.microphone"),
      "handover_middle_pose": [0, -0.05, 0.98, 0, 1, 0, 0]},
     (RIGHT, LEFT)),
    ("turn_switch", old_turn_switch, ("envs.turn_switch", "turn_switch"),
     {"arm_tag": RIGHT, "switch": Sym("self.switch")}, (RIGHT, None)),
    ("put_object_cabinet", old_put_object_cabinet,
     ("envs.put_object_cabinet", "put_object_cabinet"),
     {"arm_tag": RIGHT, "object": Sym("self.object"), "cabinet": Sym("self.cabinet")},
     (RIGHT, None)),
    ("peg_insertion", old_peg_insertion, ("envs._peg_insertion_base", "_PegInsertionBase"),
     {"arm_tag": LEFT, "peg": Sym("self.peg"), "socket": Sym("self.socket")}, (LEFT, None)),
    ("lift_pot", old_lift_pot, ("envs.lift_pot", "lift_pot"),
     {"pot": Pot(0.75)}, (LEFT, RIGHT)),
]


def test_stages_reproduce_play_once():
    for name, old_fn, (module, class_name), attrs, arms in CASES:
        cls = task_class(module, class_name)
        constants = {key: getattr(cls, key) for key in dir(cls)
                     if key.isupper() and not key.startswith("_")}
        new = Trace(**{**constants, **attrs})
        for stage in cls.scripted_stages(new):
            stage()
        old = Trace(**{**constants, **attrs})
        old_fn(old, *arms)
        assert old.calls == new.calls, (
            f"{name}: stages diverge from play_once\n"
            + "\n".join(f"  [{i}] old {o}\n      new {n}"
                        for i, (o, n) in enumerate(itertools.zip_longest(old.calls, new.calls))
                        if o != n))
        print(f"  {name}: {len(new.calls)} calls identical")


# ======================================= resume_stage =======================================

class Pose:
    def __init__(self, p):
        self.p = p


class Held:
    """A scene stub whose predicates are set per case rather than read off physics."""

    def __init__(self, cls, **state):
        self.cls = cls
        self.__dict__.update(state)

    def scripted_stages(self):
        return self.cls.scripted_stages(self)

    def resume_stage(self):
        return self.cls.resume_stage(self)


def stage_count(module, class_name, attrs):
    cls = task_class(module, class_name)
    constants = {key: getattr(cls, key) for key in dir(cls)
                 if key.isupper() and not key.startswith("_")}
    return len(cls.scripted_stages(Trace(**{**constants, **attrs}))), cls, constants


def check_branches(name, module, class_name, attrs, states):
    """Every state a task can be rewound into must name a stage that exists."""
    total, cls, constants = stage_count(module, class_name, attrs)
    seen = set()
    for state in states:
        stub = Held(cls, **{**constants, **attrs, **state})
        stage = stub.resume_stage()
        assert isinstance(stage, int), f"{name}: resume_stage returned {stage!r}"
        assert 0 <= stage <= total, f"{name}: stage {stage} outside 0..{total} for {state}"
        seen.add(stage)
    print(f"  {name}: {len(seen)} distinct stages over {len(states)} states, all within 0..{total}")
    return seen


def robot_at(distance):
    """A robot whose TCPs sit `distance` from the origin, for the withdrawal tests."""
    return type("R", (), {
        "get_left_tcp_pose": lambda s: [distance, 0.0, 0.0, 1, 0, 0, 0],
        "get_right_tcp_pose": lambda s: [distance, 0.0, 0.0, 1, 0, 0, 0],
    })()


def test_resume_stage_handover_block():
    attrs = {"grasp_arm_tag": LEFT, "place_arm_tag": RIGHT, "box": Sym("self.box"),
             "target_box": Sym("self.target_box"),
             "block_middle_pose": [0, 0.0, 0.9, 0, 1, 0, 0]}
    states = []
    for success, holds_place, holds_grasp, closed, at_handover, far in itertools.product(
            [False, True], repeat=6):
        states.append({
            "check_success": lambda s=None, v=success: v,
            "_holding": lambda arm, hp=holds_place, hg=holds_grasp: hp if arm == RIGHT else hg,
            "_at_handover": lambda v=at_handover: v,
            "is_left_gripper_close": lambda v=closed: v,
            "is_right_gripper_close": lambda v=closed: v,
            "robot": robot_at(1.0 if far else 0.0),
            "box": type("B", (), {"get_pose": lambda s: Pose([0.0, 0.0, 0.0])})(),
        })
    seen = check_branches("handover_block", "envs.handover_block", "handover_block", attrs, states)
    # The point of the ordering: with the box in the receiving arm the expert must land on the
    # release/withdraw/place tail, never back at the handing arm's own grasp.
    assert seen >= {0, 1, 3, 4, 5, 6, 7}


def test_resume_stage_handover_mic():
    attrs = {"grasp_arm_tag": RIGHT, "handover_arm_tag": LEFT,
             "microphone": Sym("self.microphone"),
             "handover_middle_pose": [0, -0.05, 0.98, 0, 1, 0, 0]}
    states = []
    for success, holds_handover, holds_grasp, closed, at_handover in itertools.product(
            [False, True], repeat=5):
        states.append({
            "check_success": lambda s=None, v=success: v,
            "_holding": lambda arm, hh=holds_handover, hg=holds_grasp: hh if arm == LEFT else hg,
            "_at_handover": lambda v=at_handover: v,
            "is_left_gripper_close": lambda v=closed: v,
            "is_right_gripper_close": lambda v=closed: v,
        })
    seen = check_branches("handover_mic", "envs.handover_mic", "handover_mic", attrs, states)
    assert seen >= {0, 1, 3, 4, 5, 6}


def test_resume_stage_turn_switch():
    attrs = {"arm_tag": RIGHT, "switch": Sym("self.switch")}
    states = [{
        "check_success": lambda s=None, v=success: v,
        "is_left_gripper_close": lambda v=closed: v,
        "is_right_gripper_close": lambda v=closed: v,
    } for success, closed in itertools.product([False, True], repeat=2)]
    seen = check_branches("turn_switch", "envs.turn_switch", "turn_switch", attrs, states)
    assert seen == {0, 1, 2}


def test_resume_stage_put_object_cabinet():
    attrs = {"arm_tag": RIGHT, "object": Sym("self.object"), "cabinet": Sym("self.cabinet")}
    states = []
    for success, holds_object, holds_bar, lifted in itertools.product([False, True], repeat=4):
        for pulls in range(5):
            states.append({
                "check_success": lambda s=None, v=success: v,
                "gripper_holds": (lambda arm, actor, hold_dis=0.0, point=None,
                                  ho=holds_object, hb=holds_bar:
                                  ho if arm == RIGHT else hb),
                "_pulls_done": lambda v=pulls: v,
                "origin_z": 0.0,
                "object": type("O", (), {
                    "get_pose": lambda s, v=lifted: Pose([0.0, 0.0, 0.15 if v else 0.0])})(),
                "cabinet": type("C", (), {
                    "get_contact_point": lambda s, i: [0.0, 0.0, 0.0, 1, 0, 0, 0]})(),
            })
    seen = check_branches("put_object_cabinet", "envs.put_object_cabinet",
                          "put_object_cabinet", attrs, states)
    # One stage per pull, so a half-open drawer resumes at the pull it reached rather than
    # re-running all four against a drawer already part of the way out.
    assert seen >= {0, 1, 2, 3, 4, 5, 6, 7, 8}


def test_resume_stage_peg_insertion():
    from envs import _peg_insertion_base as peg

    attrs = {"arm_tag": LEFT, "peg": Sym("self.peg"), "socket": Sym("self.socket")}
    states = []
    for success, grasped, seated, lifted in itertools.product([False, True], repeat=4):
        states.append({
            "check_success": lambda s=None, v=success: v,
            "_peg_grasped": lambda v=grasped: v,
            "_insertion_state": lambda: (0.0, 0.0),
            "_bore_depth": lambda lateral, depth, v=seated: 0.02 if v else 0.0,
            "peg_rest_z": 0.0,
            "peg": type("P", (), {
                "get_functional_point": lambda s, i, kind=None, v=lifted: Pose(
                    [0.0, 0.0, peg.LIFT_Z if v else 0.0])})(),
        })
    seen = check_branches("peg_insertion", "envs._peg_insertion_base", "_PegInsertionBase",
                          attrs, states)
    # 3 is the one that matters: a peg standing released in the bore must NOT send the arm
    # back down to re-grasp it out of the hole.
    assert seen >= {0, 1, 2, 3, 6}


def test_resume_stage_lift_pot():
    attrs = {"pot": Pot(0.75)}
    states = []
    for success, gripping in itertools.product([False, True], repeat=2):
        states.append({
            "check_success": lambda s=None, v=success: v,
            "_gripping": lambda v=gripping: v,
        })
    seen = check_branches("lift_pot", "envs.lift_pot", "lift_pot", attrs, states)
    # 0 while already gripping would command both grippers back to 0.5 and drop the pot;
    # the holding state must land on the lift instead.
    assert seen == {0, 2, 3}


if __name__ == "__main__":
    print("stages reproduce play_once:")
    test_stages_reproduce_play_once()
    print("resume_stage stays inside its own stage list:")
    test_resume_stage_handover_block()
    test_resume_stage_handover_mic()
    test_resume_stage_turn_switch()
    test_resume_stage_put_object_cabinet()
    test_resume_stage_peg_insertion()
    test_resume_stage_lift_pot()
    print("all resume-stage checks passed")
