"""End-effector contact wrench, read off the physics scene.

Shared by the eval driver's debug plots (`script/eval_policy.py`, one sample per policy call),
by rollout-dataset collection (`script/collect_dataset.py`, one sample per primitive step) and
by expert-demo collection (`script/collect_data.py`, likewise per primitive step, stacked into
the HDF5's `wrench/<link>` groups), so all three record the exact same quantity.

Everything that records it keeps it **per end-effector link** (`link_wrench_vector`), not
summed per arm: two fingers squeezing the same object push against each other, and those equal
and opposite forces cancel in the arm total.
"""

from collections import Counter

import numpy as np

# Component order of the flat (6,) vector `link_wrench_vector` / `tcp_wrench_vector` return.
WRENCH_COMPONENTS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

ARM_TAGS = ("left", "right")


def _ee_links(robot, arm_tag):
    """The links that make up one arm's end-effector assembly, in report order.

    Returns ``[(link_name, entity_id), ...]`` — the gripper finger links (aloha: ``fl_link7``,
    ``fl_link8``) plus any ``fix_gripper_name`` links. Matching is by ``entity.per_scene_id``
    rather than by name, because a non-dual-arm embodiment loads the same URDF twice and the
    two arms then carry identical link names on distinct entities.
    """
    entity = getattr(robot, f"{arm_tag}_entity")
    # get the child links of the ee joint (i.e. wrist link)
    # links = [getattr(robot, f"{arm_tag}_ee").child_link]
    # get the child links of the gripper
    links = []
    links += [joint.child_link for joint, _, _ in getattr(robot, f"{arm_tag}_gripper")]
    links += [entity.find_link_by_name(n) for n in getattr(robot, f"{arm_tag}_fix_gripper_name", [])]
    out, seen = [], set()
    for link in links:
        if link is None or link.entity.per_scene_id in seen:
            continue
        seen.add(link.entity.per_scene_id)
        out.append((link.get_name(), link.entity.per_scene_id))
    return out


def ee_link_labels(robot):
    """Plot/column labels for every end-effector link: ``{arm_tag: [(label, entity_id), ...]}``.

    The label is the URDF link name (``fl_link7``), which for a dual-arm URDF already names the
    arm. When both arms carry the *same* name — the same single-arm URDF loaded twice — it is
    prefixed with the arm tag (``left_link7``) so the two stay distinguishable in a legend or an
    npz key.

    Memoized on the robot object: the links are fixed for the life of a robot, and since the
    wrench is now sampled on every physics step of a collection run this would otherwise redo
    the URDF name lookups thousands of times per episode. A new episode builds a new robot, so
    the cache invalidates itself.
    """
    cached = getattr(robot, "_ee_link_labels_cache", None)
    if cached is not None:
        return cached
    per_arm = {arm: _ee_links(robot, arm) for arm in ARM_TAGS}
    shared = {name for name, count in
              Counter(name for links in per_arm.values() for name, _ in links).items() if count > 1}
    labels = {arm: [(f"{arm}_{name}" if name in shared else name, link_id) for name, link_id in links]
              for arm, links in per_arm.items()}
    try:
        robot._ee_link_labels_cache = labels
    except AttributeError:  # a robot wrapper that forbids new attributes: just recompute
        pass
    return labels


def compute_link_wrench(task_env):
    """Net contact wrench on each end-effector **link**, resolved in the **world** frame.

    SAPIEN reports per-contact-point *impulses* accumulated over the last physics step and
    applied to ``contact.bodies[0]``; dividing by the scene timestep turns them into the
    average force over that step. Force is summed over every contact point on the link, and
    torque is taken about that arm's TCP origin (``sum (p - p_tcp) x f``) — the same reference
    point for all of the arm's links, so the fingers are comparable with each other and with
    their sum (`compute_tcp_wrench`, which is exactly this summed per arm). Only the moment arm
    is TCP-relative — the components themselves stay in world axes, so a trace is comparable
    across steps even as the gripper rotates.

    Returns ``{arm_tag: {link_label: (force(3,), torque(3,))}}`` in N and N*m, labelled by
    `ee_link_labels`. A link touching nothing reads as all zeros — free-space motion produces no
    wrench here, only contact does.
    """
    contacts = task_env.scene.get_contacts()
    dt = task_env.scene.get_timestep()
    robot = task_env.robot
    out = {}
    for arm_tag, links in ee_link_labels(robot).items():  # [fl_link7, fl_link8] or [fr_link7, fr_link8]
        label_of = {link_id: label for label, link_id in links}
        tcp_p = np.asarray(getattr(robot, f"get_{arm_tag}_tcp_pose")(), dtype=np.float64)[:3]
        per_link = {label: (np.zeros(3), np.zeros(3)) for label, _ in links}
        for contact in contacts:
            ids = [body.entity.per_scene_id for body in contact.bodies]
            # impulses are applied to the first actor
            if ids[0] in label_of and ids[1] in label_of:
                continue  # self-contact inside the assembly: internal, cancels out
            if ids[0] in label_of:
                label, sign = label_of[ids[0]], 1.0  # impulses applied on the robot
            elif ids[1] in label_of: # impulses applied to another object
                label, sign = label_of[ids[1]], -1.0
            else:
                continue
            force, torque = per_link[label]
            for point in contact.points:
                f = sign * np.asarray(point.impulse, dtype=np.float64) / dt
                force += f
                torque += np.cross(np.asarray(point.position, dtype=np.float64) - tcp_p, f)
        out[arm_tag] = per_link
    return out


def compute_tcp_wrench(task_env):
    """`compute_link_wrench` summed over each arm's links: the whole end effector's wrench.

    Returns ``{"left": (force(3,), torque(3,)), "right": ...}`` in N and N*m, in the world frame
    with torque about that arm's TCP. Summing is valid because every link of an arm already
    shares that reference point.

    Nothing on the recording path calls this any more — every consumer (the demo HDF5, the
    rollout dataset, the critic's `wrench.*` modality, the debug plots) keeps the wrench per
    link, because a finger squeezing against its opposite produces equal and opposite forces
    that cancel in this sum. Kept as the arm-level summary for analysis that wants it.
    """
    return {arm: (np.sum([f for f, _ in per_link.values()] or [np.zeros(3)], axis=0),
                  np.sum([t for _, t in per_link.values()] or [np.zeros(3)], axis=0))
            for arm, per_link in compute_link_wrench(task_env).items()}


def tcp_wrench_vector(task_env):
    """`compute_tcp_wrench` flattened to one ``(6,)`` vector per arm, in WRENCH_COMPONENTS order.

    Returns ``{"left": (6,), "right": (6,)}``. The recording path uses `link_wrench_vector`
    instead; see `compute_tcp_wrench` on why.
    """
    return {
        arm: np.concatenate([force, torque])
        for arm, (force, torque) in compute_tcp_wrench(task_env).items()
    }


def link_wrench_vector(task_env):
    """`compute_link_wrench` flattened to one ``(6,)`` vector per link, in WRENCH_COMPONENTS order.

    Returns ``{link_label: (6,)}`` over both arms, left arm's links first (aloha: ``fl_link7``,
    ``fl_link8``, ``fr_link7``, ``fr_link8``). This is the layout every recorded wrench uses —
    the demo HDF5's ``wrench/<link>`` groups, the rollout dataset's
    ``observation.wrench.<link>`` columns, the critic's ``wrench.<link>`` modality, and the
    debug plots and their ``.npz`` — so a gripper's two fingers can be read apart instead of
    only their sum.
    """
    return {
        label: np.concatenate([force, torque])
        for per_link in compute_link_wrench(task_env).values()
        for label, (force, torque) in per_link.items()
    }


def stack_step_wrench(step_wrench, num_steps):
    """Stack the per-step samples logged since the last drain into ``{link_label: (num_steps, 6)}``.

    ``step_wrench`` is what ``_base_task.pop_step_wrench`` collected: one ``link_wrench_vector``
    dict per primitive step. ``num_steps`` is however many steps a drain covers — ``pi0_step``
    on the rollout paths (one drain per action chunk), the task config's ``save_freq`` on the
    demo-collection path (one drain per saved frame). Padded to it with **NaN**, so the result
    has one fixed shape whether or not that span ran to completion — an episode's first drain
    has nothing behind it and carries a single sample of the current contact state, and a chunk
    cut short by success or ``step_lim`` yields fewer than ``num_steps``. NaN rather than zero,
    because zero is a meaningful reading (the arm touching nothing); consumers that cannot take
    NaN should map it to zero explicitly.

    Keyed by whatever the samples are keyed by, so it follows `link_wrench_vector`'s labels
    without knowing them.

    Returns ``{}`` for an empty log, so callers can tell "no samples" from "samples that were
    all zero".
    """
    if not step_wrench:
        return {}
    stacked = {}
    for link in step_wrench[0]:
        samples = np.asarray([sample[link] for sample in step_wrench], dtype=np.float32)[:num_steps]
        padded = np.full((num_steps, len(WRENCH_COMPONENTS)), np.nan, dtype=np.float32)
        padded[:len(samples)] = samples
        stacked[link] = padded
    return stacked
