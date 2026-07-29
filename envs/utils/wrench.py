"""End-effector contact wrench, read off the physics scene.

Shared by the eval driver's debug plots (`script/eval_policy.py`, one sample per policy call)
and by rollout-dataset collection (`script/collect_dataset.py`, one sample per primitive step),
so both record the exact same quantity.
"""

import numpy as np

# Component order of the flat (6,) vector `tcp_wrench_vector` returns.
WRENCH_COMPONENTS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

ARM_TAGS = ("left", "right")


def _ee_link_ids(robot, arm_tag):
    """Entity ids of the links that make up one arm's end-effector assembly.
    """
    entity = getattr(robot, f"{arm_tag}_entity")
    # get the child links of the ee joint (i.e. wrist link)
    # links = [getattr(robot, f"{arm_tag}_ee").child_link]
    # get the child links of the gripper
    links = []
    links += [joint.child_link for joint, _, _ in getattr(robot, f"{arm_tag}_gripper")]
    links += [entity.find_link_by_name(n) for n in getattr(robot, f"{arm_tag}_fix_gripper_name", [])]
    # return the per_scene_id (a non-dual-arm embodiment loads the same URDF twice, so the two arms carry identical link names)
    return {link.entity.per_scene_id for link in links if link is not None}


def compute_tcp_wrench(task_env):
    """Net contact wrench on each arm's end effector, resolved in the **world** frame.

    SAPIEN reports per-contact-point *impulses* accumulated over the last physics step and
    applied to ``contact.bodies[0]``; dividing by the scene timestep turns them into the
    average force over that step. Force is summed over every contact point on the
    end-effector links and torque is taken about the TCP origin (``sum (p - p_tcp) x f``).
    Only the moment arm is TCP-relative — the components themselves stay in world axes, so a
    trace is comparable across steps even as the gripper rotates.

    Returns ``{"left": (force(3,), torque(3,)), "right": ...}`` in N and N*m. An arm touching
    nothing reads as all zeros — free-space motion produces no wrench here, only contact does.
    """
    contacts = task_env.scene.get_contacts()
    dt = task_env.scene.get_timestep()
    robot = task_env.robot
    out = {}
    for arm_tag in ARM_TAGS:
        link_ids = _ee_link_ids(robot, arm_tag)
        tcp_p = np.asarray(getattr(robot, f"get_{arm_tag}_tcp_pose")(), dtype=np.float64)[:3]
        force = np.zeros(3)
        torque = np.zeros(3)
        for contact in contacts:
            ids = [body.entity.per_scene_id for body in contact.bodies]
            # impulses are applied to the first actor
            if ids[0] in link_ids and ids[1] in link_ids:
                continue  # self-contact inside the assembly: internal, cancels out
            if ids[0] in link_ids:
                sign = 1.0  # impulses applied on the robot
            elif ids[1] in link_ids: # impulses applied to another object
                sign = -1.0
            else:
                continue
            for point in contact.points:
                f = sign * np.asarray(point.impulse, dtype=np.float64) / dt
                force += f
                torque += np.cross(np.asarray(point.position, dtype=np.float64) - tcp_p, f)
        out[arm_tag] = (force, torque)
    return out


def tcp_wrench_vector(task_env):
    """`compute_tcp_wrench` flattened to one ``(6,)`` vector per arm, in WRENCH_COMPONENTS order.

    Returns ``{"left": (6,), "right": (6,)}`` — the layout both the debug ``.npz`` logs and the
    rollout dataset's wrench columns store.
    """
    return {
        arm: np.concatenate([force, torque])
        for arm, (force, torque) in compute_tcp_wrench(task_env).items()
    }


def stack_step_wrench(step_wrench, num_steps):
    """Stack the per-step samples of one action chunk into ``{arm: (num_steps, 6)}``.

    ``step_wrench`` is what ``_base_task.pop_step_wrench`` collected while the chunk executed:
    one ``tcp_wrench_vector`` dict per ``take_action``. Padded to ``num_steps`` (i.e.
    ``pi0_step``) with **NaN**, so the result has one fixed shape whether or not the chunk ran
    to completion — only an episode's last chunk is ever short, since ``take_action`` stops
    stepping once the task succeeds or ``step_lim`` is hit. NaN rather than zero, because zero
    is a meaningful reading (the arm touching nothing); consumers that cannot take NaN should
    map it to zero explicitly.

    Returns ``{}`` for an empty log, so callers can tell "no samples" from "samples that were
    all zero".
    """
    if not step_wrench:
        return {}
    stacked = {}
    for arm in step_wrench[0]:
        samples = np.asarray([sample[arm] for sample in step_wrench], dtype=np.float32)[:num_steps]
        padded = np.full((num_steps, len(WRENCH_COMPONENTS)), np.nan, dtype=np.float32)
        padded[:len(samples)] = samples
        stacked[arm] = padded
    return stacked
