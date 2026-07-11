"""Per-task episode subset selection for LeRobot datasets.

Shared by `scripts/train.py` and `scripts/compute_norm_stats.py` so that both use an
identical, deterministic subset when `episodes_per_task` is set. The selection is a pure
function of (repo_id, episodes_per_task, seed): given the same three inputs it always
returns the same episodes, which is what keeps the norm stats computed by one script
consistent with the episodes trained on by the other.
"""

import collections
import logging
import random


def select_episodes_per_task(repo_id: str, episodes_per_task: int, seed: int) -> tuple[list[int], dict]:
    """Randomly pick up to `episodes_per_task` episodes for each task_index in the dataset.

    Groups every episode by the task_index it belongs to, then samples (without replacement,
    seeded by `seed`) up to `episodes_per_task` episodes from each group. Tasks with fewer
    episodes contribute all of theirs. Returns the sorted flat list of selected episode
    indices plus a per-task breakdown (for logging).
    """
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    # Group episode indices by their task_index. Each episode's metadata carries a `tasks`
    # list (one entry for RoboTwin's single-task episodes); map it to the task_index.
    task_to_eps: dict[int, list[int]] = collections.defaultdict(list)
    for ep_idx, ep in meta.episodes.items():
        ep_tasks = ep.get("tasks") or []
        if not ep_tasks:
            logging.warning(f"Episode {ep_idx} has no task; excluding it from per-task selection.")
            continue
        task_index = meta.task_to_task_index[ep_tasks[0]]
        task_to_eps[task_index].append(ep_idx)

    rng = random.Random(seed)
    per_task: dict[int, dict] = {}
    for task_index in sorted(task_to_eps):
        available = sorted(task_to_eps[task_index])
        k = min(episodes_per_task, len(available))
        chosen = sorted(rng.sample(available, k))
        per_task[task_index] = {
            "task": meta.tasks[task_index],
            "num_available": len(available),
            "num_selected": k,
            "selected_episodes": chosen,
        }

    selected = sorted(ep for info in per_task.values() for ep in info["selected_episodes"])
    return selected, per_task
