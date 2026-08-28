"""Per-*task* episode subset selection for LeRobot datasets.

Shared by `scripts/train.py` and `scripts/compute_norm_stats.py` so that both use an
identical, deterministic subset when `episodes_per_task` is set. The selection is a pure
function of (repo_id, episodes_per_task, seed): given the same three inputs it always
returns the same episodes, which is what keeps the norm stats computed by one script
consistent with the episodes trained on by the other.

"Task" here means a *RoboTwin task* (e.g. `beat_block_hammer`), NOT a language
instruction. Each episode in the LeRobot dataset carries a single randomly-sampled
natural-language instruction as its `tasks[0]`, and one RoboTwin task expands into many
different instructions (see `description/task_instruction/<task>.json`). So the dataset's
own `task_index` is an *instruction* index, not a task index, and grouping by it would
select `episodes_per_task` episodes per instruction — far more than intended.

To recover the real task for each episode we exploit two facts:

  1. `description/task_instruction/<task>.json` lists the instruction *templates* for each
     task (with `{...}` placeholders that get filled with object descriptions from
     `description/objects_description/*.json` and `the left/right arm`). Compiling these
     into regexes lets us match a resolved instruction back to the task(s) it could have
     come from. Many episodes match exactly one task ("anchors"); some match several
     (sibling tasks share phrasings).

  2. The converter (`convert_aloha_data_to_lerobot_robotwin.py`) walks the raw data with
     `os.walk`, which emits one task directory's episodes fully before the next. So
     episodes for a given task occupy a *contiguous* block of episode indices.

Combining the two — anchor matches to identify each contiguous block's task, contiguity to
fill in the ambiguous episodes between blocks — reconstructs the episode->task map. The few
genuinely-ambiguous episodes at a boundary between two near-identical sibling tasks are
assigned to the nearer side; this is deterministic and affects at most a handful of
episodes.
"""

import collections
import functools
import json
import logging
import os
import pathlib
import random
import re


def select_episodes_per_task(repo_id: str, episodes_per_task: int, seed: int) -> tuple[list[int], dict]:
    """Randomly pick up to `episodes_per_task` episodes for each RoboTwin task in the dataset.

    Maps every episode to the RoboTwin task it belongs to (via `description/task_instruction`
    + episode-index contiguity, see the module docstring), then samples (without replacement,
    seeded by `seed`) up to `episodes_per_task` episodes from each task. Tasks with fewer
    episodes contribute all of theirs. Returns the sorted flat list of selected episode
    indices plus a per-task breakdown (for logging), keyed by task name.
    """
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    # Instruction (as stored in the dataset) for every episode, ordered by episode index.
    ordered_eps = sorted(meta.episodes.items())
    ep_indices = [ep_idx for ep_idx, _ in ordered_eps]
    instructions = []
    for ep_idx, ep in ordered_eps:
        ep_tasks = ep.get("tasks") or []
        if not ep_tasks:
            logging.warning(f"Episode {ep_idx} has no instruction; it will inherit its neighbours' task.")
            instructions.append(None)
        else:
            instructions.append(ep_tasks[0].strip())

    ep_task_names = assign_episode_tasks(instructions)

    task_to_eps: dict[str, list[int]] = collections.defaultdict(list)
    for ep_idx, task_name in zip(ep_indices, ep_task_names):
        task_to_eps[task_name].append(ep_idx)

    rng = random.Random(seed)
    per_task: dict[str, dict] = {}
    for task_name in sorted(task_to_eps):
        available = sorted(task_to_eps[task_name])
        k = min(episodes_per_task, len(available))
        chosen = sorted(rng.sample(available, k))
        per_task[task_name] = {
            "task": task_name,
            "num_available": len(available),
            "num_selected": k,
            "selected_episodes": chosen,
        }

    selected = sorted(ep for info in per_task.values() for ep in info["selected_episodes"])
    logging.info(
        "Resolved %d episodes into %d RoboTwin tasks: %s",
        len(instructions),
        len(per_task),
        {name: info["num_available"] for name, info in per_task.items()},
    )
    return selected, per_task


def assign_episode_tasks(instructions: list[str | None]) -> list[str]:
    """The RoboTwin task name of every episode, given their instructions in episode-index order.

    The module docstring explains how: regex-match the resolved instructions back to the
    templates in `description/task_instruction`, then use episode-index contiguity to settle the
    ones several tasks could have produced. Shared with inference-time demo retrieval
    (`openpi.policies.demo_retrieval`), which needs the same episode -> task map to pick
    demonstrations of the task being evaluated -- whatever instruction each of them happens to
    carry.
    """
    return _assign_episodes_to_tasks(instructions, _task_matchers())


def _assign_episodes_to_tasks(instructions: list[str | None], matchers: list[tuple[str, re.Pattern]]) -> list[str]:
    """Assign every episode (given in episode-index order) to a RoboTwin task name.

    Uses regex "anchors" (episodes matching exactly one task) to label contiguous blocks,
    then fills the ambiguous gaps between anchors of different tasks by contiguity, breaking
    remaining ties toward the nearer confidently-labelled side.
    """
    n = len(instructions)
    # Candidate task set for each episode.
    candidates: list[set[str]] = []
    for ins in instructions:
        if ins is None:
            candidates.append(set())
        else:
            candidates.append({name for name, rx in matchers if rx.match(ins)})

    # Anchors: episodes whose instruction matches exactly one task.
    anchors = [(i, next(iter(c))) for i, c in enumerate(candidates) if len(c) == 1]
    if not anchors:
        raise ValueError(
            "Could not match any dataset instruction to a task in description/task_instruction. "
            "The dataset instructions do not look like RoboTwin instructions, or the task "
            "templates are out of sync with the dataset."
        )

    # Merge consecutive same-task anchors into runs (one run per contiguous task block).
    runs: list[list] = []  # [task_name, first_anchor_idx, last_anchor_idx]
    for i, name in anchors:
        if runs and runs[-1][0] == name:
            runs[-1][2] = i
        else:
            runs.append([name, i, i])

    assign: list[str | None] = [None] * n
    # Episodes spanned by (and inside) each anchor run take that run's task.
    for name, a0, a1 in runs:
        for i in range(a0, a1 + 1):
            assign[i] = name
    # Head before the first anchor and tail after the last inherit the nearest run.
    for i in range(0, runs[0][1]):
        assign[i] = runs[0][0]
    for i in range(runs[-1][2] + 1, n):
        assign[i] = runs[-1][0]
    # Gaps between run k (task L) and run k+1 (task R): the true boundary lies somewhere in
    # here. Prefer whichever side an episode's candidate set uniquely allows; otherwise (both
    # or neither) fall back to the nearer anchor.
    for k in range(len(runs) - 1):
        left, right = runs[k][0], runs[k + 1][0]
        lo, hi = runs[k][2] + 1, runs[k + 1][1]  # [lo, hi) exclusive
        for i in range(lo, hi):
            c = candidates[i]
            if left in c and right not in c:
                assign[i] = left
            elif right in c and left not in c:
                assign[i] = right
            else:
                assign[i] = left if (i - lo) < (hi - i) else right

    return [a for a in assign]  # no None remains: every index was covered above


@functools.lru_cache(maxsize=1)
def _task_matchers() -> list[tuple[str, re.Pattern]]:
    """Compile one regex per (task, instruction template) in description/task_instruction.

    Placeholders in the templates are replaced so the regex matches the *resolved* instruction
    stored in the dataset: arm placeholders (`{a}`) become `the (left|right) arm`, and object
    placeholders (`{A}`) become either `the <object description>` (using the real descriptions
    from description/objects_description) or a permissive wildcard as a fallback.
    """
    desc_dir = _description_dir()
    obj_alt = _object_description_alternatives(desc_dir)

    def build_rx(template: str) -> re.Pattern:
        parts = re.split(r"(\{[^}]+\})", template)
        pieces = []
        for p in parts:
            m = re.fullmatch(r"\{([^}]+)\}", p)
            if m:
                key = m.group(1)
                if len(key) == 1 and key.islower():  # arm placeholder, e.g. {a}
                    pieces.append(r"the (?:left|right) arm")
                else:  # object placeholder, e.g. {A}
                    pieces.append(r"(?:the " + obj_alt + r"|.+?)")
            else:
                pieces.append(re.escape(p))
        return re.compile(r"^" + "".join(pieces) + r"$", re.IGNORECASE)

    matchers: list[tuple[str, re.Pattern]] = []
    task_dir = desc_dir / "task_instruction"
    for task_file in sorted(task_dir.glob("*.json")):
        task_name = task_file.stem
        data = json.loads(task_file.read_text())
        for key in ("seen", "unseen"):
            for template in data.get(key, []) or []:
                matchers.append((task_name, build_rx(template)))
    if not matchers:
        raise FileNotFoundError(f"No task instruction templates found under {task_dir}.")
    return matchers


def _object_description_alternatives(desc_dir: pathlib.Path) -> str:
    """A regex alternation of every object description string (longest first, so the regex
    engine prefers the most specific match)."""
    descs: set[str] = set()
    obj_dir = desc_dir / "objects_description"
    for obj_file in obj_dir.glob("*.json"):
        data = json.loads(obj_file.read_text())
        for key in ("seen", "unseen"):
            for d in data.get(key, []) or []:
                descs.add(d)
    if not descs:
        return r".+?"
    return "(?:" + "|".join(re.escape(d) for d in sorted(descs, key=len, reverse=True)) + ")"


def _description_dir() -> pathlib.Path:
    """Locate the RoboTwin `description/` directory (holds task_instruction/ and
    objects_description/). Honours ROBOTWIN_DESCRIPTION_DIR, else searches upward from here."""
    env = os.environ.get("ROBOTWIN_DESCRIPTION_DIR")
    if env:
        p = pathlib.Path(env).expanduser().resolve()
        if (p / "task_instruction").is_dir():
            return p
        raise FileNotFoundError(f"ROBOTWIN_DESCRIPTION_DIR={env} has no task_instruction/ subdir.")
    for parent in pathlib.Path(__file__).resolve().parents:
        cand = parent / "description"
        if (cand / "task_instruction").is_dir():
            return cand
    raise FileNotFoundError(
        "Could not locate RoboTwin's description/task_instruction directory. "
        "Set ROBOTWIN_DESCRIPTION_DIR to the repo's description/ folder."
    )
