"""Startup guard for the assets `script/gen_task_assets.py` produces.

Not a task (leading underscore, like `_base_task.py`): task discovery is
`importlib.import_module(f"envs.{task_name}")`, so only the concrete task files are reachable.

Both failure modes this guards are **silent**, which is why it exists:

* A missing asset directory. `create_actor` only prints "is not exist model file!" and returns
  `None` (create_actor.py:534), and `collect_data.py`'s seed-search loop is **uncapped** -- so a
  missing asset spins forever instead of failing.
* A `model_data<id>.json` with no `scale`. Twenty-four of the shipped assets carry only
  `{stable, center, extents}`, and `create_actor` swallows the resulting KeyError in a bare
  `except` (create_actor.py:531): `scale` stays at its `(1,1,1)` default and `model_data`
  becomes `None`, so the mesh loads at its raw ~1.9 m size and every `get_contact_point` /
  `get_functional_point` returns `None`. That surfaces much later and very opaquely.

Called at module scope, which sits outside `class_decorator`'s try/except, so the raise aborts
with this message rather than being turned into "No such task".
"""

import json
from pathlib import Path

_OBJECTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "objects"

_FIX = ("`assets/` is gitignored, so these are generated and annotated rather than downloaded:\n"
        "    python script/gen_task_assets.py --verify")


def require_assets(*specs: tuple[str, int]) -> None:
    """Assert each (modelname, model_id) exists and carries a usable `model_data<id>.json`."""
    problems = []
    for name, model_id in specs:
        directory = _OBJECTS_DIR / name
        if not directory.exists():
            problems.append(f"{name}: directory missing")
            continue
        data_path = directory / f"model_data{model_id}.json"
        if not data_path.exists():
            problems.append(f"{name}/model_data{model_id}.json: missing")
            continue
        try:
            with open(data_path) as handle:
                data = json.load(handle)
        except Exception as exc:  # a malformed file becomes model_data=None downstream
            problems.append(f"{name}/model_data{model_id}.json: unreadable ({exc})")
            continue
        if "scale" not in data:
            problems.append(f"{name}/model_data{model_id}.json: no `scale` -- not yet annotated")
    if problems:
        raise FileNotFoundError("; ".join(problems) + ".\n" + _FIX)
