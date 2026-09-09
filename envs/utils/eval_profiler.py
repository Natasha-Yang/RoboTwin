"""Where an eval run's wall clock actually goes.

An eval is hours long and almost none of that is the policy. This is a deterministic span
profiler that attributes it: the simulator's renders, the motion planner, the physics loop, the
contact queries, the policy forward, the critic's TD updates, the expert feasibility gate and
the periodic held-out evaluations, each with a count so a cost can be read as "per control step"
rather than as a bare total.

It is **off unless asked for** (`profile` in `deploy_policy.yml`, or `--profile true` through
`eval.sh`'s pass-through overrides). Off means not installed: `install()` is never called, no
method is wrapped, and the run is byte-identical to one that has never heard of this module.

Two things the numbers mean, both of which would otherwise mislead:

- **SAPIEN renders asynchronously.** `take_picture()` queues the work and `get_picture()` waits
  for it, so the GPU time lands on `camera.get_rgb` / `camera.get_depth`, not on
  `camera.update_picture`. Read the `render.*` and `camera.*` spans as one group.
- **JAX compiles on the first call of each shape**, which for the sampler is tens of seconds.
  That is why every span carries `first` and `max` alongside the mean: a large `first` is
  compilation, not per-step cost, and it is excluded from `mean_ex1`.

`total` is wall time inside a span including its children; `self` excludes them, so the `self`
column sums to the profiled wall clock and is what to rank by. A span entered recursively is
counted once, at its outermost entry, so nesting cannot inflate `total`.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections import OrderedDict
from pathlib import Path


class _Stat:
    """One span's accumulated cost."""

    __slots__ = ("name", "count", "total", "self_time", "first", "max", "phase_self",
                 "phase_count")

    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total = 0.0
        self.self_time = 0.0
        self.first = None
        self.max = 0.0
        self.phase_self = {}
        # Calls made in each phase, which is what makes a ratio like "physics steps per control
        # step" honest: the expert feasibility gate steps the scene too, and it has no control
        # steps at all, so a ratio over the run's total call count reads far too high.
        self.phase_count = {}

    def as_dict(self):
        return {
            "name": self.name,
            "count": self.count,
            "total_s": self.total,
            "self_s": self.self_time,
            "first_s": self.first,
            "max_s": self.max,
            "self_by_phase_s": dict(self.phase_self),
            "calls_by_phase": dict(self.phase_count),
        }


class _Frame:
    __slots__ = ("name", "start", "child")

    def __init__(self, name, start):
        self.name = name
        self.start = start
        self.child = 0.0


class Profiler:
    """Span timings, counters and phases for one run.

    Disabled by default; `span()` then returns a shared null context and costs one attribute
    lookup, so the instrumentation can stay in place on the hot paths.
    """

    def __init__(self):
        self.enabled = False
        self.stats = OrderedDict()
        self.counters = OrderedDict()
        self._stack = []
        self._depth = {}
        # Anything not inside an explicit `phase()`: startup, the per-episode `setup_demo`,
        # the video pipe, the csv / W&B / critic-checkpoint bookkeeping between episodes.
        self._phase = "other"
        self.phase_wall = OrderedDict()
        self._phase_started = None
        self.t0 = None
        self._installed = []
        self._null = contextlib.nullcontext()
        #: Wall clock spent before the seed loop started (model load + the sampler's first JAX
        #: compilation). Held apart so the per-control-step budget is over the loop alone.
        self.startup_s = 0.0

    # ----- lifecycle -----

    def start(self):
        self.enabled = True
        self.t0 = time.perf_counter()
        self._phase_started = self.t0

    def reset(self):
        self.stats.clear()
        self.counters.clear()
        self._stack.clear()
        self._depth.clear()
        self.phase_wall.clear()
        self.t0 = time.perf_counter()
        self._phase_started = self.t0

    def mark_startup_done(self):
        """Everything up to here is one-off startup, not per-episode cost."""
        if self.enabled:
            self.startup_s = self.elapsed

    @property
    def elapsed(self):
        return 0.0 if self.t0 is None else time.perf_counter() - self.t0

    # ----- measurement -----

    @contextlib.contextmanager
    def _span(self, name):
        stack = self._stack
        depth = self._depth
        d = depth.get(name, 0)
        depth[name] = d + 1
        frame = _Frame(name, time.perf_counter())
        stack.append(frame)
        try:
            yield
        finally:
            stack.pop()
            depth[name] = d
            elapsed = time.perf_counter() - frame.start
            own = elapsed - frame.child
            if stack:
                stack[-1].child += elapsed
            st = self.stats.get(name)
            if st is None:
                st = self.stats[name] = _Stat(name)
            st.self_time += own
            st.phase_self[self._phase] = st.phase_self.get(self._phase, 0.0) + own
            # Recursion: only the outermost entry contributes to count/total, so a span that
            # contains itself is one call of that span, not two.
            if d == 0:
                st.count += 1
                st.phase_count[self._phase] = st.phase_count.get(self._phase, 0) + 1
                st.total += elapsed
                if st.first is None:
                    st.first = elapsed
                if elapsed > st.max:
                    st.max = elapsed

    def span(self, name):
        if not self.enabled:
            return self._null
        return self._span(name)

    def count(self, name, n=1):
        if self.enabled:
            self.counters[name] = self.counters.get(name, 0) + n

    @contextlib.contextmanager
    def phase(self, name):
        """Mark the coarse activity the spans below belong to (rollout / expert_check / ...)."""
        if not self.enabled:
            yield
            return
        prev, prev_started = self._phase, self._phase_started
        now = time.perf_counter()
        self.phase_wall[prev] = self.phase_wall.get(prev, 0.0) + (now - prev_started)
        self._phase, self._phase_started = name, now
        try:
            yield
        finally:
            end = time.perf_counter()
            self.phase_wall[name] = self.phase_wall.get(name, 0.0) + (end - self._phase_started)
            self._phase, self._phase_started = prev, end

    def _close_phase(self):
        if self._phase_started is not None:
            now = time.perf_counter()
            self.phase_wall[self._phase] = (self.phase_wall.get(self._phase, 0.0)
                                            + (now - self._phase_started))
            self._phase_started = now

    # ----- patching -----

    def wrap(self, owner, attr, name):
        """Time `owner.attr` under span `name`. Idempotent, and recorded so `uninstall` works."""
        if not self.enabled:
            return False
        original = getattr(owner, attr, None)
        if original is None or getattr(original, "_profiled_as", None) is not None:
            return False
        prof = self

        def wrapper(*a, __orig=original, __name=name, **kw):
            with prof._span(__name):
                return __orig(*a, **kw)

        wrapper._profiled_as = name
        wrapper.__name__ = getattr(original, "__name__", attr)
        wrapper.__doc__ = getattr(original, "__doc__", None)
        try:
            setattr(owner, attr, wrapper)
        except Exception:
            return False
        self._installed.append((owner, attr, original))
        return True

    def uninstall(self):
        for owner, attr, original in reversed(self._installed):
            try:
                setattr(owner, attr, original)
            except Exception:
                pass
        self._installed.clear()

    # ----- reporting -----

    def report(self, extra=None):
        self._close_phase()
        wall = self.elapsed
        lines = []
        w = lines.append
        loop = max(wall - self.startup_s, 1e-9)
        w("=" * 96)
        w(f"EVAL PROFILE  --  {wall:.1f}s wall ({wall / 60:.1f} min)")
        if self.startup_s:
            w(f"  startup (model load + first JAX compile): {self.startup_s:.1f}s"
              f"  |  seed loop: {loop:.1f}s ({loop / 60:.1f} min)")
        w("=" * 96)

        if self.counters:
            w("")
            w("counters")
            for key, val in sorted(self.counters.items()):
                w(f"  {key:<34} {val:>12,}")

        control_steps = self.counters.get("control_steps", 0)
        # The physics tick is already counted: one `sim.scene_step` span per `scene.step()`.
        physics_steps = self.stats["sim.scene_step"].count if "sim.scene_step" in self.stats else 0

        if self.phase_wall:
            w("")
            w("phase                            wall_s      %   (\"other\" = startup and the "
              "per-episode bookkeeping between the marked phases)")
            w("-" * 52)
            for key, val in sorted(self.phase_wall.items(), key=lambda kv: -kv[1]):
                w(f"  {key:<28} {val:>10.1f} {100 * val / wall if wall else 0:>6.1f}")

        rows = sorted(self.stats.values(), key=lambda s: -s.self_time)
        w("")
        w("span                              calls     self_s    %wall      total_s"
          "    mean_ms  mean_ms_ex1     max_ms    first_ms")
        w("-" * 130)
        for st in rows:
            mean = st.total / st.count if st.count else 0.0
            first = st.first or 0.0
            mean_ex1 = ((st.total - first) / (st.count - 1)) if st.count > 1 else mean
            w(f"  {st.name:<30} {st.count:>7,} {st.self_time:>10.2f} "
              f"{100 * st.self_time / wall if wall else 0:>7.1f} {st.total:>12.2f} "
              f"{1000 * mean:>10.2f} {1000 * mean_ex1:>12.2f} "
              f"{1000 * st.max:>10.1f} {1000 * first:>11.1f}")
        accounted = sum(s.self_time for s in rows)
        w("-" * 130)
        w(f"  {'accounted (sum of self)':<30} {'':>7} {accounted:>10.1f} "
          f"{100 * accounted / wall if wall else 0:>7.1f}")
        w(f"  {'unaccounted':<30} {'':>7} {wall - accounted:>10.1f} "
          f"{100 * (wall - accounted) / wall if wall else 0:>7.1f}")

        if len(self.phase_wall) > 1:
            w("")
            w("calls by phase (top spans)      " + "".join(f"{k:>14}" for k in self.phase_wall))
            w("-" * (32 + 14 * len(self.phase_wall)))
            for st in rows[:12]:
                w(f"  {st.name:<30}"
                  + "".join(f"{st.phase_count.get(k, 0):>14,}" for k in self.phase_wall))

        if control_steps:
            w("")
            w(f"per control step ({control_steps:,} chunks executed, over the seed loop only)")
            w("  span                             ms/control_step")
            w("  " + "-" * 48)
            for st in rows[:18]:
                w(f"  {st.name:<32} {1000 * st.self_time / control_steps:>14.2f}")
            w(f"  {'TOTAL (seed loop wall)':<32} {1000 * loop / control_steps:>14.2f}")

        w("")
        w("notes")
        w("  * `self` excludes nested spans and sums to the profiled wall clock; rank by it.")
        w("  * SAPIEN renders async: take_picture() queues, get_picture() waits, so the GPU")
        w("    time lands on camera.get_rgb / camera.get_depth, not camera.update_picture.")
        w("  * A large `first_ms` on a policy/critic span is JAX compilation, not per-step")
        w("    cost -- compare `mean_ms_ex1`. A compile can also land *mid-run* (the critic's")
        w("    TD update does not compile until the replay buffer first reaches")
        w("    `start_training`), and then it shows up in `max_ms` rather than in `first_ms`,")
        w("    with `mean_ms_ex1` still carrying it.")
        def rollout_calls(name):
            st = self.stats.get(name)
            return st.phase_count.get("rollout", 0) if st else 0

        if control_steps and rollout_calls("env.get_obs"):
            per = rollout_calls("env.get_obs") / control_steps
            w(f"  * env.get_obs runs {per:.1f}x per control step.")
            if per > 1.5:
                w("    The policy's action loop renders a full observation after every")
                w("    primitive step, and only the last one conditions the next chunk")
                w("    (policy/pi05/deploy_policy.py::eval).")
        if control_steps and rollout_calls("sim.scene_step"):
            ticks = rollout_calls("sim.scene_step")
            primitive = rollout_calls("env.take_action")
            w(f"  * {ticks / control_steps:.0f} physics steps per control step in the rollouts"
              + (f" -- {primitive / control_steps:.0f} take_action calls (`pi0_step`) of "
                 f"{ticks / primitive:.0f}" if primitive else "")
              + " each,")
            w("    which is one TOPP-retimed trajectory per primitive step. Every one of those")
            w("    ticks renders and checks success, and under `data_type.wrench` runs a")
            w("    contact query as well -- so those three costs scale with the trajectory")
            w("    length, not with the number of chunks.")
            if physics_steps > ticks:
                w(f"    ({physics_steps:,} scene.step() calls in total; the rest are the expert")
                w("    feasibility gate's and the spawn settle in setup_demo.)")
        for line in (extra or []):
            w(f"  * {line}")
        w("=" * 96)
        return "\n".join(lines)

    def as_dict(self):
        self._close_phase()
        return {
            "wall_s": self.elapsed,
            "startup_s": self.startup_s,
            "counters": dict(self.counters),
            "phase_wall_s": dict(self.phase_wall),
            "spans": [s.as_dict() for s in
                      sorted(self.stats.values(), key=lambda s: -s.self_time)],
        }

    def dump(self, save_dir, stem="_profile", extra=None):
        """Write `<stem>.txt` and `<stem>.json` into `save_dir`; return the text report."""
        text = self.report(extra=extra)
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
            (save_dir / f"{stem}.json").write_text(
                json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        except Exception as exc:  # a profile is never worth failing a finished run for
            print(f"[profile] could not write the report: {exc}")
        return text


#: The one profiler an eval run uses. Disabled until `install()`.
PROFILER = Profiler()


def parse_flag(value):
    """`profile:` as it arrives from yaml or from a `--profile` CLI override.

    Returns `("off"|"spans"|"cprofile"|"both")`. A CLI override reaches the driver as a string
    (`eval()` fails on `true`, so the parser keeps the text), and `bool("false")` is True --
    exactly backwards -- so the strings are resolved here rather than by truthiness. Anything
    unrecognised raises, so a typo cannot silently leave profiling off.
    """
    if value is None or value is False:
        return "off"
    if value is True:
        return "spans"
    text = str(value).strip().lower()
    if text in ("", "false", "no", "off", "0", "none", "null"):
        return "off"
    if text in ("true", "yes", "on", "1", "spans", "span"):
        return "spans"
    if text in ("cprofile", "c", "pstats"):
        return "cprofile"
    if text in ("both", "all", "full"):
        return "both"
    raise ValueError(
        f"profile: expected one of false / true / cprofile / both, got {value!r}")


def install(mode="spans"):
    """Wrap the simulator, policy and critic hot paths. Returns the profiler, or None if off.

    Everything here is patched on a *class*, so it applies to the env and policy objects the run
    creates later as well as the ones it already has. The critic is the exception -- it does not
    exist until the first observation (its shapes follow the embodiment), so `PI0._init_critic`
    is wrapped to instrument whatever it builds.

    Each patch is attempted independently: a policy that has no critic, a SAPIEN version that
    moved a method, an env without a camera rig -- none of them should stop the rest from being
    measured, so a failure is a missing row in the report rather than a failed run.
    """
    if mode == "off":
        return None
    prof = PROFILER
    prof.start()
    wrapped, missed = [], []

    def attempt(owner_path, attr, name):
        owner = _resolve(owner_path)
        if owner is None or not prof.wrap(owner, attr, name):
            missed.append(f"{owner_path}.{attr}")
        else:
            wrapped.append(name)

    # --- simulator: the env's own loops ---
    attempt("envs._base_task:Base_Task", "get_obs", "env.get_obs")
    attempt("envs._base_task:Base_Task", "take_action", "env.take_action")
    attempt("envs._base_task:Base_Task", "_update_render", "render.update_render")
    attempt("envs._base_task:Base_Task", "close_env", "env.close_env")
    attempt("envs._base_task:Base_Task", "_init_task_env_", "env.init_task_env")
    # `setup_demo`, `play_once` and `check_success` are *per task* -- `envs/<task>.py` defines
    # each of them and `Base_Task`'s are empty stubs -- so wrapping the base class here would
    # wrap something nothing calls. `install_env` does those, once the driver has built the
    # task object and its concrete class is known.
    attempt("envs._base_task:Base_Task", "_accumulate_step_wrench", "wrench.contact_query")
    attempt("envs._base_task:Base_Task", "_log_step_wrench", "wrench.contact_query")
    attempt("envs._base_task:Base_Task", "take_dense_action", "expert.take_dense_action")
    attempt("envs._base_task:Base_Task", "together_move_to_pose", "expert.move_to_pose")

    # --- simulator: SAPIEN itself. `Scene` is a python wrapper subclass, so this is a class
    # patch like the rest; `step` is the physics tick and `update_render` the pose sync. ---
    attempt("sapien.wrapper.scene:Scene", "step", "sim.scene_step")
    attempt("sapien.wrapper.scene:Scene", "update_render", "sim.update_render")

    # --- rendering: `take_picture` only queues the work, so most of the GPU time surfaces in
    # the getters that wait for it. Both are timed; read them together. ---
    attempt("envs.camera.camera:Camera", "update_picture", "camera.take_picture")
    attempt("envs.camera.camera:Camera", "get_rgb", "camera.get_rgb")
    attempt("envs.camera.camera:Camera", "get_depth", "camera.get_depth")
    attempt("envs.camera.camera:Camera", "get_config", "camera.get_config")
    attempt("envs.camera.camera:Camera", "get_segmentation", "camera.get_segmentation")
    attempt("envs.camera.camera:Camera", "get_pcd", "camera.get_pcd")
    attempt("envs.camera.camera:Camera", "get_observer_rgb", "camera.get_observer_rgb")

    # --- motion planning: curobo for the expert's waypoints, and the TOPP retime `take_action`
    # runs per primitive step (one call per arm). ---
    attempt("envs.robot.planner:MplibPlanner", "plan_pose", "plan.mplib_plan")
    attempt("envs.robot.planner:MplibPlanner", "plan_screw", "plan.mplib_plan")
    attempt("envs.robot.planner:MplibPlanner", "plan_path", "plan.mplib_plan")
    attempt("envs.robot.planner:CuroboPlanner", "plan_path", "plan.curobo")
    attempt("envs.robot.planner:CuroboPlanner", "plan_grippers", "plan.curobo")
    _wrap_mplib_topp(prof, wrapped, missed)

    # --- the policy adapter. Imported first, and deliberately: `pi_model` and `openpi` are
    # only on `sys.path` because importing it puts them there (`deploy_policy.py` does the
    # `sys.path` surgery at module scope), so the two blocks below would find nothing to wrap
    # if this ran after them. The two functions are patched as module globals because
    # `deploy_policy.eval` looks them up that way. ---
    attempt("pi05.deploy_policy", "critic_obs_modalities", "critic.obs_modalities")
    attempt("pi05.deploy_policy", "encode_obs", "policy.encode_obs")

    # --- policy: `Policy.infer` is the honest boundary. jax dispatch is async, but `infer`
    # converts the outputs to numpy before returning, which waits for the device. ---
    attempt("openpi.policies.policy:Policy", "infer", "policy.infer")
    attempt("pi_model:PI0", "get_action", "policy.get_action")
    attempt("pi_model:PI0", "update_observation_window", "policy.update_obs_window")
    attempt("pi_model:PI0", "_refresh_demo_proposals", "retrieval.propose")
    attempt("pi_model:PI0", "_critic_extra_obs", "critic.extra_obs")
    attempt("pi_model:PI0", "set_language", "policy.set_language")

    # --- debug output (`debug: true` in the task config), which is not free ---
    attempt("envs.utils.debug_vis:RolloutFrameLog", "record", "debug.frame_log")
    attempt("envs.utils.debug_vis:RolloutFrameLog", "flush", "debug.flush")
    attempt("envs.utils.debug_vis:TCPWrenchRecorder", "record", "debug.wrench_record")
    attempt("envs.utils.debug_vis:TCPWrenchRecorder", "flush", "debug.flush")
    attempt("envs.utils.debug_vis:QValueRecorder", "record", "debug.q_record")
    attempt("envs.utils.debug_vis:QValueRecorder", "flush", "debug.flush")
    attempt("envs.utils.debug_vis:DemoRetrievalRecorder", "record", "debug.retrieval_record")
    attempt("envs.utils.debug_vis:DemoRetrievalRecorder", "flush", "debug.flush")

    # --- critic: built lazily, so instrument what `_init_critic` produces. ---
    _wrap_critic_factory(prof, wrapped, missed)

    print(f"\033[95m[profile]\033[0m ON (mode={mode}); {len(wrapped)} hook(s) installed")
    if missed:
        print(f"\033[95m[profile]\033[0m not found (skipped): {', '.join(sorted(set(missed)))}")
    return prof


def install_env(task_env):
    """Instrument the concrete task class, once the driver has built the env.

    Three of the methods that matter most are defined by `envs/<task>.py` rather than by
    `Base_Task` -- `setup_demo` (scene construction and the spawn's stability settle),
    `play_once` (the scripted expert, i.e. the whole cost of the feasibility gate) and
    `check_success`, which `take_action` calls after *every physics step*. `Base_Task`'s
    versions are empty stubs the subclass shadows, so there is nothing useful to wrap until
    here. Safe to call more than once; the wrappers are idempotent.
    """
    prof = PROFILER
    if not prof.enabled or task_env is None:
        return
    cls = type(task_env)
    for attr, name in (("setup_demo", "env.setup_demo"),
                       ("play_once", "expert.play_once"),
                       ("check_success", "env.check_success"),
                       ("step_reward", "env.step_reward")):
        prof.wrap(cls, attr, name)


def _resolve(path):
    """`"pkg.mod:Attr"` -> the attribute, or None if the module or attribute is not there.

    Import failures are expected and are not errors: `pi_model` only exists on `sys.path` for a
    pi05 run, `sapien`'s wrapper module has moved between versions, and a baseline eval has no
    critic package at all. A hook that cannot be placed is a missing row in the report.
    """
    module_name, _, attr = path.partition(":")
    try:
        import importlib
        module = importlib.import_module(module_name)
    except Exception:
        return None
    owner = module
    for part in attr.split(".") if attr else []:
        owner = getattr(owner, part, None)
        if owner is None:
            return None
    return owner


def _wrap_mplib_topp(prof, wrapped, missed):
    """Time the per-primitive-step TOPP retime, which is an *instance* attribute.

    `MplibPlanner.__init__` binds `self.TOPP = self.planner.TOPP` -- a bound method of the
    underlying mplib planner, captured at construction -- so neither `MplibPlanner.TOPP` nor
    `mplib.Planner.TOPP` is what `take_action` actually calls. The constructor is wrapped
    instead, and instruments the attribute it just bound. It runs twice per primitive step (once
    per arm), so at `pi0_step: 10` that is 20 calls per control step.
    """
    planner_cls = _resolve("envs.robot.planner:MplibPlanner")
    if planner_cls is None or getattr(planner_cls.__init__, "_profiled_as", None) is not None:
        missed.append("envs.robot.planner:MplibPlanner.TOPP")
        return
    original = planner_cls.__init__

    def wrapper(self, *a, **kw):
        original(self, *a, **kw)
        prof.wrap(self, "TOPP", "plan.TOPP")

    wrapper._profiled_as = "plan.TOPP"
    planner_cls.__init__ = wrapper
    prof._installed.append((planner_cls, "__init__", original))
    wrapped.append("plan.TOPP")


def _wrap_critic_factory(prof, wrapped, missed):
    """Instrument the online critic as soon as the policy builds one.

    The critic's shapes follow the embodiment, so it does not exist until the first observation
    -- there is no class whose methods can be wrapped ahead of time, and the object itself is
    per-run. `PI0._init_critic` is wrapped instead, and instruments whatever it left on the
    policy. `stash` / `commit` / `train_step` / `q_values` are the four the rollout loop calls
    once per control step.
    """
    pi0 = _resolve("pi_model:PI0")
    if pi0 is None or getattr(pi0._init_critic, "_profiled_as", None) is not None:
        missed.append("pi_model:PI0._init_critic")
        return
    original = pi0._init_critic

    def wrapper(self, *a, **kw):
        with prof._span("critic.init"):
            out = original(self, *a, **kw)
        critic = getattr(self, "online_critic", None)
        if critic is not None:
            for attr, name in (("stash", "critic.stash"),
                               ("commit", "critic.commit"),
                               ("train_step", "critic.train_step"),
                               ("q_values", "critic.q_values"),
                               ("save", "critic.save"),
                               ("reset_episode", "critic.reset_episode")):
                prof.wrap(critic, attr, name)
        return out

    wrapper._profiled_as = "critic.init"
    pi0._init_critic = wrapper
    prof._installed.append((pi0, "_init_critic", original))
    wrapped.append("critic.init")


@contextlib.contextmanager
def cprofile_to(path, enabled=True, top=60):
    """Run the block under cProfile and write `<path>.pstats` plus a cumulative-time listing.

    The span profiler says which *phase* is expensive; this says which function. It is a real
    per-call cost (a few percent, more on python-heavy loops like the contact query), so it is
    a separate mode rather than something the span profiler turns on as well.
    """
    if not enabled:
        yield None
        return
    import cProfile
    import io
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    try:
        yield pr
    finally:
        pr.disable()
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            pr.dump_stats(str(path.with_suffix(".pstats")))
            buf = io.StringIO()
            stats = pstats.Stats(pr, stream=buf)
            buf.write("=== by cumulative time ===\n")
            stats.sort_stats("cumulative").print_stats(top)
            buf.write("\n=== by total (self) time ===\n")
            stats.sort_stats("tottime").print_stats(top)
            path.with_suffix(".txt").write_text(buf.getvalue(), encoding="utf-8")
            print(f"\033[95m[profile]\033[0m cProfile written to {path.with_suffix('.pstats')} "
                  f"(and .txt); inspect with "
                  f"`python -m pstats {path.with_suffix('.pstats')}` or snakeviz")
        except Exception as exc:
            print(f"[profile] could not write the cProfile output: {exc}")
