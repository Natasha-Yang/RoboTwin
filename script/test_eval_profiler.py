"""Self-test for `envs/utils/eval_profiler.py`, with no GPU and no simulator.

Run it after changing the profiler:

    python script/test_eval_profiler.py

It checks the three properties the report rests on -- nested spans do not double-count, a
recursive span counts once, and the `self` column sums to the profiled wall clock -- then drives
a stand-in for one eval episode (expert gate, control steps, physics loop, held-out evaluation)
through the real wrapping machinery and prints the report that shape produces. Timings are
`sleep`s scaled to the real per-call costs, so the printed table is what an eval of that shape
would look like, not the real thing.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs.utils.eval_profiler import PROFILER, Profiler, parse_flag


def test_flag_parsing():
    for value, want in [(None, "off"), (False, "off"), ("false", "off"), ("", "off"),
                        (True, "spans"), ("true", "spans"), ("1", "spans"),
                        ("cprofile", "cprofile"), ("both", "both")]:
        got = parse_flag(value)
        assert got == want, f"parse_flag({value!r}) = {got!r}, expected {want!r}"
    try:
        parse_flag("maybe")
    except ValueError:
        pass
    else:
        raise AssertionError("an unrecognised profile flag must raise, not default to off")
    print("ok  flag parsing (and a typo raises rather than silently disabling)")


def test_disabled_costs_nothing():
    prof = Profiler()
    with prof.span("x"):
        pass
    prof.count("y")
    with prof.phase("z"):
        pass
    prof.mark_startup_done()
    assert not prof.stats and not prof.counters and not prof.phase_wall
    assert prof.span("x") is prof._null, "a disabled span must not build a generator per call"
    print("ok  disabled profiler records nothing and allocates no context manager")


def test_accounting():
    """Nesting, recursion and the self/total split."""
    prof = Profiler()
    prof.start()

    class Env:
        def outer(self):
            time.sleep(0.02)
            self.inner()
            self.inner()

        def inner(self):
            time.sleep(0.01)

        def recur(self, n):
            time.sleep(0.005)
            if n:
                self.recur(n - 1)

    prof.wrap(Env, "outer", "env.outer")
    prof.wrap(Env, "inner", "env.inner")
    prof.wrap(Env, "recur", "env.recur")
    assert not prof.wrap(Env, "outer", "env.outer"), "wrapping twice must be a no-op"

    env = Env()
    with prof.phase("rollout"):
        env.outer()
    with prof.phase("holdout"):
        env.outer()
    env.recur(3)

    outer, inner, recur = (prof.stats[k] for k in ("env.outer", "env.inner", "env.recur"))
    assert outer.count == 2 and inner.count == 4
    assert abs(outer.total - 0.08) < 0.03, f"total should include children: {outer.total}"
    assert abs(outer.self_time - 0.04) < 0.03, f"self should exclude them: {outer.self_time}"
    # Four nested calls, one span: a recursive span is counted at its outermost entry only, so
    # `total` cannot exceed the wall clock.
    assert recur.count == 1, recur.count
    assert abs(recur.total - 0.02) < 0.03, recur.total
    accounted = sum(s.self_time for s in prof.stats.values())
    assert abs(accounted - prof.elapsed) < 0.05, (accounted, prof.elapsed)
    # A span's self time is split across the phases it ran in.
    assert abs(outer.phase_self["rollout"] - outer.phase_self["holdout"]) < 0.03

    prof.uninstall()
    assert not hasattr(Env.outer, "_profiled_as"), "uninstall must restore the original"
    print("ok  nesting, recursion, phase attribution, self==wall, uninstall")


def test_report_shape():
    """One episode's worth of the real call structure, on the shared profiler."""
    prof = PROFILER
    prof.reset()
    prof.enabled = True

    pi0_step, physics_per_step, control_steps = 10, 40, 6

    class Sim:
        def scene_step(self):
            time.sleep(0.00012)

        def update_render(self):
            time.sleep(0.00025)

        def check_success(self):
            time.sleep(0.00004)

        def contact_query(self):
            time.sleep(0.00022)

        def get_obs(self):
            time.sleep(0.006)

        def topp(self):
            time.sleep(0.0015)

        def take_action(self):
            self.topp()
            self.topp()
            for _ in range(physics_per_step):
                self.scene_step()
                self.contact_query()
                self.update_render()
                self.check_success()

        def infer(self):
            time.sleep(0.081)

        def play_once(self):
            time.sleep(0.4)

    sim = Sim()
    for attr, name in (("scene_step", "sim.scene_step"), ("update_render", "render.update_render"),
                       ("check_success", "env.check_success"), ("contact_query", "wrench.contact_query"),
                       ("get_obs", "env.get_obs"), ("topp", "plan.TOPP"),
                       ("take_action", "env.take_action"), ("infer", "policy.infer"),
                       ("play_once", "expert.play_once")):
        prof.wrap(sim, attr, name)

    prof.mark_startup_done()
    with prof.phase("expert_check"):
        sim.play_once()
    with prof.phase("rollout"):
        for _ in range(control_steps):
            prof.count("control_steps")
            sim.get_obs()          # the observation the chunk is conditioned on
            sim.infer()
            for _ in range(pi0_step):   # deploy_policy.eval's action loop
                sim.take_action()
                sim.get_obs()
    prof.count("episodes")

    assert prof.stats["env.get_obs"].count == control_steps * (pi0_step + 1)
    assert prof.stats["sim.scene_step"].count == control_steps * pi0_step * physics_per_step
    text = prof.report(extra=["synthetic: sleeps scaled to the real per-call costs"])
    assert "renders a full observation after every" in text, "the get_obs note should have fired"
    assert "physics steps per control step" in text
    prof.uninstall()
    prof.enabled = False
    print("ok  report renders, and the derived notes fire\n")
    print(text)


if __name__ == "__main__":
    test_flag_parsing()
    test_disabled_costs_nothing()
    test_accounting()
    test_report_shape()
