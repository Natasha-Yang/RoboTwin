"""Per-episode debug recorders for a policy rollout (task config `debug: true`).

Each one collects a per-control-step trace and, at the end of an episode, renders it beside
the rollout: `TCPWrenchRecorder` the end-effector contact wrench, `QValueRecorder` the critic's
Q for the chunk the policy sampled, plotted against the return the episode actually realized.
Both animate the same video, so the head-camera frames live once in `RolloutFrameLog`, and the
matplotlib/GIF plumbing is shared below.

`script/eval_policy.py` owns the driver half: it constructs these, feeds them once per control
step (the wrench before the chunk runs, from `visualize_debug_obs`; the Q after, since the value
does not exist until the chunk is drawn) and flushes them per episode into
``<eval run>/debug_vis/episode<N>/``. What a feed carries differs: the Q recorder takes one
value per policy call, the wrench recorder the `pi0_step` primitive-step rows the last chunk
logged.

matplotlib and PIL are imported inside the functions that need them, so a run with `debug`
off never pays for them.
"""

from pathlib import Path
from dataclasses import asdict, is_dataclass
import json

import numpy as np

from envs.utils.wrench import WRENCH_COMPONENTS, ee_link_labels, link_wrench_vector

DEBUG_GIF_FRAME_WIDTH = 320  # rollout frames are downscaled to this before being kept in RAM
DEBUG_GIF_FPS = 5
DEBUG_GIF_MAX_FRAMES = 200  # long episodes are subsampled; the traces still cover every sample


def agg_pyplot():
    """matplotlib's pyplot on the Agg backend -- the debug outputs are always written to file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def gif_frame_indices(count):
    """Positions to animate: every sample, thinned to at most DEBUG_GIF_MAX_FRAMES."""
    return range(0, count, max(1, -(-count // DEBUG_GIF_MAX_FRAMES)))


def figure_to_image(fig):
    from PIL import Image

    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])


def write_gif(path, images):
    if not images:
        return
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / DEBUG_GIF_FPS), loop=0)


def padded_limits(*series, pad=0.08):
    """Common y limits for a set of traces, with a little headroom (never a degenerate span)."""
    vals = np.concatenate([np.asarray(s, dtype=np.float64).ravel() for s in series])
    lo, hi = float(vals.min()), float(vals.max())
    margin = max(hi - lo, 1e-6) * pad
    return lo - margin, hi + margin


class RolloutFrameLog:
    """The head-camera frames the debug GIFs are drawn on, kept once for all of them.

    Frames are keyed by sim step (`take_action_cnt`) rather than by position, so a recorder
    that misses a sample -- the wrench one skips a step whose contact query raised -- still
    lines its trace up with the right frame. ``record`` is idempotent per step, so every
    recorder can call it and whichever runs first pays for the downscale.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self.frames = {}
        self.scale = 1.0  # factor the most recent frame was downscaled by

    def record(self, observation, step_idx):
        """Keep this step's head-camera frame; returns the factor it was downscaled by.

        None means the observation carries no head camera, i.e. there is nothing to draw the
        traces beside -- the recorders then write their data files and skip the GIF.
        """
        if step_idx in self.frames:
            return self.scale
        rgb = observation.get("observation", {}).get("head_camera", {}).get("rgb", None)
        if rgb is None:
            return None
        from PIL import Image

        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        scale = 1.0
        if img.width > DEBUG_GIF_FRAME_WIDTH:  # keep the kept-in-RAM rollout small
            scale = DEBUG_GIF_FRAME_WIDTH / img.width
            img = img.resize((DEBUG_GIF_FRAME_WIDTH, max(1, round(img.height * scale))), Image.BILINEAR)
        self.frames[step_idx] = np.asarray(img, dtype=np.uint8)
        self.scale = scale
        return scale

    def frame(self, step_idx):
        return self.frames.get(step_idx)

    def flush(self):
        """Drop the episode's frames. Called after every recorder has rendered its GIF."""
        self._reset()


class ScriptedInterventionRecorder:
    """Scripted recovery attempts -> labeled GIFs, Q-drop plot and machine-readable metadata.

    This intentionally owns its frames instead of using ``RolloutFrameLog``: recovery begins
    only after the autonomous recorders have flushed, rewinds the sim-time axis, and must not
    make the failed rollout's debug GIF appear to have succeeded.
    """

    def __init__(self, debug_save_dir):
        self.debug_save_dir = Path(debug_save_dir)
        self._reset()

    def _reset(self):
        self.attempts = []

    def record_attempt(self, snapshot_index, frames, success, restore_report):
        images = []
        for frame in frames:
            rgb = frame.get("images", {}).get("cam_high")
            if rgb is None:
                continue
            rgb = np.asarray(rgb, dtype=np.uint8)
            if rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4):
                rgb = rgb.transpose(1, 2, 0)
            images.append(rgb[..., :3].copy())
        self.attempts.append({
            "snapshot_index": int(snapshot_index),
            "success": bool(success),
            "restore_exact": bool(restore_report.exact),
            "restore_max_abs_error": float(restore_report.max_abs_error),
            "restore_mismatches": list(restore_report.mismatches),
            "frames": images,
        })

    def flush(self, episode_idx, result):
        if not self.attempts and result is None:
            return
        out_dir = self.debug_save_dir / f"episode{episode_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image, ImageDraw

            for attempt_index, attempt in enumerate(self.attempts):
                rendered = []
                for frame_index in gif_frame_indices(len(attempt["frames"])):
                    image = Image.fromarray(attempt["frames"][frame_index])
                    if image.width > DEBUG_GIF_FRAME_WIDTH:
                        scale = DEBUG_GIF_FRAME_WIDTH / image.width
                        image = image.resize(
                            (DEBUG_GIF_FRAME_WIDTH, max(1, round(image.height * scale))),
                            Image.BILINEAR,
                        )
                    draw = ImageDraw.Draw(image)
                    label = (f"expert attempt {attempt_index + 1} | snapshot "
                             f"{attempt['snapshot_index']} | frame {frame_index} | "
                             f"{'SUCCESS' if attempt['success'] else 'FAILED'}")
                    draw.rectangle((0, 0, image.width, 18), fill=(0, 0, 0))
                    draw.text((4, 3), label, fill=(255, 255, 255))
                    rendered.append(image)
                write_gif(
                    out_dir / (f"intervention_attempt{attempt_index + 1}_snapshot"
                               f"{attempt['snapshot_index']}.gif"),
                    rendered,
                )

            payload = asdict(result) if is_dataclass(result) else dict(result or {})
            # Under its own key: `result.attempts` already carries every attempt including the
            # ones rejected before a rollout, which are exactly the ones with no frames here.
            payload["rendered_attempts"] = [
                {k: v for k, v in attempt.items() if k != "frames"}
                | {"num_frames": len(attempt["frames"])}
                for attempt in self.attempts
            ]
            (out_dir / f"intervention_episode{episode_idx}.json").write_text(
                json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
            )
            self._save_q_plot(out_dir, episode_idx, payload)
            print(f"\033[93m[debug] scripted intervention written to "
                  f"{out_dir}/intervention_*\033[0m")
        except Exception as exc:
            print(f"[debug] scripted intervention output failed: {exc}")
        self._reset()

    @staticmethod
    def _save_q_plot(out_dir, episode_idx, payload):
        q = np.asarray(payload.get("q_values", ()), dtype=np.float64)
        if q.size == 0:
            return
        plt = agg_pyplot()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(q.size), q, marker="o", label="episode-start target Q mean")
        drop = payload.get("drop_index")
        rewind = payload.get("requested_snapshot")
        if drop is not None:
            ax.axvline(drop, color="tab:red", linestyle="--", label="largest Q drop")
        if rewind is not None:
            ax.axvline(rewind, color="tab:green", linestyle=":", label="requested rewind")
        ax.set_xlabel("autonomous control chunk")
        ax.set_ylabel("frozen target Q")
        ax.set_title(f"episode {episode_idx} — post-failure intervention schedule")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"intervention_q_episode{episode_idx}.png", dpi=100)
        plt.close(fig)


WRENCH_AXIS_LENGTH = 0.08  # metres; length of the world frame arrows drawn on the rollout
WRENCH_LABEL_OFFSET = 7  # points past the arrow tip, along the arrow, to place its label
# One colour per axis, shared by the trace lines and the arrows drawn on the rollout, so the
# x/y/z arrow and its Fx/Tx trace read as the same thing.
WRENCH_AXIS_COLORS = ("tab:blue", "tab:orange", "tab:green")  # x, y, z
# One colour per end-effector link, in the order `link_wrench_vector` reports them (left arm's
# links first): cool for the left gripper's fingers, warm for the right's.
WRENCH_LINK_COLORS = ("tab:blue", "tab:cyan", "tab:orange", "tab:red",
                      "tab:green", "tab:purple", "tab:brown", "tab:olive")


class TCPWrenchRecorder:
    """Per-episode end-effector wrench log -> component histograms + a rollout/wrench GIF.

    The wrench is kept **per gripper link** (aloha: ``fl_link7`` / ``fl_link8`` and
    ``fr_link7`` / ``fr_link8``) rather than summed per arm, so a finger pushing against its
    opposite -- forces that cancel in the arm total -- is still visible. Torque is about that
    arm's TCP for all of its links, so the traces stay comparable across the fingers.

    The trace runs at the **primitive-step** rate, not the policy-call rate: the env commits one
    row per `take_action` -- the average contact wrench over that step's whole TOPP trajectory,
    34-196 physics steps of it -- and this drains the ``pi0_step`` rows the last chunk left,
    once per policy call. That is the same signal at the same rate as the critic's ``wrench.*``
    modality and a rollout dataset's ``observation.wrench.*`` column, so what the plots show is
    what those consume; the single instantaneous reading this used to take per policy call saw
    only whatever the grippers happened to be touching at that instant. With
    ``data_type.wrench`` off there is no log and it falls back to that one reading.

    Row ``i`` of a drain is control step ``step_idx - n + 1 + i``, so the x axis is exact
    ``take_action_cnt`` -- the same axis the Q trace uses, and the same one the frames, the
    world-axis overlay and the GIF cursor sit on. A drain is capped at ``wrench_trace_len``,
    dropping its oldest rows: the same bound and the same rule as the critic's own view of it,
    so the two stay the same picture.

    ``flush`` writes three files into the episode's own debug dir, alongside the
    image/point-cloud dumps `visualize_debug_obs` puts there: ``<debug_save_dir>/episode<N>/``
    gets ``wrench_hist_episode<N>.png``, ``wrench_episode<N>.gif`` and
    ``wrench_episode<N>.npz``.
    """

    def __init__(self, debug_save_dir, frame_log):
        self.debug_save_dir = Path(debug_save_dir)
        self.frame_log = frame_log
        self._reset()

    def episode_dir(self, episode_idx):
        return self.debug_save_dir / f"episode{episode_idx}"

    def _reset(self):
        self.steps = []  # one control step (`take_action_cnt`) per policy call
        # One `{link label: (drained rows, 6)}` per policy call -- normally `pi0_step` of them
        # -- stacked as it arrives rather than kept as loose per-row vectors. Filled on the
        # first drain, since the links are the embodiment's and are not known before the env
        # exists.
        self.bursts = []
        self.sample_steps = []  # the matching (rows,) of control-step positions
        self.world_axes = {}  # sim step -> the projected triads for that step's frame

    def record(self, task_env, observation, step_idx):
        """Drain the primitive-step rows the previous chunk left, and keep this step's frame.

        The env's debug log is separate from the one the critic drains (`_base_task` keeps two
        copies of the same rows) precisely so this can run first -- `visualize_debug_obs` is
        called before the policy -- without taking what `critic_obs_modalities` is about to.
        """
        try:
            rows = task_env.pop_debug_step_wrench() or [link_wrench_vector(task_env)]
            # The links alone, in the order `link_wrench_vector` reports them: the logged rows
            # also carry the two arm totals, which are just their links summed.
            links = [label for arm_links in ee_link_labels(task_env.robot).values()
                     for label, _ in arm_links if label in rows[0]]
            burst = {link: np.asarray([row[link] for row in rows], dtype=np.float32)
                     for link in links}
        except Exception as e:
            print(f"[debug] end-effector wrench sampling failed: {e}")
            return
        # The rows are the primitive steps ending at this observation, so they land on
        # `take_action_cnt` exactly. An episode's first call has none behind it and carries a
        # single instantaneous reading of the contact state, which sits on the step itself.
        self.sample_steps.append(np.arange(step_idx - len(rows) + 1, step_idx + 1, dtype=float))
        self.steps.append(step_idx)
        self.bursts.append(burst)

        scale = self.frame_log.record(observation, step_idx)
        if scale is not None:
            self.world_axes[step_idx] = self._project_world_axes(task_env, observation, scale)

    @staticmethod
    def _project_world_axes(task_env, observation, scale):
        """The world axes, anchored at each TCP, as head-camera pixels at the frame's scale.

        The wrench is resolved in world axes, so those are what the rollout should show; they
        are anchored at each arm's TCP because that is the point the wrench acts on (and it
        keeps the triad in frame, unlike the world origin). Both triads therefore point the
        same way and only their origins differ. Returns ``{arm: (origin_uv, tips_uv(3, 2))}``,
        skipping an arm whose TCP is behind the camera; anything merely outside the image is
        clipped when it is drawn.
        """
        cam = observation.get("observation", {}).get("head_camera", {})
        if "intrinsic_cv" not in cam or "extrinsic_cv" not in cam:
            return {}  # camera matrices missing: draw no axes rather than guess
        K = np.asarray(cam["intrinsic_cv"], dtype=np.float64)
        ext = np.asarray(cam["extrinsic_cv"], dtype=np.float64)[:3]  # world -> camera (OpenCV)

        out = {}
        for arm_tag in ("left", "right"):
            origin = np.asarray(getattr(task_env.robot, f"get_{arm_tag}_tcp_pose")(), dtype=np.float64)[:3]
            # (4, 3): origin, then the world x/y/z unit axes stepped out from it
            pts_world = np.vstack([origin, origin + WRENCH_AXIS_LENGTH * np.eye(3)])
            pts_cam = pts_world @ ext[:, :3].T + ext[:, 3]
            if np.any(pts_cam[:, 2] <= 1e-6):  # at or behind the image plane: not projectable
                continue
            uv = (pts_cam @ K.T)[:, :2] / pts_cam[:, 2:3] * scale
            out[arm_tag] = (uv[0], uv[1:])
        return out

    def flush(self, episode_idx):
        """Render this episode's outputs and start a fresh episode. No-op with no samples."""
        if not self.steps or not self.bursts:
            self._reset()
            return
        out_dir = self.episode_dir(episode_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        steps = np.asarray(self.steps)  # one per policy call: frames, cursor, world axes
        sample_steps = np.concatenate(self.sample_steps)  # one per primitive step, as is `series`
        series = {link: np.concatenate([burst[link] for burst in self.bursts])
                  for link in self.bursts[0]}
        try:
            self._save_histograms(out_dir, episode_idx, series)
            self._save_gif(out_dir, episode_idx, steps, sample_steps, series)
            np.savez_compressed(
                out_dir / f"wrench_episode{episode_idx}.npz",
                step=sample_steps,  # the control step each row of the link arrays belongs to
                control_step=steps,  # the policy calls the rows were drained at
                rows_per_call=np.array([len(b) for b in self.sample_steps]),
                components=np.array(WRENCH_COMPONENTS),
                links=np.array(list(series)),  # the order the colours/columns follow
                **series,
            )
            print(f"\033[93m[debug] wrench log written to {out_dir}/wrench_*\033[0m")
        except Exception as e:
            print(f"[debug] end-effector wrench output failed: {e}")
        self._reset()

    def _save_histograms(self, out_dir, episode_idx, series):
        """One histogram per wrench component, every gripper link overlaid.

        Outlined rather than filled: with a link per finger there are more series than the two
        the filled bars stayed readable at.
        """
        plt = agg_pyplot()

        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        for i, name in enumerate(WRENCH_COMPONENTS):
            ax = axes[i // 3, i % 3]
            for (link, vals), color in zip(series.items(), WRENCH_LINK_COLORS):
                ax.hist(vals[:, i], bins=40, histtype="step", linewidth=1.4, color=color,
                        label=f"{link}: {vals[:, i].mean():+.3g} ± {vals[:, i].std():.3g}")
            ax.set_xlabel(f"{name} [{'N' if i < 3 else 'N·m'}]")
            ax.set_ylabel("primitive steps")
            # Most of an episode is free space, i.e. an exact zero; log counts keep the
            # contact tail readable next to that spike.
            ax.set_yscale("log")
            ax.legend(fontsize="x-small")
        fig.suptitle(f"episode {episode_idx} — per-link contact wrench distribution, world frame "
                     f"({len(next(iter(series.values())))} primitive steps)")
        fig.tight_layout()
        fig.savefig(out_dir / f"wrench_hist_episode{episode_idx}.png", dpi=100)
        plt.close(fig)

    def _draw_world_axes(self, ax, step_idx):
        """Overlay the world frame on the rollout as labelled x/y/z arrows, one triad per TCP.

        These are the axes the force and torque traces are resolved in: the Fx trace is the
        contact force along this arrow, Tx the moment about it (taken about the TCP the triad
        sits on).
        """
        import matplotlib.patheffects as pe

        for arm_tag, (origin, tips) in self.world_axes.get(step_idx, {}).items():
            for tip, label, color in zip(tips, "xyz", WRENCH_AXIS_COLORS):
                ax.annotate("", xy=tip, xytext=origin, annotation_clip=True,
                            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6,
                                            shrinkA=0, shrinkB=0))
                # Offset the label along its own arrow rather than a fixed direction: a world
                # axis pointing near the camera projects short, and two such arrows can end up
                # close together, so a fixed offset lets one arm's label drift onto its
                # neighbour's arrow and read as swapped. (dy flips: image y grows downward,
                # offset-point y grows upward.)
                d = np.asarray(tip, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
                norm = float(np.linalg.norm(d)) or 1.0
                ax.annotate(f"{arm_tag[0]}{label}", xy=tip,
                            xytext=WRENCH_LABEL_OFFSET * d / norm * (1, -1), textcoords="offset points",
                            ha="center", va="center",
                            color=color, fontsize="x-small", fontweight="bold", annotation_clip=True,
                            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])

    def _save_gif(self, out_dir, episode_idx, steps, sample_steps, series):
        """Rollout on the left, one trace column per gripper link with a step cursor, right.

        One GIF frame per policy call (thinned to DEBUG_GIF_MAX_FRAMES), but the traces behind
        the cursor carry every primitive step, ``pi0_step`` of them per call.
        """
        plt = agg_pyplot()

        n = len(steps)
        links = list(series)
        # Fixed limits across frames so only the cursor moves, and shared by every link so the
        # columns can be read against each other.
        lims = {}
        for row, sl in (("force", slice(0, 3)), ("torque", slice(3, 6))):
            vals = np.concatenate([vals[:, sl].ravel() for vals in series.values()])
            span = max(float(np.abs(vals).max()), 1e-6) * 1.1
            lims[row] = (-span, span)

        gif_frames = []
        for k in gif_frame_indices(n):
            frame = self.frame_log.frame(steps[k])
            if frame is None:  # no head camera on that step: nothing to animate against
                continue
            # The rollout keeps a fixed share of the figure; the width grows with the number of
            # links so each column stays as wide as it was when there were two.
            fig = plt.figure(figsize=(4.5 + 2.4 * len(links), 5.5))
            gs = fig.add_gridspec(2, 1 + len(links), width_ratios=[1.6] + [1] * len(links))
            ax_img = fig.add_subplot(gs[:, 0])
            ax_img.imshow(frame)
            ax_img.axis("off")
            ax_img.set_title(f"rollout — step {steps[k]}")
            self._draw_world_axes(ax_img, steps[k])
            for r, (row, sl) in enumerate((("force", slice(0, 3)), ("torque", slice(3, 6)))):
                for c, link in enumerate(links):
                    ax = fig.add_subplot(gs[r, c + 1])
                    for j, comp in enumerate(WRENCH_COMPONENTS[sl]):
                        ax.plot(sample_steps, series[link][:, sl][:, j], linewidth=1.0,
                                color=WRENCH_AXIS_COLORS[j], label=comp)
                    # The cursor is the control step the frame belongs to, which is exactly
                    # where that step's burst ends.
                    ax.axvline(steps[k], color="k", linewidth=1.2)
                    ax.set_xlim(sample_steps[0], max(sample_steps[-1], sample_steps[0] + 1))
                    ax.set_ylim(*lims[row])
                    if r == 0:  # units live on the y axis, so the title only names the link
                        ax.set_title(f"{link} (world frame)", fontsize="small",
                                     color=WRENCH_LINK_COLORS[c % len(WRENCH_LINK_COLORS)])
                    ax.set_xlabel("sim step", fontsize="x-small")
                    ax.set_ylabel(f"{row} [{'N' if row == 'force' else 'N·m'}]", fontsize="x-small")
                    ax.tick_params(labelsize="x-small")
                    ax.legend(fontsize="xx-small", ncol=3, loc="upper right")
            fig.tight_layout()
            gif_frames.append(figure_to_image(fig))
            plt.close(fig)

        write_gif(out_dir / f"wrench_episode{episode_idx}.gif", gif_frames)


class QValueRecorder:
    """Per-episode critic Q log -> a trace plot, a rollout/Q GIF and the raw series.

    Only meaningful when a critic is scoring the sampler's chunks (`guidance_scale != 0` or
    `best_of_n > 1`); the driver builds one only then. Each control step contributes the
    ensemble's Q for the chunk the policy just sampled -- the value the guidance was climbing
    and/or best-of-N selected on, evaluated at the action it settled on -- plus the reward that
    chunk earned and the guidance scale in force while it was drawn. ``flush`` writes
    ``q_episode<N>.png``, ``q_episode<N>.gif`` and ``q_episode<N>.npz`` into the same
    ``<debug_save_dir>/episode<N>/`` the wrench outputs and the image dumps go to.

    Unlike the wrench, this is sampled *after* the control step: the value does not exist until
    the policy has drawn the chunk it scores. The frame it is paired with is still the one the
    chunk was drawn from -- the shared RolloutFrameLog is keyed by sim step, and this step's
    frame was already captured by `visualize_debug_obs` before the action ran.

    What the plots are for: Q predicts the discounted return still to come, so the realized
    return-to-go is drawn against it. Q tracking that curve is a calibrated critic; a flat Q
    means it is not distinguishing the states it is steering through, a persistent gap means it
    is over- or under-valuing them, and an ensemble spread that stays wide means the members do
    not agree on states the guidance is nevertheless following.
    """

    def __init__(self, debug_save_dir, frame_log):
        self.debug_save_dir = Path(debug_save_dir)
        self.frame_log = frame_log
        # Q may be in the normalized return space an offline checkpoint was trained in; the
        # plots undo that so Q and the realized return share units. Identity when the critic
        # was trained online from scratch. `gamma_h` is the per-control-step discount
        # (discount ** horizon), i.e. what the return-to-go must be summed with.
        self.return_mean, self.return_std, self.gamma_h = 0.0, 1.0, 1.0
        self._reset()

    def episode_dir(self, episode_idx):
        return self.debug_save_dir / f"episode{episode_idx}"

    def _reset(self):
        self.steps = []
        self.q = []
        self.rewards = []
        self.guidance = []

    def record(self, model, observation, step_idx, reward):
        """Log the Q of the chunk that just executed. No-op until a critic has scored one.

        ``model.last_q_values`` is written by the policy's own sampling path (pi05:
        `PI0.get_action`, gated on `record_q_values`), so this stays a plain read -- it never
        runs the critic itself, and a policy that does not expose one simply records nothing.
        It is consumed here so a control step that somehow sampled no chunk cannot re-log the
        previous step's value.
        """
        q = getattr(model, "last_q_values", None)
        if q is None:
            return
        model.last_q_values = None
        self.frame_log.record(observation, step_idx)
        self.steps.append(step_idx)
        self.q.append(np.asarray(q, dtype=np.float32).ravel())
        self.rewards.append(float(reward))
        self.guidance.append(float(model.scheduled_guidance_scale()))
        critic = getattr(model, "online_critic", None)
        if critic is not None:
            self.return_mean = float(getattr(critic, "return_mean", 0.0))
            self.return_std = float(getattr(critic, "return_std", 1.0))
            self.gamma_h = float(getattr(critic, "gamma_h", 1.0))

    def flush(self, episode_idx):
        """Render this episode's outputs and start a fresh episode. No-op with no samples."""
        if not self.steps:
            self._reset()
            return
        out_dir = self.episode_dir(episode_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        steps = np.asarray(self.steps)
        q = np.stack(self.q)  # (samples, num_qs), in the critic's own output space
        rewards = np.asarray(self.rewards, dtype=np.float32)
        guidance = np.asarray(self.guidance, dtype=np.float32)
        # Both in return units, so they can share an axis.
        q_return = q * self.return_std + self.return_mean
        returns = self._return_to_go(rewards)
        try:
            self._save_plot(out_dir, episode_idx, steps, q_return, returns, rewards, guidance)
            self._save_gif(out_dir, episode_idx, steps, q_return, returns, rewards)
            np.savez_compressed(
                out_dir / f"q_episode{episode_idx}.npz",
                step=steps,
                q=q,  # raw ensemble output, as the guidance sees it
                reward=rewards,
                guidance_scale=guidance,
                return_to_go=returns,
                return_mean=self.return_mean,
                return_std=self.return_std,
                gamma_h=self.gamma_h,
            )
            print(f"\033[93m[debug] critic Q log written to {out_dir}/q_*\033[0m")
        except Exception as e:
            print(f"[debug] critic Q output failed: {e}")
        self._reset()

    def _return_to_go(self, rewards):
        """Realized discounted return from each control step on, at the critic's own discount.

        The episode's own outcome, so a truncated episode's tail is genuinely short -- it is
        what happened, not an estimate, which is the point of plotting it against Q.
        """
        out = np.zeros_like(rewards)
        acc = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            acc = rewards[i] + self.gamma_h * acc
            out[i] = acc
        return out

    def _plot_traces(self, ax_q, ax_r, steps, q_return, returns, rewards, cursor=None):
        """The two stacked panels both outputs share: Q vs realized return, then reward."""
        q_mean = q_return.mean(axis=1)
        ax_q.fill_between(steps, q_return.min(axis=1), q_return.max(axis=1),
                          color="tab:blue", alpha=0.2, linewidth=0,
                          label=f"ensemble range (n={q_return.shape[1]})")
        ax_q.plot(steps, q_mean, color="tab:blue", linewidth=1.6, label="Q (ensemble mean)")
        ax_q.plot(steps, returns, color="tab:red", linewidth=1.2, linestyle="--",
                  label=f"realized return-to-go (γ={self.gamma_h:.4g})")
        ax_q.set_ylabel("value [return units]", fontsize="x-small")
        ax_q.legend(fontsize="xx-small", loc="upper left")

        ax_r.plot(steps, rewards, color="tab:green", linewidth=1.0, drawstyle="steps-post",
                  label="step reward")
        ax_r.plot(steps, np.cumsum(rewards), color="tab:gray", linewidth=1.0,
                  label="cumulative reward")
        ax_r.set_ylabel("reward", fontsize="x-small")
        ax_r.set_xlabel("sim step", fontsize="x-small")
        ax_r.legend(fontsize="xx-small", loc="upper left")

        for ax in (ax_q, ax_r):
            ax.set_xlim(steps[0], max(steps[-1], steps[0] + 1))
            ax.tick_params(labelsize="x-small")
            if cursor is not None:
                ax.axvline(cursor, color="k", linewidth=1.2)

    def _save_plot(self, out_dir, episode_idx, steps, q_return, returns, rewards, guidance):
        """The whole episode in one static figure (the GIF's right-hand column, no cursor)."""
        plt = agg_pyplot()

        fig, (ax_q, ax_r) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        self._plot_traces(ax_q, ax_r, steps, q_return, returns, rewards)
        norm = ("" if (self.return_mean, self.return_std) == (0.0, 1.0)
                else f", un-normalized by ×{self.return_std:.4g}{self.return_mean:+.4g}")
        fig.suptitle(f"episode {episode_idx} — critic Q along the rollout "
                     f"({len(steps)} chunks, guidance {guidance.min():.3g}→{guidance.max():.3g}"
                     f"{norm})")
        fig.tight_layout()
        fig.savefig(out_dir / f"q_episode{episode_idx}.png", dpi=100)
        plt.close(fig)

    def _save_gif(self, out_dir, episode_idx, steps, q_return, returns, rewards):
        """Rollout on the left, the Q and reward traces with a step cursor on the right."""
        plt = agg_pyplot()

        q_mean = q_return.mean(axis=1)
        gif_frames = []
        for k in gif_frame_indices(len(steps)):
            frame = self.frame_log.frame(steps[k])
            if frame is None:  # no head camera on that step: nothing to animate against
                continue
            # The rollout axis is aspect-locked, so the figure has to be tall enough to hold it
            # plus its title -- tight_layout cannot shrink an image below its aspect ratio, and
            # a short figure simply crops the title off the top.
            fig = plt.figure(figsize=(11, 5.4))
            gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])
            ax_img = fig.add_subplot(gs[:, 0])
            ax_img.imshow(frame)
            ax_img.axis("off")
            ax_img.set_title(f"rollout — step {steps[k]}   "
                             f"Q={q_mean[k]:+.3f}   r={rewards[k]:+.3f}")
            ax_q = fig.add_subplot(gs[0, 1])
            ax_r = fig.add_subplot(gs[1, 1], sharex=ax_q)
            self._plot_traces(ax_q, ax_r, steps, q_return, returns, rewards, cursor=steps[k])
            # Fixed limits across frames so only the cursor moves.
            ax_q.set_ylim(*padded_limits(q_return, returns))
            ax_r.set_ylim(*padded_limits(rewards, np.cumsum(rewards)))
            fig.tight_layout()
            gif_frames.append(figure_to_image(fig))
            plt.close(fig)

        write_gif(out_dir / f"q_episode{episode_idx}.gif", gif_frames)


# One colour per retrieval rank, so the top-1 match reads the same in the trace plot and in the
# GIF's panel borders.
RETRIEVAL_RANK_COLORS = ("tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown")


class DemoRetrievalRecorder:
    """Per-episode log of what demo retrieval matched -> a phase plot, a GIF and the raw series.

    Only built when the policy is producing `action_proposals` / `noise_proposals` (see
    `policy/pi05/src/openpi/policies/demo_retrieval.py`): each control step contributes the
    top-K demonstration frames its observation was matched to, the averaged relative L2 distance
    to each, and a thumbnail of each so a human can see what "nearest" actually returned.

    The question this is here to answer is whether pooled-SigLIP retrieval is finding
    *corresponding* frames or merely nearby ones. Two outputs, for the two halves of that:

    * ``retrieval_episode<N>.png`` -- the retrieved demo frame index against the rollout's own
      step. A retrieval that tracks task phase draws a roughly monotone line rising with the
      rollout; a flat line means every step matched the same demo frame (the embedding is not
      separating the states), and a scattered one means it is not tracking phase at all. The
      distance of each rank is drawn underneath, which is what says whether a match is close or
      merely the least bad of a bad set -- note this is a **distance** (relative L2, averaged
      over signals), so lower is better and the panel reads the opposite way round from a
      similarity.
    * ``retrieval_episode<N>.gif`` -- the rollout's head camera beside the top-K demo head
      frames it matched, captioned with each one's distance. This is the only output that can
      show *why* a match is wrong.

    Sampled at the same point as the wrench: the retrieval happens before the chunk is drawn, so
    it belongs to the observation it was queried with, and is logged against that step.
    """

    def __init__(self, debug_save_dir, frame_log):
        self.debug_save_dir = Path(debug_save_dir)
        self.frame_log = frame_log
        # How many of the ranks drawn were actually handed to the critic as proposals; the rest
        # are shown for context. Learned from the first sample.
        self.num_proposals = 1
        self.bank_frames = 0
        self._reset()

    def episode_dir(self, episode_idx):
        return self.debug_save_dir / f"episode{episode_idx}"

    def _reset(self):
        self.steps = []
        self.indices = []
        self.distances = []
        self.demo_episode = []
        self.demo_frame = []
        self.thumbnails = {}  # step -> (k, h, w, 3), only for the steps the GIF will animate

    def record(self, model, observation, step_idx):
        """Log what the policy's retrieval matched this control step. No-op without one.

        `model.last_demo_retrieval` is written by the policy itself (pi05:
        `PI0._refresh_demo_proposals`, gated on `record_demo_retrieval`), so this is a plain
        read -- it never runs the retrieval. Consumed as it is read, so a control step that did
        not retrieve cannot re-log the previous step's match.
        """
        info = getattr(model, "last_demo_retrieval", None)
        if info is None:
            return
        model.last_demo_retrieval = None
        self.frame_log.record(observation, step_idx)
        self.steps.append(step_idx)
        self.indices.append(np.asarray(info["indices"], dtype=np.int32))
        self.distances.append(np.asarray(info["distances"], dtype=np.float32))
        self.demo_episode.append(np.asarray(info["episode"], dtype=np.int32))
        self.demo_frame.append(np.asarray(info["frame"], dtype=np.int32))
        self.bank_frames = int(info.get("bank_frames", self.bank_frames))
        self.num_proposals = int(info.get("num_proposals", self.num_proposals))
        if info.get("thumbnails") is not None:
            self.thumbnails[step_idx] = np.asarray(info["thumbnails"], dtype=np.uint8)

    def flush(self, episode_idx):
        """Render this episode's outputs and start a fresh episode. No-op with no samples."""
        if not self.steps:
            self._reset()
            return
        out_dir = self.episode_dir(episode_idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        steps = np.asarray(self.steps)
        indices = np.stack(self.indices)  # (samples, k)
        distances = np.stack(self.distances)
        demo_episode = np.stack(self.demo_episode)
        demo_frame = np.stack(self.demo_frame)
        try:
            self._save_plot(out_dir, episode_idx, steps, indices, distances, demo_frame)
            self._save_gif(out_dir, episode_idx, steps, distances, demo_episode, demo_frame)
            np.savez_compressed(
                out_dir / f"retrieval_episode{episode_idx}.npz",
                step=steps,
                bank_index=indices,
                distance=distances,
                demo_episode=demo_episode,
                demo_frame=demo_frame,
                bank_frames=self.bank_frames,
            )
            print(f"\033[93m[debug] demo retrieval log written to {out_dir}/retrieval_*\033[0m")
        except Exception as e:
            print(f"[debug] demo retrieval output failed: {e}")
        self._reset()

    def _save_plot(self, out_dir, episode_idx, steps, indices, distances, demo_frame):
        """Retrieved demo frame vs rollout step (top), and match distance (bottom, lower=better)."""
        plt = agg_pyplot()
        fig, (ax_idx, ax_sim) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for rank in range(indices.shape[1]):
            color = RETRIEVAL_RANK_COLORS[rank % len(RETRIEVAL_RANK_COLORS)]
            # Ranks past `num_proposals` are context only -- drawn thinner, since the critic
            # never saw them.
            given = rank < self.num_proposals
            ax_idx.plot(steps, demo_frame[:, rank], ".-", color=color, ms=3,
                        lw=1.4 if given else 0.7, alpha=1.0 if given else 0.6,
                        label=f"rank {rank + 1}" + (" (to critic)" if given else ""))
            ax_sim.plot(steps, distances[:, rank], "-", color=color,
                        lw=1.4 if given else 0.7, alpha=1.0 if given else 0.6)
        # The diagonal a perfectly phase-aligned retrieval would follow, if the rollout ran the
        # demo's length. Not a target -- a rollout that stalls or recovers *should* leave it --
        # but it makes "tracking phase" versus "stuck" readable at a glance.
        if len(steps) > 1 and demo_frame.size:
            span = float(demo_frame[demo_frame >= 0].max() or 1)
            ax_idx.plot(steps, np.linspace(0, span, len(steps)), "--", color="0.6", lw=1,
                        label="uniform phase")
        ax_idx.set_ylabel("retrieved demo frame")
        ax_idx.set_title(f"episode {episode_idx}: demo retrieval (top-{indices.shape[1]} of "
                         f"{self.bank_frames} bank frames; {self.num_proposals} given to critic)")
        ax_idx.legend(fontsize=7, ncol=2)
        ax_idx.grid(alpha=0.3)
        ax_sim.set_ylabel("relative L2 distance\n||q-b||/||q||  (lower = nearer)")
        ax_sim.set_xlabel("rollout step")
        ax_sim.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"retrieval_episode{episode_idx}.png", dpi=120)
        plt.close(fig)

    def _save_gif(self, out_dir, episode_idx, steps, distances, demo_episode, demo_frame):
        """The rollout frame beside the demo frames it matched, one panel per rank."""
        if not self.thumbnails:
            return
        plt = agg_pyplot()
        images = []
        for i in gif_frame_indices(len(steps)):
            step = int(steps[i])
            rollout = self.frame_log.frame(step)
            thumbs = self.thumbnails.get(step)
            if rollout is None or thumbs is None:
                continue
            k = len(thumbs)
            fig, axes = plt.subplots(1, k + 1, figsize=(3.0 * (k + 1), 3.0))
            axes = np.atleast_1d(axes)
            axes[0].imshow(rollout)
            axes[0].set_title(f"rollout  step {step}", fontsize=9)
            for rank in range(k):
                ax = axes[rank + 1]
                ax.imshow(thumbs[rank])
                ax.set_title(
                    f"#{rank + 1}{'*' if rank < self.num_proposals else ''}  "
                    f"ep{int(demo_episode[i, rank])} f{int(demo_frame[i, rank])}\n"
                    f"d {distances[i, rank]:.3f}",
                    fontsize=9,
                    color=RETRIEVAL_RANK_COLORS[rank % len(RETRIEVAL_RANK_COLORS)],
                )
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            images.append(figure_to_image(fig))
            plt.close(fig)
        write_gif(out_dir / f"retrieval_episode{episode_idx}.gif", images)
