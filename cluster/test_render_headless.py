"""Headless SAPIEN render check for the Fir cluster (run on a GPU node).

Renders a real frame through a camera and reads the pixels back, mirroring
RoboTwin's envs/_base_task.py setup. Tests the ray-tracing path RoboTwin
hardcodes ("rt" + optix) and the rasterization path ("default"), each in an
ISOLATED subprocess with a timeout so a hang/segfault in one cannot affect the
other (SAPIEN's renderer is process-global).

Requires setup_env.sh to have been sourced (gpucomp shim + system ICD).
Exit 0 if at least the rasterization path renders.
"""
import os
import subprocess
import sys

TIMEOUT = 150


def _render_once(shader: str) -> int:
    """Worker: render one frame with the given shader dir; print RESULT line."""
    import warnings
    warnings.simplefilter("ignore")
    import numpy as np
    import sapien.core as sapien

    sapien.render.set_camera_shader_dir(shader)
    if shader == "rt":
        sapien.render.set_ray_tracing_samples_per_pixel(8)
        sapien.render.set_ray_tracing_path_depth(4)
        sapien.render.set_ray_tracing_denoiser("optix")

    engine = sapien.Engine()
    # Pin to the SLURM-allocated GPU (see envs/_base_task.py) so SAPIEN doesn't
    # probe other GPUs and hang on busy multi-GPU nodes.
    try:
        renderer = sapien.SapienRenderer(device=sapien.Device("cuda:0"))
    except Exception:
        renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    scene = engine.create_scene(sapien.SceneConfig())
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5])
    scene.add_point_light([1, 0, 1.8], [1, 1, 1])
    b = scene.create_actor_builder()
    b.add_box_visual(half_size=[0.1, 0.1, 0.1], material=[0.8, 0.2, 0.2])
    box = b.build_kinematic(name="box")
    box.set_pose(sapien.Pose([0, 0, 0.1]))
    cam = scene.add_camera(name="cam", width=320, height=240,
                           fovy=np.deg2rad(60), near=0.05, far=100)
    cam.entity.set_pose(sapien.Pose([1, 0, 0.6], [0.9239, 0, 0.3827, 0]))
    scene.step()
    scene.update_render()
    cam.take_picture()
    img = np.asarray(cam.get_picture("Color"))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"render_check_{shader}.png")
    try:
        from PIL import Image
        Image.fromarray((np.clip(img[..., :3], 0, 1) * 255).astype("uint8")).save(out)
    except Exception:
        out = "(png skip)"
    print(f"RESULT shape={tuple(img.shape)} mean_rgb={float(img[..., :3].mean()):.4f} saved={out}")
    return 0


def _drive(shader: str):
    """Parent: run a worker subprocess for `shader` and report outcome."""
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", shader],
                           capture_output=True, text=True, timeout=TIMEOUT)
        tag = "\033[32mOK\033[0m" if p.returncode == 0 else "\033[31mFAIL\033[0m"
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT")), "")
        err = "" if p.returncode == 0 else (p.stderr.strip().splitlines() or [""])[-1]
        print(f"[{shader:10s}] {tag} rc={p.returncode} {line}{err}")
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[{shader:10s}] \033[31mTIMEOUT\033[0m after {TIMEOUT}s")
        return False


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        sys.exit(_render_once(sys.argv[2]))

    print("python:", sys.version.split()[0])
    print("VK_ICD_FILENAMES:", os.environ.get("VK_ICD_FILENAMES", "(unset)"))
    rt_ok = _drive("rt")
    ras_ok = _drive("default")
    print(f"\nSUMMARY: raytracing={rt_ok} rasterization={ras_ok}")
    sys.exit(0 if ras_ok else 1)
