import sys
import warnings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, "../../tools"))
import numpy as np
import pdb
import json
import torch
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import gymnasium as gym
import toppra as ta
import transforms3d as t3d
from collections import OrderedDict

import sys
import warnings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, "../../tools"))
import numpy as np
import pdb
import json
import torch
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import gymnasium as gym
import toppra as ta
import transforms3d as t3d
from collections import OrderedDict


class Sapien_TEST(gym.Env):

    def __init__(self):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        try:
            self.setup_scene()
            print("\033[32m" + "Render Well" + "\033[0m")
        except Exception as exc:
            print("\033[31m" + "Render Error" + "\033[0m")
            print(f"Render setup failed: {exc}")
            raise

    def setup_scene(self, **kwargs):
        """
        Set the scene
            - Set up the basic scene: light source, viewer.
        """
        self.engine = sapien.Engine()
        # declare sapien renderer
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        # Use the SLURM-visible GPU explicitly. On shared cluster nodes,
        # probing every Vulkan device can select or touch another job's GPU.
        try:
            self.renderer = sapien.SapienRenderer(device=sapien.Device("cuda:0"))
        except Exception:
            self.renderer = sapien.SapienRenderer()
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)

        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        # OptiX is available on the cluster GPUs; OIDN is not installed there.
        sapien.render.set_ray_tracing_denoiser("optix")

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)


if __name__ == "__main__":
    a = Sapien_TEST()
