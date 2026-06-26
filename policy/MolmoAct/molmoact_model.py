#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import numpy as np
from PIL import Image
import torch

from transformers import AutoModelForImageTextToText, AutoProcessor


class MolmoAct:

    def __init__(self, checkpoint_path, norm_tag, molmoact_step):
        self.checkpoint_path = checkpoint_path
        self.norm_tag = norm_tag

        self.processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        self.policy = AutoModelForImageTextToText.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            dtype=torch.float32,
        ).to("cuda").eval()
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.molmoact_step = molmoact_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
        )
        # img_front = np.transpose(img_front, (2, 0, 1))
        # img_right = np.transpose(img_right, (2, 0, 1))
        # img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": [
                Image.fromarray(img_front),
                Image.fromarray(img_right),
                Image.fromarray(img_left),
            ],
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.predict_action(
            processor=self.processor,
            images=self.observation_window["images"],
            task=self.instruction,
            state=self.observation_window["state"],
            norm_tag=self.norm_tag,
            inference_action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=10,
            normalize_language=True,
            enable_cuda_graph=True,
        ).actions.cpu()[0]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
