import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

repo_id = "allenai/MolmoAct2-BimanualYAM"

top_rgb = Image.open(
    hf_hub_download(repo_id, "assets/sample_top_rgb.png")
).convert("RGB")
left_rgb = Image.open(
    hf_hub_download(repo_id, "assets/sample_left_rgb.png")
).convert("RGB")
right_rgb = Image.open(
    hf_hub_download(repo_id, "assets/sample_right_rgb.png")
).convert("RGB")
task = "Place cups and plate in dishwasher rack, dispose of food waste, and organize remaining items."
robot_state = np.array(
    [
        -0.06656748056411743,
        0.014686808921396732,
        0.016594186425209045,
        -0.08602273464202881,
        -0.014686808921396732,
        0.13904783129692078,
        0.9922363758087158,
        0.19512474536895752,
        0.010872052982449532,
        0.010872052982449532,
        -0.06771191209554672,
        -0.07305257022380829,
        -0.08945601433515549,
        0.9888537526130676,
    ],
    dtype=np.float32,
)

processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    repo_id,
    trust_remote_code=True,
    dtype=torch.float32,
).to("cuda").eval()

out = model.predict_action(
    processor=processor,
    images=[top_rgb, left_rgb, right_rgb],
    task=task,
    state=robot_state,
    norm_tag="yam_dual_molmoact2",
    inference_action_mode="continuous",
    enable_depth_reasoning=False,
    num_steps=10,
    normalize_language=True,
    enable_cuda_graph=True,
)

actions = out.actions
