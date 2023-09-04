import torch
import numpy as np
from functools import partial
from torchmetrics.functional.multimodal import clip_score
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
import pandas as pd
import os
model_base = "CompVis/stable-diffusion-v1-4"
model_path = "sd-pokemon-model-lora/"
pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")
from PIL import Image
# prompts = [
#     "drug class is Breast Feeding and Lactation and Antiarrhythmics.generate smile image.",
#     "drug class is Breast Feeding and Lactation and Milk  Human and Antipsychotic Agents and Dopamine Antagonists.generate smile image.",
#
# ]
df = pd.read_csv("datasets/LactMed_data.csv")
prompts = df['drug class'].to_list()[:100]
# images = pipe(prompts, num_inference_steps=50, output_type="numpy").images

img_path = "/home/c2/Desktop/DrugDiffusers/datasets/LactMed-gen"
image_path = sorted([os.path.join(img_path, x) for x in os.listdir(img_path)])
images = [np.array(Image.open(img).convert("RGB")) for img in image_path]
images = np.array(images)[:100]


############
### Clip Score ###
############
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    # images = np.expand_dims(images, axis=0)
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

# for image in images:
#     sd_clip_score = calculate_clip_score(image, prompts)
#     print(f"CLIP score: {sd_clip_score}")

sd_clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {sd_clip_score}")

