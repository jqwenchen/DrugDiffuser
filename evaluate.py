import torch
import numpy as np
from functools import partial
from torchmetrics.functional.multimodal import clip_score
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torchmetrics.image.fid import FrechetInceptionDistance


model_base = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompts = [
    "drug class is Antineoplastic Agents and Protein Kinase Inhibitors.generate smile image.",
    "drug class is Antineoplastic Agents and Protein Kinase Inhibitors.generate smile image.",
]

images = pipe(prompts, num_inference_steps=50, output_type="numpy").images



############
############
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


sd_clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {sd_clip_score}")



############
############
fid = FrechetInceptionDistance(normalize=True)
_imgs = torch.from_numpy(images).permute(0, 3, 1, 2)
fid.update(_imgs, real=True)  # 这里替换成自己真实的图片
fid.update(_imgs, real=False)

print(f"FID: {float(fid.compute())}")
