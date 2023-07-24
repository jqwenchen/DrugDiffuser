import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# LoRA weights ~3 MB
model_path = "sd-pokemon-model-lora/"
# model_path = "sd-pokemon-model-lora/"
model_base = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

image = pipe("drug class is Breast Feeding and Lactation and Anti-Inflammatory Agents  Non-Steroidal and Uricosuric Agents.generate smile image.", num_inference_steps=50).images[0]
image.save("test.jpg")
