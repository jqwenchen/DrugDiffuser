import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import pandas as pd
from matplotlib import pyplot as plt
import os
import shutil

# LoRA weights ~3 MB
model_path = "sd-LiverTox-1000-lora/"
# model_path = "sd-pokemon-model-lora/"
# model_base = "CompVis/stable-diffusion-v1-4"
model_base = "./checkpoint" # base model dir
pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

df = pd.read_csv("datasets/LiverTox.csv")
for idx, row in df.iterrows():
    prompt = row["drug class"]
    name = row["drug name"]
    if os.path.exists(f"datasets/LiverTox/{name}.jpg"):
        shutil.copyfile(f"datasets/LiverTox/{name}.jpg", f"datasets/LiverTox_tmp/{name}.jpg")

    image = pipe(prompt, num_inference_steps=50, output_type="numpy").images[0]
    plt.imsave(f"datasets/LiverTox-gen/{idx}.png", image)




# prompts = df['drug class'].to_list()
# for idx, prompt in enumerate(prompts):
#     image = pipe(prompt, num_inference_steps=50, output_type="numpy").images[0]
# #image = pipe("drug class is Breast Feeding and Lactation and Anti-Inflammatory Agents  Non-Steroidal and Uricosuric Agents.generate smile image.", num_inference_steps=50).images[0]
#     # image.save(f"datasets/LactMed-gen/{idx}.jpg")
#     plt.imsave(f"datasets/LiverTox-gen/{idx}.png", image)

