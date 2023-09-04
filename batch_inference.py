import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import csv
import os


# LoRA weights ~3 MB
model_path = "sd-pokemon-model-lora/"
model_base = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

# dataset_name_list = ['LactMed_data', 'LiverTox_data']
dataset_name_list = ['LiverTox_data']
dataset_name = dataset_name_list[0]
dataset_root = '/home/c2/Desktop/DrugDiffusers/datasets/{}.csv'.format(dataset_name)
out_root = '/home/c2/Desktop/DrugDiffusers/test_results/'
out_img_root = os.path.join(out_root, dataset_name)
os.makedirs(out_img_root, exist_ok=True)
with open(dataset_root, 'r') as file:
    reader = csv.DictReader(file)
    data = []
    for row in reader:
        data.append(row)

for i in range(len(data)):
    data_name = data[i]['drug name']
    data_prompt = data[i]['drug class']
    image = pipe(data_prompt, num_inference_steps=50).images[0]
    image.save(os.path.join(out_img_root, "{}.jpg".format(data_name)))
    print(data_name)
