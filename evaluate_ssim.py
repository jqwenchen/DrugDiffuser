import torch
from pytorch_msssim import ms_ssim, ssim, SSIM, MS_SSIM
import os
from PIL import Image
import numpy as np

gen_img_dir = "/home/c2/Desktop/DrugDiffusers/datasets/LactMed-gen"
orig_img_dir = "/home/c2/Desktop/DrugDiffusers/datasets/LactMed"
# gen_img_dir = "/home/c2/Desktop/DrugDiffusers/datasets/LiverTox-gen"
# orig_img_dir = "/home/c2/Desktop/DrugDiffusers/datasets/LiverTox"

gen_img_paths = sorted(os.listdir(gen_img_dir))
orig_img_paths = sorted(os.listdir(orig_img_dir))

ssim_scores = []
ms_ssim_scores = []
for gen_img_path, orig_img_path in zip(gen_img_paths, orig_img_paths):
    gen_img = Image.open(os.path.join(gen_img_dir, gen_img_path))
    orig_img = Image.open(os.path.join(orig_img_dir, orig_img_path))

    gen_img = gen_img.convert("RGB").resize((256,256))
    orig_img = orig_img.convert("RGB").resize((256,256))

    gen_img = torch.from_numpy(np.array(gen_img)).permute(2, 0, 1).float().unsqueeze(0)
    orig_img = torch.from_numpy(np.array(orig_img)).permute(2, 0, 1).float().unsqueeze(0)

    ssim_score = ssim(gen_img, orig_img, data_range=255)
    ssim_scores.append(ssim_score)

    ms_ssim_score = ms_ssim(gen_img, orig_img, data_range=255, win_size=3)
    ms_ssim_scores.append(ms_ssim_score)

print(f"avg ssim score:{np.mean(ssim_scores)}")
print(f"avg ms_ssim score:{np.mean(ms_ssim_scores)}")
