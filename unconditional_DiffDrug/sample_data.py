import shutil
import os
import random
from tqdm import tqdm

source_dir_path = "../pretrain/data/224"
target_dir_path = "../pretrain/data/sample_data"
os.makedirs(target_dir_path, exist_ok=True)
sample_len = 10000

all_file_name = os.listdir(source_dir_path)
sample_file_name = random.choices(all_file_name, k=sample_len)
for filename in tqdm(sample_file_name):
    source_file_path = os.path.join(source_dir_path, filename)
    target_file_path = os.path.join(target_dir_path, filename)
    shutil.copyfile(source_file_path, target_file_path)