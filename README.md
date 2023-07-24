## Environment Variable: train_text_to_image_lora.py

http_proxy=http://127.0.0.1:7890;https_proxy=http://127.0.0.1:7890

Install diffusers

```
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .

https://huggingface.co/CompVis/stable-diffusion-v1-4
```

```commandline
python --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
--train_data_dir="./datasets"
--caption_column="drug class"
--resolution=512
--random_flip
--train_batch_size=2
--num_train_epochs=10
--checkpointing_steps=200
--learning_rate=1e-04
--lr_scheduler="constant"
--lr_warmup_steps=0
--seed=42
--output_dir="sd-pokemon-model-lora"
--report_to="wandb"
--image_column="drug name"
--dataset_name="LiverTox"
```

## Environment Variable : inference.py

http_proxy=http://127.0.0.1:7890;https_proxy=http://127.0.0.1:7890

```commandline
python inference.py
```


## Environment Variable : process_data.py
PYTHONUNBUFFERED=1

````commandline
python process_data.py
````
