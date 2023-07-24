## Environment Setup
Install diffusers

```
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

## Data Preparation
Get molecule SMILES representation, then convert to images for data preprocessing 
``` 
python smiles2img_pretrain.py --dataroot ./datasets/pretraining/ --dataset data
```

## Run
### Fine-tuning

```commandline
python train_text_to_image_lora.py
--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
--train_data_dir="./datasets"
--caption_column="drug class"
--resolution=512
--random_flip
--train_batch_size=2
--num_train_epochs=100
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

### Mol-Image Generation
```commandline
python inference.py
```


### Evaluation
````commandline
python evaluate.py
````
