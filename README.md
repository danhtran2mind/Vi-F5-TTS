# Vietnamese F5-TTS (Vi-F5-TTS)

## Dataset

You can explore more in this HuggingFace Dataset available at the given link for further details: [![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-htdung167%2Fvin100h--preprocessed--v2-yellow?style=flat&logo=huggingface)](https://huggingface.co/htdung167/vin100h-preprocessed-v2).

## Usage Guide

### Setup Instructions

#### Step 1: Clone the Repository
Clone the project repository and navigate to the project directory:
```bash
git clone https://github.com/danhtran2mind/Vi-F5-TTS.git
cd Vi-F5-TTS
```

#### Step 2: Install Dependencies
Install the required Python packages:
```bash
pip install -e . 
```

#### Step 3: Configure the Environment
Run the following scripts to set up the project:
- **Install Third-Party Dependencies**  
  ```bash
  python scripts/setup_third_party.py
  ```
- **Download Model Checkpoints**
    - Use `SWivid/F5-TTS`:
    ```bash
    python scripts/download_ckpts.py \
        --repo_id "SWivid/F5-TTS" --local_dir "./ckpts" \
        --folder_name "F5TTS_v1_Base_no_zero_init"
    ```
    - Use `danhtran2mind/Vi-F5-TTS`:
    ```bash
    python scripts/download_ckpts.py \
        --repo_id "danhtran2mind/Vi-F5-TTS" \
        --local_dir "./ckpts" --download_all
    ```

- **Prepare Dataset (Optional, for Training)**  
  ```bash
  python scripts/process_dataset.py
  ```

### Training

```bash
accelerate config default
```
Create `configs` folder in `src/f5_tts`:

```bash
mkdir -p src/f5_tts/configs
cp configs/configs/vi-fine-tuned-f5-tts.yaml src/f5_tts/configs
```
To train the model:
```bash
accelerate launch ./src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_Base \
    --dataset_name vin100h-preprocessed-v2 \
    --finetune \
    --tokenizer pinyin \
    --learning_rate 1e-05 \
    --batch_size_type frame \
    --batch_size_per_gpu 3200 \
    --max_samples 64 \
    --grad_accumulation_steps 2 \
    --max_grad_norm 1 \
    --epochs 80 \
    --num_warmup_updates 2761 \
    --save_per_updates 4000 \
    --keep_last_n_checkpoints 1 \
    --last_per_updates 4000 \
    --log_samples \
    --pretrain "<your_pretrain_model_path>" # such as./ckpts/F5TTS_v1_Base_no_zero_init/model_1250000.safetensors
```
- Training Hyperparameters
Refer to the [Training Documents](docs/training/training_doc.md) for detailed hyperparameters used in fine-tuning the model. ⚙️

### Inference
To generate videos using the trained model:
```bash
python src/text2video_ghibli_style/inference.py
```

## Project Description

This repository is trained from [![GitHub Repo](https://img.shields.io/badge/GitHub-danhtran2mind%2FMotionDirector-blue?style=flat&logo=github)](https://github.com/danhtran2mind/MotionDirector), a fork of [![GitHub Repo](https://img.shields.io/badge/GitHub-showlab%2FMotionDirector-blue?style=flat&logo=github)](https://github.com/showlab/MotionDirector), with numerous bug fixes and rewritten code for improved performance and stability. You can download the `zeroscope_v2_576w` model from the [![HuggingFace: cerspense/zeroscope_v2_576w](https://img.shields.io/badge/HuggingFace-cerspense%2Fzeroscope__v2__576w-yellow?logo=huggingface)](https://huggingface.co/cerspense/zeroscope_v2_576w). Explore more models on [![HuggingFace Hub](https://img.shields.io/badge/HuggingFace-cerspense-yellow?style=flat&logo=huggingface)](https://huggingface.co/cerspense).
