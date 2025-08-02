# TrainingArguments

## Overview
The `TrainingArguments` configuration is used to define the parameters for training the CFM (Conditional Flow Matching) model. These arguments are parsed using Python's `argparse` module and control various aspects of the training process, including model selection, dataset, learning rate, batch size, and more.

## Arguments

| Argument                  | Type        | Default Value                     | Description                                                                 |
|---------------------------|-------------|-----------------------------------|-----------------------------------------------------------------------------|
| `--exp_name`              | `str`       | `"F5TTS_v1_Base"`                 | Experiment name. Choices: `["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"]`. |
| `--dataset_name`          | `str`       | `"Emilia_ZH_EN"`                  | Name of the dataset to use.                                                |
| `--learning_rate`         | `float`     | `1e-5`                            | Learning rate for training.                                                |
| `--batch_size_per_gpu`    | `int`       | `3200`                            | Batch size per GPU.                                                        |
| `--batch_size_type`       | `str`       | `"frame"`                         | Batch size type. Choices: `["frame", "sample"]`.                           |
| `--max_samples`           | `int`       | `64`                              | Maximum sequences per batch.                                               |
| `--grad_accumulation_steps` | `int`     | `1`                               | Number of gradient accumulation steps.                                     |
| `--max_grad_norm`         | `float`     | `1.0`                             | Maximum gradient norm for clipping.                                        |
| `--epochs`                | `int`       | `100`                             | Number of training epochs.                                                 |
| `--num_warmup_updates`    | `int`       | `20000`                           | Number of warmup updates.                                                  |
| `--save_per_updates`      | `int`       | `50000`                           | Save checkpoint every N updates.                                           |
| `--keep_last_n_checkpoints` | `int`     | `-1`                              | Number of checkpoints to keep (-1 for all, 0 for none, >0 for last N).     |
| `--last_per_updates`      | `int`       | `5000`                            | Save last checkpoint every N updates.                                      |
| `--finetune`              | `flag`      | `False`                           | Enable finetuning mode.                                                    |
| `--pretrain`              | `str`       | `None`                            | Path to the pretrained checkpoint.                                         |
| `--tokenizer`             | `str`       | `"pinyin"`                        | Tokenizer type. Choices: `["pinyin", "char", "custom"]`.                   |
| `--tokenizer_path`        | `str`       | `None`                            | Path to custom tokenizer vocab file (used if tokenizer is `"custom"`).     |
| `--log_samples`           | `flag`      | `False`                           | Log inferenced samples per checkpoint save updates.                        |
| `--logger`                | `str`       | `None`                            | Logger type. Choices: `[None, "wandb", "tensorboard"]`.                    |
| `--bnb_optimizer`         | `flag`      | `False`                           | Use 8-bit Adam optimizer from bitsandbytes.                                |

## Dataset Settings
The following settings are defined globally in the script and are used for dataset processing:

| Setting                 | Value       | Description                                                                 |
|-------------------------|-------------|-----------------------------------------------------------------------------|
| `target_sample_rate`    | `24000`     | Target sample rate for audio processing (in Hz).                            |
| `n_mel_channels`        | `100`       | Number of mel-spectrogram channels.                                        |
| `hop_length`            | `256`       | Hop length for mel-spectrogram computation.                                |
| `win_length`            | `1024`      | Window length for mel-spectrogram computation.                             |
| `n_fft`                 | `1024`      | Number of FFT components for mel-spectrogram computation.                  |
| `mel_spec_type`         | `"vocos"`   | Type of mel-spectrogram vocoder. Choices: `["vocos", "bigvgan"]`.          |

## Usage
To run the training script with custom arguments, you can use the following command-line example:

```bash
python train.py --exp_name F5TTS_Base --dataset_name Emilia_ZH_EN --learning_rate 1e-5 --batch_size_per_gpu 3200 --epochs 100 --finetune
```

This command specifies the experiment name, dataset, learning rate, batch size, number of epochs, and enables finetuning.

## Notes
- The script supports three model configurations: `F5TTS_v1_Base`, `F5TTS_Base`, and `E2TTS_Base`, each with specific model architectures (`DiT` or `UNetT`) and parameters.
- When `--finetune` is enabled, a pretrained checkpoint is required, either provided via `--pretrain` or automatically downloaded from a default URL.
- The `--tokenizer` argument determines the tokenization method, with `"custom"` requiring a `--tokenizer_path`.
- The script creates a checkpoint directory based on the dataset name and copies the pretrained model if finetuning is enabled.