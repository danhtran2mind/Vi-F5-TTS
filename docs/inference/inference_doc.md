# Inference Arguments

The following table describes the command-line arguments available for the `infer-cli.py` script, which is used for text-to-speech (TTS) inference with advanced batch processing capabilities. These arguments allow users to override settings defined in the configuration file (`basic.toml` by default).

| Argument | Description | Type | Default Value | Notes |
|----------|-------------|------|---------------|-------|
| `-c`, `--config` | Path to the configuration file. | `str` | `f5_tts/infer/examples/basic/basic.toml` | Specifies the TOML configuration file to use. |
| `-m`, `--model` | Model name to use for inference. | `str` | `F5TTS_v1_Base` (from config) | Options: `F5TTS_v1_Base`, `F5TTS_Base`, `E2TTS_Base`, etc. |
| `-mc`, `--model_cfg` | Path to the model's YAML configuration file. | `str` | `configs/<model>.yaml` (from config) | Defines model-specific settings. |
| `-p`, `--ckpt_file` | Path to the model checkpoint file (.pt). | `str` | (from config) | Leave blank to use default checkpoint. |
| `-v`, `--vocab_file` | Path to the vocabulary file (.txt). | `str` | (from config) | Leave blank to use default vocabulary. |
| `-r`, `--ref_audio` | Path to the reference audio file. | `str` | `infer/examples/basic/basic_ref_en.wav` (from config) | Used as a reference for voice synthesis. |
| `-s`, `--ref_text` | Transcript or subtitle for the reference audio. | `str` | `Some call me nature, others call me mother nature.` (from config) | Text corresponding to the reference audio. |
| `-t`, `--gen_text` | Text to synthesize into speech. | `str` | `Here we generate something just for test.` (from config) | Ignored if `--gen_file` is provided. |
| `-f`, `--gen_file` | Path to a file containing text to synthesize. | `str` | (from config) | Overrides `--gen_text` if specified. |
| `-o`, `--output_dir` | Path to the output directory. | `str` | `tests` (from config) | Directory where generated audio files are saved. |
| `-w`, `--output_file` | Name of the output audio file. | `str` | `infer_cli_<timestamp>.wav` (from config) | Timestamp format: `%Y%m%d_%H%M%S`. |
| `--save_chunk` | Save individual audio chunks during inference. | `bool` | `False` (from config) | If enabled, saves chunks to `<output_dir>/<output_file>_chunks/`. |
| `--no_legacy_text` | Disable lossy ASCII transliteration for Unicode text in file names. | `bool` | `False` (from config) | If disabled, uses Unicode in file names; warns if used with `--save_chunk`. |
| `--remove_silence` | Remove long silences from the generated audio. | `bool` | `False` (from config) | Applies silence removal post-processing. |
| `--load_vocoder_from_local` | Load vocoder from a local directory. | `bool` | `False` (from config) | Uses `../checkpoints/vocos-mel-24khz` or similar if enabled. |
| `--vocoder_name` | Name of the vocoder to use. | `str` | (from config, defaults to `mel_spec_type`) | Options: `vocos`, `bigvgan`. |
| `--target_rms` | Target loudness normalization value for output speech. | `float` | (from config, defaults to `target_rms`) | Adjusts audio loudness. |
| `--cross_fade_duration` | Duration of cross-fade between audio segments (seconds). | `float` | (from config, defaults to `cross_fade_duration`) | Smooths transitions between segments. |
| `--nfe_step` | Number of function evaluation (denoising) steps. | `int` | (from config, defaults to `nfe_step`) | Controls inference quality. |
| `--cfg_strength` | Classifier-free guidance strength. | `float` | (from config, defaults to `cfg_strength`) | Influences generation quality. |
| `--sway_sampling_coef` | Sway sampling coefficient. | `float` | (from config, defaults to `sway_sampling_coef`) | Affects sampling behavior. |
| `--speed` | Speed of the generated audio. | `float` | (from config, defaults to `speed`) | Adjusts playback speed. |
| `--fix_duration` | Fixed total duration for reference and generated audio (seconds). | `float` | (from config, defaults to `fix_duration`) | Enforces a specific duration. |
| `--device` | Device to run inference on. | `str` | (from config, defaults to `device`) | E.g., `cpu`, `cuda`. |

## Notes
- Arguments without default values in the script (e.g., `--model`, `--ref_audio`) inherit defaults from the configuration file.
- The `--no_legacy_text` flag is implemented as `store_false`, so enabling it sets `use_legacy_text` to `False`.
- If `--gen_file` is provided, it overrides `--gen_text`.
- The script supports multiple voices defined in the config file under the `voices` key, with a fallback to a `main` voice.
- The output audio is saved as a WAV file, and optional chunked audio segments can be saved if `--save_chunk` is enabled.
- The script uses `cached_path` for downloading model checkpoints from Hugging Face if no local checkpoint is specified.