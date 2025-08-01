import gradio as gr
import os
import subprocess
import tempfile
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import uuid

def transcribe_audio(audio_file_path):
    """Transcribe audio using PhoWhisper-tiny model."""
    try:
        processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("vinai/PhoWhisper-tiny")
        audio, sr = librosa.load(audio_file_path, sr=16000)
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0] if transcription else ""
    except Exception as e:
        return f"Error during transcription: {str(e)}"

def get_model_configs():
    """Scan ckpts directory for valid model configurations."""
    ckpts_dir = Path("ckpts")
    if not ckpts_dir.exists():
        return []
    
    yaml_files = list(ckpts_dir.glob("*.yaml"))
    ckpt_files = list(ckpts_dir.glob("*.pt")) + list(ckpts_dir.glob("*.safetensors"))
    vocab_files = list(ckpts_dir.glob("*.txt")) + list(ckpts_dir.glob("*.safetensors"))
    
    configs = []
    for yaml_file in yaml_files:
        for ckpt_file in ckpt_files:
            for vocab_file in vocab_files:
                config_name = f"{yaml_file.stem} | {ckpt_file.stem} | {vocab_file.stem}"
                configs.append({
                    "name": config_name,
                    "model_cfg": str(yaml_file),
                    "ckpt_file": str(ckpt_file),
                    "vocab_file": str(vocab_file)
                })
    return configs

def run_tts_inference(ref_audio, ref_text, gen_text, speed, model_selection, use_upload, model_cfg_file, ckpt_file, vocab_file):
    """Run the F5-TTS inference script with provided inputs and return the output audio path."""
    if use_upload:
        if not (model_cfg_file and ckpt_file and vocab_file):
            return None, "Please upload all required model files (model_cfg, ckpt_file, vocab_file)."
        config = {
            "model_cfg": model_cfg_file,
            "ckpt_file": ckpt_file,
            "vocab_file": vocab_file
        }
    else:
        configs = get_model_configs()
        selected_config = next((c for c in configs if c["name"] == model_selection), None)
        if not selected_config:
            return None, f"Invalid model selection: {model_selection}"
        config = {
            "model_cfg": selected_config["model_cfg"],
            "ckpt_file": selected_config["ckpt_file"],
            "vocab_file": selected_config["vocab_file"]
        }

    output_dir = "apps/gradio_app/temp_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"infer_audio_{str(uuid.uuid4())[:8]}.mp3"
    output_path = os.path.join(output_dir, output_file)

    if not ref_audio:
        return None, "Reference audio is required"

    try:
        command = [
            "python", "src/f5_tts/infer/infer_cli.py",
            "--model_cfg", config["model_cfg"],
            "--ckpt_file", config["ckpt_file"],
            "--vocab_file", config["vocab_file"],
            "--ref_audio", ref_audio,
            "--ref_text", ref_text,
            "--gen_text", gen_text,
            "--speed", str(speed),
            "--output_dir", output_dir,
            "--output_file", output_file
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            return None, f"Error running inference: {result.stderr}"

        if not os.path.exists(output_path):
            return None, f"Output audio file not found at {output_path}"

        return output_path, "Audio generated successfully!"

    except Exception as e:
        return None, f"Error during inference: {str(e)}"

def create_gradio_app():
    """Create and return a Gradio interface for the F5-TTS inference with optional Whisper ASR."""
    custom_css = """
    body {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%) !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }
    [data-theme="dark"] body {
        background: linear-gradient(135deg, #1e2a44 0%, #2e3b55 100%) !important;
        color: #e0e6ed !important;
    }
    .gradio-container {
        background: transparent !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 20px !important;
    }
    h1, .gr-title {
        color: #007bff !important;
        font-weight: 700 !important;
        text-align: center !important;
        margin-bottom: 20px !important;
    }
    [data-theme="dark"] h1, [data-theme="dark"] .gr-title {
        color: #4da8ff !important;
    }
    .gr-description {
        color: #333333 !important;
        font-size: 1.1em !important;
        text-align: center !important;
        margin-bottom: 30px !important;
    }
    [data-theme="dark"] .gr-description {
        color: #b0b8c4 !important;
    }
    .gr-input, .gr-output {
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        background: #ffffff !important;
        padding: 15px !important;
        transition: all 0.3s ease !important;
    }
    [data-theme="dark"] .gr-input, [data-theme="dark"] .gr-output {
        background: #2a3b5a !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        color: #e0e6ed !important;
    }
    .gr-button {
        background: linear-gradient(90deg, #007bff 0%, #00c4cc 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .gr-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(0,123,255,0.3) !important;
    }
    [data-theme="dark"] .gr-button {
        background: linear-gradient(90deg, #4da8ff 0%, #00d4dd 100%) !important;
    }
    .gr-checkbox label, .gr-dropdown label, .gr-slider label, .gr-textbox label, .gr-audio label {
        color: #333333 !important;
        font-weight: 500 !important;
    }
    [data-theme="dark"] .gr-checkbox label, [data-theme="dark"] .gr-dropdown label,
    [data-theme="dark"] .gr-slider label, [data-theme="dark"] .gr-textbox label,
    [data-theme="dark"] .gr-audio label {
        color: #d0d8e4 !important;
    }
    .gr-row {
        gap: 20px !important;
    }
    .gr-column {
        padding: 15px !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.8) !important;
    }
    [data-theme="dark"] .gr-column {
        background: rgba(42,59,90,0.8) !important;
    }
    """
    
    def update_ref_text(audio_file_path, use_whisper):
        """Conditionally transcribe audio based on Whisper checkbox."""
        if use_whisper and audio_file_path:
            return transcribe_audio(audio_file_path)
        return gr.update()

    def toggle_model_inputs(use_upload):
        """Show/hide model file upload inputs or dropdown based on toggle."""
        if use_upload:
            return (
                gr.update(visible=False),  # Hide dropdown
                gr.update(visible=True),   # Show model_cfg upload
                gr.update(visible=True),   # Show ckpt_file upload
                gr.update(visible=True)    # Show vocab_file upload
            )
        else:
            return (
                gr.update(visible=True),   # Show dropdown
                gr.update(visible=False),  # Hide model_cfg upload
                gr.update(visible=False),  # Hide ckpt_file upload
                gr.update(visible=False)   # Hide vocab_file upload
            )

    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("# F5-TTS Audio Generation App")
        gr.Markdown("Generate audio using a fine-tuned F5-TTS model. Upload a reference audio, enable Whisper ASR for auto-transcription or manually enter reference text, provide generated text, and adjust the speed. Select a model configuration or upload custom model files.")

        with gr.Row():
            with gr.Column():
                ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                use_whisper = gr.Checkbox(label="Use Whisper ASR for Reference Text", value=False)
                ref_text = gr.Textbox(label="Reference Text", placeholder="e.g., Sau nhà Ngô, lần lượt các triều Đinh...")
                gen_text = gr.Textbox(label="Generated Text", placeholder="e.g., Nhà Tiền Lê, Lý và Trần đã chống trả...")
                speed = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Speed")
                
                use_upload = gr.Checkbox(label="Upload Custom Model Files", value=False)
                
                model_configs = get_model_configs()
                model_choices = [c["name"] for c in model_configs] if model_configs else ["No configurations found"]
                model_selection = gr.Dropdown(
                    choices=model_choices,
                    label="Model Configuration",
                    value=model_choices[0] if model_choices else None,
                    visible=not use_upload
                )
                
                model_cfg_file = gr.File(label="Model Config File (*.yaml)", file_types=[".yaml"], visible=False)
                ckpt_file = gr.File(label="Checkpoint File (*.pt or *.safetensors)", file_types=[".pt", ".safetensors"], visible=False)
                vocab_file = gr.File(label="Vocab File (*.txt or *.safetensors)", file_types=[".txt", ".safetensors"], visible=False)
                
                generate_btn = gr.Button("Generate Audio")

            with gr.Column():
                output_audio = gr.Audio(label="Generated Audio")
                output_text = gr.Textbox(label="Status")

        ref_audio.change(
            fn=update_ref_text,
            inputs=[ref_audio, use_whisper],
            outputs=ref_text
        )
        use_whisper.change(
            fn=update_ref_text,
            inputs=[ref_audio, use_whisper],
            outputs=ref_text
        )
        
        use_upload.change(
            fn=toggle_model_inputs,
            inputs=[use_upload],
            outputs=[model_selection, model_cfg_file, ckpt_file, vocab_file]
        )

        generate_btn.click(
            fn=run_tts_inference,
            inputs=[ref_audio, ref_text, gen_text, speed, model_selection, use_upload, model_cfg_file, ckpt_file, vocab_file],
            outputs=[output_audio, output_text]
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(share=True)