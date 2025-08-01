import gradio as gr
import os
import subprocess
import uuid
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import shutil

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
        return f"Transcription error: {str(e)}"

def get_files_in_ckpts(extensions, include_subdirs=False):
    """List files in ckpts directory with specified extensions, optionally including subdirectories."""
    ckpts_dir = Path("ckpts")
    if not ckpts_dir.exists():
        return ["No files found"]
    files = []
    for ext in extensions:
        if include_subdirs:
            files.extend([str(f) for f in ckpts_dir.glob(f"**/*{ext}")])
        else:
            files.extend([str(f) for f in ckpts_dir.glob(f"*{ext}")])
    return files if files else ["No files found"]

def handle_file_upload(file_obj, allowed_extensions):
    """Copy uploaded file to a permanent location and validate extension."""
    if not file_obj:
        return None, "No file uploaded."
    try:
        file_ext = os.path.splitext(file_obj.name)[1].lower()
        if file_ext not in allowed_extensions:
            return None, f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_name = f"upload_{str(uuid.uuid4())[:8]}{file_ext}"
        dest_path = upload_dir / file_name
        shutil.copyfile(file_obj.name, dest_path)
        return str(dest_path), None
    except Exception as e:
        return None, f"File upload error: {str(e)}"

def run_tts_inference(ref_audio, ref_text, gen_text, speed, use_upload, model_cfg, ckpt_file, vocab_file):
    """Run F5-TTS inference with selected or uploaded model files."""
    if use_upload:
        # Handle uploaded files with specific allowed extensions
        model_cfg_path, model_cfg_error = handle_file_upload(model_cfg, [".yaml"])
        ckpt_file_path, ckpt_file_error = handle_file_upload(ckpt_file, [".pt", ".safetensors"])
        vocab_file_path, vocab_file_error = handle_file_upload(vocab_file, [".txt", ".safetensors"])
        if model_cfg_error or ckpt_file_error or vocab_file_error:
            return None, model_cfg_error or ckpt_file_error or vocab_file_error
        if not (model_cfg_path and ckpt_file_path and vocab_file_path):
            return None, "Please upload all model files (model_cfg, ckpt_file, vocab_file)."
        config = {"model_cfg": model_cfg_path, "ckpt_file": ckpt_file_path, "vocab_file": vocab_file_path}
    else:
        if any(f == "No files found" for f in [model_cfg, ckpt_file, vocab_file]):
            return None, "No valid model files found in ckpts. Upload custom files or add files to ckpts."
        config = {"model_cfg": model_cfg, "ckpt_file": ckpt_file, "vocab_file": vocab_file}
    
    if not ref_audio:
        return None, "Reference audio is required."
    
    output_dir = "apps/gradio_app/temp_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"infer_audio_{str(uuid.uuid4())[:8]}.mp3"
    output_path = os.path.join(output_dir, output_file)
    
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
            return None, f"Inference error: {result.stderr}"
        if not os.path.exists(output_path):
            return None, f"Output audio not found at {output_path}"
        return output_path, "Audio generated successfully!"
    except Exception as e:
        return None, f"Inference error: {str(e)}"

def create_gradio_app():
    """Create Gradio interface for F5-TTS inference with Whisper ASR."""
    def update_ref_text(audio_file_path, use_whisper):
        if use_whisper and audio_file_path:
            return transcribe_audio(audio_file_path)
        return gr.update()

    def toggle_model_inputs(use_upload):
        return (
            gr.update(visible=not use_upload),  # Model cfg dropdown
            gr.update(visible=not use_upload),  # Ckpt file dropdown
            gr.update(visible=not use_upload),  # Vocab file dropdown
            gr.update(visible=use_upload),      # Model cfg upload
            gr.update(visible=use_upload),      # Ckpt file upload
            gr.update(visible=use_upload)       # Vocab file upload
        )

    # Load CSS
    CSS = open("apps/gradio_app/static/styles.css", "r").read()

    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("# F5-TTS Audio Generation")
        gr.Markdown("Generate high-quality audio with a fine-tuned F5-TTS model. Upload reference audio, use Whisper ASR for transcription, enter text, adjust speed, and select or upload model files.")

        with gr.Row():
            with gr.Column():
                ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                
                # Group for Whisper ASR, Reference Text, and Generated Text
                with gr.Group():
                    use_whisper = gr.Checkbox(label="Use Whisper ASR for Transcription", value=False)
                    ref_text = gr.Textbox(
                        label="Reference Text", 
                        placeholder="e.g., Sau nhà Ngô, lần lượt các triều Đinh...",
                        lines=1
                    )
                    gen_text = gr.Textbox(
                        label="Generated Text", 
                        placeholder="e.g., Nhà Tiền Lê, Lý và Trần đã chống trả...",
                        lines=1
                    )
                
                generate_btn = gr.Button("Generate Audio")

            with gr.Column():
                output_audio = gr.Audio(label="Generated Audio")
                output_text = gr.Textbox(label="Status", interactive=False)
                
                # Group for Speed, Model Config, Checkpoint File, Vocab File, and Upload Checkbox
                with gr.Group():
                    speed = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Speed")
                    model_cfg = gr.Dropdown(
                        choices=get_files_in_ckpts([".yaml"]),
                        label="Model Config (*.yaml)",
                        value=get_files_in_ckpts([".yaml"])[0],
                        visible=True
                    )
                    ckpt_file = gr.Dropdown(
                        choices=get_files_in_ckpts([".pt", ".safetensors"], include_subdirs=True),
                        label="Checkpoint File (*.pt or *.safetensors)",
                        value=get_files_in_ckpts([".pt", ".safetensors"], include_subdirs=True)[0],
                        visible=True
                    )
                    vocab_file = gr.Dropdown(
                        choices=get_files_in_ckpts([".txt", ".safetensors"]),
                        label="Vocab File (*.txt or *.safetensors)",
                        value=get_files_in_ckpts([".txt", ".safetensors"])[0],
                        visible=True
                    )
                    use_upload = gr.Checkbox(label="Upload Custom Model Files", value=False)
                
                # File upload inputs (hidden by default, with specific file extensions)
                model_cfg_upload = gr.File(label="Model Config (*.yaml)", file_types=[".yaml"], visible=False)
                ckpt_file_upload = gr.File(label="Checkpoint File (*.pt or *.safetensors)", file_types=[".pt", ".safetensors"], visible=False)
                vocab_file_upload = gr.File(label="Vocab File (*.txt or *.safetensors)", file_types=[".txt", ".safetensors"], visible=False)

        ref_audio.change(fn=update_ref_text, inputs=[ref_audio, use_whisper], outputs=ref_text)
        use_whisper.change(fn=update_ref_text, inputs=[ref_audio, use_whisper], outputs=ref_text)
        use_upload.change(
            fn=toggle_model_inputs,
            inputs=[use_upload],
            outputs=[model_cfg, ckpt_file, vocab_file, model_cfg_upload, ckpt_file_upload, vocab_file_upload]
        )
        generate_btn.click(
            fn=run_tts_inference,
            inputs=[ref_audio, ref_text, gen_text, speed, use_upload, model_cfg, ckpt_file, vocab_file],
            outputs=[output_audio, output_text]
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch()