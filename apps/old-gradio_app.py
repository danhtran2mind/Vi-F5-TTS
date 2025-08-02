import gradio as gr
import os
import subprocess
import tempfile
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

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

def run_tts_inference(ref_audio, ref_text, gen_text, speed, model_option):
    """
    Run the F5-TTS inference script with provided inputs and return the output audio path.
    """
    model_configs = {
        "Vietnamese Fine-Tuned": {
            "model_cfg": "ckpts/vi-fine-tuned-f5-tts.yaml",
            "ckpt_file": "ckpts/Vi_F5_TTS_ckpts/pruning_model.pt",
            "vocab_file": "ckpts/vocab.txt"
        },
    }
    
    if model_option not in model_configs:
        return None, f"Invalid model option: {model_option}"
    
    config = model_configs[model_option]
    
    output_dir = "apps/gradio_app/temp_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = "infer_audio.mp3"
    output_path = os.path.join(output_dir, output_file)
    
    if ref_audio:
        temp_audio = ref_audio
    else:
        return None, "Reference audio is required"
    
    # with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_ref_text:
    #     temp_ref_text.write(ref_text or "")
    #     temp_ref_text_path = temp_ref_text.name
    
    # with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_gen_text:
    #     temp_gen_text.write(gen_text or "")
    #     temp_gen_text_path = temp_gen_text.name
    
    try:
        command = [
            "python", "src/f5_tts/infer/infer_cli.py",
            "--model_cfg", config["model_cfg"],
            "--ckpt_file", config["ckpt_file"],
            "--vocab_file", config["vocab_file"],
            "--ref_audio", temp_audio,
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
    """
    Create and return a Gradio interface for the F5-TTS inference with optional Whisper ASR.
    """
    def update_ref_text(audio_file_path, use_whisper):
        """Conditionally transcribe audio based on Whisper checkbox."""
        if use_whisper and audio_file_path:
            return transcribe_audio(audio_file_path)
        return gr.update()  # Keep current text if Whisper is disabled or no audio

    with gr.Blocks() as demo:
        gr.Markdown("# F5-TTS Audio Generation App")
        gr.Markdown("Generate audio using a fine-tuned F5-TTS model. Upload a reference audio, enable Whisper ASR for auto-transcription or manually enter reference text, provide generated text, and adjust the speed.")

        with gr.Row():
            with gr.Column():
                ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                use_whisper = gr.Checkbox(label="Use Whisper ASR for Reference Text", value=False)
                ref_text = gr.Textbox(label="Reference Text", placeholder="e.g., Sau nhà Ngô, lần lượt các triều Đinh...")
                gen_text = gr.Textbox(label="Generated Text", placeholder="e.g., Nhà Tiền Lê, Lý và Trần đã chống trả...")
                speed = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Speed")
                model_option = gr.Dropdown(
                    choices=["Vietnamese Fine-Tuned"],
                    label="Model Option",
                    value="Vietnamese Fine-Tuned"
                )
                generate_btn = gr.Button("Generate Audio")
            
            with gr.Column():
                output_audio = gr.Audio(label="Generated Audio")
                output_text = gr.Textbox(label="Status")
        
        # Update reference text when audio is uploaded or Whisper checkbox changes
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
        
        generate_btn.click(
            fn=run_tts_inference,
            inputs=[ref_audio, ref_text, gen_text, speed, model_option],
            outputs=[output_audio, output_text]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(share=True)
