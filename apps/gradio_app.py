import gradio as gr
import os
import subprocess
import tempfile
from pathlib import Path

def run_tts_inference(ref_audio, ref_text, gen_text, speed, model_option):
    """
    Run the F5-TTS inference script with provided inputs and return the output audio path.
    """
    # Define model configurations based on the selected option
    model_configs = {
        "Vietnamese Fine-Tuned": {
            "model_cfg": "ckpts/vi-fine-tuned-f5-tts.yaml",
            "ckpt_file": "ckpts/Vi_F5_TTS_ckpts/model_last.pt",
            "vocab_file": "ckpts/vocab.txt"
        },
        # Add more model options here if needed
    }
    
    if model_option not in model_configs:
        return None, f"Invalid model option: {model_option}"
    
    config = model_configs[model_option]
    
    # Create temporary directory for input and output files
    output_dir = "apps/gradio_app/temp_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = "infer_audio.mp3"
    output_path = os.path.join(output_dir, output_file)
    
    # Use the provided ref_audio path directly (it's already a file path)
    if ref_audio:
        temp_audio = ref_audio  # No need to save, it's already a file path
    else:
        return None, "Reference audio is required"
    
    # Save reference and generated text to temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_ref_text:
        temp_ref_text.write(ref_text or "")
        temp_ref_text_path = temp_ref_text.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_gen_text:
        temp_gen_text.write(gen_text or "")
        temp_gen_text_path = temp_gen_text.name
    
    try:
        # Construct the command
        command = [
            "python", "src/f5_tts/infer/infer_cli.py",
            "--model_cfg", config["model_cfg"],
            "--ckpt_file", config["ckpt_file"],
            "--vocab_file", config["vocab_file"],
            "--ref_audio", temp_audio,
            "--ref_text", temp_ref_text_path,
            "--gen_text", temp_gen_text_path,
            "--speed", str(speed),
            "--output_dir", output_dir,
            "--output_file", output_file
        ]
        
        # Run the inference command
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            return None, f"Error running inference: {result.stderr}"
        
        if not os.path.exists(output_path):
            return None, f"Output audio file not found at {output_path}"
        
        return output_path, "Audio generated successfully!"
        
    except Exception as e:
        return None, f"Error during inference: {str(e)}"
    
    finally:
        # Clean up temporary text files (but not the ref_audio, as it's managed by Gradio)
        if os.path.exists(temp_ref_text_path):
            os.remove(temp_ref_text_path)
        if os.path.exists(temp_gen_text_path):
            os.remove(temp_gen_text_path)

def create_gradio_app():
    """
    Create and return a Gradio interface for the F5-TTS inference.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# F5-TTS Audio Generation App")
        gr.Markdown("Generate audio using a fine-tuned F5-TTS model. Upload a reference audio, provide reference and generated text, and adjust the speed.")
        
        with gr.Row():
            with gr.Column():
                ref_audio = gr.Audio(label="Reference Audio", type="filepath")
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
        
        generate_btn.click(
            fn=run_tts_inference,
            inputs=[ref_audio, ref_text, gen_text, speed, model_option],
            outputs=[output_audio, output_text]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch()
