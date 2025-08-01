import os
import sys
import subprocess
import uuid
from pathlib import Path
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_setup_script():
    setup_script = os.path.join(os.path.dirname(__file__), "setup_scripts.py")
    try:
        result = subprocess.run(["python", setup_script], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Setup script failed: {e.stderr}"


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