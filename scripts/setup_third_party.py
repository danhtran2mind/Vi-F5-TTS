import os
import shutil
import subprocess
import argparse
import sys

def install_editable_package():
    """Install the f5-tts package in editable mode without dependencies."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
            check=True,
            capture_output=True,
            text=True
        )
        print("Successfully installed f5-tts package in editable mode")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install f5-tts package: {e.stderr}")
        sys.exit(1)

def clone_repository(repo_url, target_dir, branch="main"):
    """Clone a git repository to the specified directory with specific branch."""
    if os.path.exists(target_dir):
        print(f"Directory {target_dir} already exists. Skipping clone.")
        return
    
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    
    try:
        subprocess.run(
            ["git", "clone", "-b", branch, repo_url, target_dir],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully cloned {repo_url} (branch: {branch}) to {target_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e.stderr}")
        sys.exit(1)

def handle_remove_readonly(func, path, _):
    os.chmod(path, 0o666)
    func(path)

def setup_python_path():
    """Add the src directory to sys.path to allow module imports."""
    src_dir = os.path.abspath("src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        print(f"Added {src_dir} to sys.path")
    else:
        print(f"{src_dir} already in sys.path")

def main(args):
    # Define target directories
    temp_f5_tts_target_dir = os.path.join("src", "danhtran2mind_f5_tts")
    bigvgan_target_dir = os.path.join("src", "third_party", "BigVGAN")
    f5_tts_target_dir = os.path.join("src", "f5_tts")

    # Clone F5-TTS repository
    clone_repository(args.f5_tts_url, temp_f5_tts_target_dir, args.f5_tts_branch)
    
    # Clone BigVGAN repository
    clone_repository(args.bigvgan_url, bigvgan_target_dir, args.bigvgan_branch)
    
    # Move the directory
    shutil.move(os.path.join(temp_f5_tts_target_dir, "src", "f5_tts"), f5_tts_target_dir)
    shutil.copytree(os.path.join(temp_f5_tts_target_dir, "data"), "./data", dirs_exist_ok=True)
    # Remove the nested f5_tts directory
    if os.path.exists(os.path.join(f5_tts_target_dir, "f5_tts")):
        shutil.rmtree(os.path.join(f5_tts_target_dir, "f5_tts"), onerror=handle_remove_readonly)
    # Remove the parent directory
    shutil.rmtree(temp_f5_tts_target_dir, onerror=handle_remove_readonly)

    # Set up Python path to include src directory
    install_editable_package()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone F5-TTS and BigVGAN repositories and set up Python path")
    parser.add_argument(
        "--f5-tts-url",
        default="https://github.com/danhtran2mind/F5-TTS",
        help="URL for F5-TTS repository"
    )
    parser.add_argument(
        "--bigvgan-url",
        default="https://github.com/NVIDIA/BigVGAN",
        help="URL for BigVGAN repository"
    )
    parser.add_argument(
        "--f5-tts-branch",
        default="main",
        help="Branch for F5-TTS repository"
    )
    parser.add_argument(
        "--bigvgan-branch",
        default="main",
        help="Branch or commit for BigVGAN repository"
    )
    
    args = parser.parse_args()
    main(args)