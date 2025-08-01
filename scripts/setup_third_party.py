import os
import shutil
import subprocess
import argparse
import sys

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
    shutil.move(os.path.join(temp_f5_tts_target_dir, "data"), ".")
    # Remove the parent directory
    shutil.rmtree(temp_f5_tts_target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clone F5-TTS and BigVGAN repositories")
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
        # default="7d2b454564a6c7d014227f635b7423881f14bdac",
        default="main",
        help="Branch or commit for BigVGAN repository"
    )
    
    args = parser.parse_args()
    main(args)