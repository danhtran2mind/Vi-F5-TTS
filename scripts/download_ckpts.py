from huggingface_hub import snapshot_download
import os
import argparse

def download_ckpts(repo_id, local_dir, folder_name=None, pruning_model=False):
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Initialize allow_patterns
    allow_patterns = None

    if pruning_model and repo_id == "danhtran2mind/Vi-F5-TTS":
        # Download only specific files when pruning_model is enabled
        allow_patterns = [
            "Vi_F5_TTS_ckpts/pruning_model.pt",
            ".gitattributes",
            "README.md",
            "vi-fine-tuned-f5-tts.yaml",
            "vocab.txt"
        ]
        print(f"Downloading only {', '.join(allow_patterns)} from {repo_id}")
    elif folder_name:
        # Download only the specific folder
        allow_patterns = [f"{folder_name}/*"]
        print(f"Downloading {folder_name} from {repo_id}")
    else:
        # Download entire repository
        print(f"Downloading entire repository {repo_id}")

    # Perform the download
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Ensure files are copied, not symlinked
    )

    # Print completion message
    if pruning_model:
        print(f"Downloaded specified files from {repo_id} to {local_dir}")
    elif folder_name:
        print(f"Downloaded {folder_name} from {repo_id} to {local_dir}")
    else:
        print(f"Downloaded entire repository {repo_id} to {local_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model checkpoints from HuggingFace")
    parser.add_argument("--repo_id", type=str, default="danhtran2mind/Vi-F5-TTS",
                        help="HuggingFace repository ID")
    parser.add_argument("--local_dir", type=str, default="./ckpts",
                        help="Local directory to save checkpoints")
    parser.add_argument("--folder_name", type=str, default="Vi_F5_TTS_ckpts",
                        help="Specific folder to download (optional, ignored if --pruning_model is used)")
    parser.add_argument("--pruning_model", action="store_true",
                        help="Download only Vi_F5_TTS_ckpts/pruning_model.pt, .gitattributes, README.md, vi-fine-tuned-f5-tts.yaml, and vocab.txt from danhtran2mind/Vi-F5-TTS")

    args = parser.parse_args()

    # Override folder_name for default repo, unless pruning_model is enabled
    if args.repo_id == "danhtran2mind/Vi-F5-TTS" and not args.folder_name and not args.pruning_model:
        folder_name = "Vi_F5_TTS_ckpts"
    else:
        folder_name = args.folder_name

    download_ckpts(args.repo_id, args.local_dir, folder_name, args.pruning_model)