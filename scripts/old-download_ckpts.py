from huggingface_hub import snapshot_download
import os
import argparse

def download_ckpts(repo_id, local_dir, folder_name=None):
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    if folder_name:
        # Download only the specific folder
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{folder_name}/*"],  # Download only files in this folder
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Ensure files are copied, not symlinked
        )
        print(f"Downloaded {folder_name} from {repo_id} to {local_dir}")
    else:
        # Download entire repository
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Ensure files are copied, not symlinked
        )
        print(f"Downloaded entire repository {repo_id} to {local_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model checkpoints from HuggingFace")
    parser.add_argument("--repo_id", type=str, default="SWivid/F5-TTS", 
                        help="HuggingFace repository ID")
    parser.add_argument("--local_dir", type=str, default="./ckpts", 
                        help="Local directory to save checkpoints")
    parser.add_argument("--folder_name", type=str, default="F5TTS_v1_Base_no_zero_init", 
                        help="Specific folder to download (optional)")
    parser.add_argument("--download_all", action="store_true", 
                        help="Download entire repository instead of specific folder")
    
    args = parser.parse_args()
    
    # If download_all is specified, don't use folder filtering
    folder_name = args.folder_name if not args.download_all else None
    
    # Override folder_name for default repo
    if args.repo_id == "SWivid/F5-TTS" and not args.download_all and not args.folder_name:
        folder_name = "F5TTS_v1_Base_no_zero_init"
    
    download_ckpts(args.repo_id, args.local_dir, folder_name)
