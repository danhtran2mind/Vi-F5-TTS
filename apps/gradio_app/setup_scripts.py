import subprocess
import sys
import os

def run_script(script_path, args=None):
    """
    Run a Python script using subprocess with optional arguments and handle errors.
    Returns True if successful, False otherwise.
    """
    try:
        command = [sys.executable, script_path]
        if args:
            command.extend(args)
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True
        )
        print(f"Successfully executed {script_path}")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_path}:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"Script not found: {script_path}")
        return False

def main():
    """
    Main function to execute setup_third_party.py and download_ckpts.py in sequence.
    """
    scripts_dir = "scripts"
    scripts = [
        {
            "path": os.path.join(scripts_dir, "setup_third_party.py"),
            "args": None
        },
        {
            "path": os.path.join(scripts_dir, "download_ckpts.py"),
            "args": [
                "--repo_id", "danhtran2mind/Vi-F5-TTS",
                "--local_dir", "./ckpts",
                "--pruning_model"
            ]
        }
    ]

    for script in scripts:
        script_path = script["path"]
        args = script["args"]
        print(f"Start running {script_path} {' '.join(args) if args else ''}\n")
        if not run_script(script_path, args):
            print(f"Stopping execution due to error in {script_path}")
            sys.exit(1)
        print(f"Completed {script_path}\n")

if __name__ == "__main__":
    main()