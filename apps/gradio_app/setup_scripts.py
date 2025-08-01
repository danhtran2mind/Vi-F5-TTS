import subprocess
import sys
import os

def run_script(script_path):
    """
    Run a Python script using subprocess and handle potential errors.
    Returns True if successful, False otherwise.
    """
    try:
        result = subprocess.run(
            [sys.executable, script_path],
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
        os.path.join(scripts_dir, "setup_third_party.py"),
        os.path.join(scripts_dir, "download_ckpts.py")
    ]

    for script in scripts:
        print(f"Start running {script}\n")
        if not run_script(script):
            print(f"Stopping execution due to error in {script}")
            sys.exit(1)
        print(f"Completed {script}\n")

if __name__ == "__main__":
    main()