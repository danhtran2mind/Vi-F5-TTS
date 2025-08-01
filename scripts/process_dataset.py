import json
import os
import sys
from pathlib import Path
import shutil
import torchaudio
from datasets import load_dataset
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm
import soundfile as sf
import csv
import subprocess
import argparse

def save_dataset_to_local_disk(output_dir, base_model, audio_header, text_header):
    """
    Saves a dataset to a local directory.

    Args:
        output_dir (str): The directory to save the dataset to.
        base_model (str): The base model to load the dataset from.
        audio_header (str): The header for the audio data in the dataset.
        text_header (str): The header for the text data in the dataset.
    """
    wavs_dir = os.path.join(output_dir, "wavs")
    metadata_path = os.path.join(output_dir, "metadata.csv")
    os.makedirs(wavs_dir, exist_ok=True)

    try:
        ds = load_dataset(base_model)['train']
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return

    metadata = []
    for idx, sample in tqdm(enumerate(ds), total=len(ds), desc="Saving samples to directory"):
        try:
            audio_array = sample[audio_header]['array']
            sampling_rate = sample[audio_header]['sampling_rate']
            filename = f"audio_{idx:06d}.wav"
            sf.write(os.path.join(wavs_dir, filename), audio_array, sampling_rate)
            metadata.append([f"wavs/{filename}", sample[text_header]])
        except Exception as e:
            print(f"Error processing sample {idx}: {e}", file=sys.stderr)
            continue

    try:
        with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f, delimiter='|').writerows(metadata)
        print(f"Dataset saved to {output_dir}")
    except Exception as e:
        print(f"Error writing metadata: {e}", file=sys.stderr)

def run_preprocess(input_dir, output_dir, workers):
    """
    Runs the preprocessing script with real-time output.

    Args:
        input_dir (str): Input directory for preprocessing.
        output_dir (str): Output directory for processed data.
        workers (int): Number of parallel processes.
    """
    script_path = "./src/f5_tts/train/datasets/prepare_csv_wavs.py"
    if not os.path.exists(script_path):
        print(f"Preprocessing script not found at {script_path}", file=sys.stderr)
        return

    command = [
        "python", script_path,
        input_dir, output_dir,
        "--workers", str(workers)
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Real-time output for stdout and stderr
        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()

            if stdout_line:
                print(stdout_line, end='', flush=True)
            if stderr_line:
                print(stderr_line, end='', flush=True, file=sys.stderr)

            if process.poll() is not None:
                break

        # Capture any remaining output
        stdout, stderr = process.communicate()
        if stdout:
            print(stdout, end='', flush=True)
        if stderr:
            print(stderr, end='', flush=True, file=sys.stderr)

        if process.returncode == 0:
            print("\nPreprocessing completed successfully.")
        else:
            print(f"\nPreprocessing failed with return code {process.returncode}.", file=sys.stderr)
    except Exception as e:
        print(f"Error during preprocessing: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Prepare dataset for training.")
    parser.add_argument("--command", type=str, choices=["save", "preprocess"], required=True,
                        help="Command to execute: 'save' or 'preprocess'")
    parser.add_argument("--output_dir", type=str, default="./data/vin100h-preprocessed-v2",
                        help="Output directory for save command")
    parser.add_argument("--base_model", type=str, default="htdung167/vin100h-preprocessed-v2",
                        help="Base model for save command")
    parser.add_argument("--audio_header", type=str, default="audio",
                        help="Audio header for save command")
    parser.add_argument("--text_header", type=str, default="preprocessed_sentence_v2",
                        help="Text header for save command")
    parser.add_argument("--prepare_csv_input_dir", type=str,
                        default="./data/vin100h-preprocessed-v2",
                        help="Input directory for preprocess command")
    parser.add_argument("--prepare_csv_output_dir", type=str,
                        default="./data/vin100h-preprocessed-v2_pinyin",
                        help="Output directory for preprocess command")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel processes for preprocess command")

    args = parser.parse_args()

    if args.command == "save":
        save_dataset_to_local_disk(args.output_dir, args.base_model, args.audio_header, args.text_header)
    elif args.command == "preprocess":
        run_preprocess(args.prepare_csv_input_dir, args.prepare_csv_output_dir, args.workers)