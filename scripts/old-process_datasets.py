# import json
# import os
# from pathlib import Path
# import shutil
# import torchaudio
# from datasets import load_dataset
# from datasets.arrow_writer import ArrowWriter
# from tqdm import tqdm
# import soundfile as sf
# import csv

# def save_dataset_to_local_disk(output_dir="./data/vin100h-preprocessed-v2",
#                                base_model="htdung167/vin100h-preprocessed-v2",
#                                audio_header='audio', text_header='transcription'):
  
#     wavs_dir = os.path.join(output_dir, "wavs")
#     metadata_path = os.path.join(output_dir, "metadata.csv")
#     os.makedirs(wavs_dir, exist_ok=True)

#     ds = load_dataset(base_model)['train']
#     metadata = []

#     for idx, sample in tqdm(enumerate(ds), total=len(ds),
#                             desc="Saving samples to directory"):
#         audio_array = sample[audio_header]['array']
#         sampling_rate = sample[audio_header]['sampling_rate']
#         filename = f"audio_{idx:06d}.wav"
#         sf.write(os.path.join(wavs_dir, filename), audio_array, sampling_rate)
#         metadata.append([f"wavs/{filename}", sample[text_header]])
        
#     with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
#         csv.writer(f, delimiter='|').writerows(metadata)
#     print(f"Dataset saved to {output_dir}")


# # !python ./src/f5_tts/train/datasets/prepare_csv_wavs.py \
# #     "./data/vin100h-preprocessed-v2" \
# #     "./data/vin100h-preprocessed-v2_pinyin" \
# #     --workers 4 # Sets the number of parallel processes for preprocessing.

# # if __name__ == "__main__":
# # Define the output directory and tokenizer type
# output_dir = "./data/vin100h-preprocessed-v2"
# # tokenizer_type = "pinyin"

# save_dataset_to_local_disk(output_dir=output_dir,
#                         base_model="htdung167/vin100h-preprocessed-v2",
#                         text_header="preprocessed_sentence_v2"
#                         )
    


#     #############

# import subprocess
# import argparse

# def run_preprocess(input_dir, output_dir, workers):
#     command = [
#         "python", "./src/f5_tts/train/datasets/prepare_csv_wavs.py",
#         input_dir,
#         output_dir,
#         "--workers", str(workers)
#     ]
#     process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     stdout, stderr = process.communicate()
    
#     if process.returncode == 0:
#         print("Preprocessing completed successfully.")
#         print(stdout)
#     else:
#         print("Error during preprocessing:")
#         print(stderr)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run preprocessing script for dataset.")
#     parser.add_argument("input_dir", type=str, help="Input directory for preprocessing")
#     parser.add_argument("output_dir", type=str, help="Output directory for processed data")
#     parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes")
    
#     args = parser.parse_args()
#     run_preprocess(args.input_dir, args.output_dir, args.workers)

######################################3
# prepare_dataset.py

import json
import os
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
    - output_dir (str): The directory to save the dataset to.
    - base_model (str): The base model to load the dataset from.
    - audio_header (str): The header for the audio data in the dataset.
    - text_header (str): The header for the text data in the dataset.
    """
    wavs_dir = os.path.join(output_dir, "wavs")
    metadata_path = os.path.join(output_dir, "metadata.csv")
    os.makedirs(wavs_dir, exist_ok=True)

    ds = load_dataset(base_model)['train']
    metadata = []

    for idx, sample in tqdm(enumerate(ds), total=len(ds),
                            desc="Saving samples to directory"):
        audio_array = sample[audio_header]['array']
        sampling_rate = sample[audio_header]['sampling_rate']
        filename = f"audio_{idx:06d}.wav"
        sf.write(os.path.join(wavs_dir, filename), audio_array, sampling_rate)
        metadata.append([f"wavs/{filename}", sample[text_header]])
        
    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f, delimiter='|').writerows(metadata)
    print(f"Dataset saved to {output_dir}")


# def run_preprocess(input_dir, output_dir, workers):
#     """
#     Runs the preprocessing script for the dataset.

#     Args:
#     - input_dir (str): The input directory for preprocessing.
#     - output_dir (str): The output directory for processed data.
#     - workers (int): The number of parallel processes.
#     """
#     command = [
#         "python", "./src/f5_tts/train/datasets/prepare_csv_wavs.py",
#         input_dir,
#         output_dir,
#         "--workers", str(workers)
#     ]
#     process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     stdout, stderr = process.communicate()
    
#     if process.returncode == 0:
#         print("Preprocessing completed successfully.")
#         print(stdout)
#     else:
#         print("Error during preprocessing:")
#         print(stderr)

def run_preprocess(input_dir, output_dir, workers):
    command = [
        "python", "./src/f5_tts/train/datasets/prepare_csv_wavs.py",
        input_dir,
        output_dir,
        "--workers", str(workers)
    ]
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
        print("\nError during preprocessing.", file=sys.stderr)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Prepare dataset for training.")
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand to save dataset to local disk
    save_parser = subparsers.add_parser("save", help="Save dataset to local disk")
    save_parser.add_argument("--output_dir", type=str, default="./data/vin100h-preprocessed-v2", help="Output directory")
    save_parser.add_argument("--base_model", type=str, default="htdung167/vin100h-preprocessed-v2", help="Base model")
    save_parser.add_argument("--audio_header", type=str, default="audio", help="Audio header")
    save_parser.add_argument("--text_header", type=str, default="preprocessed_sentence_v2", help="Text header")

    # Subcommand to run preprocessing
    preprocess_parser = subparsers.add_parser("preprocess", help="Run preprocessing script")
    preprocess_parser.add_argument("--prepare_csv_input_dir", type=str,
                                    default="./data/vin100h-preprocessed-v2",
                                    help="Input directory for preprocessing")
    preprocess_parser.add_argument("--prepare_csv_output_dir", type=str, 
                                   default="./data/vin100h-preprocessed-v2_pinyin",
                                   help="Output directory for processed data")
    preprocess_parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes")

    args = parser.parse_args()

    if args.command == "save":
        save_dataset_to_local_disk(args.output_dir, args.base_model, args.audio_header, args.text_header)
    elif args.command == "preprocess":
        run_preprocess(args.prepare_csv_input_dir, args.prepare_csv_output_dir, args.workers)
    else:
        parser.print_help()
