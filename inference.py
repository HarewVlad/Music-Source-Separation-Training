# coding: utf-8
__author__ = "Vladislav Petrenko (SBER)"

import argparse
import time
import librosa
from tqdm.auto import tqdm
import sys
import os
import glob
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn
from utils import prefer_target_instrument
import torchaudio

# Using the embedded version of Python can also correctly import the utils module
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
from utils import demix, get_model_from_config

import warnings

warnings.filterwarnings("ignore")


def run_audio_processing(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()

    # Handle input audio files (single file or multiple files)
    if args.input_audio_path:
        audio_files = [args.input_audio_path]
    else:
        audio_files = glob.glob(os.path.join(args.input_folder, "*.*"))
        audio_files.sort()

    print(f"Total files found: {len(audio_files)}")

    instruments = prefer_target_instrument(config)

    os.makedirs(args.output_path, exist_ok=True)

    if not verbose:
        audio_files = tqdm(audio_files, desc="Total progress")

    detailed_pbar = not args.disable_detailed_pbar

    for path in audio_files:
        if verbose:
            print("Starting processing track: ", path)
        else:
            audio_files.set_postfix({"track": os.path.basename(path)})

        try:
            mix_tensor, sr = torchaudio.load(path)
            if sr != 44100:
                resampler = torchaudio.transforms.Resample(sr, 44100)
                mix_tensor = resampler(mix_tensor)
                sr = 44100
            mix = mix_tensor.numpy()
        except Exception as e:
            print(f"Cannot read track: {path}")
            print(f"Error message: {str(e)}")
            continue

        # Ensure stereo
        if mix.shape[0] == 1:
            mix = np.tile(mix, (2, 1))

        mix_orig = mix.copy()

        # Normalization
        if config.inference.get("normalize", False):
            mono = mix.mean(0)
            mean = mono.mean()
            std = mono.std()
            mix = (mix - mean) / std

        # Test Time Augmentation (TTA)
        if args.use_tta:
            track_proc_list = [mix, mix[::-1], -mix]
        else:
            track_proc_list = [mix]

        full_result = []

        with torch.no_grad():
            for mix_proc in track_proc_list:
                waveforms = demix(
                    config,
                    model,
                    mix_proc,
                    device,
                    pbar=detailed_pbar,
                    model_type=args.model_type,
                )
                full_result.append(waveforms)

        # Average the results
        waveforms = {key: np.zeros_like(full_result[0][key]) for key in full_result[0]}
        for i, result in enumerate(full_result):
            for key in result:
                if i == 1:
                    waveforms[key] += result[key][::-1]
                elif i == 2:
                    waveforms[key] -= result[key]
                else:
                    waveforms[key] += result[key]
        num_results = len(full_result)
        for key in waveforms:
            waveforms[key] /= num_results

        # Extract instrumental if required
        if args.extract_instrumental:
            if "instrumental" not in instruments:
                instruments.append("instrumental")

            waveforms["instrumental"] = mix_orig - waveforms[args.instrument]

        # Save outputs
        for instrument in instruments:
            estimates = waveforms[instrument].T
            if config.inference.get("normalize", False):
                estimates = estimates * std + mean
            file_name, _ = os.path.splitext(os.path.basename(path))
            if args.flac_file:
                output_file = os.path.join(args.output_path, f"{instrument}.flac")
                subtype = "PCM_16" if args.pcm_type == "PCM_16" else "PCM_24"
                sf.write(output_file, estimates, sr, subtype=subtype)
            else:
                output_file = os.path.join(args.output_path, f"{instrument}.wav")
                sf.write(output_file, estimates, sr, subtype="FLOAT")

    print(f"Elapsed time: {time.time() - start_time:.2f} sec")


def process_audio_files(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="mdx23c",
        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer, scnet, scnet_unofficial, segm_models, swin_upernet, torchseg",
    )
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument(
        "--start_checkpoint",
        type=str,
        default="",
    )
    parser.add_argument(
        "--input_folder", type=str,
    )
    parser.add_argument(
        "--input_audio_path", type=str,
    )
    parser.add_argument(
        "--output_path", default="", type=str,
    )
    parser.add_argument(
        "--device_ids", nargs="+", type=int, default=0,
    )
    parser.add_argument(
        "--extract_instrumental",
        action="store_true",
        help="Invert vocals to get instrumental if provided",
    )
    parser.add_argument(
        "--disable_detailed_pbar",
        action="store_true",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force the use of CPU even if CUDA is available",
    )
    parser.add_argument(
        "--flac_file", action="store_true", help="Output FLAC file instead of WAV"
    )
    parser.add_argument(
        "--pcm_type",
        type=str,
        choices=["PCM_16", "PCM_24"],
        default="PCM_24",
        help="PCM type for FLAC files (PCM_16 or PCM_24)",
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Flag adds test time augmentation during inference (polarity and channel inverse). While this triples the runtime, it reduces noise and slightly improves prediction quality.",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        choices=["vocals", "drums", "bass", "piano", "guitar"],
        default="",
        help="Instrument to extract from initial audio",
    )

    args = parser.parse_args(args)

    # Device selection optimization
    if args.force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        print("CUDA is available, use --force_cpu to disable it.")
        device_ids = (
            args.device_ids if isinstance(args.device_ids, list) else [args.device_ids]
        )
        device = torch.device(f"cuda:{device_ids[0]}")
        torch.cuda.set_device(device)

    print("Using device:", device)

    model_load_start_time = time.time()

    # Disable CuDNN benchmark during model loading to reduce overhead
    torch.backends.cudnn.benchmark = False

    # Efficient model loading
    model, config = get_model_from_config(args.model_type, args.config_path)

    # Load checkpoint if provided
    if args.start_checkpoint:
        print("Loading checkpoint:", args.start_checkpoint)
        checkpoint_load_start_time = time.time()
        # Use 'map_location' to load directly to the desired device
        state_dict = torch.load(args.start_checkpoint, map_location=device)
        # Handle model-specific keys
        if args.model_type in ["htdemucs", "apollo"]:
            state_dict = state_dict.get("state", state_dict)
            state_dict = state_dict.get("state_dict", state_dict)
        # Remove any unnecessary keys like 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        print(
            f"Checkpoint loaded in {time.time() - checkpoint_load_start_time:.2f} sec"
        )
    else:
        print("No checkpoint provided. Using model's default initialization.")

    print("Instruments:", config.training.instruments)

    # Move model to device efficiently
    model.to(device)

    # DataParallel if multiple GPUs are available and specified
    if (
        isinstance(args.device_ids, list)
        and len(args.device_ids) > 1
        and not args.force_cpu
    ):
        model = nn.DataParallel(model, device_ids=args.device_ids)

    # Enable CuDNN benchmark after model is loaded
    torch.backends.cudnn.benchmark = True

    print(
        f"Model loaded and moved to {device} in {time.time() - model_load_start_time:.2f} sec"
    )

    run_audio_processing(model, args, config, device, verbose=True)


if __name__ == "__main__":
    process_audio_files(None)
