"""Compare the legacy native runtime with the PyTorch Beatrice inference path."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

from beatrice_distiller.pytorch_inference import IN_SAMPLE_RATE, PyTorchConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Paraphernalia directory")
    parser.add_argument("input_wav", type=Path, help="Mono or stereo source WAV")
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--formant-shift", type=float, default=0.0)
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    parser.add_argument("--vq-neighbors", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--native-extension-dir", type=Path)
    parser.add_argument("--output-wav", type=Path)
    return parser.parse_args()


def configure(converter, args: argparse.Namespace) -> None:
    converter.load_model(args.model_dir)
    converter.set_target_speaker(args.speaker)
    converter.set_formant_shift(args.formant_shift)
    converter.set_pitch_shift(args.pitch_shift)
    converter.set_vq_num_neighbors(args.vq_neighbors)
    converter.reset()


def read_input(path: Path) -> torch.Tensor:
    wav, sample_rate = torchaudio.load(path, backend="soundfile")
    wav = wav.mean(0) if wav.size(0) != 1 else wav.squeeze(0)
    if sample_rate != IN_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sample_rate, IN_SAMPLE_RATE)
    return wav.contiguous()


def best_aligned(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    best = None
    max_lag = min(480, reference.numel() // 4, candidate.numel() // 4)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            ref, actual = reference[lag:], candidate[: reference.numel() - lag]
        else:
            ref, actual = reference[: reference.numel() + lag], candidate[-lag:]
        length = min(ref.numel(), actual.numel())
        ref, actual = ref[:length], actual[:length]
        mse = (ref - actual).square().mean()
        if best is None or mse < best[0]:
            best = mse, ref, actual, lag
    assert best is not None
    _, reference, candidate, lag = best
    return reference, candidate, lag


def print_metrics(reference: torch.Tensor, candidate: torch.Tensor, lag: int) -> None:
    error = reference - candidate
    reference_power = reference.square().mean().clamp_min(1e-12)
    error_power = error.square().mean().clamp_min(1e-12)
    window = torch.hann_window(1024)
    reference_stft = torch.stft(reference, 1024, 256, window=window, return_complex=True)
    candidate_stft = torch.stft(candidate, 1024, 256, window=window, return_complex=True)
    log_magnitude_mae = (
        reference_stft.abs().clamp_min(1e-7).log()
        - candidate_stft.abs().clamp_min(1e-7).log()
    ).abs().mean()
    print(f"aligned_samples={reference.numel()}")
    print(f"best_lag_samples={lag}")
    print(f"waveform_mae={error.abs().mean().item():.8g}")
    print(f"waveform_rmse={error_power.sqrt().item():.8g}")
    print(f"signal_to_error_db={(10.0 * torch.log10(reference_power / error_power)).item():.4f}")
    print(f"log_stft_magnitude_mae={log_magnitude_mae.item():.8g}")


def main() -> None:
    args = parse_args()
    source = read_input(args.input_wav)
    pytorch_converter = PyTorchConverter(device=args.device)
    configure(pytorch_converter, args)
    pytorch_output = torch.from_numpy(pytorch_converter.process(source))
    print(f"pytorch_output_samples={pytorch_output.numel()}")

    if args.output_wav:
        torchaudio.save(args.output_wav, pytorch_output[None], 24000, backend="soundfile")

    if args.native_extension_dir is None:
        print("Native comparison skipped; pass --native-extension-dir to compare a legacy extension.")
        return
    sys.path.insert(0, str(args.native_extension_dir))
    from _beatrice_inference import Converter as NativeConverter

    native_converter = NativeConverter()
    configure(native_converter, args)
    native_output = torch.from_numpy(native_converter.process(source.numpy()))
    reference, candidate, lag = best_aligned(native_output, pytorch_output)
    print_metrics(reference, candidate, lag)


if __name__ == "__main__":
    main()