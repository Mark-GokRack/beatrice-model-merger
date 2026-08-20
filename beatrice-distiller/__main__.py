"""Train a multi-speaker Beatrice model from source/teacher-output audio pairs.

Dataset layout:
    dataset/
      target_speaker_a/
        source/example.wav
        target/example.wav
      target_speaker_b/
        source/example.wav
        target/example.wav
"""

import argparse
import gzip
import json
import math
import os
import shutil
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from beatrice_trainer.__main__ import (
    AUDIO_FILE_SUFFIXES,
    AttrDict,
    ConverterNetwork,
    GradBalancer,
    MultiPeriodDiscriminator,
    PhoneExtractor,
    PARAPHERNALIA_VERSION,
    PitchEstimator,
    get_compressed_optimizer_state_dict,
    get_resampler,
    repo_root,
)


DEFAULT_CONFIG = {
    "learning_rate_g": 5e-5,
    "learning_rate_d": 5e-5,
    "learning_rate_decay": 0.999999,
    "adam_betas": [0.8, 0.99],
    "adam_eps": 1e-6,
    "batch_size": 8,
    "grad_weight_loudness": 1.0,
    "grad_weight_mel": 50.0,
    "grad_weight_ap": 100.0,
    "grad_weight_adv": 150.0,
    "grad_weight_fm": 150.0,
    "grad_balancer_ema_decay": 0.995,
    "use_amp": True,
    "num_workers": 8,
    "n_steps": 10000,
    "warmup_steps": 5000,
    "save_interval": 2000,
    "in_sample_rate": 16000,
    "out_sample_rate": 24000,
    "wav_length": 96000,
    "segment_length": 100,
    "phone_noise_ratio": 0.5,
    "vq_topk": 4,
    "training_time_vq": "none",
    "floor_noise_level": 1e-3,
    "pitch_bins": 448,
    "hidden_channels": 256,
    "san": False,
    "phone_extractor_file": "assets/pretrained/122_checkpoint_03000000.pt",
    "pitch_estimator_file": "assets/pretrained/104_3_checkpoint_00300000.pt",
    "pretrained_file": "assets/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--data-dir", required=True, type=Path)
    parser.add_argument("-o", "--out-dir", required=True, type=Path)
    parser.add_argument("-c", "--config", type=Path)
    return parser.parse_args()


def load_config(config_file: Path | None) -> AttrDict:
    config = DEFAULT_CONFIG.copy()
    if config_file is not None:
        with open(config_file, encoding="utf-8") as file:
            supplied = json.load(file)
        unknown = set(supplied) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config.update(supplied)
    return AttrDict(config)


class PairedDistillationDataset(Dataset):
    """Aligned source and teacher-output pairs for direct voice-conversion distillation."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path, int]],
        in_sample_rate: int,
        out_sample_rate: int,
        wav_length: int,
        segment_length: int,
    ) -> None:
        self.pairs = pairs
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.wav_length = wav_length
        self.segment_length = segment_length
        self.in_hop_length = in_sample_rate // 100
        self.out_hop_length = out_sample_rate // 100

    @staticmethod
    def _load_mono(file: Path) -> tuple[torch.Tensor, int]:
        wav, sample_rate = torchaudio.load(file, backend="soundfile")
        if wav.size(0) != 1:
            wav = wav.mean(0, keepdim=True)
        return wav, sample_rate

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        source_file, target_file, speaker_id = self.pairs[index]
        source, source_rate = self._load_mono(source_file)
        target, target_rate = self._load_mono(target_file)
        source = get_resampler(source_rate, self.in_sample_rate)(source).squeeze(0)
        target = get_resampler(target_rate, self.out_sample_rate)(target).squeeze(0)

        common_frames = min(
            source.numel() // self.in_hop_length,
            target.numel() // self.out_hop_length,
        )
        if common_frames == 0:
            raise ValueError(f"Empty aligned pair: {source_file}, {target_file}")
        source = source[: common_frames * self.in_hop_length]
        target = target[: common_frames * self.out_hop_length]

        # Keep the input/output amplitude relationship produced by the teacher.
        peak = torch.maximum(source.abs().max(), target.abs().max()).clamp_min(1e-5)
        scale = (torch.rand(()) * 0.899 + 0.1) / peak
        return source * scale, target * scale, speaker_id

    def __len__(self) -> int:
        return len(self.pairs)

    def collate(
        self, batch: list[tuple[torch.Tensor, torch.Tensor, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_wavs, target_wavs, speaker_ids, slice_starts = [], [], [], []
        total_frames = self.wav_length // self.out_hop_length
        for source, target, speaker_id in batch:
            frames = min(
                source.numel() // self.in_hop_length,
                target.numel() // self.out_hop_length,
            )
            if frames < total_frames:
                pad_frames = total_frames - frames
                source = F.pad(source, (0, pad_frames * self.in_hop_length))
                target = F.pad(target, (0, pad_frames * self.out_hop_length))
                frames = total_frames
            start = torch.randint(0, frames - total_frames + 1, ()).item()
            source_start = start * self.in_hop_length
            target_start = start * self.out_hop_length
            source_wavs.append(source[source_start : source_start + total_frames * self.in_hop_length])
            target_wavs.append(target[target_start : target_start + self.wav_length])

            # The loss is calculated on a random 1-second region of this window.
            slice_starts.append(torch.randint(0, total_frames - self.segment_length + 1, ()).item())
            speaker_ids.append(speaker_id)

        return (
            torch.stack(source_wavs),
            torch.stack(target_wavs),
            torch.tensor(slice_starts),
            torch.tensor(speaker_ids),
        )


def find_pairs(data_dir: Path) -> tuple[list[str], list[tuple[Path, Path, int]]]:
    speakers, pairs = [], []
    for speaker_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        source_dir, target_dir = speaker_dir / "source", speaker_dir / "target"
        if not source_dir.is_dir() or not target_dir.is_dir():
            continue
        speaker_pairs = []
        for source_file in source_dir.rglob("*"):
            if not source_file.is_file() or source_file.suffix.lower() not in AUDIO_FILE_SUFFIXES:
                continue
            relative_path = source_file.relative_to(source_dir)
            target_file = target_dir / relative_path
            if not target_file.is_file():
                raise ValueError(f"Missing target pair for {source_file}: expected {target_file}")
            speaker_pairs.append((source_file, target_file, len(speakers)))
        if not speaker_pairs:
            raise ValueError(f"No paired audio found for {speaker_dir}")
        speakers.append(speaker_dir.name)
        pairs.extend(speaker_pairs)
    if not speakers:
        raise ValueError("No speaker directories containing source/ and target/ were found.")
    return speakers, pairs


def target_audio_by_speaker(pairs: Iterable[tuple[Path, Path, int]], n_speakers: int):
    files = [[] for _ in range(n_speakers)]
    for _, target_file, speaker_id in pairs:
        files[speaker_id].append(target_file)
    return files


def load_frozen_models(h: AttrDict, device: torch.device) -> tuple[PhoneExtractor, PitchEstimator]:
    phone_extractor = PhoneExtractor().to(device).eval().requires_grad_(False)
    checkpoint = torch.load(repo_root() / h.phone_extractor_file, map_location="cpu", weights_only=True)
    phone_extractor.load_state_dict(checkpoint["phone_extractor"], strict=False)
    pitch_estimator = PitchEstimator().to(device).eval().requires_grad_(False)
    checkpoint = torch.load(repo_root() / h.pitch_estimator_file, map_location="cpu", weights_only=True)
    pitch_estimator.load_state_dict(checkpoint["pitch_estimator"])
    return phone_extractor, pitch_estimator


def prepare_pretrained_generator(checkpoint: dict, n_speakers: int) -> dict:
    """Expand the single-speaker conditional tables used by the released pretrain."""
    state = checkpoint["net_g"]
    for name in (
        "vq.codebooks",
        "embed_speaker.weight",
        "key_value_speaker_embedding.weight",
    ):
        state[name] = state[name][:1].expand(n_speakers, *state[name].shape[1:]).clone()
    return state


def export_paraphernalia(
    output_dir: Path,
    net_g: ConverterNetwork,
    phone_extractor: PhoneExtractor,
    pitch_estimator: PitchEstimator,
    speakers: list[str],
    h: AttrDict,
) -> None:
    """Write the same inference files emitted by the regular trainer."""
    output_dir.mkdir()
    export_net = ConverterNetwork(
        torch.nn.Module(),
        torch.nn.Module(),
        len(speakers),
        h.pitch_bins,
        h.hidden_channels,
        h.vq_topk,
        h.training_time_vq,
        h.phone_noise_ratio,
        h.floor_noise_level,
    )
    export_net.load_state_dict(net_g.state_dict())
    export_net.merge_weights()
    export_net.half()
    export_net.dump(output_dir / "waveform_generator.bin")
    export_net.dump_speaker_embeddings(output_dir / "speaker_embeddings.bin")
    export_net.dump_embedding_setter(output_dir / "embedding_setter.bin")
    export_phone_extractor = PhoneExtractor()
    export_phone_extractor.load_state_dict(phone_extractor.state_dict())
    export_phone_extractor.remove_weight_norm()
    export_phone_extractor.merge_weights()
    export_phone_extractor.half()
    export_phone_extractor.dump(output_dir / "phone_extractor.bin")
    export_pitch_estimator = PitchEstimator()
    export_pitch_estimator.load_state_dict(pitch_estimator.state_dict())
    export_pitch_estimator.merge_weights()
    export_pitch_estimator.half()
    export_pitch_estimator.dump(output_dir / "pitch_estimator.bin")
    shutil.copy(repo_root() / "assets" / "images" / "noimage.png", output_dir / "noimage.png")
    with open(output_dir / "beatrice_paraphernalia_distilled.toml", "w", encoding="utf-8") as file:
        file.write(f'[model]\nversion = "{PARAPHERNALIA_VERSION}"\nname = "distilled"\n')
        for speaker_id, speaker in enumerate(speakers):
            file.write(
                f'\n[voice.{speaker_id}]\nname = "{speaker}"\n'
                'description = """Distilled from a teacher model."""\n'
                'average_pitch = 69.0\n'
                f'\n[voice.{speaker_id}.portrait]\npath = "noimage.png"\n'
            )


def main() -> None:
    args = parse_args()
    h = load_config(args.config)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {args.data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    speakers, pairs = find_pairs(args.data_dir)
    print(f"device={device}; speakers={len(speakers)}; pairs={len(pairs)}")
    for speaker_id, speaker in enumerate(speakers):
        print(f"  {speaker_id}: {speaker}")

    dataset = PairedDistillationDataset(pairs, h.in_sample_rate, h.out_sample_rate, h.wav_length, h.segment_length)
    loader = DataLoader(
        dataset,
        batch_size=h.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=min(h.num_workers, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=h.num_workers > 0,
        collate_fn=dataset.collate,
    )
    if len(loader) == 0:
        raise ValueError("Not enough pairs for one batch. Reduce batch_size or add pairs.")

    phone_extractor, pitch_estimator = load_frozen_models(h, device)
    net_g = ConverterNetwork(phone_extractor, pitch_estimator, len(speakers), h.pitch_bins, h.hidden_channels, h.vq_topk, h.training_time_vq, h.phone_noise_ratio, h.floor_noise_level).to(device)
    net_d = MultiPeriodDiscriminator(san=h.san).to(device)
    with gzip.open(repo_root() / h.pretrained_file, "rb") as file:
        pretrained = torch.load(file, map_location="cpu", weights_only=True)
    net_g.load_state_dict(prepare_pretrained_generator(pretrained, len(speakers)), strict=False)
    net_d.load_state_dict(pretrained["net_d"], strict=False)

    # Unlike normal training, codebooks describe each teacher's target voice.
    def wav_iterator(files: list[Path]):
        for file in files:
            wav, sample_rate = torchaudio.load(file, backend="soundfile")
            if wav.size(0) != 1:
                wav = wav.mean(0, keepdim=True)
            yield get_resampler(sample_rate, h.in_sample_rate)(wav).to(device)[:, None]

    net_g.initialize_vq([wav_iterator(files) for files in target_audio_by_speaker(pairs, len(speakers))])
    optim_g = torch.optim.AdamW(net_g.parameters(), h.learning_rate_g, betas=h.adam_betas, eps=h.adam_eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), h.learning_rate_d, betas=h.adam_betas, eps=h.adam_eps)
    scaler = torch.amp.GradScaler("cuda", enabled=h.use_amp)
    balancer = GradBalancer(
        {"loss_loudness": h.grad_weight_loudness, "loss_mel": h.grad_weight_mel, "loss_adv": h.grad_weight_adv, "loss_fm": h.grad_weight_fm}
        | ({"loss_ap": h.grad_weight_ap} if h.grad_weight_ap else {}),
        ema_decay=h.grad_balancer_ema_decay,
    )
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optim_g, lambda step: step / h.warmup_steps if step < h.warmup_steps else h.learning_rate_decay ** (step - h.warmup_steps))
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optim_d, lambda step: step / h.warmup_steps if step < h.warmup_steps else h.learning_rate_decay ** (step - h.warmup_steps))
    writer = SummaryWriter(args.out_dir)
    with open(args.out_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(dict(h), file, indent=2)
    shutil.copy(__file__, args.out_dir / "distiller.py")

    data_iterator = iter(loader)
    for iteration in tqdm(range(h.n_steps), desc="Distilling"):
        try:
            source_wavs, target_wavs, slice_starts, speaker_ids = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            source_wavs, target_wavs, slice_starts, speaker_ids = next(data_iterator)
        source_wavs, target_wavs, slice_starts, speaker_ids = (value.to(device, non_blocking=True) for value in (source_wavs, target_wavs, slice_starts, speaker_ids))
        formant_shifts = torch.zeros_like(speaker_ids, dtype=torch.float)
        with torch.amp.autocast("cuda", enabled=h.use_amp):
            y, y_hat, y_hat_for_backward, loss_loudness, loss_mel, loss_ap, _ = net_g.forward_and_compute_loss(
                source_wavs[:, None], speaker_ids, formant_shifts, slice_starts, h.segment_length, target_wavs[:, None], h.grad_weight_ap != 0.0
            )
            loss_d, loss_adv, loss_fm, _ = net_d.forward_and_compute_loss(y, y_hat)

        scaler.scale(loss_d).backward(retain_graph=True, inputs=list(net_d.parameters()))
        scaler.unscale_(optim_d)
        scaler.step(optim_d)
        optim_d.zero_grad(set_to_none=True)
        balancer.backward(
            {"loss_loudness": loss_loudness, "loss_mel": loss_mel, "loss_adv": loss_adv, "loss_fm": loss_fm}
            | ({"loss_ap": loss_ap} if h.grad_weight_ap else {}),
            y_hat_for_backward,
            scaler,
            skip_update_ema=iteration > 10 and iteration % 5 != 0,
        )
        scaler.unscale_(optim_g)
        scaler.step(optim_g)
        optim_g.zero_grad(set_to_none=True)
        scaler.update()
        scheduler_g.step()
        scheduler_d.step()

        if iteration == 0 or (iteration + 1) % 100 == 0:
            writer.add_scalar("loss/generator_loudness", loss_loudness.item(), iteration + 1)
            writer.add_scalar("loss/generator_mel", loss_mel.item(), iteration + 1)
            writer.add_scalar("loss/generator_adversarial", loss_adv.item(), iteration + 1)
            writer.add_scalar("loss/generator_feature_matching", loss_fm.item(), iteration + 1)
            writer.add_scalar("loss/discriminator", loss_d.item(), iteration + 1)
            writer.add_scalar("learning_rate/generator", scheduler_g.get_last_lr()[0], iteration + 1)
        if (iteration + 1) % h.save_interval == 0 or iteration + 1 == h.n_steps:
            checkpoint_file = args.out_dir / f"checkpoint_distilled_{iteration + 1:08d}.pt.gz"
            with gzip.open(checkpoint_file, "wb") as file:
                torch.save({"iteration": iteration + 1, "net_g": net_g.state_dict(), "net_d": {key: value.half() for key, value in net_d.state_dict().items()}, "optim_g": get_compressed_optimizer_state_dict(optim_g), "optim_d": get_compressed_optimizer_state_dict(optim_d), "h": dict(h)}, file)
            export_paraphernalia(
                args.out_dir / f"paraphernalia_distilled_{iteration + 1:08d}",
                net_g,
                phone_extractor,
                pitch_estimator,
                speakers,
                h,
            )
    writer.close()


if __name__ == "__main__":
    main()