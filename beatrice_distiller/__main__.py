"""Train a multi-speaker Beatrice model from source audio and teacher models."""

import argparse
import gzip
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

DISTILLER_DIR = Path(__file__).resolve().parent
TRAINER_DIR = DISTILLER_DIR.parent / "beatrice-trainer"
if str(DISTILLER_DIR) not in sys.path:
    sys.path.insert(0, str(DISTILLER_DIR))
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

from beatrice_trainer.__main__ import (
    AUDIO_FILE_SUFFIXES,
    AttrDict,
    ConverterNetwork,
    GradBalancer,
    MultiPeriodDiscriminator,
    PhoneExtractor,
    PARAPHERNALIA_VERSION,
    PitchEstimator,
    augment_audio,
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
    "num_workers": 4,
    "n_steps": 10000,
    "warmup_steps": 5000,
    "save_interval": 2000,
    "in_sample_rate": 16000,
    "out_sample_rate": 24000,
    "wav_length": 96000,
    "segment_length": 100,
    "phone_noise_ratio": 0.5,
    "formant_shift_candidates": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
    "augmentation_snr_candidates": [20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
    "augmentation_formant_shift_probability": 0.5,
    "augmentation_formant_shift_semitone_min": -3.0,
    "augmentation_formant_shift_semitone_max": 3.0,
    "augmentation_reverb_probability": 0.5,
    "augmentation_lpf_probability": 0.2,
    "augmentation_lpf_cutoff_freq_candidates": [2000.0, 3000.0, 4000.0, 6000.0],
    "in_ir_wav_dir": "beatrice-trainer/assets/ir",
    "in_noise_wav_dir": "beatrice-trainer/assets/noise",
    "vq_topk": 4,
    "vq_init_from_bin": True,
    "vq_init_max_files": 128,
    "vq_init_wav_length": 16000,
    "generator_init_mode": "pretrained",
    "generator_init_model": None,
    "training_time_vq": "none",
    "floor_noise_level": 1e-3,
    "pitch_bins": 448,
    "hidden_channels": 256,
    "san": False,
    "phone_extractor_file": "beatrice-trainer/assets/pretrained/122_checkpoint_03000000.pt",
    "pitch_estimator_file": "beatrice-trainer/assets/pretrained/104_3_checkpoint_00300000.pt",
    "pretrained_file": "beatrice-trainer/assets/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--data-dir", required=True, type=Path, help="Directory searched recursively for source WAV files.")
    parser.add_argument("-w", "--weights-dir", required=True, type=Path, help="Directory searched recursively for teacher paraphernalia directories.")
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


class DistillationDataset(Dataset):
    """Source audio with targets generated on demand by teacher models."""

    def __init__(
        self,
        examples: list[tuple[Path, Path, int, int]],
        in_sample_rate: int,
        out_sample_rate: int,
        wav_length: int,
        segment_length: int,
        formant_shift_candidates: list[float],
        vq_topk: int,
        noise_files: list[Path],
        ir_files: list[Path],
        augmentation_snr_candidates: list[float],
        augmentation_formant_shift_probability: float,
        augmentation_formant_shift_semitone_min: float,
        augmentation_formant_shift_semitone_max: float,
        augmentation_reverb_probability: float,
        augmentation_lpf_probability: float,
        augmentation_lpf_cutoff_freq_candidates: list[float],
    ) -> None:
        self.examples = examples
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.wav_length = wav_length
        self.segment_length = segment_length
        self.in_hop_length = in_sample_rate // 100
        self.out_hop_length = out_sample_rate // 100
        self.formant_shift_candidates = tuple(formant_shift_candidates)
        self.vq_topk = vq_topk
        self.noise_files = noise_files
        self.ir_files = ir_files
        self.augmentation_snr_candidates = augmentation_snr_candidates
        self.augmentation_formant_shift_probability = augmentation_formant_shift_probability
        self.augmentation_formant_shift_semitone_min = augmentation_formant_shift_semitone_min
        self.augmentation_formant_shift_semitone_max = augmentation_formant_shift_semitone_max
        self.augmentation_reverb_probability = augmentation_reverb_probability
        self.augmentation_lpf_probability = augmentation_lpf_probability
        self.augmentation_lpf_cutoff_freq_candidates = augmentation_lpf_cutoff_freq_candidates
        self._converters = {}
        if not self.formant_shift_candidates:
            raise ValueError("formant_shift_candidates must not be empty")
        if not 1 <= self.vq_topk <= 8:
            raise ValueError("vq_topk must be between 1 and 8 for teacher inference")
        for semitones in self.formant_shift_candidates:
            index = (semitones + 2.0) * 2.0
            if not -2.0 <= semitones <= 2.0 or not math.isclose(index, round(index)):
                raise ValueError(
                    "formant_shift_candidates must use 0.5-semitone values from -2.0 to 2.0"
                )

    @staticmethod
    def _load_mono(file: Path) -> tuple[torch.Tensor, int]:
        wav, sample_rate = torchaudio.load(file, backend="soundfile")
        if wav.size(0) != 1:
            wav = wav.mean(0, keepdim=True)
        return wav, sample_rate

    def _generate_target(
        self,
        source: torch.Tensor,
        model_dir: Path,
        teacher_speaker_id: int,
        formant_shift: float = 0.0,
    ) -> torch.Tensor:
        converter_key = (model_dir, teacher_speaker_id)
        converter = self._converters.get(converter_key)
        if converter is None:
            from beatrice_distiller.pytorch_inference import PyTorchConverter

            converter = PyTorchConverter()
            converter.load_model(model_dir)
            converter.set_target_speaker(teacher_speaker_id)
            converter.set_vq_num_neighbors(self.vq_topk)
            self._converters[converter_key] = converter
        converter.reset()
        converter.set_formant_shift(formant_shift)
        converter.set_pitch_shift(0.0)
        target = converter.process(source.numpy().astype("float32", copy=False))
        return torch.from_numpy(target)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, float]:
        source_file, model_dir, teacher_speaker_id, speaker_id = self.examples[index]
        source, source_rate = self._load_mono(source_file)
        source = get_resampler(source_rate, self.in_sample_rate)(source).squeeze(0)
        if source.abs().max() == 0:
            raise ValueError(f"Silent source audio: {source_file}")
        formant_shift = self.formant_shift_candidates[
            torch.randint(len(self.formant_shift_candidates), ()).item()
        ]
        target = self._generate_target(
            source, model_dir, teacher_speaker_id, formant_shift
        )
        source = augment_audio(
            source[None],
            self.in_sample_rate,
            self.noise_files,
            self.ir_files,
            self.augmentation_snr_candidates,
            self.augmentation_formant_shift_probability,
            self.augmentation_formant_shift_semitone_min,
            self.augmentation_formant_shift_semitone_max,
            self.augmentation_reverb_probability,
            self.augmentation_lpf_probability,
            self.augmentation_lpf_cutoff_freq_candidates,
        ).squeeze(0)

        common_frames = min(
            source.numel() // self.in_hop_length,
            target.numel() // self.out_hop_length,
        )
        if common_frames == 0:
            raise ValueError(f"Empty source audio: {source_file}")
        source = source[: common_frames * self.in_hop_length]
        target = target[: common_frames * self.out_hop_length]

        # Keep the input/output amplitude relationship produced by the teacher.
        peak = torch.maximum(source.abs().max(), target.abs().max()).clamp_min(1e-5)
        scale = (torch.rand(()) * 0.899 + 0.1) / peak
        return source * scale, target * scale, speaker_id, formant_shift

    def __len__(self) -> int:
        return len(self.examples)

    def collate(
        self, batch: list[tuple[torch.Tensor, torch.Tensor, int, float]]
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        source_wavs, target_wavs, speaker_ids, formant_shifts, slice_starts = [], [], [], [], []
        total_frames = self.wav_length // self.out_hop_length
        for source, target, speaker_id, formant_shift in batch:
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
            formant_shifts.append(formant_shift)

        return (
            torch.stack(source_wavs),
            torch.stack(target_wavs),
            torch.tensor(slice_starts),
            torch.tensor(speaker_ids),
            torch.tensor(formant_shifts),
        )


REQUIRED_MODEL_FILES = {
    "phone_extractor.bin",
    "pitch_estimator.bin",
    "waveform_generator.bin",
    "embedding_setter.bin",
    "speaker_embeddings.bin",
}
VOICE_TABLE_RE = re.compile(r"(?m)^\[voice\.(\d+)\]\s*$")
CODEBOOK_SIZE = 512
PHONE_CHANNELS = 128
FORMANT_EMBEDDING_COUNT = 9
HIDDEN_CHANNELS = 256
KEY_VALUE_EMBEDDING_LENGTH = 384
KEY_VALUE_EMBEDDING_CHANNELS = 128


def find_source_files(data_dir: Path) -> list[Path]:
    files = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not files:
        raise ValueError(f"No WAV files found under {data_dir}")
    return files


def find_augmentation_files(directory: Path, label: str) -> list[Path]:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_FILE_SUFFIXES
    )
    if not files:
        raise ValueError(f"No audio files found for {label}: {directory}")
    return files


def find_teachers(weights_dir: Path) -> list[tuple[Path, int, dict]]:
    teachers = []
    for model_dir in sorted(path for path in weights_dir.rglob("*") if path.is_dir()):
        if not all((model_dir / filename).is_file() for filename in REQUIRED_MODEL_FILES):
            continue
        toml_files = sorted(model_dir.glob("*.toml"))
        if len(toml_files) != 1:
            raise ValueError(f"Expected exactly one TOML file in teacher model directory: {model_dir}")
        with open(toml_files[0], "rb") as file:
            metadata = tomllib.load(file)
        model_metadata = metadata.get("model", {})
        model_version = model_metadata.get("version") if isinstance(model_metadata, dict) else None
        if model_version != PARAPHERNALIA_VERSION:
            print(
                f"Warning: Teacher model version must be {PARAPHERNALIA_VERSION}, "
                f"but {model_dir} declares {model_version!r}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        voices = metadata.get("voice", {})
        for voice_id in sorted((int(key) for key in voices)):
            teachers.append((model_dir, voice_id, {"toml_file": toml_files[0], "voice": voices[str(voice_id)]}))
    if not teachers:
        raise ValueError(f"No teacher paraphernalia directories found under {weights_dir}")
    return teachers


def load_teacher_codebooks(teachers: list[tuple[Path, int, dict]]) -> torch.Tensor:
    """Load each selected teacher voice's codebook from speaker_embeddings.bin."""
    codebook_elements = CODEBOOK_SIZE * PHONE_CHANNELS
    per_speaker_elements = (
        codebook_elements
        + HIDDEN_CHANNELS
        + KEY_VALUE_EMBEDDING_LENGTH * KEY_VALUE_EMBEDDING_CHANNELS
    )
    formant_elements = FORMANT_EMBEDDING_COUNT * HIDDEN_CHANNELS
    codebooks = []

    for model_dir, teacher_speaker_id, _ in teachers:
        embedding_file = model_dir / "speaker_embeddings.bin"
        values = np.fromfile(embedding_file, dtype=np.float16)
        if values.size < formant_elements or (values.size - formant_elements) % per_speaker_elements:
            raise ValueError(f"Invalid speaker_embeddings.bin: {embedding_file}")
        n_speakers = (values.size - formant_elements) // per_speaker_elements
        if teacher_speaker_id >= n_speakers:
            raise ValueError(
                f"Teacher speaker {teacher_speaker_id} is not present in {embedding_file}"
            )
        start = teacher_speaker_id * codebook_elements
        codebooks.append(
            torch.from_numpy(
                values[start : start + codebook_elements]
                .reshape(CODEBOOK_SIZE, PHONE_CHANNELS)
                .copy()
            )
        )

    return torch.stack(codebooks)


def make_examples(source_files: Iterable[Path], teachers: list[tuple[Path, int, dict]]) -> list[tuple[Path, Path, int, int]]:
    return [
        (source_file, model_dir, teacher_speaker_id, speaker_id)
        for speaker_id, (model_dir, teacher_speaker_id, _) in enumerate(teachers)
        for source_file in source_files
    ]


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


class _BinaryTensorReader:
    """Read one or more same-layout FP16 inference binaries as FP32 tensors."""

    def __init__(self, files: list[Path]) -> None:
        arrays = [np.fromfile(file, dtype=np.float16) for file in files]
        if not arrays or any(array.size != arrays[0].size for array in arrays[1:]):
            raise ValueError(f"Inference binaries do not have a matching layout: {files}")
        self.values = np.mean(np.stack(arrays, dtype=np.float32), axis=0)
        self.files = files
        self.offset = 0

    def read(self, shape: torch.Size) -> torch.Tensor:
        count = math.prod(shape)
        end = self.offset + count
        if end > self.values.size:
            raise ValueError(f"Unexpected end of inference binary: {self.files}")
        value = torch.from_numpy(self.values[self.offset:end].reshape(tuple(shape)).copy())
        self.offset = end
        return value

    def finish(self) -> None:
        if self.offset != self.values.size:
            raise ValueError(
                f"Unexpected trailing values in inference binary: {self.files} "
                f"({self.values.size - self.offset} FP16 values)"
            )


def _copy_parameter(parameter: torch.Tensor, value: torch.Tensor) -> None:
    parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _load_standard_layer(reader: _BinaryTensorReader, layer: torch.nn.Module) -> None:
    _copy_parameter(layer.weight, reader.read(layer.weight.shape))
    if layer.bias is not None:
        _copy_parameter(layer.bias, reader.read(layer.bias.shape))
    if hasattr(layer, "gain"):
        weight = layer.weight.data
        dims = tuple(range(1, weight.ndim))
        variance, mean = torch.var_mean(weight, dims, keepdim=True)
        gain = (variance * math.prod(weight.shape[1:]) + 1e-8).sqrt()
        layer.weight.data.copy_(weight - mean)
        _copy_parameter(layer.gain, gain)


def _load_cross_attention(reader: _BinaryTensorReader, attention: torch.nn.Module) -> None:
    scale = math.sqrt(math.sqrt(attention.head_qk_channels))
    _copy_parameter(
        attention.q_projection.weight,
        reader.read(attention.q_projection.weight.shape) * scale,
    )
    _copy_parameter(
        attention.q_projection.bias,
        reader.read(attention.q_projection.bias.shape) * scale,
    )
    _load_standard_layer(reader, attention.out_projection)


def _load_embedding_setter(reader: _BinaryTensorReader, prenet: torch.nn.Module) -> None:
    for block in prenet.convnext:
        attention = block.mha
        scale = math.sqrt(math.sqrt(attention.head_qk_channels))
        key_weights = []
        value_weights = []
        for _ in range(attention.num_heads):
            key_weights.append(
                reader.read(torch.Size((attention.head_qk_channels, attention.in_kv_channels)))
                * scale
            )
            value_weights.append(
                reader.read(torch.Size((attention.head_vo_channels, attention.in_kv_channels)))
            )
        key_biases = []
        value_biases = []
        for _ in range(attention.num_heads):
            key_biases.append(reader.read(torch.Size((attention.head_qk_channels,))) * scale)
            value_biases.append(reader.read(torch.Size((attention.head_vo_channels,))))
        _copy_parameter(
            attention.kv_projection.weight,
            torch.cat((torch.cat(key_weights), torch.cat(value_weights))),
        )
        _copy_parameter(
            attention.kv_projection.bias,
            torch.cat((torch.cat(key_biases), torch.cat(value_biases))),
        )


def _load_convnext_stack(reader: _BinaryTensorReader, stack: torch.nn.Module) -> None:
    _load_standard_layer(reader, stack.embed)
    if not stack.use_weight_standardization:
        _load_standard_layer(reader, stack.norm)
    for block in stack.convnext:
        if block.use_mha:
            if not block.cross_attention:
                raise ValueError("Self-attention is not supported by the inference-binary loader")
            _load_cross_attention(reader, block.mha)
            block.attn_norm.weight.data.fill_(1.0)
            block.attn_norm.bias.data.zero_()
        _load_standard_layer(reader, block.dwconv)
        _load_standard_layer(reader, block.pwconv1)
        _load_standard_layer(reader, block.pwconv2)
        if hasattr(block, "gamma"):
            block.gamma.data.fill_(1.0)
        if hasattr(block, "pre_scale"):
            block.pre_scale.fill_(1.0)
            block.post_scale.fill_(1.0)
            block.post_scale_weight.data.fill_(1.0)
        if not block.use_weight_standardization:
            block.norm.weight.data.fill_(1.0)
            block.norm.bias.data.zero_()
    if not stack.use_weight_standardization:
        _load_standard_layer(reader, stack.final_layer_norm)


def _load_waveform_generator(reader: _BinaryTensorReader, net_g: ConverterNetwork) -> None:
    _load_standard_layer(reader, net_g.embed_phone)
    _copy_parameter(
        net_g.embed_quantized_pitch.weight,
        reader.read(net_g.embed_quantized_pitch.weight.shape),
    )
    _load_standard_layer(reader, net_g.embed_pitch_features)
    vocoder = net_g.vocoder
    _load_convnext_stack(reader, vocoder.prenet)
    _load_convnext_stack(reader, vocoder.ir_generator)
    _load_standard_layer(reader, vocoder.ir_generator_post)
    _copy_parameter(vocoder.ir_window, reader.read(vocoder.ir_window.shape))
    _load_convnext_stack(reader, vocoder.aperiodicity_generator)
    _load_standard_layer(reader, vocoder.aperiodicity_generator_post)
    _load_convnext_stack(reader, vocoder.post_filter_generator)
    _load_standard_layer(reader, vocoder.post_filter_generator_post)
    vocoder.ir_scale.fill_(1.0)
    vocoder.aperiodicity_scale.fill_(1.0)
    vocoder.post_filter_scale.fill_(1.0)


def _load_teacher_speaker_embeddings(
    net_g: ConverterNetwork,
    teachers: list[tuple[Path, int, dict]],
    model_dir: Path,
) -> None:
    """Load non-codebook speaker tables for each selected voice in one teacher model."""
    embedding_file = model_dir / "speaker_embeddings.bin"
    values = np.fromfile(embedding_file, dtype=np.float16)
    codebook_elements = math.prod(net_g.vq.codebooks.shape[1:])
    speaker_elements = net_g.embed_speaker.embedding_dim
    key_value_elements = net_g.key_value_speaker_embedding.embedding_dim
    formant_elements = math.prod(net_g.embed_formant_shift.weight.shape)
    per_speaker_elements = codebook_elements + speaker_elements + key_value_elements
    if values.size < formant_elements or (values.size - formant_elements) % per_speaker_elements:
        raise ValueError(f"Invalid speaker_embeddings.bin: {embedding_file}")
    n_speakers = (values.size - formant_elements) // per_speaker_elements
    speaker_offset = n_speakers * codebook_elements
    formant_offset = speaker_offset + n_speakers * speaker_elements
    key_value_offset = formant_offset + formant_elements

    _copy_parameter(
        net_g.embed_formant_shift.weight,
        torch.from_numpy(values[formant_offset:key_value_offset].reshape(
            net_g.embed_formant_shift.weight.shape
        ).copy()),
    )
    for student_speaker_id, (teacher_dir, teacher_speaker_id, _) in enumerate(teachers):
        if teacher_dir.resolve() != model_dir.resolve():
            continue
        if teacher_speaker_id >= n_speakers:
            raise ValueError(
                f"Teacher speaker {teacher_speaker_id} is not present in {embedding_file}"
            )
        speaker_start = speaker_offset + teacher_speaker_id * speaker_elements
        key_value_start = key_value_offset + teacher_speaker_id * key_value_elements
        _copy_parameter(
            net_g.embed_speaker.weight[student_speaker_id],
            torch.from_numpy(values[speaker_start:speaker_start + speaker_elements].copy()),
        )
        _copy_parameter(
            net_g.key_value_speaker_embedding.weight[student_speaker_id],
            torch.from_numpy(values[key_value_start:key_value_start + key_value_elements].copy()),
        )


def initialize_generator_from_teachers(
    net_g: ConverterNetwork,
    teachers: list[tuple[Path, int, dict]],
    weights_dir: Path,
    mode: str,
    model: str | None,
) -> None:
    model_dirs = list(dict.fromkeys(model_dir for model_dir, _, _ in teachers))
    if mode == "teacher":
        if not model:
            raise ValueError("generator_init_model is required when generator_init_mode is 'teacher'")
        model_dir = Path(model)
        if not model_dir.is_absolute():
            model_dir = weights_dir / model_dir
        model_dir = model_dir.resolve()
        if model_dir not in {path.resolve() for path in model_dirs}:
            raise ValueError(f"generator_init_model is not a discovered teacher model: {model_dir}")
        model_dirs = [model_dir]
    elif mode != "average":
        raise ValueError("generator_init_mode must be 'pretrained', 'teacher', or 'average'")

    waveform_reader = _BinaryTensorReader(
        [model_dir / "waveform_generator.bin" for model_dir in model_dirs]
    )
    embedding_reader = _BinaryTensorReader(
        [model_dir / "embedding_setter.bin" for model_dir in model_dirs]
    )
    with torch.no_grad():
        _load_waveform_generator(waveform_reader, net_g)
        _load_embedding_setter(embedding_reader, net_g.vocoder.prenet)
        if mode == "teacher":
            _load_teacher_speaker_embeddings(net_g, teachers, model_dirs[0])
    waveform_reader.finish()
    embedding_reader.finish()
    print(f"Initialized waveform generator from {mode}: {', '.join(map(str, model_dirs))}")


def export_paraphernalia(
    output_dir: Path,
    net_g: ConverterNetwork,
    phone_extractor: PhoneExtractor,
    pitch_estimator: PitchEstimator,
    teachers: list[tuple[Path, int, dict]],
    h: AttrDict,
) -> None:
    """Write the same inference files emitted by the regular trainer."""
    output_dir.mkdir()
    export_net = ConverterNetwork(
        torch.nn.Module(),
        torch.nn.Module(),
        len(teachers),
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
    shutil.copy(repo_root() / "beatrice-trainer" / "assets" / "images" / "noimage.png", output_dir / "noimage.png")
    write_merged_toml(output_dir, teachers)


def write_merged_toml(output_dir: Path, teachers: list[tuple[Path, int, dict]]) -> None:
    """Keep each original voice's TOML metadata, renumbered in export order."""
    voice_sections = {}
    for _, _, teacher in teachers:
        toml_file = teacher["toml_file"]
        if toml_file not in voice_sections:
            text = toml_file.read_text(encoding="utf-8")
            matches = list(VOICE_TABLE_RE.finditer(text))
            voice_sections[toml_file] = {
                int(match.group(1)): text[match.start() : matches[index + 1].start() if index + 1 < len(matches) else len(text)]
                for index, match in enumerate(matches)
            }
    with open(output_dir / "beatrice_paraphernalia_distilled.toml", "w", encoding="utf-8") as file:
        file.write(
            f'[model]\nversion = "{PARAPHERNALIA_VERSION}"\n'
            'name = "distilled"\n'
            'description = ""\n'
        )
        for speaker_id, (model_dir, teacher_speaker_id, teacher) in enumerate(teachers):
            section = voice_sections[teacher["toml_file"]].get(teacher_speaker_id)
            if section is None:
                raise ValueError(f"Missing [voice.{teacher_speaker_id}] section in {teacher['toml_file']}")
            portrait = teacher["voice"].get("portrait", {}).get("path")
            if portrait and (model_dir / portrait).is_file():
                portrait_file = Path(portrait)
                merged_portrait = f"voice_{speaker_id}{portrait_file.suffix}"
                shutil.copy(model_dir / portrait, output_dir / merged_portrait)
                section = re.sub(
                    rf'(?m)^(\[voice\.{teacher_speaker_id}\.portrait\][\s\S]*?^path\s*=\s*)"[^"]*"',
                    rf'\1"{merged_portrait}"',
                    section,
                )
            section = re.sub(rf'(?m)^\[voice\.{teacher_speaker_id}', f"[voice.{speaker_id}", section)
            file.write("\n" + section.strip() + "\n")


def main() -> None:
    args = parse_args()
    h = load_config(args.config)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    if not args.weights_dir.is_dir():
        raise ValueError(f"Weights directory does not exist: {args.weights_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    source_files = find_source_files(args.data_dir)
    teachers = find_teachers(args.weights_dir)
    noise_files = find_augmentation_files(
        repo_root() / h.in_noise_wav_dir, "input noise augmentation"
    )
    ir_files = find_augmentation_files(
        repo_root() / h.in_ir_wav_dir, "input reverb augmentation"
    )
    examples = make_examples(source_files, teachers)
    if not h.vq_init_from_bin:
        if h.vq_init_max_files < 1:
            raise ValueError("vq_init_max_files must be at least 1")
        if h.vq_init_wav_length < h.in_sample_rate // 100:
            raise ValueError("vq_init_wav_length must contain at least one 10 ms frame")
        if h.vq_init_wav_length % (h.in_sample_rate // 100) != 0:
            raise ValueError("vq_init_wav_length must be a multiple of the 10 ms frame length")
    print(f"device={device}; sources={len(source_files)}; speakers={len(teachers)}; examples={len(examples)}")
    for speaker_id, (model_dir, teacher_speaker_id, teacher) in enumerate(teachers):
        print(f"  {speaker_id}: {teacher['voice'].get('name', f'{model_dir.name}:{teacher_speaker_id}')} ({model_dir})")

    dataset = DistillationDataset(
        examples,
        h.in_sample_rate,
        h.out_sample_rate,
        h.wav_length,
        h.segment_length,
        h.formant_shift_candidates,
        h.vq_topk,
        noise_files,
        ir_files,
        h.augmentation_snr_candidates,
        h.augmentation_formant_shift_probability,
        h.augmentation_formant_shift_semitone_min,
        h.augmentation_formant_shift_semitone_max,
        h.augmentation_reverb_probability,
        h.augmentation_lpf_probability,
        h.augmentation_lpf_cutoff_freq_candidates,
    )
    loader_workers = 0 if os.name == "nt" else min(h.num_workers, os.cpu_count() or 1)
    if os.name == "nt" and h.num_workers != 0:
        print("Windows uses num_workers=0 because teacher inference datasets cannot be spawned safely.")
    loader = DataLoader(
        dataset,
        batch_size=h.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=loader_workers,
        pin_memory=True,
        persistent_workers=loader_workers > 0,
        collate_fn=dataset.collate,
    )
    if len(loader) == 0:
        raise ValueError("Not enough pairs for one batch. Reduce batch_size or add pairs.")

    phone_extractor, pitch_estimator = load_frozen_models(h, device)
    net_g = ConverterNetwork(phone_extractor, pitch_estimator, len(teachers), h.pitch_bins, h.hidden_channels, h.vq_topk, h.training_time_vq, h.phone_noise_ratio, h.floor_noise_level).to(device)
    net_d = MultiPeriodDiscriminator(san=h.san).to(device)
    with gzip.open(repo_root() / h.pretrained_file, "rb") as file:
        pretrained = torch.load(file, map_location="cpu", weights_only=True)
    net_g.load_state_dict(prepare_pretrained_generator(pretrained, len(teachers)), strict=False)
    if h.generator_init_mode != "pretrained":
        initialize_generator_from_teachers(
            net_g,
            teachers,
            args.weights_dir,
            h.generator_init_mode,
            h.generator_init_model,
        )
    net_d.load_state_dict(pretrained["net_d"], strict=False)

    if h.vq_init_from_bin:
        codebooks = load_teacher_codebooks(teachers)
        expected_shape = (len(teachers), net_g.vq.codebook_size, net_g.vq.channels)
        if tuple(codebooks.shape) != expected_shape:
            raise ValueError(
                f"Loaded codebooks have shape {tuple(codebooks.shape)}, "
                f"expected {expected_shape}"
            )
        with torch.no_grad():
            net_g.vq.codebooks.copy_(codebooks.to(device))
        net_g.enable_hook()
    else:
        # Bound the activation set so K-means does not materialize a GPU-sized corpus.
        vq_source_files = [
            source_files[index * len(source_files) // min(len(source_files), h.vq_init_max_files)]
            for index in range(min(len(source_files), h.vq_init_max_files))
        ]

        # Codebooks describe the 16 kHz targets generated by each teacher voice.
        def wav_iterator(speaker_id: int):
            for source_file in vq_source_files:
                source, sample_rate = dataset._load_mono(source_file)
                source = get_resampler(sample_rate, h.in_sample_rate)(source).squeeze(0)
                model_dir, teacher_speaker_id, _ = teachers[speaker_id]
                target = dataset._generate_target(source, model_dir, teacher_speaker_id)
                target = get_resampler(h.out_sample_rate, h.in_sample_rate)(target[None]).squeeze(0)
                yield target[: h.vq_init_wav_length].to(device)[None, None]

        net_g.initialize_vq([wav_iterator(speaker_id) for speaker_id in range(len(teachers))])
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
            source_wavs, target_wavs, slice_starts, speaker_ids, formant_shifts = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            source_wavs, target_wavs, slice_starts, speaker_ids, formant_shifts = next(data_iterator)
        source_wavs, target_wavs, slice_starts, speaker_ids, formant_shifts = (
            value.to(device, non_blocking=True)
            for value in (
                source_wavs,
                target_wavs,
                slice_starts,
                speaker_ids,
                formant_shifts,
            )
        )
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
                teachers,
                h,
            )
    writer.close()


if __name__ == "__main__":
    main()