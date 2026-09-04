"""Offline Beatrice 2.0.0-rc.0 inference implemented with PyTorch."""

import math
import sys
from pathlib import Path

import numpy as np
import torch

DISTILLER_DIR = Path(__file__).resolve().parent
TRAINER_DIR = DISTILLER_DIR.parent / "beatrice-trainer"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

from beatrice_trainer.__main__ import ConverterNetwork, PhoneExtractor, PitchEstimator


IN_HOP_LENGTH = 160
IN_SAMPLE_RATE = 16000
OUT_HOP_LENGTH = 240
OUT_SAMPLE_RATE = 24000


class _BinaryTensorReader:
    """Read FP16 inference tensors in the order used by the Beatrice runtime."""

    def __init__(self, file: Path) -> None:
        self.file = file
        self.values = np.fromfile(file, dtype=np.float16)
        self.offset = 0

    def read(self, shape: torch.Size) -> torch.Tensor:
        count = math.prod(shape)
        end = self.offset + count
        if end > self.values.size:
            raise ValueError(f"Unexpected end of inference binary: {self.file}")
        value = torch.from_numpy(self.values[self.offset:end].reshape(tuple(shape)).copy())
        self.offset = end
        return value

    def finish(self) -> None:
        if self.offset != self.values.size:
            raise ValueError(
                f"Unexpected trailing values in inference binary: {self.file} "
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


class PyTorchConverter:
    """Load Beatrice paraphernalia and convert complete waveforms with PyTorch.

    This class intentionally performs offline inference. Unlike the official C API,
    ``process`` does not preserve a streaming context between calls.
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        phone_extractor_file: Path | str | None = None,
        pitch_estimator_file: Path | str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        root = DISTILLER_DIR.parent
        self.phone_extractor_file = Path(
            phone_extractor_file
            or root / "beatrice-trainer/assets/pretrained/122_checkpoint_03000000.pt"
        )
        self.pitch_estimator_file = Path(
            pitch_estimator_file
            or root / "beatrice-trainer/assets/pretrained/104_3_checkpoint_00300000.pt"
        )
        self.net_g: ConverterNetwork | None = None
        self.target_speaker = 0
        self.formant_shift = 0.0
        self.pitch_shift = 0.0
        self.vq_num_neighbors = 4

    @property
    def n_speakers(self) -> int:
        if self.net_g is None:
            raise RuntimeError("load_model() must be called first")
        return self.net_g.vq.n_speakers

    def load_model(self, directory: Path | str) -> None:
        directory = Path(directory)
        values = np.fromfile(directory / "speaker_embeddings.bin", dtype=np.float16)
        codebook_elements = 512 * 128
        speaker_elements = 256
        formant_elements = 9 * 256
        key_value_elements = 384 * 128
        per_speaker_elements = codebook_elements + speaker_elements + key_value_elements
        if values.size < formant_elements or (values.size - formant_elements) % per_speaker_elements:
            raise ValueError(f"Invalid speaker_embeddings.bin: {directory}")
        n_speakers = (values.size - formant_elements) // per_speaker_elements
        if n_speakers == 0:
            raise ValueError(f"speaker_embeddings.bin contains no speakers: {directory}")

        phone_extractor = PhoneExtractor().to(self.device).eval().requires_grad_(False)
        phone_checkpoint = torch.load(self.phone_extractor_file, map_location="cpu", weights_only=True)
        phone_extractor.load_state_dict(phone_checkpoint["phone_extractor"], strict=False)
        pitch_estimator = PitchEstimator().to(self.device).eval().requires_grad_(False)
        pitch_checkpoint = torch.load(self.pitch_estimator_file, map_location="cpu", weights_only=True)
        pitch_estimator.load_state_dict(pitch_checkpoint["pitch_estimator"])
        net_g = ConverterNetwork(
            phone_extractor,
            pitch_estimator,
            n_speakers,
            pitch_bins=448,
            hidden_channels=256,
            vq_topk=self.vq_num_neighbors,
        ).to(self.device)

        waveform_reader = _BinaryTensorReader(directory / "waveform_generator.bin")
        embedding_reader = _BinaryTensorReader(directory / "embedding_setter.bin")
        with torch.no_grad():
            _load_waveform_generator(waveform_reader, net_g)
            _load_embedding_setter(embedding_reader, net_g.vocoder.prenet)
            waveform_reader.finish()
            embedding_reader.finish()
            codebook_end = n_speakers * codebook_elements
            speaker_end = codebook_end + n_speakers * speaker_elements
            formant_end = speaker_end + formant_elements
            _copy_parameter(net_g.vq.codebooks, torch.from_numpy(values[:codebook_end].reshape(net_g.vq.codebooks.shape).copy()))
            _copy_parameter(net_g.embed_speaker.weight, torch.from_numpy(values[codebook_end:speaker_end].reshape(net_g.embed_speaker.weight.shape).copy()))
            _copy_parameter(net_g.embed_formant_shift.weight, torch.from_numpy(values[speaker_end:formant_end].reshape(net_g.embed_formant_shift.weight.shape).copy()))
            _copy_parameter(net_g.key_value_speaker_embedding.weight, torch.from_numpy(values[formant_end:].reshape(net_g.key_value_speaker_embedding.weight.shape).copy()))
        net_g.enable_hook()
        self.net_g = net_g.eval()
        self.target_speaker = 0

    def reset(self) -> None:
        """Match the native API; offline PyTorch inference has no retained state."""

    def set_target_speaker(self, speaker_id: int) -> None:
        if not 0 <= speaker_id < self.n_speakers:
            raise ValueError("speaker_id is outside the model's voice range")
        self.target_speaker = speaker_id

    def set_formant_shift(self, semitones: float) -> None:
        self.formant_shift = float(np.clip(semitones, -2.0, 2.0))

    def set_pitch_shift(self, semitones: float) -> None:
        self.pitch_shift = float(np.clip(semitones, -24.0, 24.0))

    def set_vq_num_neighbors(self, neighbors: int) -> None:
        self.vq_num_neighbors = int(np.clip(neighbors, 1, 8))
        if self.net_g is not None:
            self.net_g.vq.topk = self.vq_num_neighbors

    @torch.inference_mode()
    def process(self, input: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.net_g is None:
            raise RuntimeError("load_model() must be called first")
        wav = torch.as_tensor(input, dtype=torch.float32)
        if wav.ndim != 1:
            raise ValueError("input must be a one-dimensional waveform")
        output_length = math.ceil(wav.numel() / IN_HOP_LENGTH) * OUT_HOP_LENGTH
        pad = -wav.numel() % IN_HOP_LENGTH
        if pad:
            wav = torch.nn.functional.pad(wav, (0, pad))
        minimum_input_length = 5 * IN_HOP_LENGTH
        if wav.numel() < minimum_input_length:
            wav = torch.nn.functional.pad(wav, (0, minimum_input_length - wav.numel()))
        target_speaker = torch.tensor([self.target_speaker], device=self.device)
        formant_shift = torch.tensor([self.formant_shift], device=self.device)
        pitch_shift = torch.tensor([self.pitch_shift], device=self.device)
        output = self.net_g(
            wav[None, None].to(self.device),
            target_speaker,
            formant_shift,
            pitch_shift,
        )
        return output.squeeze().cpu().numpy()[:output_length]