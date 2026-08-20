#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "beatricelib/beatrice.h"

namespace py = pybind11;
namespace fs = std::filesystem;

namespace {

constexpr int kFormantEmbeddingCount = 9;

auto ErrorMessage(const Beatrice_ErrorCode error) -> const char* {
  switch (error) {
    case Beatrice_kSuccess:
      return "success";
    case Beatrice_kFileOpenError:
      return "could not open file";
    case Beatrice_kFileTooSmall:
      return "file is too small";
    case Beatrice_kFileTooLarge:
      return "file is too large";
    case Beatrice_kInvalidFileSize:
      return "invalid file size or unsupported model version";
  }
  return "unknown Beatrice library error";
}

void Check(const Beatrice_ErrorCode error, const std::string& operation) {
  if (error != Beatrice_kSuccess) {
    throw std::runtime_error(operation + ": " + ErrorMessage(error));
  }
}

auto Utf8Path(const fs::path& path) -> std::string {
  const auto text = path.u8string();
  return {reinterpret_cast<const char*>(text.data()), text.size()};
}

class Converter {
 public:
  Converter() {
    phone_extractor_.reset(Beatrice20rc0_CreatePhoneExtractor());
    pitch_estimator_.reset(Beatrice20rc0_CreatePitchEstimator());
    waveform_generator_.reset(Beatrice20rc0_CreateWaveformGenerator());
    embedding_setter_.reset(Beatrice20rc0_CreateEmbeddingSetter());
    if (!phone_extractor_ || !pitch_estimator_ || !waveform_generator_ ||
        !embedding_setter_) {
      throw std::runtime_error("could not create Beatrice inference objects");
    }
    CreateContexts();
  }

  Converter(const Converter&) = delete;
  auto operator=(const Converter&) -> Converter& = delete;

  void LoadModel(const fs::path& directory) {
    if (!fs::is_directory(directory)) {
      throw std::invalid_argument("model_dir is not a directory: " +
                                  directory.string());
    }
    Check(Beatrice20rc0_ReadPhoneExtractorParameters(
              phone_extractor_.get(),
              Utf8Path(directory / "phone_extractor.bin").c_str()),
          "reading phone_extractor.bin");
    Check(Beatrice20rc0_ReadPitchEstimatorParameters(
              pitch_estimator_.get(),
              Utf8Path(directory / "pitch_estimator.bin").c_str()),
          "reading pitch_estimator.bin");
    Check(Beatrice20rc0_ReadWaveformGeneratorParameters(
              waveform_generator_.get(),
              Utf8Path(directory / "waveform_generator.bin").c_str()),
          "reading waveform_generator.bin");
    Check(Beatrice20rc0_ReadEmbeddingSetterParameters(
              embedding_setter_.get(),
              Utf8Path(directory / "embedding_setter.bin").c_str()),
          "reading embedding_setter.bin");

    const auto embeddings = directory / "speaker_embeddings.bin";
    Check(
        Beatrice20rc0_ReadNSpeakers(Utf8Path(embeddings).c_str(), &n_speakers_),
        "reading speaker count");
    if (n_speakers_ <= 0) {
      throw std::runtime_error(
          "speaker_embeddings.bin does not contain a speaker");
    }
    codebooks_.resize(n_speakers_ * BEATRICE_20RC0_CODEBOOK_SIZE *
                      BEATRICE_20RC0_PHONE_CHANNELS);
    additive_embeddings_.resize(n_speakers_ *
                                BEATRICE_WAVEFORM_GENERATOR_HIDDEN_CHANNELS);
    formant_embeddings_.resize(kFormantEmbeddingCount *
                               BEATRICE_WAVEFORM_GENERATOR_HIDDEN_CHANNELS);
    key_value_embeddings_.resize(n_speakers_ * BEATRICE_20RC0_KV_LENGTH *
                                 BEATRICE_20RC0_KV_SPEAKER_EMBEDDING_CHANNELS);
    Check(Beatrice20rc0_ReadSpeakerEmbeddings(
              Utf8Path(embeddings).c_str(), codebooks_.data(),
              additive_embeddings_.data(), formant_embeddings_.data(),
              key_value_embeddings_.data()),
          "reading speaker_embeddings.bin");
    loaded_ = true;
    target_speaker_ = 0;
    formant_shift_ = 0.0;
    pitch_shift_ = 0.0;
    Reset();
  }

  void Reset() {
    RequireLoaded();
    DestroyContexts();
    CreateContexts();
    ApplySpeaker(target_speaker_);
    ApplyFormantShift(formant_shift_);
    Beatrice20rc0_SetMinQuantizedPitch(pitch_context_.get(), 1);
    Beatrice20rc0_SetMaxQuantizedPitch(pitch_context_.get(),
                                       BEATRICE_20RC0_PITCH_BINS - 1);
    Beatrice20rc0_SetVQNumNeighbors(phone_context_.get(), vq_num_neighbors_);
    FlushKeyValueEmbedding();
  }

  void SetTargetSpeaker(const int speaker_id) {
    RequireLoaded();
    if (speaker_id < 0 || speaker_id >= n_speakers_) {
      throw std::out_of_range("speaker_id is outside the model's voice range");
    }
    target_speaker_ = speaker_id;
    ApplySpeaker(speaker_id);
  }

  void SetFormantShift(const double semitones) {
    RequireLoaded();
    formant_shift_ = std::clamp(semitones, -2.0, 2.0);
    ApplyFormantShift(formant_shift_);
  }

  void SetPitchShift(const double semitones) {
    pitch_shift_ = std::clamp(semitones, -24.0, 24.0);
  }

  void SetVQNumNeighbors(const int neighbors) {
    RequireLoaded();
    vq_num_neighbors_ = std::clamp(neighbors, 0, 8);
    Beatrice20rc0_SetVQNumNeighbors(phone_context_.get(), vq_num_neighbors_);
  }

  [[nodiscard]] auto NSpeakers() const -> int { return n_speakers_; }

  auto Process(const py::array_t<float, py::array::c_style |
                                            py::array::forcecast>& input)
      -> py::array_t<float> {
    RequireLoaded();
    const auto buffer = input.request();
    if (buffer.ndim != 1) {
      throw std::invalid_argument(
          "input must be a one-dimensional float32 array");
    }
    const auto sample_count = static_cast<size_t>(buffer.shape[0]);
    const auto blocks =
        (sample_count + BEATRICE_IN_HOP_LENGTH - 1) / BEATRICE_IN_HOP_LENGTH;
    auto output = py::array_t<float>(blocks * BEATRICE_OUT_HOP_LENGTH);
    const auto* input_data = static_cast<const float*>(buffer.ptr);
    auto* output_data = static_cast<float*>(output.request().ptr);
    std::array<float, BEATRICE_IN_HOP_LENGTH> padded_input{};
    std::array<float, BEATRICE_20RC0_PHONE_CHANNELS> phone{};
    std::array<float, 4> pitch_features{};
    for (size_t block = 0; block < blocks; ++block) {
      const auto offset = block * BEATRICE_IN_HOP_LENGTH;
      const auto available =
          std::min<size_t>(BEATRICE_IN_HOP_LENGTH, sample_count - offset);
      const float* block_input = input_data + offset;
      if (available != BEATRICE_IN_HOP_LENGTH) {
        padded_input.fill(0.0f);
        std::copy_n(block_input, available, padded_input.data());
        block_input = padded_input.data();
      }
      SetNextKeyValueEmbeddingBlock();
      Beatrice20rc0_ExtractPhone1(phone_extractor_.get(), block_input,
                                  phone.data(), phone_context_.get());
      int quantized_pitch = 1;
      Beatrice20rc0_EstimatePitch1(pitch_estimator_.get(), block_input,
                                   &quantized_pitch, pitch_features.data(),
                                   pitch_context_.get());
      const auto shifted_pitch = static_cast<int>(
          std::round(quantized_pitch +
                     pitch_shift_ * BEATRICE_PITCH_BINS_PER_OCTAVE / 12.0));
      quantized_pitch =
          std::clamp(shifted_pitch, 1, BEATRICE_20RC0_PITCH_BINS - 1);
      Beatrice20rc0_GenerateWaveform1(
          waveform_generator_.get(), phone.data(), &quantized_pitch,
          pitch_features.data(), output_data + block * BEATRICE_OUT_HOP_LENGTH,
          waveform_context_.get());
    }
    return output;
  }

 private:
  template <typename T, void (*Destroy)(T*)>
  using Handle = std::unique_ptr<T, decltype(Destroy)>;

  Handle<Beatrice20rc0_PhoneExtractor, Beatrice20rc0_DestroyPhoneExtractor>
      phone_extractor_{nullptr, Beatrice20rc0_DestroyPhoneExtractor};
  Handle<Beatrice20rc0_PitchEstimator, Beatrice20rc0_DestroyPitchEstimator>
      pitch_estimator_{nullptr, Beatrice20rc0_DestroyPitchEstimator};
  Handle<Beatrice20rc0_WaveformGenerator,
         Beatrice20rc0_DestroyWaveformGenerator>
      waveform_generator_{nullptr, Beatrice20rc0_DestroyWaveformGenerator};
  Handle<Beatrice20rc0_EmbeddingSetter, Beatrice20rc0_DestroyEmbeddingSetter>
      embedding_setter_{nullptr, Beatrice20rc0_DestroyEmbeddingSetter};
  Handle<Beatrice20rc0_PhoneContext1, Beatrice20rc0_DestroyPhoneContext1>
      phone_context_{nullptr, Beatrice20rc0_DestroyPhoneContext1};
  Handle<Beatrice20rc0_PitchContext1, Beatrice20rc0_DestroyPitchContext1>
      pitch_context_{nullptr, Beatrice20rc0_DestroyPitchContext1};
  Handle<Beatrice20rc0_WaveformContext1, Beatrice20rc0_DestroyWaveformContext1>
      waveform_context_{nullptr, Beatrice20rc0_DestroyWaveformContext1};
  Handle<Beatrice20rc0_EmbeddingContext, Beatrice20rc0_DestroyEmbeddingContext>
      embedding_context_{nullptr, Beatrice20rc0_DestroyEmbeddingContext};

  std::vector<float> codebooks_;
  std::vector<float> additive_embeddings_;
  std::vector<float> formant_embeddings_;
  std::vector<float> key_value_embeddings_;
  int n_speakers_ = 0;
  int target_speaker_ = 0;
  int key_value_block_ = 0;
  int vq_num_neighbors_ = 0;
  double formant_shift_ = 0.0;
  double pitch_shift_ = 0.0;
  bool loaded_ = false;

  void CreateContexts() {
    phone_context_.reset(Beatrice20rc0_CreatePhoneContext1());
    pitch_context_.reset(Beatrice20rc0_CreatePitchContext1());
    waveform_context_.reset(Beatrice20rc0_CreateWaveformContext1());
    embedding_context_.reset(Beatrice20rc0_CreateEmbeddingContext());
    if (!phone_context_ || !pitch_context_ || !waveform_context_ ||
        !embedding_context_) {
      throw std::runtime_error("could not create Beatrice inference contexts");
    }
  }

  void DestroyContexts() {
    phone_context_.reset();
    pitch_context_.reset();
    waveform_context_.reset();
    embedding_context_.reset();
  }

  void RequireLoaded() const {
    if (!loaded_) {
      throw std::runtime_error("load_model() must be called first");
    }
  }

  void ApplySpeaker(const int speaker_id) {
    const auto codebook_size =
        BEATRICE_20RC0_CODEBOOK_SIZE * BEATRICE_20RC0_PHONE_CHANNELS;
    const auto kv_size =
        BEATRICE_20RC0_KV_LENGTH * BEATRICE_20RC0_KV_SPEAKER_EMBEDDING_CHANNELS;
    Beatrice20rc0_SetCodebook(phone_context_.get(),
                              codebooks_.data() + speaker_id * codebook_size);
    Beatrice20rc0_SetAdditiveSpeakerEmbedding(
        embedding_setter_.get(),
        additive_embeddings_.data() +
            speaker_id * BEATRICE_WAVEFORM_GENERATOR_HIDDEN_CHANNELS,
        embedding_context_.get(), waveform_context_.get());
    Beatrice20rc0_RegisterKeyValueSpeakerEmbedding(
        embedding_setter_.get(),
        key_value_embeddings_.data() + speaker_id * kv_size,
        embedding_context_.get());
    key_value_block_ = 0;
  }

  void ApplyFormantShift(const double semitones) {
    const auto index = static_cast<int>(std::round(semitones * 2.0 + 4.0));
    Beatrice20rc0_SetFormantShiftEmbedding(
        embedding_setter_.get(),
        formant_embeddings_.data() +
            index * BEATRICE_WAVEFORM_GENERATOR_HIDDEN_CHANNELS,
        embedding_context_.get(), waveform_context_.get());
  }

  void SetNextKeyValueEmbeddingBlock() {
    if (key_value_block_ < BEATRICE_20RC0_N_BLOCKS) {
      Beatrice20rc0_SetKeyValueSpeakerEmbedding(
          embedding_setter_.get(), key_value_block_++, embedding_context_.get(),
          waveform_context_.get());
    }
  }

  void FlushKeyValueEmbedding() {
    while (key_value_block_ < BEATRICE_20RC0_N_BLOCKS) {
      SetNextKeyValueEmbeddingBlock();
    }
  }
};

}  // namespace

PYBIND11_MODULE(_beatrice_inference, module) {
  module.doc() =
      "Bindings for the official Beatrice 2.0.0-rc.0 inference library.";
  module.attr("IN_SAMPLE_RATE") = BEATRICE_IN_SAMPLE_RATE;
  module.attr("OUT_SAMPLE_RATE") = BEATRICE_OUT_SAMPLE_RATE;
  module.attr("IN_HOP_LENGTH") = BEATRICE_IN_HOP_LENGTH;
  module.attr("OUT_HOP_LENGTH") = BEATRICE_OUT_HOP_LENGTH;
  py::class_<Converter>(module, "Converter")
      .def(py::init<>())
      .def("load_model", &Converter::LoadModel, py::arg("model_dir"))
      .def("reset", &Converter::Reset)
      .def("set_target_speaker", &Converter::SetTargetSpeaker,
           py::arg("speaker_id"))
      .def("set_formant_shift", &Converter::SetFormantShift,
           py::arg("semitones"))
      .def("set_pitch_shift", &Converter::SetPitchShift, py::arg("semitones"))
      .def("set_vq_num_neighbors", &Converter::SetVQNumNeighbors,
           py::arg("neighbors"))
      .def_property_readonly("n_speakers", &Converter::NSpeakers)
      .def(
          "process", &Converter::Process, py::arg("input"),
          "Process mono 16 kHz float32 audio and return 24 kHz float32 audio.");
}