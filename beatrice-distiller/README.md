# Beatrice Distiller

`beatrice_distiller` trains one multi-speaker student model from source audio and one or more Beatrice teacher models. Teacher outputs are generated on demand from the teacher `.bin` files, so no pre-rendered target audio is required.

## Dataset layout

Pass a source-data directory with `-d`; every `.wav` file below it is found recursively and used. Pass a teacher-model directory with `-w`; every subdirectory containing the five Beatrice inference `.bin` files and one TOML file is found recursively. Each voice in each discovered teacher model becomes one voice in the distilled model, in path order followed by original voice-number order.

```text
data/
  corpus_a/
    set_01/phrase_001.wav
  corpus_b/phrase_002.wav
weights/
  teacher_a/
    phone_extractor.bin
    pitch_estimator.bin
    waveform_generator.bin
    embedding_setter.bin
    speaker_embeddings.bin
    teacher_a.toml
  teacher_b/
    ...
```

Each source waveform is converted by every teacher voice with zero pitch/formant shift and zero VQ neighbours. The conversion is reset per waveform, then its generated output is used as the training target.

## Run

From the repository root, activate the environment and start training:

```powershell
& c:\Users\gokqo\.venv\base\Scripts\Activate.ps1
python -m beatrice_distiller -d data -w weights -o distilled_output
```

Use `-c config.json` to override keys in the distiller's `DEFAULT_CONFIG`. The pretrained generator checkpoint and the two frozen feature extractors must remain available at the configured paths.

At each `save_interval`, the distiller writes a training checkpoint and a `paraphernalia_distilled_<step>` directory. The latter contains the inference `.bin` files and a TOML file. The output TOML combines the original teacher voice sections in the same discovery order as the distilled speaker embeddings, including voice descriptions, average pitches, and portraits where their files are available.

## Notes

- Training feeds each source waveform to the student and uses the teacher's generated waveform as its target, so it learns the actual conversion task rather than self-reconstruction from pseudo voices.
- VQ codebooks are built from dynamically generated output for each teacher voice.
- Formant-shift augmentation is disabled because it would invalidate source/target alignment.
- Confirm that you have permission to generate derivatives of every teacher model and every source recording used in the dataset.

## Dynamic Teacher Inference

`native/beatrice_inference.cc` is a pybind11 binding to the official
`beatrice.lib` C API supplied by the `beatrice-vst` submodule. It supports
Beatrice `2.0.0-rc.0` paraphernalia directories, including models containing
multiple voices.

The build downloads the official `beatrice.lib` to
`beatrice-vst/lib/beatricelib/beatrice.lib` when it is not already present.
Install CMake, a C++20 compiler compatible with the library, and pybind11 in
the Python environment used by the distiller. On Windows, build it from this
directory as follows:

```powershell
python -m pip install pybind11
cmake -S native -B native/build -A x64 -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build native/build --config Release
```

The CMake project selects `/MT` automatically because it must match the
official Windows library. The resulting `_beatrice_inference` extension is
written directly into this directory. Use a separate `Converter` instance for
each teacher model because each instance owns its streaming inference state.

```python
from pathlib import Path

import numpy as np
from beatrice_inference import Converter

teacher = Converter()
teacher.load_model(Path("weights/teacher_paraphernalia"))
teacher.set_target_speaker(0)
teacher.set_formant_shift(0.0)
teacher.set_pitch_shift(0.0)
teacher.set_vq_num_neighbors(0)
teacher.reset()

# mono_source_16k is a one-dimensional np.float32 array at 16 kHz.
teacher_output_24k = teacher.process(mono_source_16k)
```

`process()` preserves the native streaming context. It zero-pads the final
partial 160-sample input hop and returns 240 samples for every processed hop.
The output therefore contains the inference engine's startup/tail delay. The
distiller calls `process()` once per source waveform, keeps this padding, and
trims the source/target pair to their shared frame count before training.