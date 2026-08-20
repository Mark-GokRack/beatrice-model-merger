"""Python import surface for the official Beatrice inference binding.

Build the extension with ``cmake --build native/build --config Release`` before
importing this module. See README.md for the complete build command.
"""

from _beatrice_inference import (
    IN_HOP_LENGTH,
    IN_SAMPLE_RATE,
    OUT_HOP_LENGTH,
    OUT_SAMPLE_RATE,
    Converter,
)

__all__ = [
    "Converter",
    "IN_HOP_LENGTH",
    "IN_SAMPLE_RATE",
    "OUT_HOP_LENGTH",
    "OUT_SAMPLE_RATE",
]