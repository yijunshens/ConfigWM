"""Narrow public exports for ``src.data``."""

from __future__ import annotations

from .preprocessor import (
    Preprocessor,
    create_preprocessor,
)

# Sample types from schema
from .schema import (
    DictMixin,
    ImageBoxesSample,
    ImageMasksSample,
    ImageTextSample,
    PreprocessedSample,
    VideoMasksSample,
    Sample,
)

__all__ = [
    "DictMixin",
    "PreprocessedSample",
    "Preprocessor",
    "Sample",
    "create_preprocessor",
]


# Backward-compatible Python names; internal code uses the canonical names above.
VLSample = Sample
PreprocessedVLSample = PreprocessedSample
__all__ += ['VLSample', 'PreprocessedVLSample']
