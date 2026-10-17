"""Sample schema definitions for Configurable Encoder data pipeline."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from typing import Any

from PIL import Image
import torch


class InvalidSampleError(ValueError):
    """Raised when one raw sample violates the shared data contract."""


class DictMixin(Mapping):
    """Mixin to make dataclasses behave like dicts for PyTorch collation.

    Inherits from Mapping so isinstance(obj, Mapping) returns True,
    which is required by PyTorch's default_collate.
    """

    def keys(self) -> list[str]:
        return [f.name for f in fields(self)]

    def values(self) -> list[Any]:
        return [getattr(self, f.name) for f in fields(self)]

    def items(self) -> list[tuple[str, Any]]:
        return [(f.name, getattr(self, f.name)) for f in fields(self)]

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(fields(self))


@dataclass
class ImageTextSample(DictMixin):
    image: Image.Image
    text: str
    meta: dict[str, Any]
    query: str | None = None


@dataclass
class ImageBoxesSample(DictMixin):
    image: Image.Image
    boxes: list[list[float]]
    labels: list[str]
    meta: dict[str, Any]
    query: str | None = None


@dataclass
class ImageMasksSample(DictMixin):
    image: Image.Image
    masks: list[torch.Tensor]
    labels: list[str]
    meta: dict[str, Any]
    query: str | None = None


@dataclass
class VideoMasksSample(DictMixin):
    frames: list[Image.Image]
    masks: list[torch.Tensor]
    labels: list[str]
    meta: dict[str, Any]
    query: str | None = None


@dataclass
class Sample(DictMixin):
    """Raw sample before preprocessing.

    Exactly one raw modality must be present:
    - text-only samples use a non-empty ``text_input`` and ``frames=[]``
    - vision/video samples use non-empty ``frames`` and ``text_input=None``
    """

    frames: List[Image.Image]  # T PIL RGB frames (empty list for text-only)
    label_map: torch.Tensor  # (T, H, W) tensor with dtype torch.long; empty 1-D long tensor for text-only
    targets: List[str]  # list of target labels
    meta: Dict[str, Any]  # additional info
    target_frames: Optional[List[Image.Image]] = None  # optional paired target frames
    target_label_map: Optional[torch.Tensor] = None  # optional paired target mask (T,H,W)
    query: Optional[str] = None  # optional query string shared by all targets
    text_input: Optional[str] = (
        None  # text input alternative to vision (for text-only samples)
    )

    # Priority list for auto-deriving meta["id"] from existing fields.
    _ID_KEYS = (
        "id",
        "sample_id",
        "example_id",
        "annotation_id",
        "key",
        "sample_key",
        "image_id",
        "video_uid",
        "question_id",
        "narration_id",
        "video_name",
        "video_id",
    )

    def __post_init__(self):
        # Auto-derive meta["id"] from existing meta fields if not set.
        if "id" not in self.meta:
            for k in self._ID_KEYS:
                if k in self.meta and self.meta[k] is not None:
                    self.meta["id"] = str(self.meta[k])
                    break
        validate_sample(self)


def validate_sample(sample: Sample) -> None:
    """Validate the raw-sample contract shared by training and visualization."""
    if not isinstance(sample, Sample):
        raise TypeError(f"Expected Sample, got {type(sample).__name__}")
    if not isinstance(sample.frames, list):
        raise TypeError("sample.frames must be a list")
    if sample.target_frames is not None and not isinstance(
        sample.target_frames, list
    ):
        raise TypeError("sample.target_frames must be a list or None")
    if not isinstance(sample.targets, list):
        raise TypeError("sample.targets must be a list[str]")
    if not isinstance(sample.meta, dict):
        raise TypeError("sample.meta must be a dict")
    if not isinstance(sample.label_map, torch.Tensor):
        raise TypeError("sample.label_map must be a torch.Tensor")
    if sample.query is not None and not isinstance(sample.query, str):
        raise TypeError("sample.query must be a string or None")
    if sample.target_label_map is not None and not isinstance(
        sample.target_label_map, torch.Tensor
    ):
        raise TypeError("sample.target_label_map must be a torch.Tensor or None")

    has_frames = len(sample.frames) > 0
    has_target_frames = (
        sample.target_frames is not None and len(sample.target_frames) > 0
    )
    if sample.text_input is None:
        has_text_input = False
    else:
        if not isinstance(sample.text_input, str):
            raise TypeError("sample.text_input must be a string or None")
        if not sample.text_input.strip():
            raise InvalidSampleError(
                "text-only samples require a non-empty text_input"
            )
        has_text_input = True

    if has_frames == has_text_input:
        if has_frames:
            raise InvalidSampleError(
                "Sample must not define both frames and text_input"
            )
        raise InvalidSampleError(
            "Sample must define either frames or a non-empty text_input"
        )

    if has_text_input:
        if has_target_frames:
            raise InvalidSampleError(
                "text-only samples must not define target_frames"
            )
        if sample.label_map.dtype != torch.long:
            raise InvalidSampleError(
                "text-only sample.label_map must use torch.long"
            )
        if sample.label_map.numel() != 0:
            raise InvalidSampleError("text-only sample.label_map must be empty")
        if not sample.targets:
            raise InvalidSampleError(
                "text-only samples require at least one target"
            )
        if any(
            not isinstance(target, str) or not target.strip()
            for target in sample.targets
        ):
            raise InvalidSampleError(
                "text-only samples require non-empty target strings"
            )
        return

    if sample.label_map.dtype != torch.long:
        raise InvalidSampleError(
            "vision/video sample.label_map must use torch.long"
        )
    if sample.label_map.dim() != 3:
        raise InvalidSampleError(
            f"label_map should be 3-D (T, H, W), got shape {tuple(sample.label_map.shape)}"
        )

    num_frames = len(sample.frames)
    if sample.label_map.shape[0] != num_frames:
        raise InvalidSampleError(
            f"label_map.shape[0] ({sample.label_map.shape[0]}) != len(frames) ({num_frames})"
        )
    if has_target_frames and len(sample.target_frames) != num_frames:
        raise InvalidSampleError(
            "len(target_frames) must match len(frames) for paired vision samples"
        )
    if sample.target_label_map is not None:
        if sample.target_label_map.dim() != 3:
            raise InvalidSampleError(
                "target_label_map should be 3-D (T,H,W), got "
                f"shape {tuple(sample.target_label_map.shape)}"
            )
        if sample.target_label_map.shape[0] != num_frames:
            raise InvalidSampleError(
                "target_label_map.shape[0] "
                f"({sample.target_label_map.shape[0]}) != len(frames) ({num_frames})"
            )

    max_label = (
        int(sample.label_map.max().item()) if sample.label_map.numel() else 0
    )
    if len(sample.targets) != max_label:
        raise InvalidSampleError(
            f"len(targets) ({len(sample.targets)}) != max(label_map) ({max_label})"
        )


@dataclass
class PreprocessedSample(DictMixin):
    """Preprocessed sample ready for model input."""

    vision_input: Optional[
        torch.Tensor
    ]  # (T, C, H, W) - None for text-only samples
    targets: List[str]  # target text labels
    query: List[str]
    target_vision_input: Optional[
        torch.Tensor
    ] = None  # optional paired target tensor (T, C, H, W)
    target_mask: Optional[torch.Tensor] = None  # optional paired target mask (T,H,W)
    text_input: Optional[str] = (
        None  # text input alternative to vision (for text-only samples)
    )
    meta: Optional[Dict[str, Any]] = None


__all__ = [
    "DictMixin",
    "ImageBoxesSample",
    "ImageMasksSample",
    "InvalidSampleError",
    "PreprocessedSample",
    "Sample",
    "VideoMasksSample",
    "validate_sample",
]


# Backward-compatible Python names; internal code uses the canonical names above.
VLSample = Sample
PreprocessedVLSample = PreprocessedSample
validate_vlsample = validate_sample
__all__ += ['VLSample', 'PreprocessedVLSample', 'validate_vlsample']
