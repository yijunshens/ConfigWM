"""Preprocessor for Configurable Encoder datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from PIL import Image
import torch

from src.data.schema import PreprocessedSample


class Preprocessor(ABC):
    @abstractmethod
    def __call__(self, frames: list[Image.Image]) -> torch.Tensor:
        """Convert PIL frames to a (T,C,H,W) tensor."""

    def process_sample(self, sample: Any) -> PreprocessedSample:
        vision_input = self(sample.frames)
        target_frames = getattr(sample, "target_frames", None)
        target_vision_input = self(target_frames) if target_frames else None
        target_mask = getattr(sample, "target_label_map", None)
        if target_mask is not None:
            target_mask = target_mask.float()
        query = getattr(sample, "query", None) or ""
        return PreprocessedSample(
            vision_input=vision_input,
            targets=sample.targets,
            query=[query],
            target_vision_input=target_vision_input,
            target_mask=target_mask,
            meta=getattr(sample, "meta", None),
        )


def create_preprocessor(config: Mapping[str, Any]) -> Preprocessor:
    config = dict(config)
    prep_id = config.pop("id", None)
    if prep_id in {"dinov2", "hf_image"}:
        from src.data.preprocessor.dinov2 import DINOv2Preprocessor

        preprocessor_cls = DINOv2Preprocessor
    elif prep_id == "imagenet_image":
        from src.data.preprocessor.imagenet_image import ImageNetImagePreprocessor

        preprocessor_cls = ImageNetImagePreprocessor
    elif prep_id == "vjepa2":
        from src.data.preprocessor.vjepa2 import VJEPA2Preprocessor

        preprocessor_cls = VJEPA2Preprocessor
    else:
        available = "dinov2, hf_image, imagenet_image, vjepa2"
        raise ValueError(
            f"Preprocessor config missing or invalid 'id'. Available: {available}"
        )
    return preprocessor_cls(**config)


__all__ = ["Preprocessor", "create_preprocessor"]
