"""DINOv2 image preprocessor using HuggingFace AutoImageProcessor."""

from __future__ import annotations

import os

from PIL import Image
from src.data.preprocessor import Preprocessor
import torch
from transformers import AutoImageProcessor


class DINOv2Preprocessor(Preprocessor):
    """Preprocessor for DINOv2 image models."""

    def __init__(
        self,
        model_name: str,
        resolution: int | None = None,
        use_fast: bool = True,
    ) -> None:
        model_name = os.path.expanduser(model_name)
        is_local = os.path.isdir(model_name)
        self.model_name = model_name
        self._processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=is_local,
            use_fast=use_fast,
        )

        if resolution is not None:
            resolution = int(resolution)
            if hasattr(self._processor, "crop_size"):
                self._processor.crop_size["height"] = resolution
                self._processor.crop_size["width"] = resolution
            if hasattr(self._processor, "size"):
                self._processor.size["shortest_edge"] = resolution

    def __call__(self, frames: list[Image.Image]) -> torch.Tensor:
        """Convert PIL frames to DINOv2 pixel tensor (T, C, H, W)."""
        processed = self._processor(images=frames, return_tensors="pt")
        return processed["pixel_values"]


__all__ = ["DINOv2Preprocessor"]
