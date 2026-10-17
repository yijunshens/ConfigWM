"""ImageNet-normalized square image preprocessor for patch-token backbones."""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch

from src.data.preprocessor import Preprocessor


class ImageNetImagePreprocessor(Preprocessor):
    def __init__(
        self,
        resolution: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.resolution = int(resolution)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        if torch.any(self.std <= 0):
            raise ValueError("Image normalization std values must be positive")

    def _resize_and_crop(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = self.resolution / min(width, height)
        resized = image.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.BILINEAR,
        )
        left = (resized.width - self.resolution) // 2
        top = (resized.height - self.resolution) // 2
        return resized.crop(
            (left, top, left + self.resolution, top + self.resolution)
        )

    def __call__(self, frames: list[Image.Image]) -> torch.Tensor:
        tensors = []
        for frame in frames:
            array = np.asarray(self._resize_and_crop(frame), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append((tensor - self.mean) / self.std)
        return torch.stack(tensors)


__all__ = ["ImageNetImagePreprocessor"]
