"""Frozen observation encoders for TS-style world models."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import load_model_and_preprocessor
from src.model.vision_encoder.dinov2 import DINOv2VisionEncoder
from .config import normalize_encoder_type, normalize_model_config

_UNSET_CHECKPOINT = object()


def _pool_dino_tokens(tokens: torch.Tensor, readout: str) -> torch.Tensor:
    if tokens.dim() != 3:
        raise ValueError(f"Expected DINO token tensor (B, S, D), got {tokens.shape}")
    if readout == "cls":
        return tokens[:, 0, :]
    if readout == "all_tokens_mean":
        return tokens.mean(dim=1)
    if readout == "patch_mean":
        return tokens[:, 1:, :].mean(dim=1)
    if readout == "patch_tokens":
        return tokens[:, 1:, :]
    raise ValueError(f"Unsupported DINO readout: {readout!r}")


@dataclass
class EncoderBundle:
    encoder: nn.Module
    output_dim: int


class FrozenFrameEncoder(nn.Module):
    output_dim: int

    def freeze(self) -> None:
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(False)
        return self


class FrozenDINOEncoder(FrozenFrameEncoder):
    def __init__(
        self,
        model_name: str,
        *,
        readout: str = "patch_mean",
        input_size: int | None = None,
    ) -> None:
        super().__init__()
        self.vision_encoder = DINOv2VisionEncoder(model_name=model_name)
        self.readout = readout
        self.input_size = None if input_size is None else int(input_size)
        self.output_dim = int(self.vision_encoder.output_dim)
        self.freeze()

    def forward(
        self,
        pixel_values: torch.Tensor,
        queries: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if pixel_values.dim() != 5:
            raise ValueError(
                "FrozenDINOEncoder expects (B, T, C, H, W) history tensors"
            )
        batch_size, num_frames, channels, height, width = pixel_values.shape
        flat = pixel_values.reshape(batch_size * num_frames, channels, height, width)
        if self.input_size is not None and (height != self.input_size or width != self.input_size):
            flat = F.interpolate(
                flat,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        with torch.no_grad():
            tokens = self.vision_encoder(flat)
        pooled = _pool_dino_tokens(tokens, self.readout)
        return pooled.reshape(batch_size, num_frames, *pooled.shape[1:])


class FrozenVisionEncoder(FrozenFrameEncoder):
    def __init__(self, checkpoint_path: str, *, readout: str = "patch_mean") -> None:
        super().__init__()
        model, _ = load_model_and_preprocessor(checkpoint_path, strict=False)
        if model.vision_encoder is None:
            raise RuntimeError("Configurable Encoder checkpoint does not contain a vision encoder")
        if model.config.vision_embedding.type != "dinov2":
            raise RuntimeError(
                "FrozenVisionEncoder currently supports only DINO-backed Configurable Encoder checkpoints"
            )
        self.vision_encoder = model.vision_encoder
        self.readout = readout
        self.output_dim = int(self.vision_encoder.output_dim)
        self.freeze()

    def forward(
        self,
        pixel_values: torch.Tensor,
        queries: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if pixel_values.dim() != 5:
            raise ValueError(
                "FrozenVisionEncoder expects (B, T, C, H, W) history tensors"
            )
        batch_size, num_frames, channels, height, width = pixel_values.shape
        flat = pixel_values.reshape(batch_size * num_frames, channels, height, width)
        with torch.no_grad():
            tokens = self.vision_encoder(flat)
        pooled = _pool_dino_tokens(tokens, self.readout)
        return pooled.reshape(batch_size, num_frames, *pooled.shape[1:])


class FrozenConfigurableEncoder(FrozenFrameEncoder):
    def __init__(
        self, checkpoint_path: str, *, query: str, readout: str = "patch_mean",
        backbone_dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        model, _ = load_model_and_preprocessor(
            checkpoint_path, strict=False, backbone_dtype=backbone_dtype
        )
        if model.vision_encoder is None:
            raise RuntimeError("Configurable Encoder checkpoint does not contain a vision encoder")
        if model.config.vision_embedding.type != "dinov2":
            raise RuntimeError(
                "FrozenConfigurableEncoder currently supports only DINO-backed Configurable Encoder checkpoints"
            )
        if model.pred_projector is None:
            raise RuntimeError("Configurable Encoder checkpoint does not contain pred_projector")
        self.model = model
        self.query = str(query)
        self.readout = str(readout)
        self.output_dim = int(self.model.config.projector.dim)
        self.freeze()

    def _flat_queries(
        self,
        queries: Sequence[str] | None,
        *,
        batch_size: int,
        num_frames: int,
    ) -> list[str]:
        if queries is None:
            return [self.query] * (batch_size * num_frames)
        if len(queries) == batch_size:
            return [
                str(query)
                for query in queries
                for _ in range(num_frames)
            ]
        if len(queries) == batch_size * num_frames:
            return [str(query) for query in queries]
        raise ValueError(
            "queries must have length B or B*T for configurable_encoder encoding; "
            f"got len={len(queries)}, B={batch_size}, T={num_frames}"
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        queries: Sequence[str] | None = None,
    ) -> torch.Tensor:
        if pixel_values.dim() != 5:
            raise ValueError(
                "FrozenConfigurableEncoder expects (B, T, C, H, W) history tensors"
            )
        batch_size, num_frames, channels, height, width = pixel_values.shape
        flat = pixel_values.reshape(batch_size * num_frames, 1, channels, height, width)
        flat_queries = self._flat_queries(
            queries,
            batch_size=batch_size,
            num_frames=num_frames,
        )
        with torch.no_grad():
            vision_embedding = self.model.vision_encoder(flat)
            predictor_out = self.model.predictor(
                vision_embedding=vision_embedding,
                text_input=None,
                queries=flat_queries,
            )
            projected = self.model._project_embeddings(
                self.model.pred_projector,
                predictor_out,
            )
        if projected.dim() == 2:
            pooled = projected
        elif projected.dim() == 3:
            if self.readout == "patch_tokens":
                pooled = projected
            elif self.readout in {"patch_mean", "all_tokens_mean", "mean"}:
                pooled = projected.mean(dim=1)
            elif self.readout == "flatten":
                pooled = projected.reshape(projected.shape[0], -1)
            else:
                raise ValueError(
                    f"Unsupported configurable_encoder token readout {self.readout!r}; "
                    "use patch_tokens, patch_mean, or flatten"
                )
        else:
            raise ValueError(f"Unexpected configurable_encoder projected shape {tuple(projected.shape)}")
        return pooled.reshape(batch_size, num_frames, *pooled.shape[1:])


def build_frozen_encoder(
    *, encoder_type: str, dino_model_name: str, dino_input_size: int | None = None,
    encoder_checkpoint: str | None = _UNSET_CHECKPOINT,
    encoder_backbone_dtype: str = "bfloat16", query: str = "",
    readout: str = "patch_mean", **legacy_kwargs,
) -> EncoderBundle:
    encoder_type = normalize_encoder_type(encoder_type)
    fields = dict(legacy_kwargs)
    if encoder_checkpoint is not _UNSET_CHECKPOINT:
        fields['encoder_checkpoint'] = encoder_checkpoint
    fields = normalize_model_config(fields)
    unexpected = set(fields) - {'encoder_checkpoint'}
    if unexpected:
        raise TypeError(f'Unexpected encoder arguments: {sorted(unexpected)}')
    encoder_checkpoint = fields.get('encoder_checkpoint')
    if encoder_type == 'dino':
        encoder = FrozenDINOEncoder(dino_model_name, readout=readout, input_size=dino_input_size)
    elif encoder_type == 'vision_only':
        if not encoder_checkpoint:
            raise ValueError('encoder_checkpoint is required for encoder_type=vision_only')
        encoder = FrozenVisionEncoder(encoder_checkpoint, readout=readout)
    elif encoder_type == 'configurable_encoder':
        if not encoder_checkpoint:
            raise ValueError('encoder_checkpoint is required for encoder_type=configurable_encoder')
        encoder = FrozenConfigurableEncoder(
            encoder_checkpoint, query=query, readout=readout,
            backbone_dtype=encoder_backbone_dtype,
        )
    else:
        raise ValueError(f'Unsupported encoder_type: {encoder_type!r}')
    return EncoderBundle(encoder=encoder, output_dim=int(encoder.output_dim))


__all__ = [
    "EncoderBundle",
    "FrozenDINOEncoder",
    "FrozenConfigurableEncoder",
    "FrozenVisionEncoder",
    "build_frozen_encoder",
]
