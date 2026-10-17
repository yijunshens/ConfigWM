"""DINOv2 vision encoder wrapper for HuggingFace transformers."""

from __future__ import annotations

import logging
import os
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class _DeterministicBicubic2d(torch.autograd.Function):
    """Keep CUDA bicubic forward values and compute the small backward on CPU."""

    @staticmethod
    def forward(ctx, values: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        ctx.input_shape = tuple(values.shape)
        ctx.output_size = tuple(output_size)
        return F.interpolate(
            values,
            size=ctx.output_size,
            mode="bicubic",
            align_corners=False,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        with torch.enable_grad():
            cpu_values = torch.zeros(
                ctx.input_shape,
                dtype=torch.float32,
                device="cpu",
                requires_grad=True,
            )
            cpu_output = F.interpolate(
                cpu_values,
                size=ctx.output_size,
                mode="bicubic",
                align_corners=False,
            )
            (cpu_grad,) = torch.autograd.grad(
                cpu_output,
                cpu_values,
                grad_outputs=grad_output.detach().to(device="cpu", dtype=torch.float32),
            )
        return cpu_grad.to(device=grad_output.device, dtype=grad_output.dtype), None


def _deterministic_interpolate_pos_encoding(
    embeddings_module: nn.Module,
    embeddings: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    num_patches = embeddings.shape[1] - 1
    num_positions = embeddings_module.position_embeddings.shape[1] - 1
    if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
        return embeddings_module.position_embeddings

    class_pos_embed = embeddings_module.position_embeddings[:, :1]
    patch_pos_embed = embeddings_module.position_embeddings[:, 1:]
    dim = embeddings.shape[-1]
    new_height = height // embeddings_module.patch_size
    new_width = width // embeddings_module.patch_size
    sqrt_num_positions = int(num_positions**0.5)
    patch_pos_embed = patch_pos_embed.reshape(
        1, sqrt_num_positions, sqrt_num_positions, dim
    ).permute(0, 3, 1, 2)
    target_dtype = patch_pos_embed.dtype
    patch_pos_embed = _DeterministicBicubic2d.apply(
        patch_pos_embed.to(torch.float32),
        (new_height, new_width),
    ).to(dtype=target_dtype)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


class DINOv2VisionEncoder(nn.Module):
    """Vision encoder using HuggingFace Dinov2Model."""

    def __init__(self, model_name: str, expected_output_dim=None, expected_patch_size=None) -> None:
        super().__init__()
        from .pretrained import check_backbone, check_loading_info

        is_local = os.path.isdir(model_name)
        self.model, loading_info = AutoModel.from_pretrained(
            model_name, local_files_only=is_local, output_loading_info=True
        )
        check_loading_info(loading_info)
        check_backbone(self.model, expected_output_dim, expected_patch_size, "dinov2")
        if os.environ.get("CONFIGWM_DETERMINISTIC_DINO_POS_INTERP") == "1":
            embeddings = self.model.embeddings
            embeddings.interpolate_pos_encoding = MethodType(
                _deterministic_interpolate_pos_encoding,
                embeddings,
            )
            logging.getLogger(__name__).info(
                "Enabled deterministic DINOv2 positional interpolation backward."
            )
        self.config = self.model.config
        self.output_dim = int(self.config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image or video-style tensors into token embeddings.

        Accepts:
        - (B, C, H, W) images
        - (B, T, C, H, W) video-style tensors; frames are flattened into tokens
        """
        if pixel_values.dim() == 4:
            outputs = self.model(pixel_values=pixel_values)
            return outputs.last_hidden_state

        if pixel_values.dim() != 5:
            raise ValueError(
                "DINOv2VisionEncoder expects (B,C,H,W) or (B,T,C,H,W) input"
            )

        batch_size, num_frames, channels, height, width = pixel_values.shape
        flat = pixel_values.reshape(batch_size * num_frames, channels, height, width)
        outputs = self.model(pixel_values=flat)
        tokens = outputs.last_hidden_state
        _, seq_len, hidden = tokens.shape
        return tokens.reshape(batch_size, num_frames * seq_len, hidden)


__all__ = ["DINOv2VisionEncoder"]
