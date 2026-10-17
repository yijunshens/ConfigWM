from __future__ import annotations

import torch


def add_patch_token_residual(
    predicted_tokens: torch.Tensor,
    vision_tokens: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Add a learned patch delta to aligned source-vision patch tokens."""
    if predicted_tokens.dim() != 3 or vision_tokens.dim() != 3:
        raise ValueError("Patch residual expects tensors with shape (B,N,D)")
    if vision_tokens.size(1) == predicted_tokens.size(1) + 1:
        base_tokens = vision_tokens[:, 1:]
    elif vision_tokens.size(1) == predicted_tokens.size(1):
        base_tokens = vision_tokens
    else:
        raise ValueError(
            "Vision and predicted token counts are not aligned: "
            f"{vision_tokens.size(1)} vs {predicted_tokens.size(1)}"
        )
    if base_tokens.shape != predicted_tokens.shape:
        raise ValueError(
            "Vision and predicted patch shapes must match, got "
            f"{tuple(base_tokens.shape)} and {tuple(predicted_tokens.shape)}"
        )
    return base_tokens.to(
        device=predicted_tokens.device,
        dtype=predicted_tokens.dtype,
    ) + float(scale) * predicted_tokens


__all__ = ["add_patch_token_residual"]
