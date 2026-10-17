"""Generic Hugging Face ViT wrapper returning CLS plus spatial patch tokens."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel


class HFPatchTokenVisionEncoder(nn.Module):
    """Load a Hugging Face vision backbone and remove non-spatial registers."""

    def __init__(
        self,
        model_name: str,
        vision_submodule: str | None = None,
        add_mean_cls_token: bool = False,
        interpolate_pos_encoding: bool = False,
        activation_checkpointing: bool = False,
        load_dtype: str | None = None,
        expected_patch_count: int | None = None,
        discard_pooling_head: bool = False,
        force_uniform_dtype: bool = False,
        expected_output_dim: int | None = None,
        expected_patch_size: int | None = None,
        expected_model_type: str | None = None,
    ) -> None:
        super().__init__()
        is_local = os.path.isdir(model_name)
        load_kwargs = {"local_files_only": is_local}
        if load_dtype is not None:
            dtype = getattr(torch, str(load_dtype), None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"Unsupported load_dtype: {load_dtype!r}")
            load_kwargs["dtype"] = dtype
        from .pretrained import check_backbone, check_loading_info

        loaded_model, loading_info = AutoModel.from_pretrained(
            model_name,
            output_loading_info=True,
            **load_kwargs,
        )
        check_loading_info(loading_info)
        check_backbone(loaded_model, expected_output_dim, expected_patch_size, expected_model_type)
        if force_uniform_dtype:
            if load_dtype is None:
                raise ValueError(
                    "force_uniform_dtype requires load_dtype to be set"
                )
            loaded_model = loaded_model.to(dtype=dtype)
        if vision_submodule is not None:
            try:
                loaded_model = getattr(loaded_model, vision_submodule)
            except AttributeError as exc:
                raise ValueError(
                    f"Model {model_name!r} has no vision submodule "
                    f"{vision_submodule!r}"
                ) from exc

        self.model = loaded_model
        if discard_pooling_head and hasattr(self.model, "head"):
            for parameter in self.model.head.parameters():
                parameter.requires_grad_(False)
            if hasattr(self.model, "use_head"):
                self.model.use_head = False
        self.config = self.model.config
        self.output_dim = int(self.config.hidden_size)
        self.add_mean_cls_token = bool(add_mean_cls_token)
        self.interpolate_pos_encoding = bool(interpolate_pos_encoding)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.expected_patch_count = (
            None if expected_patch_count is None else int(expected_patch_count)
        )
        self.num_register_tokens = int(
            getattr(self.config, "num_register_tokens", 0) or 0
        )

        # Mask tokens are pretraining-only parameters. Standard image forwards
        # never touch them, so leaving them trainable breaks DDP reduction on
        # the second step when find_unused_parameters is disabled.
        for name, parameter in self.model.named_parameters():
            if name == "mask_token" or name.endswith(".mask_token"):
                parameter.requires_grad_(False)

        self._uses_native_checkpointing = False
        if self.activation_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()
            self._uses_native_checkpointing = True

    def _forward_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        if self.interpolate_pos_encoding:
            kwargs["interpolate_pos_encoding"] = True
        output = self.model(pixel_values=pixel_values, **kwargs)
        return output.last_hidden_state

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        use_wrapper_checkpoint = (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and not self._uses_native_checkpointing
        )
        if use_wrapper_checkpoint:
            tokens = checkpoint(
                self._forward_tokens,
                pixel_values,
                use_reentrant=False,
            )
        else:
            tokens = self._forward_tokens(pixel_values)

        if self.num_register_tokens > 0:
            cls = tokens[:, :1]
            patches = tokens[:, 1 + self.num_register_tokens :]
            tokens = torch.cat((cls, patches), dim=1)

        if self.add_mean_cls_token:
            self._validate_patch_count(tokens.size(1))
            mean_cls = tokens.mean(dim=1, keepdim=True)
            tokens = torch.cat((mean_cls, tokens), dim=1)
        else:
            self._validate_patch_count(tokens.size(1) - 1)
        return tokens

    def _validate_patch_count(self, patch_count: int) -> None:
        if (
            self.expected_patch_count is not None
            and patch_count != self.expected_patch_count
        ):
            raise ValueError(
                f"Expected {self.expected_patch_count} spatial patches, "
                f"got {patch_count}"
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() == 4:
            return self._encode_images(pixel_values)
        if pixel_values.dim() != 5:
            raise ValueError(
                "HFPatchTokenVisionEncoder expects (B,C,H,W) or (B,T,C,H,W)"
            )
        batch_size, num_frames, channels, height, width = pixel_values.shape
        flat = pixel_values.reshape(batch_size * num_frames, channels, height, width)
        tokens = self._encode_images(flat)
        _, sequence_length, hidden_size = tokens.shape
        return tokens.reshape(
            batch_size,
            num_frames * sequence_length,
            hidden_size,
        )


__all__ = ["HFPatchTokenVisionEncoder"]
