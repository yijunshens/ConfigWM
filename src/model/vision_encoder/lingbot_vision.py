"""LingBot-Vision wrapper for Configurable Encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class LingBotVisionEncoder(nn.Module):
    """Return LingBot CLS plus normalized spatial patch tokens."""

    def __init__(
        self,
        model_name: str,
        code_dir: str | None = None,
        variant: str = "large",
        trainable_all: bool = False,
        trainable_last_n_blocks: int | None = None,
        activation_checkpointing: bool = False,
        expected_patch_count: int | None = None,
        expected_output_dim: int | None = None,
        expected_patch_size: int | None = None,
    ) -> None:
        super().__init__()
        standard = Path(__file__).resolve().parents[3] / "third_party/lingbot-vision"
        if code_dir is None and standard.is_dir():
            code_dir = str(standard)
        if code_dir is not None:
            code_path = str(Path(code_dir).expanduser().resolve())
            if not (Path(code_path) / "lingbot_vision").is_dir():
                raise FileNotFoundError(f"Missing lingbot_vision package in {code_path}")
            if code_path not in sys.path:
                sys.path.insert(0, code_path)
        try:
            from lingbot_vision.loader import (
                resolve_pretrained_files, load_config, load_backbone_state,
                extract_submodule,
            )
            from lingbot_vision.build import build_backbone_from_cfg
        except ModuleNotFoundError as exc:
            if exc.name == "lingbot_vision":
                raise ImportError("Clone robbyant/lingbot-vision into third_party/lingbot-vision or set --backbone-code-dir") from exc
            raise
        from .pretrained import check_backbone

        config_path, weights_path = resolve_pretrained_files(
            repo_id_or_path=model_name,
            variant=variant,
            local_files_only=Path(model_name).expanduser().exists(),
        )
        self.model, self.output_dim = build_backbone_from_cfg(load_config(config_path))
        check_backbone(self.model, expected_output_dim, expected_patch_size)
        weights = load_backbone_state(weights_path)
        backbone_weights = extract_submodule(weights, "backbone.")
        if not backbone_weights:
            for key in ("teacher", "model_state", "state_dict", "model", "backbone"):
                if isinstance(weights.get(key), dict):
                    weights = weights[key]
                    break
            backbone_weights = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
        self.model.load_state_dict(backbone_weights, strict=True)
        self.model = self.model.to(device="cpu", dtype=torch.float32).eval()
        self.model.requires_grad_(False)
        self.output_dim = int(self.output_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.expected_patch_count = (
            None if expected_patch_count is None else int(expected_patch_count)
        )
        self.trainable_last_n_blocks = (
            None
            if trainable_last_n_blocks is None
            else int(trainable_last_n_blocks)
        )
        if trainable_all and self.trainable_last_n_blocks is not None:
            raise ValueError(
                "trainable_all and trainable_last_n_blocks are mutually exclusive"
            )
        if trainable_all:
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
        elif self.trainable_last_n_blocks is not None:
            self._configure_trainable_blocks(self.trainable_last_n_blocks)
        self._has_trainable_parameters = any(
            parameter.requires_grad for parameter in self.model.parameters()
        )

    def _configure_trainable_blocks(self, last_n_blocks: int) -> None:
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise AttributeError("LingBot backbone has no blocks attribute")
        if last_n_blocks < 0 or last_n_blocks > len(blocks):
            raise ValueError(
                f"trainable_last_n_blocks must be in [0, {len(blocks)}], "
                f"got {last_n_blocks}"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if last_n_blocks > 0:
            for block in blocks[-last_n_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for name in ("norm", "cls_norm", "local_cls_norm"):
                module = getattr(self.model, name, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and self._has_trainable_parameters
        ):
            tokens = self._encode_images_checkpointed(pixel_values)
        else:
            features = self.model.forward_features(pixel_values)
            cls = features["x_norm_clstoken"].unsqueeze(1)
            patches = features["x_norm_patchtokens"]
            tokens = torch.cat((cls, patches), dim=1)
        if (
            self.expected_patch_count is not None
            and tokens.size(1) - 1 != self.expected_patch_count
        ):
            raise ValueError(
                f"Expected {self.expected_patch_count} spatial patches, "
                f"got {tokens.size(1) - 1}"
            )
        return tokens

    def _encode_images_checkpointed(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        model = self.model
        hidden, (height, width) = model.prepare_tokens_with_masks(pixel_values)
        rope_is_static = model.rope_embed is not None and not (
            model.rope_embed.training
            and any(
                value is not None
                for value in (
                    model.rope_embed.shift_coords,
                    model.rope_embed.jitter_coords,
                    model.rope_embed.rescale_coords,
                )
            )
        )
        static_rope = (
            model.rope_embed(H=height, W=width) if rope_is_static else None
        )

        for block in model.blocks:
            if model.rope_embed is None:
                rope = None
            elif static_rope is not None:
                rope = static_rope
            else:
                rope = model.rope_embed(H=height, W=width)

            def block_forward(
                block_input: torch.Tensor,
                current_block: nn.Module = block,
                current_rope=rope,
            ) -> torch.Tensor:
                return current_block(block_input, current_rope)

            hidden = checkpoint(block_forward, hidden, use_reentrant=False)

        token_offset = model.n_storage_tokens + 1
        if model.untie_cls_and_patch_norms:
            cls_and_storage = model.cls_norm(hidden[:, :token_offset])
            patches = model.norm(hidden[:, token_offset:])
        else:
            normalized = model.norm(hidden)
            cls_and_storage = normalized[:, :token_offset]
            patches = normalized[:, token_offset:]
        return torch.cat((cls_and_storage[:, :1], patches), dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() == 4:
            return self._encode_images(pixel_values)
        if pixel_values.dim() != 5:
            raise ValueError(
                "LingBotVisionEncoder expects (B,C,H,W) or (B,T,C,H,W)"
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


__all__ = ["LingBotVisionEncoder"]
