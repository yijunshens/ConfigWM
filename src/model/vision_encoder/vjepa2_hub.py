"""VJEPA2 Vision Encoder using torch.hub and the native Meta checkpoint."""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

_DEFAULT_VJEPA2_REPO = str(Path(__file__).resolve().parents[3] / "third_party/vjepa2")

# Map user-facing model names to (hub entrypoint, embed_dim, checkpoint_key)
_MODEL_REGISTRY = {
    # V-JEPA 2
    "vjepa2-vitl": ("vjepa2_vit_large", 1024, "target_encoder"),
    "vjepa2-vith": ("vjepa2_vit_huge", 1280, "target_encoder"),
    "vjepa2-vitg": ("vjepa2_vit_giant", 1408, "target_encoder"),
    "vjepa2-vitg-384": ("vjepa2_vit_giant_384", 1408, "target_encoder"),
    # V-JEPA 2.1
    "vjepa2.1-vitb-384": ("vjepa2_1_vit_base_384", 768, "ema_encoder"),
    "vjepa2.1-vitl-384": ("vjepa2_1_vit_large_384", 1024, "ema_encoder"),
    "vjepa2.1-vitg-384": ("vjepa2_1_vit_giant_384", 1408, "target_encoder"),
    "vjepa2.1-vitG-384": ("vjepa2_1_vit_gigantic_384", 1664, "target_encoder"),
}


def _clean_state_dict(state_dict: dict) -> dict:
    """Strip ``module.`` and ``backbone.`` prefixes that Meta checkpoints use."""
    cleaned = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = val
    return cleaned


class VJEPA2HubVisionEncoder(nn.Module):
    """Vision encoder that loads V-JEPA 2 via ``torch.hub.load`` (local source).

    This mirrors ``VJEPA2VisionEncoder`` but bypasses HuggingFace, building the
    ViT directly from the vjepa2 submodule's hubconf and loading native weights.

    Args:
        model_name: One of the keys in ``_MODEL_REGISTRY`` (e.g. "vjepa2-vitl").
        checkpoint_path: Path to the original ``.pth`` checkpoint.  If the path
            is a directory it looks for ``original/model.pth`` inside it (the
            layout used by the HF cache). When omitted, official Meta weights
            are downloaded once into the Torch cache and loaded strictly.
        img_size: Override spatial resolution (default from registry).
        num_frames: Number of temporal frames (default 64).
    """

    def __init__(
        self,
        model_name: str = "vjepa2-vitl",
        checkpoint_path: str | None = None,
        code_dir: str | None = None,
        add_mean_cls_token: bool = True,
        activation_checkpointing: bool = False,
        expected_patch_count: int | None = None,
        expected_output_dim: int | None = None,
        expected_patch_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_name {model_name!r}. "
                f"Choose from {list(_MODEL_REGISTRY.keys())}"
            )

        hub_entry, embed_dim, ckpt_key = _MODEL_REGISTRY[model_name]
        self.output_dim = embed_dim
        self.add_mean_cls_token = bool(add_mean_cls_token)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.expected_patch_count = (
            None if expected_patch_count is None else int(expected_patch_count)
        )
        repo_dir = os.path.abspath(
            os.path.expanduser(code_dir or _DEFAULT_VJEPA2_REPO)
        )
        if not os.path.isfile(os.path.join(repo_dir, "hubconf.py")):
            raise FileNotFoundError(
                "Clone facebookresearch/vjepa2 into third_party/vjepa2 or set "
                f"--backbone-code-dir (missing hubconf.py in {repo_dir!r})"
            )

        # Both projects use a top-level src package. Extend its search path so
        # upstream hub imports resolve without replacing ConfigWM's src module.
        import src
        vendor_src = str(Path(repo_dir) / "src")
        for name in ("src.hub", "src.models", "src.masks", "src.utils"):
            module = sys.modules.get(name)
            if module is not None:
                origins = list(getattr(module, "__path__", []))
                if not origins or not all(Path(origin).resolve().is_relative_to(Path(vendor_src).resolve()) for origin in origins):
                    raise ImportError(f"{name} was already loaded from a different project")
        if vendor_src not in src.__path__:
            src.__path__.append(vendor_src)

        # The hub entrypoints (e.g. vjepa2_vit_large) already set img_size
        # and num_frames with correct defaults.  Strip config-only keys that
        # should not be forwarded to the hub entrypoint.
        hub_kwargs = {k: v for k, v in kwargs.items() if k not in ("disabled",)}

        from .pretrained import check_backbone, resolve_vjepa_checkpoint

        # Never use the upstream pretrained=True URL; some releases point it
        # at a local testing server. Both paths use the same strict loader.
        checkpoint_path = resolve_vjepa_checkpoint(model_name, checkpoint_path)
        logger.info("Loading V-JEPA encoder (%s) from %s", model_name, checkpoint_path)
        encoder, _predictor = torch.hub.load(
            repo_dir, hub_entry, source="local", pretrained=False, **hub_kwargs
        )
        check_backbone(encoder, expected_output_dim, expected_patch_size)
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", mmap=True, weights_only=True
        )
        if not isinstance(state_dict, dict) or not isinstance(state_dict.get(ckpt_key), dict):
            raise ValueError(f"V-JEPA checkpoint must contain {ckpt_key!r} weights")
        encoder_sd = _clean_state_dict(state_dict[ckpt_key])
        encoder.load_state_dict(encoder_sd, strict=True)

        self.model = encoder
        # V-JEPA 2.1 models have a dedicated image patch embedding (tubelet_size=1)
        # so single-frame input works natively.  V-JEPA 2 models only have a 3D
        # patch embedding (tubelet_size=2) and need frame duplication for T=1.
        self._needs_frame_repeat = (
            not hasattr(encoder, "patch_embed_img")
            or encoder.patch_embed_img is None
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode video frames into vision embeddings.

        Args:
            pixel_values: Video tensor of shape (B, T, C, H, W) — same
                convention as the HF-based encoder.

        Returns:
            Vision embeddings of shape (B, N, embed_dim).
        """
        # The native ViT expects (B, C, T, H, W).  Input arrives as (B, T, C, H, W).
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if pixel_values.ndim != 5:
            raise ValueError("VJEPA2HubVisionEncoder expects (B,C,H,W) or (B,T,C,H,W)")
        x = pixel_values.permute(0, 2, 1, 3, 4)  # → (B, C, T, H, W)

        # V-JEPA 2: 3D patch embedding (tubelet_size=2), need T ≥ 2.
        # V-JEPA 2.1: has dedicated image patch embed, T=1 works natively.
        if x.shape[2] == 1 and self._needs_frame_repeat:
            x = x.repeat(1, 1, 2, 1, 1)

        if (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            tokens = checkpoint(self.model, x, use_reentrant=False)
        else:
            tokens = self.model(x)

        if (
            self.expected_patch_count is not None
            and tokens.size(1) != self.expected_patch_count
        ):
            raise ValueError(
                f"Expected {self.expected_patch_count} spatial patches, "
                f"got {tokens.size(1)}"
            )
        if self.add_mean_cls_token:
            mean_cls = tokens.mean(dim=1, keepdim=True)
            tokens = torch.cat((mean_cls, tokens), dim=1)
        return tokens
