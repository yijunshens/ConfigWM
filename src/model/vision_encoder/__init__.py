"""Vision encoder module for ConfigurableEncoder."""

from __future__ import annotations

import logging
from typing import Any


def create_vision_encoder(
    type: str,
    model_name: str,
    **kwargs: Any,
):
    """Build a configured visual encoder returning CLS plus patch tokens."""
    logging.info("Loading %s vision encoder: %s", type, model_name)

    if type == "vjepa2":
        from .vjepa2 import VJEPA2VisionEncoder

        return VJEPA2VisionEncoder(model_name=model_name, **kwargs)

    if type == "vjepa2_hub":
        from .vjepa2_hub import VJEPA2HubVisionEncoder

        return VJEPA2HubVisionEncoder(model_name=model_name, **kwargs)

    if type == "dinov2":
        from .dinov2 import DINOv2VisionEncoder

        return DINOv2VisionEncoder(model_name=model_name, **kwargs)

    if type == "hf_patch_tokens":
        from .hf_patch_tokens import HFPatchTokenVisionEncoder

        return HFPatchTokenVisionEncoder(model_name=model_name, **kwargs)

    if type == "lingbot_vision":
        from .lingbot_vision import LingBotVisionEncoder

        return LingBotVisionEncoder(model_name=model_name, **kwargs)

    raise ValueError(f"Unsupported encoder type: {type!r}")
