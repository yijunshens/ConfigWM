"""ConfigurableEncoder model package with checkpoint loading utilities."""

import logging
from pathlib import Path
from typing import Tuple

from safetensors.torch import load_file
from src.data.preprocessor import Preprocessor, create_preprocessor
from src.model.configurable_encoder import ConfigurableEncoder, ModelConfig, count_params
import yaml

logger = logging.getLogger(__name__)


def load_preprocessor(ckpt_path: str) -> Preprocessor:
    """Load preprocessor from checkpoint directory."""
    data_config_path = Path(ckpt_path) / "configs" / "data_config.yaml"
    if not data_config_path.exists():
        raise FileNotFoundError(f"Data config not found at {data_config_path}")

    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)

    preprocessor = create_preprocessor(data_config["preprocessor"])
    logger.info(f"Loaded preprocessor from {data_config_path}")
    return preprocessor


def load_model_and_preprocessor(
    ckpt_path: str,
    device: str = "cpu",
    strict: bool = False,
    *,
    backbone_dtype: str | None = None,
) -> tuple[ConfigurableEncoder, Preprocessor]:
    """Load ConfigurableEncoder model and preprocessor from checkpoint directory."""
    ckpt_path = Path(ckpt_path)
    model_config_path = ckpt_path / "configs" / "model_config.yaml"
    weights_path = ckpt_path / "model.safetensors"

    if not model_config_path.exists():
        raise FileNotFoundError(
            f"Model config not found at {model_config_path}"
        )
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    preprocessor = load_preprocessor(str(ckpt_path))
    model_config = yaml.safe_load(model_config_path.read_text())
    if backbone_dtype is not None:
        backbone = model_config["predictor"]["backbone"]
        if backbone["type"] not in {"llama3", "smollm2"}:
            raise ValueError("backbone_dtype override requires a Llama backbone")
        backbone["dtype"] = backbone_dtype
    model = ConfigurableEncoder(model_config)

    state_dict = load_file(str(weights_path), device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    logger.info(f"Loaded checkpoint from {ckpt_path}")
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    return model, preprocessor


__all__ = [
    "ConfigurableEncoder",
    "ModelConfig",
    "count_params",
    "load_model_and_preprocessor",
    "load_preprocessor",
]
