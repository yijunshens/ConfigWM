"""Configuration helpers for TS-style world-model experiments."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_dir: str
    feature_cache_dir: str | None = None
    query_metadata_path: str | None = None
    batch_size: int = 32
    num_workers: int = 4
    resolution: int = 224
    num_hist: int = 3
    frameskip: int = 5
    target_offset: int = 5
    action_dim: int = 2
    max_samples: int | None = None
    sample_mode: str = "prefix"
    normalize_actions: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = None


@dataclass
class ModelConfig:
    encoder_type: str
    dino_model_name: str
    dino_input_size: int | None = None
    encoder_checkpoint: str | None = None
    encoder_backbone_dtype: str = "bfloat16"
    query: str = "focus on the red circle"
    readout: str = "patch_mean"
    use_observation_projector: bool = True
    projector_type: str = "mlp"
    adapter_dim: int = 256
    state_dim: int = 128
    projector_spatial_hw: int | None = None
    projector_channel_dim: int = 8
    projector_agg_hidden_dim: int = 2048
    detach_target: bool = True
    projector_pool_hw: int = 1
    curvature_weight: float = 0.0
    curvature_mode: str = "cos"
    action_emb_dim: int = 10
    predictor_type: str = "vit"
    predictor_hidden_dim: int | None = None
    predictor_dim_head: int = 64
    predictor_emb_dropout: float = 0.0
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    pred_projector_type: str = "identity"
    sigreg_weight: float = 0.0
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.encoder_backbone_dtype not in {"float32", "bfloat16"}:
            raise ValueError("encoder_backbone_dtype must be float32 or bfloat16")


@dataclass
class TrainConfig:
    epochs: int = 10
    projector_lr: float = 1.0e-6
    predictor_lr: float = 5.0e-4
    weight_decay: float = 0.0
    precision: str = "bf16"
    log_every: int = 10
    output_dir: str = "outputs/configwm"
    smoke_steps: int = 3
    grad_clip_norm: float = 1.0
    save_final_state: bool = True
    save_epoch_checkpoints: bool = False
    checkpoint_epochs: list[int] | None = None
    checkpoint_every_steps: int = 0


@dataclass
class ExperimentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**normalize_model_config(raw["model"])),
        train=TrainConfig(**raw["train"]),
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "data": vars(config.data),
        "model": vars(config.model),
        "train": vars(config.train),
    }


__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainConfig",
    "config_to_dict",
    "load_experiment_config",
]


def normalize_encoder_type(value: str) -> str:
    """Validate the encoder names used by configs and command lines."""
    if value not in {"dino", "vision_only", "configurable_encoder"}:
        raise ValueError(f"Unsupported encoder_type: {value!r}")
    return value


def normalize_model_config(raw: dict) -> dict:
    """Validate a model config without mutating the caller's dictionary."""
    model = dict(raw)
    if "encoder_type" in model:
        model["encoder_type"] = normalize_encoder_type(model["encoder_type"])
    if model.get("encoder_type") == "configurable_encoder" and "encoder_backbone_dtype" not in model:
        logging.getLogger(__name__).warning(
            "Missing encoder_backbone_dtype; using bfloat16 for historical ConfigWM runs. "
            "For a run trained with FP32 Llama parameters, set encoder_backbone_dtype=float32 "
            "or pass --encoder-backbone-dtype float32."
        )
    return model
