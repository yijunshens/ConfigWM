"""ConfigurableEncoder model: Query-conditioned visual representation model."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
import logging
import os
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
from src.model.loss import get_loss_function
from src.model.predictor import create_predictor
from src.model.residual import add_patch_token_residual
from src.model.text_encoder import create_text_encoder
from src.model.vision_encoder import create_vision_encoder
import torch
from torch import nn
import torch.nn.functional as F
import yaml


class ModelConfig:
    """Config class with attribute access and to_dict() for HF Trainer compatibility."""

    def __init__(self, config_dict: dict[str, Any]):
        self._config_dict = config_dict
        for k, v in config_dict.items():
            if isinstance(v, dict):
                setattr(self, k, ModelConfig(v))
            elif isinstance(v, list):
                setattr(
                    self,
                    k,
                    [
                        ModelConfig(item) if isinstance(item, dict) else item
                        for item in v
                    ],
                )
            else:
                setattr(self, k, v)
        # Add use_cache for HF Trainer compatibility
        if not hasattr(self, "use_cache"):
            self.use_cache = False

    def to_dict(self) -> dict[str, Any]:
        """Return the original config dict (for HF Trainer/WandB)."""
        return self._config_dict

    def __repr__(self) -> str:
        return f"ModelConfig({self._config_dict})"


def count_params(module: nn.Module | None) -> int:
    """Return total number of parameters in a module (0 if None)."""
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def freeze_module(module: nn.Module | None) -> int:
    """Freeze a module and return the number of parameters touched."""
    if module is None:
        return 0
    count = 0
    for param in module.parameters():
        param.requires_grad = False
        count += param.numel()
    module.eval()
    return count


def expand_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Expand ~ in model_name and checkpoint_path values throughout config."""
    config = dict(config)  # shallow copy
    for path_key in ("model_name", "checkpoint_path"):
        if path_key in config:
            config[path_key] = os.path.expanduser(config[path_key])
    # Recurse into nested dicts (e.g., predictor.backbone)
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = expand_paths(value)
    return config


class ConfigurableEncoder(nn.Module):
    def __init__(self, config: str | os.PathLike | Mapping[str, Any]) -> None:
        super().__init__()

        logging.info("Initializing ConfigurableEncoder model with config: %s", config)

        if isinstance(config, (str, os.PathLike)):
            config = yaml.safe_load(Path(config).read_text())

        # Expand ~ in all model_name paths
        config = expand_paths(config)
        self.freeze_vision_encoder = bool(
            config.get("freeze_vision_encoder", False)
        )
        self.freeze_predictor_token_embeddings = bool(
            config.get("freeze_predictor_token_embeddings", False)
        )
        self.use_target_vision_projector = bool(
            config.get("use_target_vision_projector", True)
        )
        self.separate_target_vision_encoder = bool(
            config.get("separate_target_vision_encoder", False)
        )
        self.freeze_target_vision_encoder = bool(
            config.get("freeze_target_vision_encoder", False)
        )
        self.prediction_residual_to_vision_tokens = bool(
            config.get("prediction_residual_to_vision_tokens", False)
        )
        self.prediction_residual_scale = float(
            config.get("prediction_residual_scale", 1.0)
        )
        self.target_vision_pooling = str(
            config.get("target_vision_pooling", "cls")
        )
        valid_target_vision_pooling = {
            "cls",
            "patch_mean",
            "patch_tokens",
            "all_tokens_mean",
            "mask_weighted_cls_patch",
            "mask_normed_cls_object",
            "mask_innernorm_cls_object",
        }
        if self.target_vision_pooling not in valid_target_vision_pooling:
            raise ValueError(
                "target_vision_pooling must be one of "
                f"{sorted(valid_target_vision_pooling)}, got {self.target_vision_pooling!r}."
            )
        self.target_mask_pooling_config = dict(
            config.get("target_mask_pooling", {}) or {}
        )
        self.target_mask_cls_weight = float(
            self.target_mask_pooling_config.get("cls_weight", 1.0)
        )
        self.target_mask_object_weight = float(
            self.target_mask_pooling_config.get("object_weight", 1.0)
        )
        self.target_mask_background_weight = float(
            self.target_mask_pooling_config.get("background_weight", 0.2)
        )
        self.target_mask_patch_threshold = float(
            self.target_mask_pooling_config.get("patch_threshold", 0.02)
        )
        self.target_mask_dilate = int(
            self.target_mask_pooling_config.get("dilate", 1)
        )
        self.target_mask_normalize_weights = bool(
            self.target_mask_pooling_config.get("normalize_weights", True)
        )
        self.query_conditioned_target_consistency = bool(
            config.get("query_conditioned_target_consistency", False)
        )
        self.qtarget_consistency_detach = bool(
            config.get("qtarget_consistency_detach", True)
        )
        self.qtarget_consistency_weight = float(
            config.get("qtarget_consistency_weight", 0.0)
        )
        self.qtarget_consistency_scale = float(
            config.get("qtarget_consistency_scale", 384.0)
        )
        self.qtarget_anchor_weight = float(
            config.get("qtarget_anchor_weight", 0.0)
        )
        self.qtarget_anchor_scale = float(
            config.get("qtarget_anchor_scale", self.qtarget_consistency_scale)
        )
        self.qtarget_contrastive_weight = float(
            config.get("qtarget_contrastive_weight", 0.0)
        )
        self.qtarget_contrastive_temperature = float(
            config.get("qtarget_contrastive_temperature", 0.07)
        )
        self.qtarget_contrastive_detach_target = bool(
            config.get("qtarget_contrastive_detach_target", True)
        )
        self.same_source_contrastive_weight = float(
            config.get("same_source_contrastive_weight", 0.0)
        )
        self.same_source_contrastive_temperature = float(
            config.get("same_source_contrastive_temperature", 0.07)
        )
        self.same_source_contrastive_detach_target = bool(
            config.get("same_source_contrastive_detach_target", True)
        )
        self.same_source_clean_contrastive_weight = float(
            config.get("same_source_clean_contrastive_weight", 0.0)
        )
        self.same_source_clean_contrastive_temperature = float(
            config.get(
                "same_source_clean_contrastive_temperature",
                self.same_source_contrastive_temperature,
            )
        )
        self.same_source_clean_contrastive_detach_target = bool(
            config.get("same_source_clean_contrastive_detach_target", True)
        )
        self.same_source_clean_pred_weight = float(
            config.get("same_source_clean_pred_weight", 0.0)
        )
        self.same_source_pred_clean_weight = float(
            config.get("same_source_pred_clean_weight", 0.0)
        )
        self.same_source_pred_clean_scale = float(
            config.get("same_source_pred_clean_scale", 1.0)
        )
        self.same_source_pred_clean_loss = str(
            config.get("same_source_pred_clean_loss", "mse")
        ).strip().lower()
        self.same_source_pred_clean_detach_target = bool(
            config.get("same_source_pred_clean_detach_target", True)
        )
        self.aux_object_grounding_config = dict(
            config.get("aux_object_grounding", {}) or {}
        )
        self.aux_object_grounding_enabled = bool(
            self.aux_object_grounding_config.get("enabled", False)
        )
        self.aux_object_grounding_detach_target = bool(
            self.aux_object_grounding_config.get("detach_target", True)
        )
        self.aux_object_grounding_normalize_pred = bool(
            self.aux_object_grounding_config.get("normalize_pred", True)
        )
        self.aux_object_grounding_normalize_target = bool(
            self.aux_object_grounding_config.get("normalize_target", True)
        )
        self.aux_object_grounding_weight = float(
            self.aux_object_grounding_config.get("weight", 0.0)
        )
        self.aux_object_grounding_active = (
            self.aux_object_grounding_enabled
            and self.aux_object_grounding_weight > 0.0
        )
        if (
            self.aux_object_grounding_active
            and self.target_vision_pooling != "cls"
        ):
            raise ValueError(
                "aux_object_grounding route keeps the main planning target as "
                "pure CLS; set target_vision_pooling: cls."
            )
        default_aux_scale = config.get("loss", {}).get(
            "pred_loss_scale",
            config.get("projector", {}).get("dim", 1.0),
        )
        aux_loss_scale = self.aux_object_grounding_config.get(
            "loss_scale", default_aux_scale
        )
        if isinstance(aux_loss_scale, str):
            if aux_loss_scale != "same_as_pred_loss_scale":
                raise ValueError(
                    "aux_object_grounding.loss_scale must be numeric or "
                    "'same_as_pred_loss_scale'."
                )
            aux_loss_scale = default_aux_scale
        self.aux_object_grounding_scale = float(aux_loss_scale)

        # Check if vision encoder is disabled (text-only mode)
        vision_config = dict(config["vision_embedding"])
        vision_disabled = vision_config.pop("disabled", False)

        if vision_disabled:
            logging.info("Vision encoder disabled - running in text-only mode")
            self.vision_encoder = None
            self.target_vision_encoder = None
            vision_dim = 0
        else:
            self.vision_encoder = create_vision_encoder(**vision_config)
            self.target_vision_encoder = (
                create_vision_encoder(**vision_config)
                if self.separate_target_vision_encoder
                else None
            )
            vision_dim = self.vision_encoder.output_dim

        self.predictor = create_predictor(
            **config["predictor"],
            vision_dim=vision_dim,
        )

        text_config = dict(config["text_encoder"])
        text_disabled = text_config.pop("disabled", False)
        if text_disabled:
            self.text_encoder = None
        else:
            self.text_encoder = create_text_encoder(**text_config)

        # Projector: maps encoder outputs to a shared embedding space.
        from src.model.projector import build_projector

        proj_config = config["projector"]
        if (
            vision_dim > 0
            and not self.use_target_vision_projector
            and int(proj_config["dim"]) != int(vision_dim)
        ):
            raise ValueError(
                "Direct target vision mode requires projector.dim to match "
                f"vision_dim ({vision_dim}), got {proj_config['dim']}."
            )
        self.pred_projector = build_projector(
            self.predictor.hidden_size, proj_config
        )
        self.tgt_vision_projector = (
            build_projector(vision_dim, proj_config)
            if vision_dim > 0 and self.use_target_vision_projector
            else None
        )
        self.tgt_text_projector = (
            build_projector(self.text_encoder.output_dim, proj_config)
            if self.text_encoder is not None
            else None
        )
        self.aux_object_head = (
            self._build_aux_object_head(
                int(proj_config["dim"]),
                self.aux_object_grounding_config.get("head", {}) or {},
            )
            if self.aux_object_grounding_active
            else None
        )

        # Optional legacy prediction loss; Configurable Encoder InfoNCE is computed by the trainer.
        loss_config = config.get("loss")
        self.loss_fn = (
            get_loss_function(**loss_config) if loss_config is not None else None
        )

        if self.freeze_vision_encoder:
            n_frozen = freeze_module(self.vision_encoder)
            if self.separate_target_vision_encoder:
                n_frozen += freeze_module(self.target_vision_encoder)
            logging.info(
                "Frozen vision_encoder parameters: %d", n_frozen
            )
        elif self.freeze_target_vision_encoder and self.separate_target_vision_encoder:
            n_frozen = freeze_module(self.target_vision_encoder)
            logging.info("Frozen target_vision_encoder parameters: %d", n_frozen)
        if self.freeze_predictor_token_embeddings:
            self._freeze_predictor_token_embeddings()

        # Some large-backbone checkpoints use one parameter dtype throughout.
        # Keep this opt-in: the existing small-backbone defaults are unchanged.
        parameter_dtype = config.get("parameter_dtype", config.get("force_parameter_dtype"))
        if parameter_dtype is not None:
            dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
            if parameter_dtype not in dtypes:
                raise ValueError(f"Unsupported parameter_dtype: {parameter_dtype!r}")
            for parameter in self.parameters():
                if parameter.is_floating_point():
                    parameter.data = parameter.data.to(dtype=dtypes[parameter_dtype])

        # Retrieval similarities use normalized embeddings.
        self.normalize_for_eval = True

        self.add_query_to_target = config.get("add_query_to_target", False)
        self.use_sequence_packing = config.get("use_sequence_packing", False)
        logging.info("Target vision pooling: %s", self.target_vision_pooling)
        if self.target_vision_pooling == "mask_weighted_cls_patch":
            logging.info(
                "Target mask pooling config: cls=%.3f object=%.3f bg=%.3f "
                "threshold=%.4f dilate=%d normalize=%s",
                self.target_mask_cls_weight,
                self.target_mask_object_weight,
                self.target_mask_background_weight,
                self.target_mask_patch_threshold,
                self.target_mask_dilate,
                self.target_mask_normalize_weights,
            )
        if self.query_conditioned_target_consistency:
            logging.info(
                "Query-conditioned target consistency enabled: weight=%.4f "
                "scale=%.4f detach=%s",
                self.qtarget_consistency_weight,
                self.qtarget_consistency_scale,
                self.qtarget_consistency_detach,
            )
        if self.qtarget_anchor_weight > 0.0:
            logging.info(
                "Query-conditioned target anchor enabled: weight=%.4f scale=%.4f",
                self.qtarget_anchor_weight,
                self.qtarget_anchor_scale,
            )
        if self.qtarget_contrastive_weight > 0.0:
            logging.info(
                "Query-conditioned source-clean contrastive enabled: "
                "weight=%.4f temperature=%.4f detach_target=%s",
                self.qtarget_contrastive_weight,
                self.qtarget_contrastive_temperature,
                self.qtarget_contrastive_detach_target,
            )
        if self.same_source_clean_contrastive_weight > 0.0:
            logging.info(
                "Same-source clean-anchor contrastive enabled: "
                "weight=%.4f temperature=%.4f detach_target=%s",
                self.same_source_clean_contrastive_weight,
                self.same_source_clean_contrastive_temperature,
                self.same_source_clean_contrastive_detach_target,
            )
        if self.same_source_clean_pred_weight > 0.0:
            logging.info(
                "Same-source clean-anchor pred loss enabled: weight=%.4f",
                self.same_source_clean_pred_weight,
            )
        if self.same_source_pred_clean_weight > 0.0:
            logging.info(
                "Same-source source->clean predictor alignment enabled: "
                "weight=%.4f scale=%.4f loss=%s detach_target=%s",
                self.same_source_pred_clean_weight,
                self.same_source_pred_clean_scale,
                self.same_source_pred_clean_loss,
                self.same_source_pred_clean_detach_target,
            )
        if self.aux_object_grounding_enabled:
            head_name = (
                self.aux_object_head.__class__.__name__
                if self.aux_object_head is not None
                else "disabled(weight<=0)"
            )
            logging.info(
                "Aux object grounding enabled: weight=%.4f scale=%.4f "
                "detach=%s norm_pred=%s norm_tgt=%s head=%s",
                self.aux_object_grounding_weight,
                self.aux_object_grounding_scale,
                self.aux_object_grounding_detach_target,
                self.aux_object_grounding_normalize_pred,
                self.aux_object_grounding_normalize_target,
                head_name,
            )
        logging.info(
            "ConfigurableEncoder constructed. Params (M): vision=%.2f, target_vision=%.2f, predictor=%.2f, text=%.2f, total=%.2f",
            count_params(self.vision_encoder) / 1e6,
            (
                count_params(self.target_vision_encoder) / 1e6
                if self.separate_target_vision_encoder
                else 0.0
            ),
            count_params(self.predictor) / 1e6,
            count_params(self.text_encoder) / 1e6,
            count_params(self) / 1e6,
        )

        # Store config as ModelConfig for HF Trainer compatibility (expects model.config.use_cache, to_dict())
        self.config = ModelConfig(config)

    def train(self, mode: bool = True):
        """Keep explicitly frozen encoders in eval mode across Trainer.train()."""
        super().train(mode)
        if self.freeze_vision_encoder and self.vision_encoder is not None:
            self.vision_encoder.eval()
            if self.separate_target_vision_encoder:
                self.target_vision_encoder.eval()
        elif (
            self.freeze_target_vision_encoder
            and self.separate_target_vision_encoder
            and self.target_vision_encoder is not None
        ):
            self.target_vision_encoder.eval()
        return self

    @staticmethod
    def _build_aux_object_head(
        dim: int,
        head_config: Mapping[str, Any],
    ) -> nn.Module:
        head_type = str(head_config.get("type", "mlp")).lower()
        if head_type == "linear":
            return nn.Linear(dim, dim)
        if head_type != "mlp":
            raise ValueError(
                "aux_object_grounding.head.type must be 'linear' or 'mlp', "
                f"got {head_type!r}."
            )

        layers = int(head_config.get("layers", 2))
        hidden_dim = int(head_config.get("hidden_dim", dim))
        activation = str(head_config.get("activation", "gelu")).lower()
        norm = str(head_config.get("norm", "layernorm")).lower()
        if layers < 2:
            return nn.Linear(dim, dim)
        if activation != "gelu":
            raise ValueError(
                "aux_object_grounding.head.activation currently supports "
                f"only 'gelu', got {activation!r}."
            )

        modules: list[nn.Module] = [nn.Linear(dim, hidden_dim), nn.GELU()]
        if norm in ("layernorm", "layer_norm", "ln"):
            modules.append(nn.LayerNorm(hidden_dim))
        elif norm not in ("none", ""):
            raise ValueError(
                "aux_object_grounding.head.norm must be 'layernorm' or "
                f"'none', got {norm!r}."
            )
        for _ in range(max(0, layers - 2)):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
            if norm in ("layernorm", "layer_norm", "ln"):
                modules.append(nn.LayerNorm(hidden_dim))
        modules.append(nn.Linear(hidden_dim, dim))
        return nn.Sequential(*modules)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the predictor backbone (called by HF Trainer)."""
        backbone = self.predictor.backbone
        if hasattr(backbone, "gradient_checkpointing"):
            backbone.gradient_checkpointing = True
            logging.info(
                "Gradient checkpointing enabled on predictor backbone."
            )
        else:
            logging.warning(
                "backbone does not support gradient_checkpointing flag."
            )

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the predictor backbone."""
        backbone = self.predictor.backbone
        if hasattr(backbone, "gradient_checkpointing"):
            backbone.gradient_checkpointing = False

    # ---- Private helpers ------------------------------------------------------
    def _freeze_predictor_token_embeddings(self) -> None:
        backbone = getattr(self.predictor, "backbone", None)
        embed_tokens = getattr(backbone, "embed_tokens", None)
        if embed_tokens is None:
            logging.warning(
                "freeze_predictor_token_embeddings=true but predictor backbone has no embed_tokens."
            )
            return
        n_frozen = freeze_module(embed_tokens)
        logging.info(
            "Frozen predictor token embedding parameters: %d", n_frozen
        )

    def validate_inputs(
        self,
        vision_input: torch.Tensor | None,
        text_input: list[str | None] | None,
    ) -> list[bool]:
        """Validate vision/text inputs and determine which samples have vision.

        Returns:
            has_vision: List of booleans indicating which samples have valid vision input
        """
        # Determine batch size from whichever input is available
        if vision_input is not None:
            B = vision_input.size(0)
        elif text_input is not None:
            B = len(text_input)
        else:
            raise ValueError(
                "At least one of vision_input or text_input must be provided."
            )

        if self.vision_encoder is None:
            # Text-only mode: all samples must have text
            if text_input is None or not all(t is not None for t in text_input):
                raise ValueError(
                    "Vision encoder is disabled. All samples must have text_input."
                )
            return [False] * B

        # Validate inputs: vision and text are complementary
        if vision_input is not None:
            # zero values indicate no valid vision data
            has_vision = vision_input.view(B, -1).any(dim=1).tolist()
        else:
            has_vision = [False] * B
        has_text = [(t is not None) for t in (text_input or [None] * B)]

        for i in range(B):
            if has_vision[i] and has_text[i]:
                raise ValueError(
                    f"Sample {i}: both vision and text_input are valid. Only one should be set."
                )
            if not has_vision[i] and not has_text[i]:
                raise ValueError(
                    f"Sample {i}: neither vision nor text_input is valid. One must be set."
                )

        return has_vision

    # ---- Public API -----------------------------------------------------------
    def load_checkpoint(self, ckpt_path: str) -> ConfigurableEncoder:
        """Load model checkpoint from a given path."""
        state_dict = load_file(
            os.path.join(ckpt_path, "model.safetensors"), device="cpu"
        )
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logging.info(
            "Loaded model from %s. Missing keys: %s, Unexpected keys: %s",
            ckpt_path,
            missing,
            unexpected,
        )
        return self

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode texts through text_encoder + text target projector."""
        if self.text_encoder is None or self.tgt_text_projector is None:
            raise RuntimeError("Text encoder is disabled for this model.")
        embeds = self.text_encoder(texts=texts)
        return self._project_embeddings(self.tgt_text_projector, embeds)

    @staticmethod
    def _project_embeddings(
        projector: nn.Module,
        embeds: torch.Tensor,
    ) -> torch.Tensor:
        param = next(projector.parameters())
        embeds = embeds.to(device=param.device, dtype=param.dtype)
        return projector(embeds)

    def _pool_target_vision_embedding(
        self,
        vision_tokens: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if vision_tokens.dim() != 3:
            raise ValueError(
                "Expected target vision tokens with shape (B, N, D), got "
                f"{tuple(vision_tokens.shape)}."
            )
        if self.target_vision_pooling == "cls":
            return vision_tokens[:, 0]
        if self.target_vision_pooling == "patch_tokens":
            if vision_tokens.size(1) <= 1:
                raise ValueError("patch_tokens pooling requires patch tokens.")
            return vision_tokens[:, 1:]
        if self.target_vision_pooling == "patch_mean":
            if vision_tokens.size(1) <= 1:
                raise ValueError("patch_mean pooling requires patch tokens.")
            return vision_tokens[:, 1:].mean(dim=1)
        if self.target_vision_pooling == "mask_weighted_cls_patch":
            return self._pool_mask_weighted_cls_patch(vision_tokens, target_mask)
        if self.target_vision_pooling == "mask_normed_cls_object":
            return self._pool_mask_normed_cls_object(vision_tokens, target_mask)
        if self.target_vision_pooling == "mask_innernorm_cls_object":
            return self._pool_mask_innernorm_cls_object(vision_tokens, target_mask)
        return vision_tokens.mean(dim=1)

    def _prepare_aux_object_target(
        self,
        vision_tokens: torch.Tensor,
        target_mask: torch.Tensor | None,
        *,
        pred_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obj_mean, object_coverage = self._object_patch_mean_and_coverage_from_mask(
            vision_tokens,
            target_mask,
            pooling_name="aux_object_grounding",
        )
        if self.use_target_vision_projector:
            if self.tgt_vision_projector is None:
                raise RuntimeError(
                    "Target vision projector is enabled but was not constructed."
                )
            obj_target = self._project_embeddings(
                self.tgt_vision_projector, obj_mean
            )
        else:
            obj_target = obj_mean.to(
                device=pred_embeds.device,
                dtype=pred_embeds.dtype,
            )
        if self.aux_object_grounding_detach_target:
            obj_target = obj_target.detach()
        return obj_target, object_coverage.to(
            device=pred_embeds.device,
            dtype=pred_embeds.dtype,
        )

    def _object_patch_mean_from_mask(
        self,
        vision_tokens: torch.Tensor,
        target_mask: torch.Tensor | None,
        *,
        pooling_name: str,
    ) -> torch.Tensor:
        obj_mean, _object_coverage = self._object_patch_mean_and_coverage_from_mask(
            vision_tokens,
            target_mask,
            pooling_name=pooling_name,
        )
        return obj_mean

    def _object_patch_mean_and_coverage_from_mask(
        self,
        vision_tokens: torch.Tensor,
        target_mask: torch.Tensor | None,
        *,
        pooling_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_mask is None:
            raise ValueError(
                f"target_vision_pooling={pooling_name!r} requires target_mask."
            )
        if vision_tokens.size(1) <= 1:
            raise ValueError("mask target pooling requires patch tokens.")
        batch_size, token_count, _hidden_dim = vision_tokens.shape
        patch_count = token_count - 1
        grid = int(patch_count**0.5)
        if grid * grid != patch_count:
            raise ValueError(
                "mask target pooling expects a square patch grid; "
                f"got {patch_count} patches."
            )

        mask = target_mask.to(device=vision_tokens.device, dtype=torch.float32)
        if mask.dim() == 4:
            if mask.size(0) != batch_size:
                raise ValueError(
                    "target_mask batch does not match vision tokens: "
                    f"{mask.size(0)} != {batch_size}"
                )
            mask = mask.amax(dim=1)
        elif mask.dim() != 3:
            raise ValueError(
                "target_mask must have shape (B,T,H,W) or (B,H,W), got "
                f"{tuple(mask.shape)}."
            )
        if mask.size(0) != batch_size:
            raise ValueError(
                "target_mask batch does not match vision tokens: "
                f"{mask.size(0)} != {batch_size}"
            )

        mask = (mask > 0).to(dtype=torch.float32)
        patch_mask = F.interpolate(
            mask.unsqueeze(1),
            size=(grid, grid),
            mode="area",
        ).flatten(1)
        object_weights = (patch_mask > self.target_mask_patch_threshold).to(
            dtype=vision_tokens.dtype
        )
        if self.target_mask_dilate > 0:
            dilation = int(self.target_mask_dilate)
            object_weights = F.max_pool2d(
                object_weights.reshape(batch_size, 1, grid, grid),
                kernel_size=2 * dilation + 1,
                stride=1,
                padding=dilation,
            ).flatten(1).clamp(max=1)

        patch_tokens = vision_tokens[:, 1:]
        object_counts = object_weights.sum(dim=1, keepdim=True)
        obj_den = object_counts.clamp_min(1.0)
        obj_mean = (patch_tokens * object_weights.unsqueeze(-1)).sum(dim=1) / obj_den
        object_coverage = object_counts.squeeze(1) / float(patch_count)
        return obj_mean, object_coverage

    def _pool_mask_normed_cls_object(
        self,
        vision_tokens: torch.Tensor,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        obj_mean = self._object_patch_mean_from_mask(
            vision_tokens,
            target_mask,
            pooling_name="mask_normed_cls_object",
        )
        cls = F.normalize(vision_tokens[:, 0], dim=-1)
        obj = F.normalize(obj_mean, dim=-1)
        pooled = self.target_mask_cls_weight * cls + self.target_mask_object_weight * obj
        return F.normalize(pooled, dim=-1)

    def _pool_mask_innernorm_cls_object(
        self,
        vision_tokens: torch.Tensor,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        obj_mean = self._object_patch_mean_from_mask(
            vision_tokens,
            target_mask,
            pooling_name="mask_innernorm_cls_object",
        )
        cls = F.normalize(vision_tokens[:, 0], dim=-1)
        obj = F.normalize(obj_mean, dim=-1)
        return self.target_mask_cls_weight * cls + self.target_mask_object_weight * obj

    def _pool_mask_weighted_cls_patch(
        self,
        vision_tokens: torch.Tensor,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if target_mask is None:
            raise ValueError(
                "target_vision_pooling='mask_weighted_cls_patch' requires target_mask."
            )
        if vision_tokens.size(1) <= 1:
            raise ValueError("mask-weighted target pooling requires patch tokens.")
        batch_size, token_count, hidden_dim = vision_tokens.shape
        patch_count = token_count - 1
        grid = int(patch_count**0.5)
        if grid * grid != patch_count:
            raise ValueError(
                "mask-weighted target pooling expects a square patch grid; "
                f"got {patch_count} patches."
            )

        mask = target_mask.to(device=vision_tokens.device, dtype=torch.float32)
        if mask.dim() == 4:
            if mask.size(0) != batch_size:
                raise ValueError(
                    "target_mask batch does not match vision tokens: "
                    f"{mask.size(0)} != {batch_size}"
                )
            mask = mask.amax(dim=1)
        elif mask.dim() != 3:
            raise ValueError(
                "target_mask must have shape (B,T,H,W) or (B,H,W), got "
                f"{tuple(mask.shape)}."
            )
        if mask.size(0) != batch_size:
            raise ValueError(
                "target_mask batch does not match vision tokens: "
                f"{mask.size(0)} != {batch_size}"
            )

        mask = (mask > 0).to(dtype=torch.float32)
        patch_mask = F.interpolate(
            mask.unsqueeze(1),
            size=(grid, grid),
            mode="area",
        ).flatten(1)
        object_weights = (patch_mask > self.target_mask_patch_threshold).to(
            dtype=vision_tokens.dtype
        )
        if self.target_mask_dilate > 0:
            dilation = int(self.target_mask_dilate)
            object_weights = F.max_pool2d(
                object_weights.reshape(batch_size, 1, grid, grid),
                kernel_size=2 * dilation + 1,
                stride=1,
                padding=dilation,
            ).flatten(1).clamp(max=1)

        patch_tokens = vision_tokens[:, 1:]
        obj_den = object_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        obj_mean = (patch_tokens * object_weights.unsqueeze(-1)).sum(dim=1) / obj_den

        bg_weights = (1.0 - object_weights).to(dtype=vision_tokens.dtype)
        bg_den = bg_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        bg_mean = (patch_tokens * bg_weights.unsqueeze(-1)).sum(dim=1) / bg_den

        pooled = (
            self.target_mask_cls_weight * vision_tokens[:, 0]
            + self.target_mask_object_weight * obj_mean
            + self.target_mask_background_weight * bg_mean
        )
        if self.target_mask_normalize_weights:
            total = (
                self.target_mask_cls_weight
                + self.target_mask_object_weight
                + self.target_mask_background_weight
            )
            if total > 0:
                pooled = pooled / total
        return pooled

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Return number of parameters (for WandB logging compatibility)."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        vision_input: torch.Tensor | None = None,
        target_vision_input: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        text_input: list[str | None] | None = None,
        targets: list[str] | None = None,
        queries: list[str] | None = None,
        compute_loss: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass supporting vision OR text input per sample.

        Args:
            vision_input: Video frames (B, T, C, H, W). None or zeros indicate no valid vision.
            target_vision_input: Optional paired target images (B, T, C, H, W).
            text_input: List of text strings or None per sample. Complements vision_input.
            targets: Ground-truth texts for computing loss.
            queries: Text queries for conditional prediction.
            compute_loss: Whether to compute loss if targets provided.

        Returns:
            dict with predicted_embeddings, and optionally loss and target_embeddings.
        """
        has_vision = self.validate_inputs(vision_input, text_input)

        # Encode vision if at least one sample needs it
        if any(has_vision):
            if self.freeze_vision_encoder:
                with torch.no_grad():
                    vision_embedding = self.vision_encoder(
                        pixel_values=vision_input
                    )
            else:
                vision_embedding = self.vision_encoder(
                    pixel_values=vision_input
                )
            vision_embedding = vision_embedding.to(
                device=next(self.predictor.parameters()).device,
                dtype=next(self.predictor.parameters()).dtype,
            )
        else:
            vision_embedding = None

        # Encode and project (use packed forward for text-only if enabled)
        if (
            self.use_sequence_packing
            and vision_embedding is None
            and text_input is not None
            and queries is None
        ):
            predictor_out = self.predictor.forward_packed(text_input)
        else:
            predictor_out = self.predictor(
                vision_embedding=vision_embedding,
                text_input=text_input,
                queries=queries,
            )
        pred_embeds = self._project_embeddings(
            self.pred_projector, predictor_out
        )
        if self.prediction_residual_to_vision_tokens:
            if vision_embedding is None:
                raise RuntimeError(
                    "prediction_residual_to_vision_tokens requires vision input"
                )
            pred_embeds = add_patch_token_residual(
                pred_embeds,
                vision_embedding,
                scale=self.prediction_residual_scale,
            )

        out: dict[str, torch.Tensor] = {"predicted_embeddings": pred_embeds}

        if target_vision_input is not None:
            target_encoder = (
                self.target_vision_encoder
                if self.separate_target_vision_encoder
                else self.vision_encoder
            )
            if target_encoder is None:
                raise RuntimeError(
                    "Target vision inputs require an enabled vision encoder."
                )
            if vision_input is None or vision_input.dim() != 5:
                raise ValueError(
                    "Paired target vision mode expects source vision_input with shape (B, T, C, H, W)."
                )
            if target_vision_input.dim() != 5:
                raise ValueError(
                    "Paired target vision mode expects target_vision_input with shape (B, T, C, H, W)."
                )
            if vision_input.size(1) != 1 or target_vision_input.size(1) != 1:
                raise ValueError(
                    "Paired DINOv2 training currently supports single-frame inputs only (T=1)."
                )
            target_encoder_context = (
                torch.no_grad()
                if (
                    not self.separate_target_vision_encoder
                    or self.freeze_vision_encoder
                    or self.freeze_target_vision_encoder
                )
                else nullcontext()
            )
            with target_encoder_context:
                target_vision_embedding = target_encoder(
                    pixel_values=target_vision_input
                )
            if self.query_conditioned_target_consistency:
                if queries is None:
                    raise ValueError(
                        "queries must be provided when "
                        "query_conditioned_target_consistency=True."
                    )
                target_predictor_input = target_vision_embedding.to(
                    device=next(self.predictor.parameters()).device,
                    dtype=next(self.predictor.parameters()).dtype,
                )
                if self.qtarget_consistency_detach:
                    with torch.no_grad():
                        qtarget_predictor_out = self.predictor(
                            vision_embedding=target_predictor_input,
                            text_input=None,
                            queries=queries,
                        )
                        qtarget_pred_embeds = self._project_embeddings(
                            self.pred_projector, qtarget_predictor_out
                        )
                    qtarget_pred_embeds = qtarget_pred_embeds.detach()
                else:
                    qtarget_predictor_out = self.predictor(
                        vision_embedding=target_predictor_input,
                        text_input=None,
                        queries=queries,
                    )
                    qtarget_pred_embeds = self._project_embeddings(
                        self.pred_projector, qtarget_predictor_out
                    )
                out["query_target_predicted_embeddings"] = qtarget_pred_embeds
            target_pooled = self._pool_target_vision_embedding(
                target_vision_embedding, target_mask=target_mask
            )
            if self.use_target_vision_projector:
                if self.tgt_vision_projector is None:
                    raise RuntimeError(
                        "Target vision projector is enabled but was not constructed."
                    )
                target_embeds = self._project_embeddings(
                    self.tgt_vision_projector, target_pooled
                )
            else:
                target_embeds = target_pooled.to(
                    device=pred_embeds.device,
                    dtype=pred_embeds.dtype,
                )
            out["target_embeddings"] = target_embeds
            if self.aux_object_grounding_active:
                if self.aux_object_head is None:
                    raise RuntimeError(
                        "aux_object_grounding is enabled but aux head was not constructed."
                    )
                aux_object_target, aux_object_coverage = (
                    self._prepare_aux_object_target(
                        target_vision_embedding,
                        target_mask,
                        pred_embeds=pred_embeds,
                    )
                )
                out["aux_object_target_embeddings"] = aux_object_target
                out["aux_object_mask_coverage"] = aux_object_coverage
                out["aux_object_pred_embeddings"] = self.aux_object_head(
                    pred_embeds
                )
        # If text targets provided, get text embeddings and compute loss
        elif targets is not None:
            if self.text_encoder is None or self.tgt_text_projector is None:
                raise RuntimeError(
                    "Text targets were provided but the text encoder is disabled."
                )
            if self.add_query_to_target:
                if queries is None:
                    raise ValueError(
                        "Queries must be provided if add_query_to_target is True."
                    )
                targets = [
                    f"Query: {q}\nTarget: {t}"
                    for t, q in zip(targets, queries, strict=False)
                ]
            target_embeds = self._project_embeddings(
                self.tgt_text_projector,
                self.text_encoder(texts=targets),
            )
            out["target_embeddings"] = target_embeds

        if compute_loss and "target_embeddings" in out:
            if self.loss_fn is None:
                raise ValueError(
                    "Configurable Encoder InfoNCE requires ConfigurableEncoderTrainer and contrastive "
                    "group IDs. Use compute_loss=False to return embeddings."
                )
            if pred_embeds.size(-1) != out["target_embeddings"].size(-1):
                raise ValueError(
                    "Predicted and target embeddings must have matching dims, got "
                    f"{pred_embeds.size(-1)} and {out['target_embeddings'].size(-1)}."
                )
            loss_out = self.loss_fn(pred_embeds, out["target_embeddings"])
            out["loss"] = (
                loss_out["loss"] if isinstance(loss_out, dict) else loss_out
            )

        # Ensure learnable loss parameters (e.g., logit_scale, logit_bias)
        # participate in the DDP forward graph. Without this, DDP errors with
        # "marked as ready twice" because the trainer computes loss outside
        # the DDP-wrapped forward (to do all_gather first).
        if self.loss_fn is not None:
            for p in self.loss_fn.parameters():
                out["predicted_embeddings"] = (
                    out["predicted_embeddings"] + (0.0 * p).sum()
                )

        return out
