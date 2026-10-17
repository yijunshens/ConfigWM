"""Argument parsing utilities for Configurable Encoder training."""

from dataclasses import dataclass, field
from typing import Any

from transformers import HfArgumentParser, TrainingArguments
import yaml

from src.data.seeding import validate_encoder_seed


@dataclass
class ConfigurableEncoderTrainingArguments(TrainingArguments):
    """Extended TrainingArguments with Configurable Encoder specific settings.

    All training config in one place - HF args + custom args.
    """

    shuffle_dataset: bool = field(
        default=True, metadata={"help": "Shuffle training dataset"}
    )
    gather_embeddings: bool = field(
        default=True, metadata={"help": "Gather embeddings across GPUs"}
    )
    main_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the primary Configurable Encoder prediction/SigReg loss"},
    )
    main_sigreg_only_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Add only the SigReg regularizer from the primary loss path, "
                "scaled by the loss function's internal sigreg_weight. This is "
                "useful for ablations with main_loss_weight=0 where primary "
                "prediction loss should stay disabled."
            )
        },
    )
    main_sigreg_only_lambda: float = field(
        default=-1.0,
        metadata={
            "help": (
                "Optional explicit lambda for main_sigreg_only_weight. Values "
                ">=0 override the loss function's internal sigreg_weight so "
                "pred losses can stay pure while a controlled SigReg term is "
                "added separately."
            )
        },
    )
    main_loss_force_all_rows: bool = field(
        default=False,
        metadata={
            "help": (
                "Ignore per-row main_loss_mask from the collator and apply the "
                "primary Configurable Encoder loss to every row. Useful for all-query source5 "
                "smokes where every query row should regress to its own clean target."
            )
        },
    )
    main_loss_include_clean_pred: bool = field(
        default=False,
        metadata={
            "help": (
                "When clean target prediction is enabled, concatenate source "
                "pred rows and clean-pred rows into one primary prediction loss "
                "instead of adding clean pred as a separate auxiliary loss."
            )
        },
    )
    same_source_contrastive_target_space: str = field(
        default="target_encoder",
        metadata={
            "help": (
                "Target-side space for same-source contrastive loss: "
                "'target_encoder' uses target_embeddings; 'predictor' uses "
                "predicted_embeddings from a second Configurable Encoder pass on target images."
            )
        },
    )
    same_source_contrastive_pooling: str = field(
        default="mean",
        metadata={
            "help": (
                "Spatial unit for same-source InfoNCE: 'mean' contrasts one "
                "pooled vector per image; 'patch_position' contrasts matching "
                "positions across images; 'patch_all' also contrasts positions "
                "within each image. All modes remain pure InfoNCE."
            )
        },
    )
    same_source_contrastive_patch_chunk_size: int = field(
        default=16,
        metadata={
            "help": (
                "Number of matching patch positions evaluated together for "
                "patch_position InfoNCE. Chunking is mathematically exact and "
                "only controls peak logits memory."
            )
        },
    )
    same_source_contrastive_target_query_mode: str = field(
        default="same",
        metadata={
            "help": (
                "Query passed to target-side predictor for same-source contrastive "
                "loss when target_space='predictor': 'same' reuses source queries, "
                "'none' passes no query/text, and any other value is used as a "
                "literal neutral query string."
            )
        },
    )
    same_source_margin_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight for same-source target-relative hinge margin loss. "
                "For each source/query prediction P_i, enforces sim(P_i,T_i) "
                "to exceed sim(P_i,T_j) for j != i by same_source_margin_value."
            )
        },
    )
    same_source_margin_value: float = field(
        default=0.02,
        metadata={
            "help": "Cosine-similarity margin for same-source hinge ranking loss."
        },
    )
    same_source_margin_mode: str = field(
        default="mean",
        metadata={
            "help": (
                "Same-source margin aggregation: 'mean' averages all violating "
                "off-diagonal pairs; 'hard' uses only the hardest wrong target per row."
            )
        },
    )
    embedding_diagnostics_on_epoch_end: bool = field(
        default=True,
        metadata={
            "help": (
                "At each epoch end, run a small no-grad embedding diagnostic pass "
                "and log effective-rank/collapse metrics via Trainer.log()."
            )
        },
    )
    embedding_diagnostics_max_samples: int = field(
        default=2048,
        metadata={
            "help": (
                "Maximum global samples used for epoch-end embedding diagnostics. "
                "Set to 0 to skip the diagnostic pass."
            )
        },
    )
    lr_multipliers: dict[str, float] = field(
        default_factory=dict,
        metadata={"help": "Learning rate multipliers by param name pattern"},
    )
    keywords_to_freeze: list[str] = field(
        default_factory=list,
        metadata={"help": "Keywords to match parameter names for freezing"},
    )
    compile_backbone: bool = field(
        default=False,
        metadata={
            "help": (
                "Compile the predictor backbone with torch.compile(inductor) before training. "
                "This bypasses HF Trainer's torch_compile mechanism (which breaks in "
                "transformers 5.x via output_capturing.py). Set torch_compile=false when using this."
            )
        },
    )
    compile_text_encoder: bool = field(
        default=False,
        metadata={
            "help": (
                "Compile the inner SentenceTransformer forward with torch.compile(inductor). "
                "Targets text_encoder.model.forward (pure tensor path, post-tokenization). "
                "Uses dynamic=True to handle variable token-length batches. "
                "Beneficial when text encoder is large (e.g. Qwen3-Emb-4B) or unfrozen."
            )
        },
    )
    compile_vision_encoder: bool = field(
        default=False,
        metadata={
            "help": (
                "Compile the vision encoder forward with torch.compile(inductor). "
                "Uses dynamic=False (fixed patch grid shape). "
                "Only relevant when vision encoder is enabled (image pretraining)."
            )
        },
    )

    @classmethod
    def from_yaml(cls, path: str) -> "ConfigurableEncoderTrainingArguments":
        """Load from YAML config file."""
        with open(path) as f:
            config = yaml.safe_load(f)

        seed = validate_encoder_seed(config.get("seed"))
        if config.get("data_seed") not in (None, seed):
            raise ValueError("Configurable Encoder uses one seed; remove the independent data_seed.")
        # Hugging Face's sampler field is always bound to the same run seed.
        config["data_seed"] = seed

        # HF Trainer treats negative max_steps as epoch-based scheduling.
        # Keep backward compatibility for older configs that used 0 here.
        if config.get("max_steps") == 0:
            config["max_steps"] = -1

        # Normalize lr_multipliers from various YAML formats
        if "learning_rate_scaling" in config:
            config["lr_multipliers"] = cls._normalize_lr_multipliers(
                config.pop("learning_rate_scaling")
            )

        # Ensure keywords_to_freeze is a list
        if (
            "keywords_to_freeze" in config
            and config["keywords_to_freeze"] is None
        ):
            config["keywords_to_freeze"] = []

        return cls(**config)

    @staticmethod
    def _normalize_lr_multipliers(x: Any) -> dict[str, float]:
        """Normalize learning_rate_scaling from YAML to dict[pattern, scale]."""
        if x is None:
            return {}
        if isinstance(x, dict):
            return {str(k): float(v) for k, v in x.items()}
        out = {}
        for item in x:  # list format
            if isinstance(item, dict) and "pattern" in item and "scale" in item:
                out[str(item["pattern"])] = float(item["scale"])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                k, v = item
                out[str(k)] = float(v)
        return out


@dataclass
class LaunchArgs:
    """CLI launch arguments (paths only).

    Use load_trainer_config() to populate training_args from YAML.
    """

    data_config_path: str = field(
        metadata={"help": "Path to data config YAML file"}
    )
    model_config_path: str = field(
        metadata={"help": "Path to model config YAML file"}
    )
    trainer_config_path: str = field(
        metadata={"help": "Path to trainer config YAML file"}
    )
    where_am_i: str | None = field(
        default=None,
        metadata={
            "help": "Cluster environment (fluidstack, nebius) for auto data_dir routing"
        },
    )
    run_description: str = field(
        default="",
        metadata={"help": "Description of the run for logging purposes"},
    )
    initialization: str | None = field(
        default=None,
        metadata={"help": "Optional checkpoint dir to initialize WEIGHTS ONLY"},
    )
    deepspeed: str | None = field(
        default=None,
        metadata={
            "help": "DeepSpeed config JSON path (for multi-node training)"
        },
    )
    output_dir: str | None = field(
        default=None,
        metadata={
            "help": "Override output directory (for multi-node, set by submitit)"
        },
    )
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={"help": "Checkpoint directory for full-state resumption"},
    )

    # Loaded from YAML (not CLI)
    training_args: ConfigurableEncoderTrainingArguments | None = field(
        default=None, repr=False
    )

    def load_trainer_config(self) -> "LaunchArgs":
        """Load trainer config from YAML. Returns self for chaining."""
        self.training_args = ConfigurableEncoderTrainingArguments.from_yaml(
            self.trainer_config_path
        )
        # Override deepspeed from CLI if provided
        if self.deepspeed:
            self.training_args.deepspeed = self.deepspeed
        # Override output_dir from CLI if provided (multi-node)
        if self.output_dir:
            self.training_args.output_dir = self.output_dir
        # Explicit full-state resume checkpoint
        if self.resume_from_checkpoint:
            self.training_args.resume_from_checkpoint = (
                self.resume_from_checkpoint
            )
        return self


def parse_launch_args() -> LaunchArgs:
    """Parse CLI args and load trainer config from YAML."""
    args = HfArgumentParser(LaunchArgs).parse_args_into_dataclasses()[0]
    return args.load_trainer_config()


__all__ = [
    "LaunchArgs",
    "ConfigurableEncoderTrainingArguments",
    "parse_launch_args",
]
