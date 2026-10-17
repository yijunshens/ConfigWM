from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


import torch.nn as nn
from torch.utils.data import Dataset

from src.data.config import resolve_data_config as _resolve_data_config
from src.data.seeding import seed_encoder_data
from src.data.manifest.core import collect_prebuilt_runtime_requirements
import src.data.sharding as sharding_utils
from src.model import ConfigurableEncoder
from src.train.args import LaunchArgs, parse_launch_args
from src.train.collate import DataCollator
from src.train.data import TrainDataset
from src.train.logging import (
    get_params_count_summary,
    prepare_run_dirs,
    setup_logging,
)
from src.train.trainer import ConfigurableEncoderTrainer
import torch
from transformers import set_seed



def freeze_by_keyword(model: nn.Module, keywords: list[str]) -> None:
    """Freeze parameters whose full name contains any of the given keywords."""
    if not keywords:
        return
    n_frozen = 0
    for name, p in model.named_parameters():
        if any(kw in name for kw in keywords):
            p.requires_grad = False
            n_frozen += 1
    logging.info(
        "Frozen %d params by keywords: %s", n_frozen, ", ".join(keywords)
    )


def prepare_model(args: LaunchArgs) -> ConfigurableEncoder:
    """Build the model from config path and (optionally) load weights.

    Uses main_process_first() so rank 0 downloads HF models first,
    then other ranks use the cached files (avoids rate limiting).

    Returns:
      model
    """
    from accelerate import PartialState

    state = PartialState()

    # Rank 0 downloads HF models first, others wait then use cache
    with state.main_process_first():
        # Build model directly from config path
        model = ConfigurableEncoder(args.model_config_path)

        # Optional weights-only initialization
        if args.initialization:
            logging.info(
                f"Initializing model WEIGHTS from {args.initialization} (no optimizer/scheduler state)"
            )
            model.load_checkpoint(args.initialization)
        else:
            logging.info(
                "No initialization checkpoint provided; training from scratch."
            )

    # Freeze parameters by keyword (training concern, not model concern)
    keywords_to_freeze = getattr(args.training_args, "keywords_to_freeze", [])
    if keywords_to_freeze:
        freeze_by_keyword(model, keywords_to_freeze)

    logging.info(get_params_count_summary(model))

    # NOTE: compile_backbone is applied AFTER Trainer.__init__ in ConfigurableEncoderTrainer,
    # not here — accelerator.unwrap_model() in Trainer.__init__ fails on OptimizedModule.

    return model


def resolve_data_config(
    config_path: str, where_am_i: str | None
) -> dict[str, Any]:
    """Load and strictly resolve a training-style data config."""
    return _resolve_data_config(config_path, where_am_i)


def prepare_data(args: LaunchArgs) -> dict[str, Any]:
    """Build train/eval datasets and collator from data config YAML.

    The training dataset handles preprocessing automatically based on
    the config file, so no external preprocessor is needed.

    Uses a barrier to ensure rank 0 loads/caches datasets first to avoid
    filesystem lock contention on shared filesystems.

    Returns:
        Dict with keys: train_dataset, eval_dataset, data_collator
    """
    from pathlib import Path

    from accelerate import PartialState

    state = PartialState()

    # Load and resolve data config with where_am_i routing
    config = resolve_data_config(args.data_config_path, args.where_am_i)
    config = seed_encoder_data(config, args.training_args.seed)

    manifest_cache_root = config.get("manifest_cache_root")
    if manifest_cache_root:
        sharding_utils.set_default_manifest_cache_root(manifest_cache_root)
        logging.info(
            "Using manifest_cache_root=%s",
            sharding_utils.DEFAULT_MANIFEST_CACHE_ROOT,
        )
    else:
        logging.info(
            "No manifest_cache_root configured; tar datasets will use "
            "dataset-local .offsets/ directories."
        )
    cache_root = (
        Path(manifest_cache_root).expanduser().resolve()
        if manifest_cache_root
        else None
    )
    required_prebuilt_paths, _ = collect_prebuilt_runtime_requirements(
        config,
        dataset_ids=None,
        cache_root=cache_root,
        samples_per_chunk=None,
        workers=1,
    )
    sharding_utils.set_required_prebuilt_paths(required_prebuilt_paths)
    logging.info(
        "Registered %d prebuilt path(s) that training will only load and never build at runtime",
        len(required_prebuilt_paths),
    )

    # Rank 0 loads first to populate cache, others wait
    logging.info("Preparing training dataset inside main_process_first()")
    with state.main_process_first():
        # Load training dataset from resolved config
        train_dataset = TrainDataset(
            config,
            datasets_key="train_datasets",
            seed=args.training_args.seed,
            shuffle_dataset=args.training_args.shuffle_dataset,
        )
    logging.info("Training dataset is ready on all ranks")

    # Read max_str_bytes from dataset config (default 4096)
    max_str_bytes = train_dataset.config.get("max_str_bytes", 4096)

    # Get pixel shape from dataset for text-only batch support
    pixel_shape = train_dataset.pixel_shape

    # Load eval datasets if configured (use already resolved config)
    eval_dataset: dict[str, Dataset] = {}
    if config.get("eval_datasets"):
        logging.info("Preparing eval dataset inside main_process_first()")
        with state.main_process_first():
            eval_dataset = TrainDataset(
                config,  # use resolved config with data_dir routing
                preprocessor=train_dataset.preprocessor,  # reuse preprocessor
                datasets_key="eval_datasets",
                seed=args.training_args.seed,
                shuffle_dataset=False,
            )
        logging.info("Eval dataset is ready on all ranks")

    data_collator = DataCollator(
        max_str_bytes=max_str_bytes, pixel_shape=pixel_shape
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }


def prepare_trainer(
    args: LaunchArgs, model: ConfigurableEncoder, **data_kwargs
) -> ConfigurableEncoderTrainer:
    """Construct a Trainer from launch args.

    Args:
        args: Launch arguments
        model: The model to train
        **data_kwargs: train_dataset, eval_dataset, data_collator from prepare_data()

    Returns:
        trainer - ready for trainer.train()
    """
    trainer = ConfigurableEncoderTrainer(
        model=model,
        args=args.training_args,
        **data_kwargs,
    )

    if args.training_args.resume_from_checkpoint is None:
        checkpoints = list(
            Path(args.training_args.output_dir).glob("checkpoint-*")
        )
        if checkpoints:
            raise FileExistsError(
                "Output directory already contains checkpoints. Refusing an implicit "
                "resume; pass --resume_from_checkpoint explicitly. "
                f"output_dir={args.training_args.output_dir}"
            )

    if args.training_args.resume_from_checkpoint:
        logging.info(
            "Trainer will resume from checkpoint: %s",
            args.training_args.resume_from_checkpoint,
        )
    else:
        logging.info("Trainer will start fresh (no checkpoint resumption).")

    return trainer


def main() -> None:
    args = parse_launch_args()
    set_seed(args.training_args.seed)
    output_dir = prepare_run_dirs(args)

    # For multi-node (output_dir provided), skip file logging - slurm handles it
    # For single-node, log to train.log
    log_dir = None if args.output_dir else args.training_args.logging_dir
    setup_logging(
        log_dir=log_dir,
        level=logging.INFO,
        rank0_only=True,
    )

    logging.info("Launch args: %s", args)
    logging.info("Output directory: %s", output_dir)

    data_kwargs = prepare_data(args)

    model = prepare_model(args)

    trainer = prepare_trainer(args, model, **data_kwargs)

    trainer.train(
        resume_from_checkpoint=args.training_args.resume_from_checkpoint
    )
    final_step = int(trainer.state.global_step)
    final_checkpoint = Path(args.training_args.output_dir) / f"checkpoint-{final_step}"
    save_steps = max(1, int(args.training_args.save_steps))
    final_was_regular_save = (
        args.training_args.save_strategy == "steps"
        and final_step > 0
        and final_step % save_steps == 0
    )
    if final_step > 0 and not final_was_regular_save:
        logging.info("Saving final full-state checkpoint at step %d", final_step)
        trainer._save_checkpoint(trainer.model, trial=None)
    else:
        logging.info("Final checkpoint was already covered by the save schedule: %s", final_checkpoint)


if __name__ == "__main__":
    main()
