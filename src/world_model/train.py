from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.world_model.config import config_to_dict, load_experiment_config
from src.world_model.data import TransitionDataset
from src.world_model.model import build_world_model
from src.world_model.config import normalize_encoder_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConfigWM dynamics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--encoder-type", choices=["dino", "configurable_encoder"], default=None, type=normalize_encoder_type)
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--encoder-backbone-dtype", choices=["float32", "bfloat16"], default=None)
    parser.add_argument("--dino-path", default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--query-metadata", default=None)
    parser.add_argument("--sample-mode", choices=["prefix", "random"], default=None)
    parser.add_argument("--normalize-actions", action="store_true")
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    return parser.parse_args()


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("observation_encoder.")
    }


def save_checkpoint(model, config, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainable_state(model), output_dir / "configwm.pt")
    (output_dir / "config.json").write_text(
        json.dumps(config_to_dict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    config.train.output_dir = str(Path(args.output_dir).expanduser().resolve())
    config.data.data_dir = str(Path(args.data_root).expanduser())
    data_manifest = Path(config.data.data_dir) / "manifest.json"
    if (args.query_metadata is None
            and config.data.query_metadata_path == "query_metadata.jsonl"
            and data_manifest.is_file()
            and "field_map" in json.loads(data_manifest.read_text())):
        # The release manifest supplies the portable episode-query reference.
        config.data.query_metadata_path = None
    if config.data.query_metadata_path is not None:
        metadata_path = Path(config.data.query_metadata_path).expanduser()
        if not metadata_path.is_absolute():
            config.data.query_metadata_path = str(
                Path(config.data.data_dir) / metadata_path
            )
    if args.encoder_type is not None:
        config.model.encoder_type = args.encoder_type
    if args.encoder_checkpoint is not None:
        config.model.encoder_checkpoint = args.encoder_checkpoint
    if args.encoder_backbone_dtype is not None:
        config.model.encoder_backbone_dtype = args.encoder_backbone_dtype
    if args.dino_path is not None:
        config.model.dino_model_name = args.dino_path
    if args.query is not None:
        config.model.query = args.query
    if args.query_metadata is not None:
        config.data.query_metadata_path = args.query_metadata
    if args.sample_mode is not None:
        config.data.sample_mode = args.sample_mode
    if args.normalize_actions:
        config.data.normalize_actions = True
    if args.action_dim is not None:
        config.data.action_dim = args.action_dim
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.max_samples is not None:
        config.data.max_samples = args.max_samples

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device(args.device)
    model = build_world_model(config).to(device)
    dataset = TransitionDataset(
        config.data.data_dir,
        dino_model_name=config.model.dino_model_name,
        resolution=config.data.resolution,
        num_hist=config.data.num_hist,
        frameskip=config.data.frameskip,
        target_offset=config.data.target_offset,
        max_samples=config.data.max_samples,
        sample_mode=config.data.sample_mode,
        seed=config.seed,
        normalize_actions=config.data.normalize_actions,
        query_metadata_path=config.data.query_metadata_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=(config.data.persistent_workers and config.data.num_workers > 0),
        **({"prefetch_factor": config.data.prefetch_factor}
           if config.data.num_workers > 0 and config.data.prefetch_factor is not None else {}),
    )
    projector_params = list(model.projector.parameters())
    dynamics_params = (
        list(model.predictor.parameters())
        + list(model.action_encoder.parameters())
        + list(model.pred_proj.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": projector_params, "lr": config.train.projector_lr},
            {"params": dynamics_params, "lr": config.train.predictor_lr},
        ],
        weight_decay=config.train.weight_decay,
    )
    model.train()
    output_dir = Path(config.train.output_dir)
    step = 0
    for epoch in range(config.train.epochs):
        epoch_steps = 0
        for batch in loader:
            step += 1
            epoch_steps += 1
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda" and config.train.precision == "bf16"),
            ):
                outputs = model.forward_pixels(
                    batch["history_pixel_values"],
                    batch["future_pixel_values"],
                    batch["action_chunk"],
                    action_context=batch.get("action_context"),
                    queries=batch.get("queries"),
                )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.train.grad_clip_norm,
            )
            optimizer.step()
            print(json.dumps({"epoch": epoch + 1, "step": step,
                              "loss": float(outputs.loss.detach().cpu()),
                              "pred_loss": float(outputs.pred_loss.detach().cpu()),
                              "sigreg_loss": (None if outputs.sigreg_loss is None else float(outputs.sigreg_loss.detach().cpu()))}), flush=True)
            if args.steps is not None and step >= args.steps:
                break
        reached_step_limit = args.steps is not None and step >= args.steps
        if epoch_steps > 0 and (
            config.train.save_epoch_checkpoints
            or reached_step_limit
            or epoch + 1 == config.train.epochs
        ):
            epoch_dir = output_dir / f"epoch{epoch + 1:02d}"
            save_checkpoint(model, config, epoch_dir)
            if epoch_steps < len(loader):
                print(f"Stopped during epoch {epoch + 1}; checkpoint saved to {epoch_dir}", flush=True)
        if reached_step_limit:
            break


if __name__ == "__main__":
    main()
