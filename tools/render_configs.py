from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_layout import prepare_data_root
from src.model.vision_encoder.pretrained import resolve_backbone


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def query_split(root: Path) -> Path:
    if (root / "train" / "manifest.json").is_file():
        return root / "train"
    return root


def configure_data(env: str, config: dict, data_root: Path, vision_model: Path) -> None:
    data_root = data_root / env
    config["preprocessor"]["model_name"] = str(vision_model)
    dataset = config["train_datasets"][0]
    # Public presets share option names; the existing adapters keep their arguments.
    field_names = {
        "pusht": {
            "core_source_variants": "core_variant_names",
            "core_temporal_center_stride": "core_center_frame_stride",
            "query_frame_stride": "query_support_frame_stride",
            "query_group_size": "query_support_query_group_size",
            "query_anchor_mode": "query_support_anchor_mode",
        },
        "tworoom": {
            "core_source_variants": "core_variant_names",
            "query_frame_stride": "source9_frame_stride",
            "query_anchor_mode": "source9_anchor_mode",
        },
    }
    for public_name, adapter_name in field_names.get(env, {}).items():
        if public_name in dataset:
            value = dataset.pop(public_name)
            if adapter_name in dataset and dataset[adapter_name] != value:
                raise ValueError(f"Conflicting options: {public_name} and {adapter_name}")
            dataset[adapter_name] = value
    if env == "tworoom":
        dataset.setdefault("source9_query_group_size", dataset["query_group_size"])
    if env == "pusht":
        dataset["core_data_dirs"] = [
            {"local": str(data_root / variant)}
            for variant in dataset["core_variant_names"]
        ]
        clean = data_root / "clean"
        dataset["core_target_data_dir"] = {"local": str(clean)}
        dataset["core_query_metadata_path"] = str(clean)
        dataset["query_support_data_dir"] = {"local": str(query_split(data_root))}
    elif env == "tworoom":
        dataset["core_data_dirs"] = [
            {"local": str(data_root / variant)}
            for variant in dataset["core_variant_names"]
        ]
        dataset["source9_data_dir"] = {"local": str(data_root)}
        dataset["source9_parent_data_root"] = {"local": str(data_root / "d8")}
    elif env == "ogbcube":
        split = query_split(data_root)
        dataset["core_data_root"] = {"local": str(data_root)}
        dataset["core_split_manifest_path"] = str(split / "manifest.json")
        dataset["query9_data_dir"] = {"local": str(split)}


def configure_backbone(args, data: dict, model: dict, trainer: dict) -> None:
    if args.backbone is None:
        if any(value is not None for value in (args.backbone_code_dir, args.backbone_checkpoint, args.resolution)):
            raise ValueError("Backbone overrides require --backbone")
        return
    preset, local_model, local_checkpoint = resolve_backbone(args.backbone)
    profile = load_yaml(args.package_root / "configs/backbones" / f"{preset}.yaml")
    vision = dict(profile["vision_embedding"])
    if local_model is not None:
        vision["model_name"] = local_model
    if local_checkpoint is not None:
        vision["checkpoint_path"] = local_checkpoint
    if args.vision_model is not None:
        vision["model_name"] = str(args.vision_model)
    if args.backbone_code_dir is not None:
        if vision["type"] not in {"lingbot_vision", "vjepa2_hub"}:
            raise ValueError("--backbone-code-dir applies to LingBot and V-JEPA")
        vision["code_dir"] = str(args.backbone_code_dir.resolve())
    elif vision["type"] in {"lingbot_vision", "vjepa2_hub"}:
        folder = "lingbot-vision" if vision["type"] == "lingbot_vision" else "vjepa2"
        standard = args.package_root / "third_party" / folder
        if standard.is_dir():
            vision["code_dir"] = str(standard.resolve())
    if args.backbone_checkpoint is not None:
        if vision["type"] != "vjepa2_hub":
            raise ValueError("--backbone-checkpoint applies to V-JEPA; use --vision-model for HF weights")
        vision["checkpoint_path"] = str(args.backbone_checkpoint.resolve())
    if vision["type"] == "vjepa2_hub":
        if "code_dir" in vision and not (Path(vision["code_dir"]) / "hubconf.py").is_file():
            raise FileNotFoundError("V-JEPA code directory must contain hubconf.py")
        if "checkpoint_path" in vision and not Path(vision["checkpoint_path"]).is_file():
            raise FileNotFoundError(vision["checkpoint_path"])
    vision["expected_output_dim"] = profile["projector_dim"]
    vision["expected_patch_size"] = profile["patch_size"]
    if vision["type"] == "hf_patch_tokens":
        vision["expected_model_type"] = "dinov3_vit" if preset == "dinov3-7b" else "dinov2"
    prep = dict(profile["preprocessor"])
    resolution = args.resolution if args.resolution is not None else prep["resolution"]
    patch_size = profile["patch_size"]
    if resolution < patch_size or resolution % patch_size:
        raise ValueError(f"resolution must be a positive multiple of patch size {patch_size}")
    prep["resolution"] = resolution
    patch_count = (resolution // patch_size) ** 2
    if vision["type"] != "dinov2":
        vision["expected_patch_count"] = patch_count
    if prep["id"] in {"dinov2", "hf_image"}:
        prep["model_name"] = vision["model_name"]
    model["vision_embedding"] = vision
    model["projector"]["dim"] = profile["projector_dim"]
    model["predictor"]["max_input_tokens"] = patch_count + 1
    model["predictor"]["max_query_tokens"] = profile["max_query_tokens"]
    data["preprocessor"] = prep
    trainer.update(profile.get("trainer", {}))
    if profile.get("parameter_dtype"):
        model["parameter_dtype"] = profile["parameter_dtype"]
        model["predictor"]["backbone"]["dtype"] = profile["parameter_dtype"]
    if trainer.get("fsdp") and args.world_size < 2:
        raise ValueError("This backbone preset uses FSDP; provide at least two GPUs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--env", choices=("pusht", "tworoom", "ogbcube", "unified", "realworld"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vision-model", type=str)
    parser.add_argument("--backbone", help="Preset name, supported model ID/official URL, or local model path")
    parser.add_argument("--backbone-code-dir", type=Path)
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--batch-size", type=int, help="Per-process batch size")
    parser.add_argument("--group-size", type=int, default=8, help="Real-world contrastive group size across all processes")
    parser.add_argument("--steps", type=int, help="Stop after this many optimizer updates")
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of training processes used for the global contrastive bank",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="One Configurable Encoder seed for initialization, data order, groups and anchors")
    args = parser.parse_args()
    if args.world_size < 1:
        parser.error("--world-size must be positive")
    if args.vision_model is None and args.backbone is None:
        parser.error("provide --vision-model or --backbone")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.env == "realworld":
        args.data_root = args.data_root.expanduser().resolve()
        if not (args.data_root / "manifest.jsonl").is_file():
            parser.error("realworld --data-root must contain manifest.jsonl")
    else:
        args.data_root = prepare_data_root(
            args.data_root, args.output_dir / "data_views", args.env, "configurable_encoder"
        )
    if args.seed is None:
        args.seed = load_yaml(args.package_root / "configs/configurable_encoder/seed.yaml")["seed"]
    if type(args.seed) is not int or not 0 <= args.seed < 2**32:
        parser.error("--seed must be an integer in [0, 2**32).")

    environments = ("pusht", "tworoom", "ogbcube") if args.env == "unified" else (args.env,)
    parts = []
    for env in environments:
        part = load_yaml(args.package_root / "configs" / "data" / f"{env}.yaml")
        if env == "realworld":
            dataset = part["train_datasets"][0]
            dataset["data_dir"] = {"local": str(args.data_root)}
            dataset["group_size"] = args.group_size
            dataset["shuffle_seed"] = args.seed
            part["preprocessor"]["model_name"] = str(args.vision_model or "facebook/dinov2-small")
        else:
            configure_data(env, part, args.data_root, args.vision_model)
        parts.append(part)
    data = dict(parts[0])
    data["seed"] = args.seed
    data["train_datasets"] = [dataset for part in parts for dataset in part["train_datasets"]]
    if args.env == "unified":
        data["mix_inbatch_groups_atomically"] = True
    model = load_yaml(args.package_root / "configs" / "configurable_encoder" / "model.yaml")
    trainer = load_yaml(args.package_root / "configs" / "configurable_encoder" / "trainer.yaml")

    if args.env == "realworld":
        trainer.update(load_yaml(args.package_root / "configs/configurable_encoder/trainer_realworld.yaml"))
        if args.group_size < 2 or args.group_size % args.world_size:
            parser.error("--group-size must be >=2 and divisible by the GPU count")
        trainer["per_device_train_batch_size"] = args.group_size // args.world_size
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        trainer["per_device_train_batch_size"] = args.batch_size
    if args.env == "realworld" and trainer["per_device_train_batch_size"] * args.world_size != args.group_size:
        parser.error("realworld requires batch-size * GPU count == group-size")
    if args.steps is not None:
        trainer["max_steps"] = args.steps
    global_batch_size = (
        int(trainer["per_device_train_batch_size"])
        * int(args.world_size)
    )
    for dataset in data.get("train_datasets", []):
        if dataset.get("bank_size") == "auto":
            dataset["bank_size"] = global_batch_size
    model["vision_embedding"]["model_name"] = str(args.vision_model)
    model["predictor"]["backbone"]["model_name"] = str(args.llama_model)
    configure_backbone(args, data, model, trainer)
    trainer["output_dir"] = str(args.output_dir)
    trainer["seed"] = args.seed

    args.config_dir.mkdir(parents=True, exist_ok=False)
    dump_yaml(args.config_dir / "data_config.yaml", data)
    dump_yaml(args.config_dir / "model_config.yaml", model)
    dump_yaml(args.config_dir / "trainer_config.yaml", trainer)
    (args.config_dir / "render_args.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
