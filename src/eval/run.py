from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from src.world_model.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from src.world_model.model import build_world_model
from .common import TSWMProjectorCostModel
from src.world_model.config import normalize_model_config
from src.data_layout import prepare_data_root


def main():
    parser = argparse.ArgumentParser(description="Evaluate matched success rate")
    parser.add_argument("--env", choices=["pusht", "tworoom", "ogbcube"], required=True)
    parser.add_argument(
        "--variant", choices=["d" + str(i) for i in range(9)], required=True
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", "--state-path", dest="checkpoint", type=Path,
                        help="Evaluate one checkpoint; otherwise evaluate all saved checkpoints in --run-dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-backbone-dtype", choices=["float32", "bfloat16"], default=None,
                        help="Override the saved Llama parameter dtype, e.g. for older checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--eval-budget", type=int, default=50)
    args = parser.parse_args()
    if args.checkpoint is None:
        from .batch import evaluate_checkpoints

        evaluate_checkpoints(args)
        return
    config = json.loads((args.run_dir / "config.json").read_text())
    if args.encoder_backbone_dtype is not None:
        config["model"]["encoder_backbone_dtype"] = args.encoder_backbone_dtype
    # Older checkpoint metadata may contain this retired training-only field.
    config["data"].pop("sample_seed", None)
    config = ExperimentConfig(
        seed=config["seed"],
        data=DataConfig(**config["data"]),
        model=ModelConfig(**normalize_model_config(config["model"])),
        train=TrainConfig(**config["train"]),
    )
    model = build_world_model(config)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=False
    )
    model.eval().requires_grad_(False).to(args.device)
    bridge = TSWMProjectorCostModel(
        model=model,
        history_len=config.data.num_hist,
        action_block=config.data.target_offset,
        eval_query=config.model.query,
    ).to(args.device)
    env = importlib.import_module("." + args.env, __package__)
    if args.env == "tworoom":
        from .tworoom_renderer import make_instance_overlay_class

        env.TwoRoomMatchedDatasetOverlay = make_instance_overlay_class(
            env.TwoRoomMatchedDatasetOverlay
        )
    root = prepare_data_root(
        args.data_root, args.output.parent / "data_views", args.env, "control", args.variant
    )
    matched = args.variant != "d0"
    options = dict(
        run_dir=args.run_dir,
        seed=args.seed,
        num_eval=args.num_eval,
        goal_offset_steps=25,
        eval_budget=args.eval_budget,
        horizon=5,
        receding_horizon=5,
        img_size=224,
        world_history_size=1,
        world_frame_skip=1,
        num_samples=args.num_samples,
        cem_steps=args.cem_steps,
        topk=30,
        solver_batch_size=50,
        var_scale=1.0,
        warm_start=True,
        decouple_donor_trajectory=matched,
        donor_trajectory_seed=args.seed,
        matched_dataset_overlay_dir=None,
        matched_clean_dataset_dir=None,
        matched_core_color_filter=None,
        matched_overlay_threshold=12.0,
        ogbcube_2x2_base=None,
        ogbcube_live_2x2_variant=None,
        ogbcube_core_color_filter=None,
    )
    if args.env == "pusht":
        options.update(
            env_name="swm/PushT-v1",
            eval_data_dir=root / "eval/pusht/pusht_expert_train.h5",
            keys_to_cache=["action", "proprio", "state"],
        )
        if matched:
            options.update(
                matched_dataset_overlay_dir=root / "eval/pusht" / args.variant,
                matched_clean_dataset_dir=root / "eval/pusht/d0",
                matched_core_color_filter="gray",
            )
    elif args.env == "tworoom":
        options.update(
            env_name="swm/TwoRoom-v1",
            eval_data_dir=root / "eval/tworoom/tworoom.h5",
            keys_to_cache=["action", "proprio", "pos_agent"],
        )
        if matched:
            options.update(
                matched_dataset_overlay_dir=root / "eval/tworoom" / args.variant,
                matched_core_color_filter="red",
            )
    else:
        options.update(
            env_name="swm/OGBCube-v0",
            eval_data_dir=root / "ogbcube/d0",
            keys_to_cache=[
                "action",
                "qpos",
                "qvel",
                "observation",
                "privileged_block_0_pos",
            ],
        )
        # The fixed silver-arm query also requires silver samples in d0.
        options.update(
            ogbcube_2x2_base=root / "ogbcube",
            ogbcube_core_color_filter="silver",
        )
        if matched:
            options.update(ogbcube_live_2x2_variant=args.variant)
    result = {
        "matched_sr": env.run_control_eval(SimpleNamespace(**options), bridge, config)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
