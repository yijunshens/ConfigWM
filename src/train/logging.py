import datetime
import logging
import os
from pathlib import Path
from shutil import copyfile
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.train.args import LaunchArgs


def _dist_info():
    """Return (rank, local_rank, world_size) with robust fallbacks."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Fallbacks when not under torch.distributed
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, local_rank, world_size


class _DistFields(logging.Filter):
    """Inject rank/local_rank/world_size fields into each LogRecord."""

    def __init__(self, rank: int, local_rank: int, world_size: int) -> None:
        super().__init__()
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        record.local_rank = self.local_rank
        record.world_size = self.world_size
        return True


LOG_FORMAT = "%(asctime)s | %(levelname)-7s | rank %(rank)s/%(world_size)s (local %(local_rank)s) | %(name)s | %(message)s"


def setup_logging(
    log_dir: str | None = None,
    level: int = logging.INFO,
    rank0_only: bool = True,
) -> None:
    """Setup logging to stdout and optionally to a file.

    Args:
        log_dir: If provided, also log to {log_dir}/train.log (rank 0 only).
        level: Logging level.
        rank0_only: If True, silence non-zero ranks.
    """
    logger = logging.getLogger()
    if getattr(logger, "_logging_initialized", False):
        return  # already configured

    rank, local_rank, world_size = _dist_info()
    if rank0_only and rank != 0:
        # Silence non-zero ranks entirely if desired
        logging.getLogger().disabled = True
        return

    logger.setLevel(level)
    # Remove any default handlers added by libraries
    for h in list(logger.handlers):
        logger.removeHandler(h)

    dist_filter = _DistFields(rank, local_rank, world_size)
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    console_handler.addFilter(dist_filter)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir:
        log_file = Path(log_dir) / "train.log"
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.addFilter(dist_filter)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Also capture warnings via logging
    logging.captureWarnings(True)

    # Mark as initialized
    logger._logging_initialized = True


def _compose_run_paths(args: "LaunchArgs") -> tuple[Path, Path, str]:
    """Compose output_dir, logging_dir, and run_name from launch args."""
    exp_label = getattr(args, "experiment_name", None)
    if not exp_label:
        exp_label = Path(args.data_config_path).parent.stem

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_desc = ts + (f"-{args.run_description}" if args.run_description else "")

    output_dir = Path(args.training_args.output_dir) / exp_label / run_desc
    logging_dir = output_dir / "logs"
    return output_dir, logging_dir, f"{exp_label}/{run_desc}"


def prepare_run_dirs(args: "LaunchArgs") -> Path:
    """Finalize output/logging dirs and copy config files for reproducibility.

    If args.output_dir is set (multi-node), use it directly without generating timestamps.
    Only rank 0 performs file I/O operations.
    """
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))

    # If output_dir is provided (multi-node), use it directly
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        logging_dir = output_dir / "logs"
        # Use experiment_name for wandb run_name if available.
        # Submitit embeds experiment_name into the output_dir path as the
        # parent directory (e.g., .../experiment_name/timestamp-desc/), so
        # we can recover it from there when LaunchArgs doesn't carry it.
        exp_label = getattr(args, "experiment_name", None)
        if not exp_label:
            # output_dir = .../output_base/experiment_name/run_id
            # output_dir.parent.name gives the experiment_name segment
            exp_label = output_dir.parent.name
        if not exp_label or exp_label == output_dir.name:
            exp_label = Path(args.data_config_path).parent.stem
        run_name = f"{exp_label}/{output_dir.name}"
    else:
        # Single-node: generate timestamped directory
        output_dir, logging_dir, run_name = _compose_run_paths(args)
        output_dir = output_dir.resolve()
        logging_dir = logging_dir.resolve()

    args.training_args.output_dir = str(output_dir)
    args.training_args.logging_dir = str(logging_dir)
    args.training_args.run_name = run_name

    # Only rank 0 creates dirs and copies configs
    if rank == 0:
        configs_dir = output_dir / "configs"
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        # Save training args
        with open(os.path.join(configs_dir, "training_args.json"), "w") as f:
            f.write(args.training_args.to_json_string())

        # Copy config files for reproducibility
        configs_to_copy = [
            (args.model_config_path, "model_config.yaml"),
            (args.data_config_path, "data_config.yaml"),
            (args.trainer_config_path, "trainer_config.yaml"),
        ]
        for src_path, dest_name in configs_to_copy:
            try:
                copyfile(src_path, os.path.join(configs_dir, dest_name))
            except Exception as e:
                logging.warning(f"Failed to copy {dest_name}: {e}")

    return output_dir


def get_params_count_summary(model, max_name_len: int = 96) -> str:
    """Pretty-print a table of all parameters in ``model``.

    Includes trainable flag, dtype, device, shape, and
    individual & total parameter counts.
    """
    rows = []
    total_trainable = 0
    total_nontrainable = 0

    for name, p in model.named_parameters():
        row = {
            "name": name[:max_name_len],
            "trainable": p.requires_grad,
            "dtype": str(p.dtype),
            "device": str(p.device),
            "shape": str(tuple(p.shape)),
            "count": p.numel(),
        }
        rows.append(row)

        if p.requires_grad:
            total_trainable += p.numel()
        else:
            total_nontrainable += p.numel()

    # ---------- compute column widths ----------
    # Guard against an empty model (e.g. nn.Sequential())
    def _max_len(key: str, header: str, default: int = 0) -> int:
        if rows:
            return max(len(header), max(len(r[key]) for r in rows))
        return max(len(header), default)

    col_w = {
        "name": max(max_name_len, _max_len("name", "Module")),
        "trainable": len("Trainable"),
        "dtype": _max_len("dtype", "Dtype"),
        "device": _max_len("device", "Device"),
        "shape": _max_len("shape", "Shape"),
        "count": len("Param Count"),
    }

    # Table width (accounts for „| “ and spaces between columns and ending „|“)
    width = (
        sum(col_w.values())  # actual content widths
        + 3 * len(col_w)  # spaces after each „|“
        + len(col_w)
        + 1  # the „|“ separators themselves
    )

    ruler = "=" * width
    lines = [ruler]

    # Header
    lines.append(
        f"| {'Module':<{col_w['name']}} "
        f"| {'Trainable':<{col_w['trainable']}} "
        f"| {'Dtype':<{col_w['dtype']}} "
        f"| {'Device':<{col_w['device']}} "
        f"| {'Shape':>{col_w['shape']}} "
        f"| {'Param Count':>{col_w['count']}} |"
    )
    lines.append("-" * width)

    # Body
    for r in rows:
        lines.append(
            f"| {r['name']:<{col_w['name']}} "
            f"| {r['trainable']!s:<{col_w['trainable']}} "
            f"| {r['dtype']:<{col_w['dtype']}} "
            f"| {r['device']:<{col_w['device']}} "
            f"| {r['shape']:>{col_w['shape']}} "
            f"| {r['count']:>{col_w['count']},} |"
        )

    # Totals
    lines.append("-" * width)
    lines.append(
        f"| {'Total trainable params':<{col_w['name']}} "
        f"| {'':<{col_w['trainable']}} | {'':<{col_w['dtype']}} | {'':<{col_w['device']}} "
        f"| {'':>{col_w['shape']}} | {total_trainable:>{col_w['count']},} |"
    )
    lines.append(
        f"| {'Total non-trainable params':<{col_w['name']}} "
        f"| {'':<{col_w['trainable']}} | {'':<{col_w['dtype']}} | {'':<{col_w['device']}} "
        f"| {'':>{col_w['shape']}} | {total_nontrainable:>{col_w['count']},} |"
    )
    lines.append(ruler)

    return "\n".join(lines)
