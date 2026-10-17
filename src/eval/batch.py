"""Evaluate the checkpoints saved by one ConfigWM training run."""
from __future__ import annotations

import filecmp
import json
import math
from pathlib import Path
import re
import subprocess
import sys


def discover_checkpoints(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training output directory does not exist: {run_dir}")
    epochs = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and re.fullmatch(r"epoch\d+", p.name)),
        key=lambda p: int(p.name[5:]),
    )
    checkpoints = [p / "configwm.pt" for p in epochs]
    final = run_dir / "configwm.pt"
    if final.is_file():
        checkpoints.append(final)
    if not checkpoints:
        raise FileNotFoundError(f"No configwm.pt checkpoints found in {run_dir}")
    for checkpoint in checkpoints:
        for required in (checkpoint, checkpoint.parent / "config.json"):
            if not required.is_file():
                raise FileNotFoundError(f"Incomplete checkpoint: missing {required}")
    # Training also saves the final epoch at the root. Evaluate identical copies once.
    if epochs and final.is_file():
        last = epochs[-1] / "configwm.pt"
        same_config = json.loads((final.parent / "config.json").read_text()) == json.loads(
            (last.parent / "config.json").read_text()
        )
        if same_config and filecmp.cmp(final, last, shallow=False):
            checkpoints.pop()
    return checkpoints


def evaluate_checkpoints(args) -> dict:
    checkpoints = discover_checkpoints(args.run_dir)
    if args.output.resolve() in {
        p.resolve() for checkpoint in checkpoints
        for p in (checkpoint, checkpoint.parent / "config.json")
    } | {(args.run_dir / name).resolve() for name in ("config.json", "configwm.pt")}:
        raise ValueError("--output must not overwrite checkpoint weights or config.json")
    results_dir = args.output.parent / (args.output.stem + "_checkpoints")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(
        env=args.env, variant=args.variant, seed=args.seed, num_eval=args.num_eval,
        checkpoint_count=len(checkpoints), complete=False, results=[],
        matched_sr=None, best_checkpoint=None,
    )

    def save():
        args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    save()
    for index, checkpoint in enumerate(checkpoints, 1):
        name = checkpoint.parent.name if checkpoint.parent != args.run_dir else "final"
        output = results_dir / (name + ".json")
        command = [sys.executable, "-m", "src.eval.run"]
        for key, value in vars(args).items():
            if key not in {"run_dir", "checkpoint", "output"} and value is not None:
                command.extend(["--" + key.replace("_", "-"), str(value)])
        command.extend([
            "--run-dir", str(checkpoint.parent), "--checkpoint", str(checkpoint),
            "--output", str(output),
        ])
        print(f"Evaluating checkpoint {index}/{len(checkpoints)}: {checkpoint}", flush=True)
        try:
            # A separate process releases simulator/GPU state between checkpoints.
            subprocess.run(command, check=True)
            result = json.loads(output.read_text())
            score = result["matched_sr"]
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= 100):
                raise ValueError(f"Invalid matched_sr in {output}: {score!r}")
        except (subprocess.CalledProcessError, OSError, ValueError, KeyError) as error:
            summary.update(failed_checkpoint=str(checkpoint), error=str(error))
            save()
            raise
        summary["results"].append(dict(checkpoint=str(checkpoint), matched_sr=score))
        best = max(summary["results"], key=lambda row: row["matched_sr"])
        summary.update(matched_sr=best["matched_sr"], best_checkpoint=best["checkpoint"])
        save()
    summary["complete"] = True
    save()
    print(json.dumps(summary), flush=True)
    return summary
