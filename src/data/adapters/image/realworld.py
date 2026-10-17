"""Task-conditioned source/target image pairs from a portable JSONL manifest."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.schema import Sample


class RealWorldDataset(Dataset[Sample]):
    """One source/query matches its target; other targets form grouped negatives."""

    def __init__(
        self,
        data_dir: str | Path,
        manifest_name: str = "manifest.jsonl",
        split: str = "train",
        group_size: int = 8,
        shuffle_records: bool = True,
        shuffle_seed: int = 42,
        num_frames: int = 1,
    ) -> None:
        self.root = Path(data_dir).expanduser().resolve()
        self.inbatch_group_size = int(group_size)
        self.shuffle_seed = int(shuffle_seed)
        if num_frames != 1:
            raise ValueError("realworld uses single-frame image pairs (num_frames=1)")
        if self.inbatch_group_size < 2:
            raise ValueError("group_size must be at least 2 for InfoNCE")
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be train, val, test, or all")

        records = []
        case_splits: dict[str, str] = {}
        manifest = self._file(manifest_name)
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("each record must be a JSON object")
                    for field in ("source", "target", "query"):
                        if not isinstance(row.get(field), str) or not row[field].strip():
                            raise ValueError(f"missing non-empty string {field!r}")
                    row_split = row.get("split", "train")
                    if row_split not in {"train", "val", "test"}:
                        raise ValueError("record split must be train, val, or test")
                    case_id = row.get("case_id", row["target"])
                    if not isinstance(case_id, str) or not case_id.strip():
                        raise ValueError("case_id must be a non-empty string")
                    if case_splits.setdefault(case_id, row_split) != row_split:
                        raise ValueError(f"case {case_id!r} occurs in multiple splits")
                    if split != "all" and split != row_split:
                        continue
                    records.append({
                        "source": self._file(row["source"]),
                        "target": self._file(row["target"]),
                        "query": row["query"],
                        "case_id": case_id,
                        "split": row_split,
                        "line": line_number,
                    })
                except (ValueError, TypeError, FileNotFoundError) as exc:
                    raise ValueError(f"{manifest}:{line_number}: {exc}") from exc
        if shuffle_records:
            random.Random(self.shuffle_seed).shuffle(records)
        usable = len(records) // self.inbatch_group_size * self.inbatch_group_size
        if usable == 0:
            raise ValueError(
                f"Split {split!r} has {len(records)} pairs; at least "
                f"group_size={self.inbatch_group_size} are required"
            )
        self.selection_stats = {"selected": len(records), "usable": usable,
                                "dropped_incomplete_group": len(records) - usable}
        if usable != len(records):
            logging.warning("realworld: omitting %d pairs from an incomplete InfoNCE group",
                            len(records) - usable)
        self._records = records[:usable]

    def _file(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError(f"use a path relative to data_dir: {value!r}")
        path = (self.root / path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"path escapes data_dir: {value!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @property
    def base_length(self) -> int:
        return len(self._records) // self.inbatch_group_size

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Sample:
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self._records[index]
        with Image.open(row["source"]) as image:
            source = image.convert("RGB")
        with Image.open(row["target"]) as image:
            target = image.convert("RGB")
        # Each image is independently resized by the backbone preprocessor.
        return Sample(
            frames=[source], target_frames=[target], targets=[], query=row["query"],
            label_map=torch.zeros((1, 1, 1), dtype=torch.long),
            meta={
                "id": f"realworld:{row['line']}", "task": "realworld",
                "case_id": row["case_id"], "split": row["split"],
                "contrast_group_key": f"realworld:{self.root}:{self.shuffle_seed}:{index // self.inbatch_group_size}",
                "contrast_positive_key": f"realworld:{row['target']}",
                "main_loss_weight": 0.0,
            },
        )
