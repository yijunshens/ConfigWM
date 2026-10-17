"""TwoRoom9 anchored clean-bank InfoNCE mix for Configurable Encoder.

This is the TwoRoom9 counterpart of the PushT anchored
clean-bank adapter: target-frozen pure InfoNCE sees a fixed 512-row bank with
mostly core source->clean rows plus a small number of same-source query rows.
The query rows are routed by bank core color, not appended blindly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401 - registers compressed HDF5 filters
import numpy as np
import torch
from torch.utils.data import Dataset

from ...schema import Sample
from ...cleanbank import resolve_cleanbank_rows
from .temporal import (
    build_trajectory_windows,
    validate_temporal_contract,
)
from .tworoom_core import (
    TwoRoomNativeGaussianCoreQueryBalancedHDF5Dataset,
)
from .tworoom_queries import (
    TwoRoomNativeGaussianUnifiedSource5QueryHDF5Dataset,
)


class TwoRoomNativeGaussianAnchoredSource9TemporalCleanBankInfoNCEMixHDF5Dataset(
    Dataset[Sample]
):
    """D1-DK core temporal hard-negatives plus anchored source9 query rows.

    Core rows are built like the PushT old anchored path: a local temporal
    window from one source variant and one core color forms hard negatives. The
    source variant is selected round-robin per temporal group, instead of
    expanding every variant for every center. Source9 groups are selected from
    scenes whose core color matches evenly spaced core rows in the same bank.
    """

    def __init__(
        self,
        *,
        core_data_dirs: list[str | Path | dict[str, Any]],
        source9_data_dir: str | Path | dict[str, Any] | None = None,
        source9_parent_data_root: str | Path | dict[str, Any] | None = None,
        core_variant_names: list[str] | None = None,
        num_frames: int = 1,
        core_frame_stride: int = 5,
        core_temporal_offsets: list[int] | tuple[int, ...] = (-2, -1, 0, 1, 2),
        core_temporal_center_stride: int = 1,
        query_group_size: int = 9,
        source9_frame_stride: int = 1,
        source9_query_group_size: int | None = None,
        scene_split_path: str | Path | None = None,
        scene_split_name: str = "train",
        core_shuffle_groups: bool = True,
        core_balance_seed: int = 6274,
        source9_anchor_seed: int = 20260621,
        source9_anchor_mode: str = "bank_core_color",
        bank_size: int = 512,
        query_ratio: float = 0.10,
        max_supergroups: int | None = None,
        contrast_supergroup_merge_factor: int = 1,
        max_open_files: int = 16,
        **_unused: Any,
    ) -> None:
        if not core_data_dirs:
            raise ValueError("core_data_dirs must contain at least one TwoRoom variant")
        self.variant_names = (
            [f"variant_{idx}" for idx in range(len(core_data_dirs))]
            if core_variant_names is None
            else [str(name) for name in core_variant_names]
        )
        if len(self.variant_names) != len(core_data_dirs):
            raise ValueError("core_variant_names length must match core_data_dirs")

        self.core_temporal_offsets, self.core_temporal_center_stride = (
            validate_temporal_contract(
                core_temporal_offsets, core_temporal_center_stride
            )
        )
        self.core_shuffle_groups = bool(core_shuffle_groups)
        self.core_balance_seed = int(core_balance_seed)

        self.core_children = [
            TwoRoomNativeGaussianCoreQueryBalancedHDF5Dataset(
                data_dir,
                num_frames=num_frames,
                frame_stride=core_frame_stride,
                query_group_size=query_group_size,
                scene_split_path=scene_split_path,
                scene_split_name=scene_split_name,
                balance_core_colors=True,
                balance_seed=core_balance_seed,
                shuffle_refs=False,
                max_open_files=max_open_files,
            )
            for data_dir in core_data_dirs
        ]
        child_colors = {tuple(child.colors) for child in self.core_children}
        if len(child_colors) != 1:
            raise ValueError(f"Core children do not share colors: {sorted(child_colors)}")
        self._validate_core_alignment()
        self.core_colors = tuple(next(iter(child_colors)))
        self._core_groups, self._core_color_counts = self._build_core_temporal_groups()
        self._core_group_size = len(self.core_temporal_offsets)
        self._core_total_rows = int(self._core_groups.shape[0]) * self._core_group_size
        if self._core_total_rows <= 0:
            raise ValueError("No TwoRoom9 core temporal groups available")
        self._source_target_label_map = self._make_source_target_label_map()

        self.source9 = None
        self.source9_parent_data_root: Path | None = None
        if source9_parent_data_root is not None:
            parent_root = source9_parent_data_root
            if isinstance(parent_root, dict):
                parent_root = parent_root.get("local", parent_root.get("path"))
            if parent_root is None:
                raise ValueError("source9_parent_data_root must provide local or path")
            self.source9_parent_data_root = Path(str(parent_root))
        source9_group_size = int(source9_query_group_size or query_group_size)
        (
            self.cleanbank_size,
            self.core_rows_per_bank,
            self.source9_rows_per_bank,
        ) = resolve_cleanbank_rows(bank_size, query_ratio, source9_group_size)
        self.contrast_supergroup_merge_factor = max(1, int(contrast_supergroup_merge_factor))
        self.source9_anchor_seed = int(source9_anchor_seed)
        self.source9_anchor_mode = str(source9_anchor_mode)
        if self.source9_anchor_mode != "bank_core_color":
            raise ValueError(f"Unsupported source9_anchor_mode: {self.source9_anchor_mode!r}")
        if self.cleanbank_size <= 1:
            raise ValueError("cleanbank_size must be at least 2")
        if self.core_rows_per_bank <= 0:
            raise ValueError("core_rows_per_bank must be positive")
        if self.source9_rows_per_bank < 0:
            raise ValueError("source9_rows_per_bank must be non-negative")
        if self.core_rows_per_bank + self.source9_rows_per_bank != self.cleanbank_size:
            raise ValueError("core_rows_per_bank + source9_rows_per_bank must equal cleanbank_size")

        self.source9_groups_per_bank = 0
        self._source9_units_by_color: dict[str, np.ndarray] = {}
        if self.source9_rows_per_bank > 0:
            if source9_data_dir is None:
                raise ValueError("source9_data_dir is required when source9 rows are used")
            self.source9 = TwoRoomNativeGaussianUnifiedSource5QueryHDF5Dataset(
                data_dir=source9_data_dir,
                num_frames=num_frames,
                frame_stride=source9_frame_stride,
                query_group_size=source9_group_size,
                scene_split_path=scene_split_path,
                scene_split_name=scene_split_name,
                max_open_files=max_open_files,
            )
            if self.source9_rows_per_bank % int(self.source9.inbatch_group_size) != 0:
                raise ValueError("source9_rows_per_bank must preserve complete query groups")
            self.source9_groups_per_bank = int(self.source9_rows_per_bank) // int(
                self.source9.inbatch_group_size
            )
            self._source9_units_by_color = self._build_source9_units_by_core_color()
            missing = [color for color in self.core_colors if color not in self._source9_units_by_color]
            if missing:
                raise ValueError(f"source9 anchor colors missing from source9 data: {missing}")

        base_length = self._core_total_rows // self.core_rows_per_bank
        if self.source9_rows_per_bank > 0:
            assert self.source9 is not None
            base_length = min(base_length, len(self.source9) // self.source9_rows_per_bank)
        if max_supergroups is not None:
            base_length = min(base_length, int(max_supergroups))
        if base_length <= 0:
            raise ValueError("No complete TwoRoom9 anchored clean-bank groups available")
        self._base_length = int(base_length)
        self._group_size = int(self.cleanbank_size)
        self._length = self._base_length * self._group_size

    def _validate_core_alignment(self) -> None:
        """Ensure shared child indices identify the same scene and raw frame."""
        reference = self.core_children[0]
        for child, variant_name in zip(
            self.core_children[1:], self.variant_names[1:], strict=False
        ):
            if not np.array_equal(reference._refs, child._refs):
                raise ValueError(
                    f"TwoRoom variant {variant_name} is not aligned by scene/raw frame"
                )
            if len(reference._shards) != len(child._shards):
                raise ValueError(f"TwoRoom variant {variant_name} has different shard count")
            for shard_idx in range(len(reference._shards)):
                reference_meta = reference._metadata_for_shard(shard_idx)
                child_meta = child._metadata_for_shard(shard_idx)
                for key in (
                    "scene_source_offset",
                    "scene_ep_len",
                    "scene_global_idx",
                    "scene_core_color_idx",
                ):
                    if not np.array_equal(reference_meta[key], child_meta[key]):
                        raise ValueError(
                            "TwoRoom variants are not aligned: "
                            f"variant={variant_name}, shard={shard_idx}, field={key}"
                        )

    def _build_core_temporal_groups(self) -> tuple[np.ndarray, dict[str, int]]:
        reference_child = self.core_children[0]
        rows_by_color_scene: dict[int, dict[tuple[int, int], list[tuple[int, int]]]] = {
            idx: {} for idx in range(len(self.core_colors))
        }
        color_counts = {color: 0 for color in self.core_colors}
        for child_index, ref in enumerate(reference_child._refs):
            shard_idx, scene_idx, _source_local_index, frame_delta = (int(v) for v in ref)
            shard_index = reference_child._load_index(shard_idx)
            color_idx = int(shard_index["scene_core_color_idx"][scene_idx])
            if color_idx < 0 or color_idx >= len(self.core_colors):
                continue
            color_counts[self.core_colors[color_idx]] += 1
            key = (shard_idx, scene_idx)
            rows_by_color_scene[color_idx].setdefault(key, []).append((child_index, frame_delta))

        groups: list[np.ndarray] = []
        for color_idx in range(len(self.core_colors)):
            trajectories: list[np.ndarray] = []
            for key in sorted(rows_by_color_scene[color_idx]):
                rows = sorted(rows_by_color_scene[color_idx][key], key=lambda item: item[1])
                trajectories.append(
                    np.asarray([int(child_index) for child_index, _ in rows], dtype=np.int64)
                )
            candidates = build_trajectory_windows(
                trajectories,
                offsets=self.core_temporal_offsets,
                center_stride=self.core_temporal_center_stride,
            )
            if candidates.size == 0:
                continue
            for candidate in candidates:
                variant_idx = int(len(groups) % len(self.core_children))
                groups.append(
                    np.asarray([int(color_idx), variant_idx, *candidate], dtype=np.int64)
                )
        if not groups:
            raise ValueError("No temporal core groups selected")
        group_array = np.stack(groups, axis=0).astype(np.int64, copy=False)
        if self.core_shuffle_groups:
            rng = np.random.default_rng(self.core_balance_seed)
            group_array = group_array[rng.permutation(group_array.shape[0])]
        drop_multiple = 4
        keep = (group_array.shape[0] // drop_multiple) * drop_multiple
        if keep <= 0:
            raise ValueError("No temporal core groups remain after drop-last")
        group_array = group_array[:keep]
        return group_array, {str(k): int(v) for k, v in color_counts.items()}

    @staticmethod
    def _decode_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if hasattr(value, "decode"):
            return value.decode("utf-8")
        return str(value)

    def _source9_parent_scene_colors(self, shard_idx: int) -> np.ndarray:
        assert self.source9 is not None
        shard = self.source9._shards[int(shard_idx)]
        raw_path = shard.get("parent_h5_path")
        if not raw_path and isinstance(shard.get("paths"), dict):
            raw_path = shard["paths"].get("parent_h5_path")
        if not raw_path:
            raise KeyError(f"source9 shard {shard_idx} has no parent_h5_path for anchoring")
        path = Path(str(raw_path))
        if self.source9_parent_data_root is not None:
            path = self.source9_parent_data_root / path.parent.name / path.name
        elif not path.is_absolute():
            path = self.source9.data_dir / path
        with h5py.File(path, "r") as handle:
            if "scene_core_color" not in handle:
                raise KeyError(f"parent H5 lacks scene_core_color: {path}")
            return np.asarray(
                [self._decode_text(value) for value in handle["scene_core_color"][:]],
                dtype=object,
            )

    def _build_source9_units_by_core_color(self) -> dict[str, np.ndarray]:
        assert self.source9 is not None
        groups: dict[str, list[int]] = {str(color): [] for color in self.core_colors}
        parent_color_cache: dict[int, np.ndarray] = {}
        for source_unit, ref in enumerate(self.source9._source_refs):
            shard_idx = int(ref[0])
            scene_idx = int(ref[1])
            if shard_idx not in parent_color_cache:
                parent_color_cache[shard_idx] = self._source9_parent_scene_colors(shard_idx)
            colors = parent_color_cache[shard_idx]
            if scene_idx < 0 or scene_idx >= int(colors.shape[0]):
                continue
            color = str(colors[scene_idx])
            if color in groups:
                groups[color].append(int(source_unit))
        return {
            color: np.asarray(units, dtype=np.int64)
            for color, units in groups.items()
            if units
        }

    def __len__(self) -> int:
        return self._length

    @property
    def base_length(self) -> int:
        return self._base_length

    @property
    def inbatch_group_size(self) -> int:
        return self._group_size

    @property
    def color_counts(self) -> dict[str, int]:
        return dict(self._core_color_counts)

    def _core_positive_key(self, meta: dict[str, Any]) -> str:
        # scene_global_idx is shard-local; the original episode spans variants.
        episode = meta["original_episode_idx"]
        frame = meta.get("frame_delta", meta.get("source_local_index", "unknown"))
        color = meta.get("core_color", meta.get("target_color", meta.get("query", "unknown")))
        return f"tworoom9_coreclean:{episode}:{frame}:{color}"

    def _source9_positive_key(self, meta: dict[str, Any]) -> str:
        shard = meta.get("shard_idx", "unknown")
        target = meta.get("target_local_index", meta.get("id", "unknown"))
        color = meta.get("target_color", meta.get("query", "unknown"))
        return f"tworoom9_source9target:{shard}:{target}:{color}"

    def _core_sample_for_linear_index(self, child_index: int) -> tuple[Sample, int, int, str]:
        group_idx = int(child_index) // self._core_group_size
        temporal_slot = int(child_index) % self._core_group_size
        group = self._core_groups[group_idx]
        color_idx = int(group[0])
        variant_idx = int(group[1])
        core_child_index = int(group[2 + temporal_slot])
        sample = self.core_children[variant_idx][core_child_index]
        return sample, variant_idx, temporal_slot, str(self.core_colors[color_idx])

    def _make_source_target_label_map(self) -> torch.Tensor:
        for group in self._core_groups[:64]:
            variant_idx = int(group[1])
            for core_child_index in group[2:]:
                sample = self.core_children[variant_idx][int(core_child_index)]
                if sample.target_label_map is not None:
                    return torch.ones_like(sample.target_label_map)
        return torch.ones((1, 224, 224), dtype=torch.long)

    def _bank_core_color_for_source9_group(self, super_idx: int, group_offset: int) -> str:
        anchor_slot = int((int(group_offset) * int(self.core_rows_per_bank)) // max(1, self.source9_groups_per_bank))
        anchor_slot = min(max(anchor_slot, 0), int(self.core_rows_per_bank) - 1)
        core_child_index = int(super_idx) * int(self.core_rows_per_bank) + anchor_slot
        group_idx = int(core_child_index) // self._core_group_size
        color_idx = int(self._core_groups[group_idx][0])
        return str(self.core_colors[color_idx])

    def _source9_child_index(self, super_idx: int, local_slot: int) -> tuple[int, str]:
        assert self.source9 is not None
        query_group_size = int(self.source9.inbatch_group_size)
        group_offset = int(local_slot) // query_group_size
        query_slot = int(local_slot) % query_group_size
        color = self._bank_core_color_for_source9_group(int(super_idx), int(group_offset))
        units = self._source9_units_by_color[color]
        unit_pos = (
            int(super_idx) * 104729
            + int(group_offset) * 9176
            + self.source9_anchor_seed
        ) % int(units.shape[0])
        source_unit = int(units[int(unit_pos)])
        return source_unit * query_group_size + query_slot, color

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError("TwoRoom9 anchored clean-bank index out of range")
        super_idx = index // self._group_size
        slot = index % self._group_size
        if slot < self.core_rows_per_bank:
            child_index = super_idx * self.core_rows_per_bank + slot
            sample, variant_idx, temporal_slot, anchor_color = self._core_sample_for_linear_index(child_index)
            meta = dict(sample.meta)
            positive_key = self._core_positive_key(meta)
            family = "tworoom9_core_temporal_exactbank"
            meta.update(
                {
                    "core_temporal_variant": self.variant_names[variant_idx],
                    "core_temporal_variant_idx": int(variant_idx),
                    "core_temporal_slot": int(temporal_slot),
                    "core_temporal_offset": int(self.core_temporal_offsets[temporal_slot]),
                    "core_temporal_anchor_color": str(anchor_color),
                }
            )
        else:
            if self.source9 is None:
                raise RuntimeError("source9 rows requested but source9 dataset is absent")
            local_slot = slot - self.core_rows_per_bank
            child_index, anchor_color = self._source9_child_index(int(super_idx), int(local_slot))
            sample = self.source9[child_index]
            meta = dict(sample.meta)
            if not str(meta.get("core_color", "")):
                meta["core_color"] = str(anchor_color)
            positive_key = self._source9_positive_key(meta)
            family = "tworoom9_source9_query_exactbank_anchored"
            meta.update(
                {
                    "source9_anchor_color": str(anchor_color),
                    "source9_anchor_mode": self.source9_anchor_mode,
                }
            )

        contrast_super_idx = int(super_idx) // int(self.contrast_supergroup_merge_factor)
        meta.update(
            {
                "real_cleanbank_shard_idx": meta.get("shard_idx"),
                "real_cleanbank_source_local_index": meta.get("source_local_index"),
                "exact_cleanbank_child_index": int(child_index),
                "mix_family": family,
                "dataset_task": "tworoom9_anchored_source9_temporal_cleanbank_infonce_mix",
                "cleanbank_super_idx": int(super_idx),
                "cleanbank_slot": int(slot),
                "cleanbank_group_size": int(self._group_size),
                "cleanbank_core_rows": int(self.core_rows_per_bank),
                "cleanbank_source9_rows": int(self.source9_rows_per_bank),
                "cleanbank_contrast_super_idx": int(contrast_super_idx),
                "cleanbank_contrast_merge_factor": int(self.contrast_supergroup_merge_factor),
                "task": "pusht_source5_query",
                "shard_idx": f"tworoom9_anchored_cleanbank{int(self._group_size)}",
                "source_local_index": int(contrast_super_idx),
                "contrast_positive_key": positive_key,
                "main_loss_weight": 0.0,
            }
        )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map
            if sample.target_label_map is not None
            else self._source_target_label_map,
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


__all__ = ["TwoRoomNativeGaussianAnchoredSource9TemporalCleanBankInfoNCEMixHDF5Dataset"]
