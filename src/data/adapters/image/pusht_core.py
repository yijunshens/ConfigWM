"""Color-balanced PushT D1-D5 bag adapter.

This adapter mirrors the successful TwoRoom pooled core-query setup for PushT.
Each in-batch group contains every requested source variant for every PushT
core color:

    5 source variants x 5 core colors = 25 samples per group

Every sample still uses the episode's own core-color query and maps to the
aligned clean target frame.  This adapter adds color/query balance to the
existing aligned variant bag; it does not create counterfactual all-query
targets.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from torch.utils.data import Dataset

from ...schema import Sample
from .pusht_pairs import PushTPairedHDF5Dataset


PUSHT_COLORS = ("gray", "yellow", "pink", "brown", "gold")


class PushTCoreColorBalancedVariantBagHDF5Dataset(Dataset[Sample]):
    """Pool aligned PushT variants into complete color-balanced groups."""

    def __init__(
        self,
        data_dirs: list[str | Path | dict[str, Any]],
        target_data_dir: str | Path | dict[str, Any],
        *,
        variant_names: list[str] | None = None,
        colors: list[str] | tuple[str, ...] | None = None,
        query: str = "focus on the colored T block",
        query_metadata_path: str | Path | None = None,
        query_template: str = "focus on the {color} T block",
        num_frames: int = 1,
        num_hard_negatives: int = 0,
        frame_stride: int = 1,
        max_groups: int | None = None,
        variant_schedule: str = "full",
        shuffle_groups: bool = True,
        balance_seed: int = 3072,
        drop_last_groups_to_multiple: int = 1,
    ) -> None:
        if not data_dirs:
            raise ValueError("data_dirs must contain at least one PushT variant")
        self.data_dirs = [self._resolve_path(path) for path in data_dirs]
        self.target_data_dir = self._resolve_path(target_data_dir)
        self.variant_names = (
            [f"variant_{idx}" for idx in range(len(self.data_dirs))]
            if variant_names is None
            else [str(name) for name in variant_names]
        )
        if len(self.variant_names) != len(self.data_dirs):
            raise ValueError(
                "variant_names length must match data_dirs length: "
                f"{len(self.variant_names)} != {len(self.data_dirs)}"
            )
        self.colors = tuple(PUSHT_COLORS if colors is None else [str(c) for c in colors])
        if len(set(self.colors)) != len(self.colors):
            raise ValueError(f"Duplicate colors are not allowed: {self.colors}")
        if len(self.colors) <= 0:
            raise ValueError("At least one color is required")

        self.frame_stride = max(1, int(frame_stride))
        self.max_groups = None if max_groups is None else int(max_groups)
        if self.max_groups is not None and self.max_groups <= 0:
            raise ValueError("max_groups must be positive when provided")
        self.variant_schedule = str(variant_schedule).lower()
        if self.variant_schedule not in {"full", "diagonal"}:
            raise ValueError(
                "variant_schedule must be either 'full' or 'diagonal', "
                f"got {variant_schedule!r}"
            )
        self.shuffle_groups = bool(shuffle_groups)
        self.balance_seed = int(balance_seed)
        self.drop_last_groups_to_multiple = max(1, int(drop_last_groups_to_multiple))
        self.query_template = str(query_template)

        self.children = [
            PushTPairedHDF5Dataset(
                data_dir,
                self.target_data_dir,
                query=query,
                query_metadata_path=query_metadata_path,
                query_template=query_template,
                num_frames=num_frames,
                num_hard_negatives=num_hard_negatives,
            )
            for data_dir in self.data_dirs
        ]
        child_lengths = [len(child) for child in self.children]
        if len(set(child_lengths)) != 1:
            raise ValueError(f"Aligned PushT variant lengths differ: {child_lengths}")
        if child_lengths[0] <= 0:
            raise ValueError("No aligned PushT frames available")

        color_by_episode = getattr(self.children[0], "_query_color_by_episode", {})
        if not color_by_episode:
            raise ValueError("query_metadata_path is required for color-balanced bags")
        unknown = sorted({str(c) for c in color_by_episode.values()} - set(self.colors))
        if unknown:
            raise ValueError(
                f"Metadata contains colors outside configured PushT colors: {unknown}"
            )

        self._group_indices, self._group_color_indices, self._color_counts = (
            self._build_balanced_groups(self.children[0], color_by_episode)
        )
        self._child_group_size = len(self.colors)
        if self.variant_schedule == "diagonal":
            if len(self.children) != self._child_group_size:
                raise ValueError(
                    "diagonal variant_schedule requires the number of variants "
                    "to match the number of colors: "
                    f"{len(self.children)} != {self._child_group_size}"
                )
            self._group_size = self._child_group_size
        else:
            self._group_size = len(self.children) * self._child_group_size
        self._base_length = int(self._group_indices.shape[0])
        self._length = self._base_length * self._group_size
        if self._base_length <= 0:
            raise ValueError("No balanced PushT color groups selected")

    @staticmethod
    def _resolve_path(value: str | Path | dict[str, Any]) -> Path:
        if isinstance(value, dict):
            for key in ("local", "path", "root"):
                candidate = value.get(key)
                if candidate:
                    return Path(candidate).expanduser().resolve()
            raise ValueError(f"path mapping must contain local/path/root: {value}")
        return Path(value).expanduser().resolve()

    @staticmethod
    def _episode_key(handle: h5py.File) -> str:
        for key in ("episode_idx", "ep_idx", "episode", "global_episode_idx"):
            if key in handle:
                return key
        raise KeyError("Could not find episode key in PushT HDF5 file")

    def _build_balanced_groups(
        self,
        reference_child: PushTPairedHDF5Dataset,
        color_by_episode: dict[int, str],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        by_episode: dict[int, list[tuple[int, int]]] = defaultdict(list)
        source_manifest = getattr(reference_child, "_source_manifest")
        step_starts = getattr(reference_child, "_step_starts")

        for shard_idx, shard in enumerate(source_manifest.get("shards", [])):
            shard_path = reference_child.data_dir / reference_child._shard_path(shard)
            with h5py.File(shard_path, "r") as handle:
                episode_values = np.asarray(
                    handle[self._episode_key(handle)][:],
                    dtype=np.int64,
                )
                step_values = np.asarray(handle["step_idx"][:], dtype=np.int64)
            global_offset = int(step_starts[shard_idx])
            for local_index, (episode_idx, step_idx) in enumerate(
                zip(episode_values.tolist(), step_values.tolist(), strict=False)
            ):
                by_episode[int(episode_idx)].append(
                    (global_offset + int(local_index), int(step_idx))
                )

        buckets: dict[str, list[int]] = {color: [] for color in self.colors}
        for episode_idx in sorted(by_episode):
            color = str(color_by_episode[int(episode_idx)])
            if color not in buckets:
                continue
            rows = sorted(by_episode[episode_idx], key=lambda item: item[1])
            selected = rows[:: self.frame_stride]
            buckets[color].extend(int(global_index) for global_index, _step in selected)

        rng = np.random.default_rng(self.balance_seed)
        bucket_arrays: dict[str, np.ndarray] = {}
        for color in self.colors:
            values = np.asarray(buckets[color], dtype=np.int64)
            if values.size <= 0:
                raise ValueError(f"No PushT frames found for color {color!r}")
            if self.shuffle_groups:
                values = values[rng.permutation(values.size)]
            bucket_arrays[color] = values

        group_count = min(int(bucket_arrays[color].shape[0]) for color in self.colors)
        if self.max_groups is not None:
            group_count = min(group_count, self.max_groups)
        if self.drop_last_groups_to_multiple > 1:
            group_count -= group_count % self.drop_last_groups_to_multiple
        if group_count <= 0:
            raise ValueError("No PushT color-balanced groups remain after truncation")

        group_indices = np.empty((group_count, len(self.colors)), dtype=np.int64)
        group_color_indices = np.empty((group_count, len(self.colors)), dtype=np.int64)
        for group_idx in range(group_count):
            order = (np.arange(len(self.colors), dtype=np.int64) + group_idx) % len(
                self.colors
            )
            for slot, color_idx in enumerate(order.tolist()):
                color = self.colors[color_idx]
                group_indices[group_idx, slot] = int(bucket_arrays[color][group_idx])
                group_color_indices[group_idx, slot] = int(color_idx)

        color_counts = {color: int(bucket_arrays[color].shape[0]) for color in self.colors}
        return group_indices, group_color_indices, color_counts

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
        return dict(self._color_counts)

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError(
                "PushTCoreColorBalancedVariantBagHDF5Dataset index out of range"
            )

        group_idx = index // self._group_size
        slot = index % self._group_size
        if self.variant_schedule == "diagonal":
            color_slot = slot
            variant_idx = slot % len(self.children)
        else:
            variant_idx = slot // self._child_group_size
            color_slot = slot % self._child_group_size
        frame_index = int(self._group_indices[group_idx, color_slot])
        color_idx = int(self._group_color_indices[group_idx, color_slot])
        expected_color = self.colors[color_idx]

        sample = self.children[variant_idx][frame_index]
        expected_query = self.query_template.format(color=expected_color)
        if sample.query != expected_query:
            raise ValueError(
                "Balanced bag resolved to the wrong query: "
                f"expected {expected_query!r}, got {sample.query!r}, "
                f"group_idx={group_idx}, variant={self.variant_names[variant_idx]}, "
                f"frame_index={frame_index}"
            )

        meta = dict(sample.meta)
        meta.update(
            {
                "id": (
                    f"PushTColorBag:{self.variant_names[variant_idx]}:"
                    f"{expected_color}:{meta.get('id', index)}"
                ),
                "task": "pusht_corecolor_balanced_variant_bag",
                "balanced_bag_group_idx": int(group_idx),
                "balanced_bag_frame_index": int(frame_index),
                "balanced_bag_group_size": int(self._group_size),
                "balanced_bag_child_group_size": int(self._child_group_size),
                "balanced_bag_variant": self.variant_names[variant_idx],
                "balanced_bag_variant_idx": int(variant_idx),
                "balanced_bag_num_variants": len(self.children),
                "balanced_bag_color": expected_color,
                "balanced_bag_color_idx": int(color_idx),
                "balanced_bag_color_slot": int(color_slot),
                "balanced_bag_colors": list(self.colors),
                "balanced_bag_frame_stride": int(self.frame_stride),
                "balanced_bag_max_groups": self.max_groups,
                "balanced_bag_variant_schedule": self.variant_schedule,
                "balanced_bag_shuffle_groups": self.shuffle_groups,
                "balanced_bag_drop_last_groups_to_multiple": (
                    self.drop_last_groups_to_multiple
                ),
            }
        )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map,
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


__all__ = ["PushTCoreColorBalancedVariantBagHDF5Dataset", "PUSHT_COLORS"]
