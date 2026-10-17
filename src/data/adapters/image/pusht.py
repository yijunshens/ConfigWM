"""PushT data-grouping adapters for pure-InfoNCE ablations.

These adapters deliberately do not introduce a new loss.  They only reshape
which samples share the collator's existing ``contrast_group_ids``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch.utils.data import Dataset

from ...schema import Sample
from ...cleanbank import resolve_cleanbank_rows
from .pusht_core import (
    PUSHT_COLORS,
    PushTCoreColorBalancedVariantBagHDF5Dataset,
)
from .pusht_pairs import PushTPairedHDF5Dataset
from .pusht_queries import PushTSource5QueryHDF5Dataset


class PushTSource5MultiStateQueryHDF5Dataset(Dataset[Sample]):
    """Group multiple source states, each with all query colors.

    The base Source5 adapter groups rows as ``1 source x 5 queries``.  This
    wrapper exposes groups as ``N sources x 5 queries`` and spoofs the metadata
    keys consumed by the existing collator so the whole block becomes one
    contrastive group.  This adds same-color different-state negatives while
    keeping the loss implementation unchanged.
    """

    def __init__(
        self,
        data_dir: str | Path | dict[str, Any],
        *,
        num_frames: int = 1,
        frame_stride: int = 1,
        query_group_size: int = 5,
        state_group_size: int = 4,
        max_groups: int | None = None,
    ) -> None:
        self.child = PushTSource5QueryHDF5Dataset(
            data_dir=data_dir,
            num_frames=num_frames,
            frame_stride=frame_stride,
            query_group_size=query_group_size,
        )
        self.query_group_size = int(query_group_size)
        self.state_group_size = int(state_group_size)
        if self.state_group_size <= 0:
            raise ValueError("state_group_size must be positive")
        self._group_size = self.state_group_size * self.query_group_size
        base_length = int(self.child.base_length) // self.state_group_size
        if max_groups is not None:
            base_length = min(base_length, int(max_groups))
        if base_length <= 0:
            raise ValueError("No complete multi-state Source5 groups available")
        self._base_length = base_length
        self._length = self._base_length * self._group_size

    def __len__(self) -> int:
        return self._length

    @property
    def base_length(self) -> int:
        return self._base_length

    @property
    def inbatch_group_size(self) -> int:
        return self._group_size

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError("PushTSource5MultiStateQueryHDF5Dataset index out of range")
        group_idx = index // self._group_size
        slot = index % self._group_size
        state_slot = slot // self.query_group_size
        query_slot = slot % self.query_group_size
        source_unit = group_idx * self.state_group_size + state_slot
        child_index = source_unit * self.query_group_size + query_slot
        sample = self.child[child_index]
        meta = dict(sample.meta)
        meta.update(
            {
                "real_task": meta.get("task"),
                "real_shard_idx": meta.get("shard_idx"),
                "real_source_local_index": meta.get("source_local_index"),
                "task": "pusht_source5_query",
                "shard_idx": f"source5_multistate_{self.state_group_size}",
                "source_local_index": int(group_idx),
                "multi_state_group_idx": int(group_idx),
                "multi_state_slot": int(state_slot),
                "multi_state_query_slot": int(query_slot),
                "multi_state_group_size": int(self._group_size),
                "main_loss_weight": 1.0,
            }
        )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


class PushTCoreColorInfoNCEBagHDF5Dataset(Dataset[Sample]):
    """Expose D1-D5 core-clean balanced bags as contrastive groups."""

    def __init__(
        self,
        data_dirs: list[str | Path | dict[str, Any]],
        target_data_dir: str | Path | dict[str, Any],
        *,
        variant_names: list[str] | None = None,
        colors: list[str] | tuple[str, ...] | None = None,
        query: str = "gray T block",
        query_metadata_path: str | Path | None = None,
        query_template: str = "{color} T block",
        num_frames: int = 1,
        num_hard_negatives: int = 0,
        frame_stride: int = 5,
        max_groups: int | None = None,
        variant_schedule: str = "full",
        shuffle_groups: bool = True,
        balance_seed: int = 3072,
        drop_last_groups_to_multiple: int = 4,
    ) -> None:
        self.child = PushTCoreColorBalancedVariantBagHDF5Dataset(
            data_dirs=data_dirs,
            target_data_dir=target_data_dir,
            variant_names=variant_names,
            colors=colors,
            query=query,
            query_metadata_path=query_metadata_path,
            query_template=query_template,
            num_frames=num_frames,
            num_hard_negatives=num_hard_negatives,
            frame_stride=frame_stride,
            max_groups=max_groups,
            variant_schedule=variant_schedule,
            shuffle_groups=shuffle_groups,
            balance_seed=balance_seed,
            drop_last_groups_to_multiple=drop_last_groups_to_multiple,
        )

    def __len__(self) -> int:
        return len(self.child)

    @property
    def base_length(self) -> int:
        return int(self.child.base_length)

    @property
    def inbatch_group_size(self) -> int:
        return int(self.child.inbatch_group_size)

    def __getitem__(self, index: int) -> Sample:
        sample = self.child[index]
        group_idx = index // self.inbatch_group_size
        slot = index % self.inbatch_group_size
        meta = dict(sample.meta)
        meta.update(
            {
                "real_task": meta.get("task"),
                "task": "pusht_source5_query",
                "shard_idx": "core_color_bag",
                "source_local_index": int(group_idx),
                "core_infonce_group_idx": int(group_idx),
                "core_infonce_slot": int(slot),
                "pair_is_core_clean": True,
                "main_loss_weight": 1.0,
            }
        )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


class PushTCoreStateInfoNCEGroupHDF5Dataset(Dataset[Sample]):
    """Group same-color but different-state PushT core-clean pairs.

    Unlike ``PushTCoreColorInfoNCEBagHDF5Dataset``, a contrastive group here is
    not a complete color bag.  Each group contains many different source states
    that share one core color, so color alone cannot solve the within-group
    InfoNCE objective.
    """

    def __init__(
        self,
        data_dirs: list[str | Path | dict[str, Any]],
        target_data_dir: str | Path | dict[str, Any],
        *,
        variant_names: list[str] | None = None,
        colors: list[str] | tuple[str, ...] | None = None,
        query: str = "gray T block",
        query_metadata_path: str | Path | None = None,
        query_template: str = "{color} T block",
        num_frames: int = 1,
        num_hard_negatives: int = 0,
        frame_stride: int = 5,
        samples_per_group: int = 25,
        max_groups_per_color: int | None = None,
        shuffle_groups: bool = True,
        balance_seed: int = 4073,
        drop_last_groups_to_multiple: int = 4,
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
            raise ValueError("variant_names length must match data_dirs")
        self.colors = tuple(PUSHT_COLORS if colors is None else [str(c) for c in colors])
        if len(set(self.colors)) != len(self.colors):
            raise ValueError(f"Duplicate colors are not allowed: {self.colors}")
        self.frame_stride = max(1, int(frame_stride))
        self.samples_per_group = int(samples_per_group)
        if self.samples_per_group < 2:
            raise ValueError("samples_per_group must be at least 2")
        self.max_groups_per_color = (
            None if max_groups_per_color is None else int(max_groups_per_color)
        )
        if self.max_groups_per_color is not None and self.max_groups_per_color <= 0:
            raise ValueError("max_groups_per_color must be positive when provided")
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
        self._validate_aligned_children()

        color_by_episode = getattr(self.children[0], "_query_color_by_episode", {})
        if not color_by_episode:
            raise ValueError("query_metadata_path is required for state InfoNCE groups")
        unknown = sorted({str(c) for c in color_by_episode.values()} - set(self.colors))
        if unknown:
            raise ValueError(
                f"Metadata contains colors outside configured PushT colors: {unknown}"
            )

        self._groups, self._color_counts = self._build_same_color_state_groups(
            self.children[0],
            color_by_episode,
        )
        self._group_size = self.samples_per_group
        self._base_length = int(self._groups.shape[0])
        self._length = self._base_length * self._group_size
        if self._base_length <= 0:
            raise ValueError("No same-color PushT state groups selected")

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

    def _validate_aligned_children(self) -> None:
        """Reject variants whose raw episode/frame identity is not aligned."""
        reference = self.children[0]
        reference_shards = list(reference._source_manifest.get("shards", []))
        for child, variant_name in zip(
            self.children[1:], self.variant_names[1:], strict=False
        ):
            child_shards = list(child._source_manifest.get("shards", []))
            if len(child_shards) != len(reference_shards):
                raise ValueError(
                    f"PushT variant {variant_name} has {len(child_shards)} shards; "
                    f"expected {len(reference_shards)}"
                )
            if list(child._step_starts) != list(reference._step_starts):
                raise ValueError(f"PushT variant {variant_name} has different shard offsets")
            for shard_idx, (reference_shard, child_shard) in enumerate(
                zip(reference_shards, child_shards, strict=False)
            ):
                reference_path = reference._manifest_shard_file(
                    reference.data_dir, reference_shard
                )
                child_path = child._manifest_shard_file(child.data_dir, child_shard)
                with h5py.File(reference_path, "r") as reference_h, h5py.File(
                    child_path, "r"
                ) as child_h:
                    reference_episode = np.asarray(
                        reference_h[self._episode_key(reference_h)][:], dtype=np.int64
                    )
                    child_episode = np.asarray(
                        child_h[self._episode_key(child_h)][:], dtype=np.int64
                    )
                    reference_step = np.asarray(reference_h["step_idx"][:], dtype=np.int64)
                    child_step = np.asarray(child_h["step_idx"][:], dtype=np.int64)
                if not np.array_equal(reference_episode, child_episode) or not np.array_equal(
                    reference_step, child_step
                ):
                    raise ValueError(
                        "PushT variants are not aligned by episode/raw frame: "
                        f"variant={variant_name}, shard={shard_idx}"
                    )

    def _build_same_color_state_groups(
        self,
        reference_child: PushTPairedHDF5Dataset,
        color_by_episode: dict[int, str],
    ) -> tuple[np.ndarray, dict[str, int]]:
        by_color: dict[str, list[int]] = {color: [] for color in self.colors}
        source_manifest = getattr(reference_child, "_source_manifest")
        step_starts = getattr(reference_child, "_step_starts")

        for shard_idx, shard in enumerate(source_manifest.get("shards", [])):
            shard_path = reference_child._manifest_shard_file(
                reference_child.data_dir, shard
            )
            with h5py.File(shard_path, "r") as handle:
                episode_values = np.asarray(
                    handle[self._episode_key(handle)][:],
                    dtype=np.int64,
                )
                step_values = np.asarray(handle["step_idx"][:], dtype=np.int64)
            global_offset = int(step_starts[shard_idx])
            rows_by_episode: dict[int, list[tuple[int, int]]] = {}
            for local_index, (episode_idx, step_idx) in enumerate(
                zip(episode_values.tolist(), step_values.tolist(), strict=False)
            ):
                rows_by_episode.setdefault(int(episode_idx), []).append(
                    (global_offset + int(local_index), int(step_idx))
                )
            for episode_idx in sorted(rows_by_episode):
                color = str(color_by_episode[int(episode_idx)])
                if color not in by_color:
                    continue
                rows = sorted(rows_by_episode[episode_idx], key=lambda item: item[1])
                by_color[color].extend(
                    int(global_index) for global_index, _step in rows[:: self.frame_stride]
                )

        rng = np.random.default_rng(self.balance_seed)
        groups: list[np.ndarray] = []
        color_counts: dict[str, int] = {}
        for color_idx, color in enumerate(self.colors):
            values = np.asarray(by_color[color], dtype=np.int64)
            if values.size < self.samples_per_group:
                raise ValueError(
                    f"Not enough PushT frames for color {color!r}: {values.size}"
                )
            if self.shuffle_groups:
                values = values[rng.permutation(values.size)]
            group_count = int(values.size // self.samples_per_group)
            if self.max_groups_per_color is not None:
                group_count = min(group_count, self.max_groups_per_color)
            if self.drop_last_groups_to_multiple > 1:
                group_count -= group_count % self.drop_last_groups_to_multiple
            color_counts[color] = int(values.size)
            if group_count <= 0:
                continue
            chunks = values[: group_count * self.samples_per_group].reshape(
                group_count,
                self.samples_per_group,
            )
            color_column = np.full((group_count, 1), color_idx, dtype=np.int64)
            groups.append(np.concatenate([color_column, chunks], axis=1))
        if not groups:
            raise ValueError("No same-color state groups remain after truncation")
        all_groups = np.concatenate(groups, axis=0)
        if self.shuffle_groups:
            all_groups = all_groups[rng.permutation(all_groups.shape[0])]
        return all_groups.astype(np.int64, copy=False), color_counts

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
            raise IndexError("PushTCoreStateInfoNCEGroupHDF5Dataset index out of range")
        group_idx = index // self._group_size
        slot = index % self._group_size
        group = self._groups[group_idx]
        color_idx = int(group[0])
        frame_index = int(group[slot + 1])
        variant_idx = int((group_idx + slot) % len(self.children))
        expected_color = self.colors[color_idx]

        sample = self.children[variant_idx][frame_index]
        expected_query = self.query_template.format(color=expected_color)
        if sample.query != expected_query:
            raise ValueError(
                "Same-color state group resolved to the wrong query: "
                f"expected {expected_query!r}, got {sample.query!r}, "
                f"group_idx={group_idx}, frame_index={frame_index}"
            )
        meta = dict(sample.meta)
        meta.update(
            {
                "id": (
                    f"PushTCoreStateInfoNCE:{expected_color}:"
                    f"{self.variant_names[variant_idx]}:{meta.get('id', index)}"
                ),
                "real_task": meta.get("task"),
                "task": "pusht_source5_query",
                "shard_idx": f"core_state_samecolor_{expected_color}",
                "source_local_index": int(group_idx),
                "core_state_group_idx": int(group_idx),
                "core_state_slot": int(slot),
                "core_state_group_size": int(self._group_size),
                "core_state_color": expected_color,
                "core_state_color_idx": int(color_idx),
                "core_state_variant": self.variant_names[variant_idx],
                "core_state_variant_idx": int(variant_idx),
                "core_state_frame_index": int(frame_index),
                "pair_is_core_clean": True,
                "main_loss_weight": 1.0,
            }
        )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


class PushTCoreStateSupConGroupHDF5Dataset(PushTCoreStateInfoNCEGroupHDF5Dataset):
    """Same-color different-state groups with D1-D5 variants as positives.

    A group contains ``states_per_group`` underlying frames for one core color.
    For each frame, all aligned D1-D5 source variants are emitted.  Rows from
    the same underlying frame share ``contrast_positive_key`` so the trainer's
    same-source InfoNCE becomes a supervised multi-positive objective.
    """

    def __init__(
        self,
        data_dirs: list[str | Path | dict[str, Any]],
        target_data_dir: str | Path | dict[str, Any],
        *,
        variant_names: list[str] | None = None,
        colors: list[str] | tuple[str, ...] | None = None,
        query: str = "gray T block",
        query_metadata_path: str | Path | None = None,
        query_template: str = "{color} T block",
        num_frames: int = 1,
        num_hard_negatives: int = 0,
        frame_stride: int = 5,
        states_per_group: int = 5,
        max_groups_per_color: int | None = None,
        shuffle_groups: bool = True,
        balance_seed: int = 5074,
        drop_last_groups_to_multiple: int = 4,
    ) -> None:
        self.states_per_group = int(states_per_group)
        if self.states_per_group < 2:
            raise ValueError("states_per_group must be at least 2")
        super().__init__(
            data_dirs=data_dirs,
            target_data_dir=target_data_dir,
            variant_names=variant_names,
            colors=colors,
            query=query,
            query_metadata_path=query_metadata_path,
            query_template=query_template,
            num_frames=num_frames,
            num_hard_negatives=num_hard_negatives,
            frame_stride=frame_stride,
            samples_per_group=self.states_per_group,
            max_groups_per_color=max_groups_per_color,
            shuffle_groups=shuffle_groups,
            balance_seed=balance_seed,
            drop_last_groups_to_multiple=drop_last_groups_to_multiple,
        )
        self._state_count = int(self.samples_per_group)
        self._variant_count = int(len(self.children))
        self._group_size = self._state_count * self._variant_count
        self._length = int(self._base_length) * int(self._group_size)

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError("PushTCoreStateSupConGroupHDF5Dataset index out of range")
        group_idx = index // self._group_size
        slot = index % self._group_size
        state_slot = slot // self._variant_count
        variant_idx = slot % self._variant_count
        group = self._groups[group_idx]
        color_idx = int(group[0])
        frame_index = int(group[state_slot + 1])
        expected_color = self.colors[color_idx]

        sample = self.children[variant_idx][frame_index]
        expected_query = self.query_template.format(color=expected_color)
        if sample.query != expected_query:
            raise ValueError(
                "State-aware SupCon group resolved to the wrong query: "
                f"expected {expected_query!r}, got {sample.query!r}, "
                f"group_idx={group_idx}, frame_index={frame_index}"
            )
        meta = dict(sample.meta)
        meta.update(
            {
                "id": (
                    f"PushTCoreStateSupCon:{expected_color}:"
                    f"{self.variant_names[variant_idx]}:{meta.get('id', index)}"
                ),
                "real_task": meta.get("task"),
                "task": "pusht_source5_query",
                "shard_idx": f"core_state_supcon_{expected_color}",
                "source_local_index": int(group_idx),
                "contrast_positive_key": f"core_state_supcon:{group_idx}:{state_slot}",
                "core_state_supcon_group_idx": int(group_idx),
                "core_state_supcon_state_slot": int(state_slot),
                "core_state_supcon_variant_slot": int(variant_idx),
                "core_state_supcon_group_size": int(self._group_size),
                "core_state_color": expected_color,
                "core_state_color_idx": int(color_idx),
                "core_state_variant": self.variant_names[variant_idx],
                "core_state_variant_idx": int(variant_idx),
                "core_state_frame_index": int(frame_index),
                "pair_is_core_clean": True,
                "main_loss_weight": 1.0,
            }
        )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


class PushTCoreTemporalHardNegativeInfoNCEGroupHDF5Dataset(Dataset[Sample]):
    """Group nearby same-episode PushT core-clean pairs as hard negatives.

    Each contrastive group is one local temporal window from a single source
    variant and core color.  The diagonal target is the aligned clean frame; the
    other rows in the same local window are the only negatives used by the
    existing group InfoNCE loss.
    """

    def __init__(
        self,
        data_dirs: list[str | Path | dict[str, Any]],
        target_data_dir: str | Path | dict[str, Any],
        *,
        variant_names: list[str] | None = None,
        colors: list[str] | tuple[str, ...] | None = None,
        query: str = "gray T block",
        query_metadata_path: str | Path | None = None,
        query_template: str = "{color} T block",
        num_frames: int = 1,
        num_hard_negatives: int = 0,
        temporal_offsets: list[int] | tuple[int, ...] | None = None,
        center_frame_stride: int = 5,
        max_groups_per_color: int | None = None,
        shuffle_groups: bool = True,
        balance_seed: int = 6171,
        drop_last_groups_to_multiple: int = 4,
        max_open_files: int = 16,
        **_unused: Any,
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
            raise ValueError("variant_names length must match data_dirs")
        self.colors = tuple(PUSHT_COLORS if colors is None else [str(c) for c in colors])
        if len(set(self.colors)) != len(self.colors):
            raise ValueError(f"Duplicate colors are not allowed: {self.colors}")
        offsets = [-2, -1, 0, 1, 2] if temporal_offsets is None else [int(v) for v in temporal_offsets]
        if len(offsets) < 2:
            raise ValueError("temporal_offsets must contain at least two offsets")
        if len(set(offsets)) != len(offsets):
            raise ValueError(f"Duplicate temporal offsets are not allowed: {offsets}")
        if 0 not in offsets:
            raise ValueError("temporal_offsets must include 0 for the anchor frame")
        self.temporal_offsets = tuple(offsets)
        self.center_frame_stride = max(1, int(center_frame_stride))
        self.max_groups_per_color = (
            None if max_groups_per_color is None else int(max_groups_per_color)
        )
        if self.max_groups_per_color is not None and self.max_groups_per_color <= 0:
            raise ValueError("max_groups_per_color must be positive when provided")
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
                max_open_files=max_open_files,
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
            raise ValueError("query_metadata_path is required for temporal hard-negative groups")
        unknown = sorted({str(c) for c in color_by_episode.values()} - set(self.colors))
        if unknown:
            raise ValueError(
                f"Metadata contains colors outside configured PushT colors: {unknown}"
            )

        self._groups, self._color_counts = self._build_temporal_groups(
            self.children[0],
            color_by_episode,
        )
        self._group_size = len(self.temporal_offsets)
        self._base_length = int(self._groups.shape[0])
        self._length = self._base_length * self._group_size
        if self._base_length <= 0:
            raise ValueError("No PushT temporal hard-negative groups selected")

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

    def _build_temporal_groups(
        self,
        reference_child: PushTPairedHDF5Dataset,
        color_by_episode: dict[int, str],
    ) -> tuple[np.ndarray, dict[str, int]]:
        rows_by_color_episode: dict[str, dict[int, list[tuple[int, int]]]] = {
            color: {} for color in self.colors
        }
        source_manifest = getattr(reference_child, "_source_manifest")
        step_starts = getattr(reference_child, "_step_starts")

        for shard_idx, shard in enumerate(source_manifest.get("shards", [])):
            shard_path = reference_child._manifest_shard_file(
                reference_child.data_dir, shard
            )
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
                color = str(color_by_episode.get(int(episode_idx), ""))
                if color not in rows_by_color_episode:
                    continue
                rows_by_color_episode[color].setdefault(int(episode_idx), []).append(
                    (global_offset + int(local_index), int(step_idx))
                )

        rng = np.random.default_rng(self.balance_seed)
        groups: list[np.ndarray] = []
        color_counts: dict[str, int] = {}
        left = max(0, -min(self.temporal_offsets))
        right = max(0, max(self.temporal_offsets))
        for color_idx, color in enumerate(self.colors):
            candidates: list[list[int]] = []
            available_rows = 0
            for episode_idx in sorted(rows_by_color_episode[color]):
                rows = sorted(rows_by_color_episode[color][episode_idx], key=lambda item: item[1])
                available_rows += len(rows)
                if len(rows) <= left + right:
                    continue
                for center_pos in range(left, len(rows) - right, self.center_frame_stride):
                    candidates.append(
                        [int(rows[center_pos + offset][0]) for offset in self.temporal_offsets]
                    )
            color_counts[color] = int(available_rows)
            if not candidates:
                continue
            order = np.arange(len(candidates), dtype=np.int64)
            if self.shuffle_groups:
                order = rng.permutation(order)
            if self.max_groups_per_color is not None:
                order = order[: self.max_groups_per_color]
            for local_order, candidate_idx in enumerate(order.tolist()):
                variant_idx = int(len(groups) % len(self.children))
                groups.append(
                    np.asarray(
                        [int(color_idx), variant_idx, *candidates[int(candidate_idx)]],
                        dtype=np.int64,
                    )
                )
        if not groups:
            raise ValueError("No temporal hard-negative groups remain after filtering")
        all_groups = np.stack(groups, axis=0).astype(np.int64, copy=False)
        if self.shuffle_groups:
            all_groups = all_groups[rng.permutation(all_groups.shape[0])]
        if self.drop_last_groups_to_multiple > 1:
            keep = all_groups.shape[0] - (all_groups.shape[0] % self.drop_last_groups_to_multiple)
            all_groups = all_groups[:keep]
        if all_groups.shape[0] <= 0:
            raise ValueError("Temporal hard-negative groups dropped to zero")
        return all_groups, color_counts

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
            raise IndexError("PushTCoreTemporalHardNegativeInfoNCEGroupHDF5Dataset index out of range")
        group_idx = index // self._group_size
        slot = index % self._group_size
        group = self._groups[group_idx]
        color_idx = int(group[0])
        variant_idx = int(group[1])
        frame_index = int(group[2 + slot])
        expected_color = self.colors[color_idx]

        sample = self.children[variant_idx][frame_index]
        expected_query = self.query_template.format(color=expected_color)
        if sample.query != expected_query:
            raise ValueError(
                "Temporal hard-negative group resolved to the wrong query: "
                f"expected {expected_query!r}, got {sample.query!r}, "
                f"group_idx={group_idx}, frame_index={frame_index}"
            )
        meta = dict(sample.meta)
        meta.update(
            {
                "id": (
                    f"PushTCoreTemporalHardNeg:{expected_color}:"
                    f"{self.variant_names[variant_idx]}:{group_idx}:{slot}:{meta.get('id', index)}"
                ),
                "real_task": meta.get("task"),
                "task": "pusht_source5_query",
                "shard_idx": f"core_temporal_hardneg_{expected_color}_{self.variant_names[variant_idx]}",
                "source_local_index": int(group_idx),
                "core_temporal_group_idx": int(group_idx),
                "core_temporal_slot": int(slot),
                "core_temporal_group_size": int(self._group_size),
                "core_temporal_color": expected_color,
                "core_temporal_color_idx": int(color_idx),
                "core_temporal_variant": self.variant_names[variant_idx],
                "core_temporal_variant_idx": int(variant_idx),
                "core_temporal_frame_index": int(frame_index),
                "core_temporal_offset": int(self.temporal_offsets[slot]),
                "pair_is_core_clean": True,
                "main_loss_weight": 0.0,
            }
        )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


class PushTCoreTemporalSource5QueryHardNegativeInfoNCEMixHDF5Dataset(Dataset[Sample]):
    """Mix temporal core hard negatives with one-source five-query groups."""

    def __init__(
        self,
        *,
        core_data_dirs: list[str | Path | dict[str, Any]],
        core_target_data_dir: str | Path | dict[str, Any],
        source5_data_dir: str | Path | dict[str, Any] | None = None,
        core_variant_names: list[str] | None = None,
        core_colors: list[str] | tuple[str, ...] | None = None,
        core_query: str = "gray T block",
        core_query_metadata_path: str | Path | None = None,
        core_query_template: str = "{color} T block",
        core_temporal_offsets: list[int] | tuple[int, ...] | None = None,
        core_center_frame_stride: int = 5,
        core_shuffle_groups: bool = True,
        core_balance_seed: int = 6171,
        source5_frame_stride: int = 1,
        source5_query_group_size: int = 5,
        core_groups_per_super: int = 9,
        source5_groups_per_super: int = 1,
        max_supergroups: int | None = None,
        **_unused: Any,
    ) -> None:
        self.core = PushTCoreTemporalHardNegativeInfoNCEGroupHDF5Dataset(
            data_dirs=core_data_dirs,
            target_data_dir=core_target_data_dir,
            variant_names=core_variant_names,
            colors=core_colors,
            query=core_query,
            query_metadata_path=core_query_metadata_path,
            query_template=core_query_template,
            temporal_offsets=core_temporal_offsets,
            center_frame_stride=core_center_frame_stride,
            shuffle_groups=core_shuffle_groups,
            balance_seed=core_balance_seed,
            drop_last_groups_to_multiple=4,
        )
        self.source5 = PushTSource5QueryHDF5Dataset(
            data_dir=source5_data_dir,
            frame_stride=source5_frame_stride,
            query_group_size=source5_query_group_size,
        )
        self.core_groups_per_super = int(core_groups_per_super)
        self.source5_groups_per_super = int(source5_groups_per_super)
        if self.core_groups_per_super <= 0 or self.source5_groups_per_super <= 0:
            raise ValueError("group counts per super-group must be positive")
        self.core_group_size = int(self.core.inbatch_group_size)
        self.source5_group_size = int(self.source5.inbatch_group_size)
        self.core_slots = self.core_groups_per_super * self.core_group_size
        self.source5_slots = self.source5_groups_per_super * self.source5_group_size
        self._group_size = self.core_slots + self.source5_slots
        base_length = min(
            int(self.core.base_length) // self.core_groups_per_super,
            int(self.source5.base_length) // self.source5_groups_per_super,
        )
        if max_supergroups is not None:
            base_length = min(base_length, int(max_supergroups))
        if base_length <= 0:
            raise ValueError("No complete temporal hard-negative mixed super-groups available")
        self._base_length = base_length
        self._length = self._base_length * self._group_size

    def __len__(self) -> int:
        return self._length

    @property
    def base_length(self) -> int:
        return self._base_length

    @property
    def inbatch_group_size(self) -> int:
        return self._group_size

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError(
                "PushTCoreTemporalSource5QueryHardNegativeInfoNCEMixHDF5Dataset "
                "index out of range"
            )
        super_idx = index // self._group_size
        slot = index % self._group_size
        if slot < self.core_slots:
            group_offset = slot // self.core_group_size
            inner = slot % self.core_group_size
            core_group_idx = super_idx * self.core_groups_per_super + group_offset
            child_index = core_group_idx * self.core_group_size + inner
            sample = self.core[child_index]
            meta = dict(sample.meta)
            meta.update(
                {
                    "mix_family": "core_temporal_hardneg",
                    "mix_super_idx": int(super_idx),
                    "source_local_index": int(core_group_idx),
                    "main_loss_weight": 0.0,
                }
            )
        else:
            local_slot = slot - self.core_slots
            group_offset = local_slot // self.source5_group_size
            inner = local_slot % self.source5_group_size
            source5_group_idx = super_idx * self.source5_groups_per_super + group_offset
            child_index = source5_group_idx * self.source5_group_size + inner
            if self.source5 is None:
                raise RuntimeError("source5 branch reached but source5 dataset is disabled")
            sample = self.source5[child_index]
            meta = dict(sample.meta)
            meta.update(
                {
                    "real_shard_idx": meta.get("shard_idx"),
                    "real_source_local_index": meta.get("source_local_index"),
                    "mix_family": "source5_query_hardneg",
                    "mix_super_idx": int(super_idx),
                    "shard_idx": "source5_query_hardneg",
                    "source_local_index": int(source5_group_idx),
                    "main_loss_weight": 0.0,
                }
            )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


class PushTCoreTemporalSource5QueryCleanBankInfoNCEMixHDF5Dataset(
    PushTCoreTemporalSource5QueryHardNegativeInfoNCEMixHDF5Dataset
):
    """Use each mixed super-group as one large clean-bank InfoNCE group.

    The parent dataset still decides which core temporal groups and source5
    query groups enter a super-group.  This thin wrapper only changes the
    contrastive grouping metadata so the existing trainer compares every row
    against all targets in the super-group instead of separate local 5-way
    groups.
    """

    def __getitem__(self, index: int) -> Sample:
        sample = super().__getitem__(index)
        super_idx = index // self._group_size
        slot = index % self._group_size
        meta = dict(sample.meta)
        contrast_super_idx = int(super_idx) // int(getattr(self, "contrast_supergroup_merge_factor", 1))
        meta.update(
            {
                "real_cleanbank_shard_idx": meta.get("shard_idx"),
                "real_cleanbank_source_local_index": meta.get("source_local_index"),
                "cleanbank_super_idx": int(super_idx),
                "cleanbank_slot": int(slot),
                "cleanbank_group_size": int(self._group_size),
                "shard_idx": f"cleanbank{int(self._group_size)}",
                "source_local_index": int(super_idx),
                "main_loss_weight": 0.0,
            }
        )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


class PushTCoreTemporalSource5QueryExactCleanBankInfoNCEMixHDF5Dataset(Dataset[Sample]):
    """Exact-size clean-bank InfoNCE mix for core-clean plus optional query support.

    Unlike the older super-group wrapper, this adapter is not constrained to a
    multiple of the child 5-way groups. Each bank contains exactly
    ``cleanbank_size`` rows, with ``core_rows_per_bank`` aligned core rows and
    ``source5_rows_per_bank`` source5/query rows. The whole bank shares one
    contrast group so every row sees the remaining bank targets as negatives.
    """

    def __init__(
        self,
        *,
        core_data_dirs: list[str | Path | dict[str, Any]],
        core_target_data_dir: str | Path | dict[str, Any],
        source5_data_dir: str | Path | dict[str, Any] | None = None,
        core_variant_names: list[str] | None = None,
        core_colors: list[str] | tuple[str, ...] | None = None,
        core_query: str = "gray T block",
        core_query_metadata_path: str | Path | None = None,
        core_query_template: str = "{color} T block",
        core_temporal_offsets: list[int] | tuple[int, ...] | None = None,
        core_center_frame_stride: int = 5,
        core_shuffle_groups: bool = True,
        core_balance_seed: int = 6171,
        source5_frame_stride: int = 1,
        source5_query_group_size: int = 5,
        cleanbank_size: int = 256,
        core_rows_per_bank: int = 240,
        source5_rows_per_bank: int = 16,
        max_supergroups: int | None = None,
        contrast_supergroup_merge_factor: int = 1,
        max_open_files: int = 16,
        **_unused: Any,
    ) -> None:
        self.core = PushTCoreTemporalHardNegativeInfoNCEGroupHDF5Dataset(
            data_dirs=core_data_dirs,
            target_data_dir=core_target_data_dir,
            variant_names=core_variant_names,
            colors=core_colors,
            query=core_query,
            query_metadata_path=core_query_metadata_path,
            query_template=core_query_template,
            temporal_offsets=core_temporal_offsets,
            center_frame_stride=core_center_frame_stride,
            shuffle_groups=core_shuffle_groups,
            balance_seed=core_balance_seed,
            drop_last_groups_to_multiple=4,
            max_open_files=max_open_files,
        )
        self.source5 = None
        self.cleanbank_size = int(cleanbank_size)
        self.core_rows_per_bank = int(core_rows_per_bank)
        self.source5_rows_per_bank = int(source5_rows_per_bank)
        self.contrast_supergroup_merge_factor = max(1, int(contrast_supergroup_merge_factor))
        if self.cleanbank_size <= 1:
            raise ValueError("cleanbank_size must be at least 2")
        if self.core_rows_per_bank < 0 or self.source5_rows_per_bank < 0:
            raise ValueError("rows per bank must be non-negative")
        if self.core_rows_per_bank + self.source5_rows_per_bank != self.cleanbank_size:
            raise ValueError(
                "core_rows_per_bank + source5_rows_per_bank must equal cleanbank_size"
            )
        if self.core_rows_per_bank <= 0:
            raise ValueError("core_rows_per_bank must be positive for source->clean alignment")
        if self.source5_rows_per_bank > 0:
            if source5_data_dir is None:
                raise ValueError("source5_data_dir is required when source5_rows_per_bank > 0")
            self.source5 = PushTSource5QueryHDF5Dataset(
                data_dir=source5_data_dir,
                frame_stride=source5_frame_stride,
                query_group_size=source5_query_group_size,
                max_open_files=max_open_files,
            )

        base_length = len(self.core) // self.core_rows_per_bank
        if self.source5_rows_per_bank > 0:
            if self.source5 is None:
                raise ValueError("source5 dataset was not initialized")
            base_length = min(
                base_length,
                len(self.source5) // self.source5_rows_per_bank,
            )
        if max_supergroups is not None:
            base_length = min(base_length, int(max_supergroups))
        if base_length <= 0:
            raise ValueError("No complete exact clean-bank groups available")
        self._base_length = base_length
        self._group_size = self.cleanbank_size
        self._length = self._base_length * self._group_size

    def __len__(self) -> int:
        return self._length

    @property
    def base_length(self) -> int:
        return self._base_length

    @property
    def inbatch_group_size(self) -> int:
        return self._group_size

    def _core_positive_key(self, meta: dict[str, Any]) -> str:
        color = meta.get("core_temporal_color") or meta.get("query") or "unknown"
        frame = meta.get("core_temporal_frame_index", meta.get("id", "unknown"))
        return f"coreclean:{color}:{frame}"

    def _source5_positive_key(self, meta: dict[str, Any]) -> str:
        shard = meta.get("shard_idx", "unknown")
        target = meta.get("target_local_index", meta.get("id", "unknown"))
        color = meta.get("target_color", meta.get("query", "unknown"))
        return f"source5target:{shard}:{target}:{color}"

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError(
                "PushTCoreTemporalSource5QueryExactCleanBankInfoNCEMixHDF5Dataset "
                "index out of range"
            )
        super_idx = index // self._group_size
        slot = index % self._group_size
        if slot < self.core_rows_per_bank:
            child_index = super_idx * self.core_rows_per_bank + slot
            sample = self.core[child_index]
            meta = dict(sample.meta)
            positive_key = self._core_positive_key(meta)
            family = "core_clean_exactbank"
        else:
            local_slot = slot - self.core_rows_per_bank
            child_index = super_idx * self.source5_rows_per_bank + local_slot
            sample = self.source5[child_index]
            meta = dict(sample.meta)
            positive_key = self._source5_positive_key(meta)
            family = "source5_query_exactbank"

        contrast_super_idx = int(super_idx) // int(self.contrast_supergroup_merge_factor)
        meta.update(
            {
                "real_cleanbank_shard_idx": meta.get("shard_idx"),
                "real_cleanbank_source_local_index": meta.get("source_local_index"),
                "exact_cleanbank_child_index": int(child_index),
                "mix_family": family,
                "cleanbank_super_idx": int(super_idx),
                "cleanbank_slot": int(slot),
                "cleanbank_group_size": int(self._group_size),
                "cleanbank_core_rows": int(self.core_rows_per_bank),
                "cleanbank_source5_rows": int(self.source5_rows_per_bank),
                "cleanbank_contrast_super_idx": int(contrast_super_idx),
                "cleanbank_contrast_merge_factor": int(self.contrast_supergroup_merge_factor),
                "task": "pusht_source5_query",
                "shard_idx": f"cleanbank{int(self._group_size)}",
                "source_local_index": int(contrast_super_idx),
                "contrast_positive_key": positive_key,
                "main_loss_weight": 0.0,
            }
        )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


class PushTCoreTemporalAnchoredSource5QueryExactCleanBankInfoNCEMixHDF5Dataset(
    PushTCoreTemporalSource5QueryExactCleanBankInfoNCEMixHDF5Dataset
):
    """Exact clean-bank mix with source5 groups anchored to bank core colors.

    The parent exact-bank adapter mixes core rows and source5 rows by absolute
    order. This variant keeps the same loss and bank shape, but chooses each
    source5 5-way group from scenes whose scene_core_color matches an
    evenly-spaced core row in the same bank. It is a data-routing change only:
    no architecture or loss changes.
    """

    def __init__(
        self,
        *,
        source5_anchor_seed: int = 20260621,
        source5_anchor_mode: str = "bank_core_color",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if self.source5 is None or self.source5_rows_per_bank <= 0:
            raise ValueError("anchored source5 requires source5_rows_per_bank > 0")
        if self.source5_rows_per_bank % int(self.source5.inbatch_group_size) != 0:
            raise ValueError(
                "source5_rows_per_bank must be a multiple of the source5 query group size"
            )
        self.source5_anchor_seed = int(source5_anchor_seed)
        self.source5_anchor_mode = str(source5_anchor_mode)
        if self.source5_anchor_mode != "bank_core_color":
            raise ValueError(f"Unsupported source5_anchor_mode: {self.source5_anchor_mode!r}")
        self.source5_groups_per_bank = int(self.source5_rows_per_bank) // int(
            self.source5.inbatch_group_size
        )
        self._source5_units_by_color = self._build_source5_units_by_core_color()
        missing = [str(c) for c in self.core.colors if str(c) not in self._source5_units_by_color]
        if missing:
            raise ValueError(f"source5 anchor colors missing from source5 data: {missing}")

    def _build_source5_units_by_core_color(self) -> dict[str, np.ndarray]:
        groups: dict[str, list[int]] = {str(color): [] for color in self.core.colors}
        for source_unit, ref in enumerate(getattr(self.source5, "_source_refs")):
            shard_idx = int(ref[0])
            scene_idx = int(ref[1])
            shard_index = self.source5._load_index(shard_idx)
            color = str(shard_index["scene_core_color"][scene_idx])
            if color in groups:
                groups[color].append(int(source_unit))
        return {
            color: np.asarray(units, dtype=np.int64)
            for color, units in groups.items()
            if units
        }

    def _bank_core_color_for_source5_group(self, super_idx: int, group_offset: int) -> str:
        anchor_slot = int((int(group_offset) * int(self.core_rows_per_bank)) // max(1, self.source5_groups_per_bank))
        anchor_slot = min(max(anchor_slot, 0), int(self.core_rows_per_bank) - 1)
        core_child_index = int(super_idx) * int(self.core_rows_per_bank) + anchor_slot
        core_group_size = int(self.core.inbatch_group_size)
        core_group_idx = core_child_index // core_group_size
        group = self.core._groups[int(core_group_idx)]
        color_idx = int(group[0])
        return str(self.core.colors[color_idx])

    def _anchored_source5_child_index(self, super_idx: int, local_slot: int) -> int:
        query_group_size = int(self.source5.inbatch_group_size)
        group_offset = int(local_slot) // query_group_size
        query_slot = int(local_slot) % query_group_size
        color = self._bank_core_color_for_source5_group(int(super_idx), int(group_offset))
        units = self._source5_units_by_color[color]
        unit_pos = (
            int(super_idx) * 104729
            + int(group_offset) * 9176
            + self.source5_anchor_seed
        ) % int(units.shape[0])
        source_unit = int(units[int(unit_pos)])
        return source_unit * query_group_size + query_slot

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError(
                "PushTCoreTemporalAnchoredSource5QueryExactCleanBankInfoNCEMixHDF5Dataset "
                "index out of range"
            )
        super_idx = index // self._group_size
        slot = index % self._group_size
        if slot < self.core_rows_per_bank:
            child_index = super_idx * self.core_rows_per_bank + slot
            sample = self.core[child_index]
            meta = dict(sample.meta)
            positive_key = self._core_positive_key(meta)
            family = "core_clean_exactbank"
            anchor_color = meta.get("core_temporal_color")
        else:
            local_slot = slot - self.core_rows_per_bank
            child_index = self._anchored_source5_child_index(int(super_idx), int(local_slot))
            sample = self.source5[child_index]
            meta = dict(sample.meta)
            positive_key = self._source5_positive_key(meta)
            family = "source5_query_exactbank_anchored"
            anchor_color = self._bank_core_color_for_source5_group(
                int(super_idx), int(local_slot) // int(self.source5.inbatch_group_size)
            )

        contrast_super_idx = int(super_idx) // int(self.contrast_supergroup_merge_factor)
        meta.update(
            {
                "real_cleanbank_shard_idx": meta.get("shard_idx"),
                "real_cleanbank_source_local_index": meta.get("source_local_index"),
                "exact_cleanbank_child_index": int(child_index),
                "mix_family": family,
                "cleanbank_super_idx": int(super_idx),
                "cleanbank_slot": int(slot),
                "cleanbank_group_size": int(self._group_size),
                "cleanbank_core_rows": int(self.core_rows_per_bank),
                "cleanbank_source5_rows": int(self.source5_rows_per_bank),
                "cleanbank_contrast_super_idx": int(contrast_super_idx),
                "cleanbank_contrast_merge_factor": int(self.contrast_supergroup_merge_factor),
                "source5_anchor_color": None if anchor_color is None else str(anchor_color),
                "source5_anchor_mode": self.source5_anchor_mode,
                "task": "pusht_source5_query",
                "shard_idx": f"anchored_cleanbank{int(self._group_size)}",
                "source_local_index": int(contrast_super_idx),
                "contrast_positive_key": positive_key,
                "main_loss_weight": 0.0,
            }
        )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


class PushTCoreTemporalAnchoredQuerySupportExactCleanBankInfoNCEMixHDF5Dataset(
    PushTCoreTemporalAnchoredSource5QueryExactCleanBankInfoNCEMixHDF5Dataset
):
    """Manifest-driven N-query alias of the audited anchored Source5 route.

    This wrapper only generalizes the external data vocabulary. The clean-bank
    construction, temporal hard negatives, anchored sampling, and loss-facing
    metadata remain delegated to the existing Source5 implementation.
    """

    def __init__(
        self,
        *,
        query_support_data_dir: str | Path | dict[str, Any],
        query_support_frame_stride: int = 1,
        query_support_query_group_size: int = 5,
        bank_size: int = 512,
        query_ratio: float = 0.10,
        query_support_anchor_seed: int = 20260621,
        query_support_anchor_mode: str = "bank_core_color",
        **kwargs: Any,
    ) -> None:
        bank_size, core_rows, query_rows = resolve_cleanbank_rows(
            bank_size,
            query_ratio,
            query_support_query_group_size,
        )
        super().__init__(
            source5_data_dir=query_support_data_dir,
            source5_frame_stride=query_support_frame_stride,
            source5_query_group_size=query_support_query_group_size,
            cleanbank_size=bank_size,
            core_rows_per_bank=core_rows,
            source5_rows_per_bank=query_rows,
            source5_anchor_seed=query_support_anchor_seed,
            source5_anchor_mode=query_support_anchor_mode,
            **kwargs,
        )
        self.query_support = self.source5
        self.query_support_rows_per_bank = self.source5_rows_per_bank
        self.query_support_anchor_seed = self.source5_anchor_seed
        self.query_support_anchor_mode = self.source5_anchor_mode

    def __getitem__(self, index: int) -> Sample:
        sample = super().__getitem__(index)
        meta = dict(sample.meta)
        family = str(meta.get("mix_family", ""))
        if family.startswith("source5_query"):
            meta["mix_family"] = family.replace("source5_query", "query_support", 1)
        meta.update(
            {
                "cleanbank_query_support_rows": int(self.query_support_rows_per_bank),
                "query_support_anchor_color": meta.get("source5_anchor_color"),
                "query_support_anchor_mode": self.query_support_anchor_mode,
                "query_support_group_size": int(self.query_support.inbatch_group_size),
                "task": "pusht_query_support",
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


class PushTCoreSource5MultiStateInfoNCEMixHDF5Dataset(Dataset[Sample]):
    """Mix core-clean InfoNCE groups with multi-state Source5 query groups."""

    def __init__(
        self,
        *,
        core_data_dirs: list[str | Path | dict[str, Any]],
        core_target_data_dir: str | Path | dict[str, Any],
        source5_data_dir: str | Path | dict[str, Any],
        core_variant_names: list[str] | None = None,
        core_colors: list[str] | tuple[str, ...] | None = None,
        core_query: str = "gray T block",
        core_query_metadata_path: str | Path | None = None,
        core_query_template: str = "{color} T block",
        core_frame_stride: int = 5,
        core_shuffle_groups: bool = True,
        core_balance_seed: int = 3072,
        source5_frame_stride: int = 1,
        source5_query_group_size: int = 5,
        source5_state_group_size: int = 4,
        core_groups_per_super: int = 2,
        source5_groups_per_super: int = 2,
        max_supergroups: int | None = None,
        **_unused: Any,
    ) -> None:
        self.core = PushTCoreColorInfoNCEBagHDF5Dataset(
            data_dirs=core_data_dirs,
            target_data_dir=core_target_data_dir,
            variant_names=core_variant_names,
            colors=core_colors,
            query=core_query,
            query_metadata_path=core_query_metadata_path,
            query_template=core_query_template,
            frame_stride=core_frame_stride,
            shuffle_groups=core_shuffle_groups,
            balance_seed=core_balance_seed,
            drop_last_groups_to_multiple=4,
        )
        self.source5 = PushTSource5MultiStateQueryHDF5Dataset(
            data_dir=source5_data_dir,
            frame_stride=source5_frame_stride,
            query_group_size=source5_query_group_size,
            state_group_size=source5_state_group_size,
        )
        self.core_groups_per_super = int(core_groups_per_super)
        self.source5_groups_per_super = int(source5_groups_per_super)
        if self.core_groups_per_super <= 0 or self.source5_groups_per_super <= 0:
            raise ValueError("group counts per super-group must be positive")
        self.core_group_size = int(self.core.inbatch_group_size)
        self.source5_group_size = int(self.source5.inbatch_group_size)
        self.core_slots = self.core_groups_per_super * self.core_group_size
        self.source5_slots = self.source5_groups_per_super * self.source5_group_size
        self._group_size = self.core_slots + self.source5_slots
        base_length = min(
            int(self.core.base_length) // self.core_groups_per_super,
            int(self.source5.base_length) // self.source5_groups_per_super,
        )
        if max_supergroups is not None:
            base_length = min(base_length, int(max_supergroups))
        if base_length <= 0:
            raise ValueError("No complete mixed InfoNCE super-groups available")
        self._base_length = base_length
        self._length = self._base_length * self._group_size

    def __len__(self) -> int:
        return self._length

    @property
    def base_length(self) -> int:
        return self._base_length

    @property
    def inbatch_group_size(self) -> int:
        return self._group_size

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError("PushTCoreSource5MultiStateInfoNCEMixHDF5Dataset index out of range")
        super_idx = index // self._group_size
        slot = index % self._group_size
        if slot < self.core_slots:
            group_offset = slot // self.core_group_size
            inner = slot % self.core_group_size
            child_index = (
                (super_idx * self.core_groups_per_super + group_offset)
                * self.core_group_size
                + inner
            )
            sample = self.core[child_index]
            meta = dict(sample.meta)
            meta.update(
                {
                    "mix_family": "core_infonce",
                    "mix_super_idx": int(super_idx),
                    "source_local_index": int(super_idx * self.core_groups_per_super + group_offset),
                }
            )
        else:
            local_slot = slot - self.core_slots
            group_offset = local_slot // self.source5_group_size
            inner = local_slot % self.source5_group_size
            child_index = (
                (super_idx * self.source5_groups_per_super + group_offset)
                * self.source5_group_size
                + inner
            )
            sample = self.source5[child_index]
            meta = dict(sample.meta)
            meta.update(
                {
                    "mix_family": "source5_multistate",
                    "mix_super_idx": int(super_idx),
                    "source_local_index": int(super_idx * self.source5_groups_per_super + group_offset),
                }
            )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


class PushTCoreSupConSource5MultiStateInfoNCEMixHDF5Dataset(Dataset[Sample]):
    """Mix same-clean SupCon core groups with multi-state Source5 query groups.

    The core side emits multiple D1-D5 source variants for the same underlying
    clean target and marks them as multi-positive rows. The Source5 side keeps
    the strict same-source multi-query negatives. This is still the existing
    same-source contrastive loss; only the in-batch grouping changes.
    """

    def __init__(
        self,
        *,
        core_data_dirs: list[str | Path | dict[str, Any]],
        core_target_data_dir: str | Path | dict[str, Any],
        source5_data_dir: str | Path | dict[str, Any],
        core_variant_names: list[str] | None = None,
        core_colors: list[str] | tuple[str, ...] | None = None,
        core_query: str = "gray T block",
        core_query_metadata_path: str | Path | None = None,
        core_query_template: str = "{color} T block",
        core_frame_stride: int = 5,
        core_shuffle_groups: bool = True,
        core_balance_seed: int = 5074,
        core_states_per_group: int = 5,
        source5_frame_stride: int = 1,
        source5_query_group_size: int = 5,
        source5_state_group_size: int = 4,
        core_groups_per_super: int = 6,
        source5_groups_per_super: int = 1,
        max_supergroups: int | None = None,
        **_unused: Any,
    ) -> None:
        self.core = PushTCoreStateSupConGroupHDF5Dataset(
            data_dirs=core_data_dirs,
            target_data_dir=core_target_data_dir,
            variant_names=core_variant_names,
            colors=core_colors,
            query=core_query,
            query_metadata_path=core_query_metadata_path,
            query_template=core_query_template,
            frame_stride=core_frame_stride,
            states_per_group=core_states_per_group,
            shuffle_groups=core_shuffle_groups,
            balance_seed=core_balance_seed,
            drop_last_groups_to_multiple=4,
        )
        self.source5 = PushTSource5MultiStateQueryHDF5Dataset(
            data_dir=source5_data_dir,
            frame_stride=source5_frame_stride,
            query_group_size=source5_query_group_size,
            state_group_size=source5_state_group_size,
        )
        self.core_groups_per_super = int(core_groups_per_super)
        self.source5_groups_per_super = int(source5_groups_per_super)
        if self.core_groups_per_super <= 0 or self.source5_groups_per_super <= 0:
            raise ValueError("group counts per super-group must be positive")
        self.core_group_size = int(self.core.inbatch_group_size)
        self.source5_group_size = int(self.source5.inbatch_group_size)
        self.core_slots = self.core_groups_per_super * self.core_group_size
        self.source5_slots = self.source5_groups_per_super * self.source5_group_size
        self._group_size = self.core_slots + self.source5_slots
        base_length = min(
            int(self.core.base_length) // self.core_groups_per_super,
            int(self.source5.base_length) // self.source5_groups_per_super,
        )
        if max_supergroups is not None:
            base_length = min(base_length, int(max_supergroups))
        if base_length <= 0:
            raise ValueError("No complete SupCon+Source5 super-groups available")
        self._base_length = base_length
        self._length = self._base_length * self._group_size

    def __len__(self) -> int:
        return self._length

    @property
    def base_length(self) -> int:
        return self._base_length

    @property
    def inbatch_group_size(self) -> int:
        return self._group_size

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError(
                "PushTCoreSupConSource5MultiStateInfoNCEMixHDF5Dataset "
                "index out of range"
            )
        super_idx = index // self._group_size
        slot = index % self._group_size
        if slot < self.core_slots:
            group_offset = slot // self.core_group_size
            inner = slot % self.core_group_size
            core_group_idx = super_idx * self.core_groups_per_super + group_offset
            child_index = core_group_idx * self.core_group_size + inner
            sample = self.core[child_index]
            meta = dict(sample.meta)
            meta.update(
                {
                    "mix_family": "core_state_supcon",
                    "mix_super_idx": int(super_idx),
                    "source_local_index": int(core_group_idx),
                }
            )
        else:
            local_slot = slot - self.core_slots
            group_offset = local_slot // self.source5_group_size
            inner = local_slot % self.source5_group_size
            source5_group_idx = (
                super_idx * self.source5_groups_per_super + group_offset
            )
            child_index = source5_group_idx * self.source5_group_size + inner
            sample = self.source5[child_index]
            meta = dict(sample.meta)
            meta.update(
                {
                    "mix_family": "source5_multistate",
                    "mix_super_idx": int(super_idx),
                    "source_local_index": int(source5_group_idx),
                }
            )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


__all__ = [
    "PushTSource5MultiStateQueryHDF5Dataset",
    "PushTCoreColorInfoNCEBagHDF5Dataset",
    "PushTCoreStateInfoNCEGroupHDF5Dataset",
    "PushTCoreStateSupConGroupHDF5Dataset",
    "PushTCoreTemporalHardNegativeInfoNCEGroupHDF5Dataset",
    "PushTCoreTemporalSource5QueryHardNegativeInfoNCEMixHDF5Dataset",
    "PushTCoreTemporalSource5QueryCleanBankInfoNCEMixHDF5Dataset",
    "PushTCoreSource5MultiStateInfoNCEMixHDF5Dataset",
    "PushTCoreSupConSource5MultiStateInfoNCEMixHDF5Dataset",
]
