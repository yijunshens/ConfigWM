"""Push-T-style Configurable Encoder clean banks for the OGB-Cube 9-color dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ...schema import Sample
from ...cleanbank import resolve_cleanbank_rows
from .ogbcube_pairs import (
    OGBCubeCoreCleanHDF5Dataset,
    OGBCubeSameSourceQuery4HDF5Dataset,
    _empty_label_map,
    _load_jsonl,
    _shard_sort_key,
)
from .temporal import (
    build_trajectory_windows,
    round_robin_variant_indices,
    validate_temporal_contract,
)


class OGBCubePushTStyleTemporalCoreCleanHDF5Dataset(
    OGBCubeCoreCleanHDF5Dataset
):
    """D0-DK -> D0 with one round-robin variant per raw temporal window."""

    def __init__(
        self,
        data_root: str | Path | dict[str, Any],
        *,
        source_variants: list[str] | tuple[str, ...],
        target_variant: str = "d0",
        temporal_offsets: list[int] | tuple[int, ...] = (-2, -1, 0, 1, 2),
        temporal_center_stride: int = 5,
        query: str = "arm",
        query_template: str = "{color} arm",
        require_query_metadata: bool = True,
        split_manifest_path: str | Path | None = None,
        shuffle_refs: bool = True,
        balance_seed: int = 6274,
        max_open_files: int = 16,
        **_unused: Any,
    ) -> None:
        self.temporal_offsets, self.temporal_center_stride = (
            validate_temporal_contract(temporal_offsets, temporal_center_stride)
        )
        self._temporal_group_size = len(self.temporal_offsets)
        super().__init__(
            data_root,
            source_variants=source_variants,
            target_variant=target_variant,
            num_frames=1,
            # The shared contract is defined in raw-frame units.
            frame_stride=1,
            temporal_offsets=self.temporal_offsets,
            temporal_center_stride=self.temporal_center_stride,
            query=query,
            query_template=query_template,
            require_query_metadata=require_query_metadata,
            split_manifest_path=split_manifest_path,
            shuffle_refs=shuffle_refs,
            balance_seed=balance_seed,
            max_open_files=max_open_files,
        )

    def _load_core_query_colors(
        self, _query_data_dir: str | Path | dict[str, Any] | None
    ) -> tuple[dict[Any, str], dict[Any, list[int]], dict[Any, dict[str, Any]]]:
        """Load episode metadata once instead of materializing a map per frame."""
        self._metadata_by_shard: dict[
            str, tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]
        ] = {}
        self._ep_offsets_by_shard_idx: dict[int, np.ndarray] = {}
        for shard in self._shards:
            shard_name = str(shard["source_shard"])
            if shard_name in self._metadata_by_shard:
                continue
            variant = str(shard["source_variant"])
            records = _load_jsonl(
                self.data_root / variant / shard_name / "episode_metadata.jsonl"
            )
            by_episode: dict[int, dict[str, Any]] = {}
            by_source_episode: dict[int, dict[str, Any]] = {}
            by_local_episode: dict[int, dict[str, Any]] = {}
            for record in records:
                if "episode" in record:
                    by_episode[int(record["episode"])] = record
                if "source_episode" in record:
                    by_source_episode[int(record["source_episode"])] = record
                if "local_episode_idx" in record:
                    by_local_episode[int(record["local_episode_idx"])] = record
            self._metadata_by_shard[shard_name] = (
                by_episode,
                by_source_episode,
                by_local_episode,
            )
        self._shard_index_by_key = {
            (str(shard["source_variant"]), str(shard["source_shard"])): int(index)
            for index, shard in enumerate(self._shards)
        }
        return {}, {}, {}

    def _query_for(
        self, shard: dict[str, Any], row_idx: int
    ) -> tuple[str, str, dict[str, Any]]:
        shard_name = str(shard["source_shard"])
        shard_idx = self._shard_index_by_key[
            (str(shard["source_variant"]), shard_name)
        ]
        handle = self._open_source(shard_idx)
        episode: int | None = None
        local_episode: int | None = None
        for key in ("ep_idx", "episode_idx", "episode", "trajectory"):
            if key in handle:
                episode = int(handle[key][int(row_idx)])
                break
        if "ep_offset" in handle:
            offsets = self._ep_offsets_by_shard_idx.get(shard_idx)
            if offsets is None:
                offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
                self._ep_offsets_by_shard_idx[shard_idx] = offsets
            local_episode = int(np.searchsorted(offsets, int(row_idx), side="right") - 1)

        by_episode, by_source_episode, by_local_episode = self._metadata_by_shard[shard_name]
        record = None
        if episode is not None:
            record = by_episode.get(episode) or by_source_episode.get(episode)
        if record is None and local_episode is not None:
            record = by_local_episode.get(local_episode)
        if record is None:
            if self.require_query_metadata:
                raise KeyError(
                    f"missing OGB episode metadata for {shard_name} row {row_idx}"
                )
            return self.query, "", {}
        color = str(record.get("core_color", "")).strip()
        if not color:
            if self.require_query_metadata:
                raise KeyError(
                    f"missing OGB core_color for {shard_name} row {row_idx}"
                )
            return self.query, "", record
        return self.query_template.format(color=color), color, record

    @staticmethod
    def _episode_rows(handle: h5py.File, row_count: int) -> list[np.ndarray]:
        if "ep_offset" in handle and "ep_len" in handle:
            offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
            lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
            if offsets.ndim != 1 or lengths.ndim != 1 or offsets.shape != lengths.shape:
                raise ValueError("OGB ep_offset/ep_len must be aligned one-dimensional arrays")
            trajectories: list[np.ndarray] = []
            covered = np.zeros(row_count, dtype=np.uint8)
            for episode_idx, (start_raw, length_raw) in enumerate(
                zip(offsets, lengths, strict=True)
            ):
                start = int(start_raw)
                length = int(length_raw)
                stop = start + length
                if length <= 0 or start < 0 or stop > row_count:
                    raise ValueError(
                        "OGB ep_offset/ep_len contains an invalid interval: "
                        f"episode={episode_idx}, start={start}, length={length}, rows={row_count}"
                    )
                rows = np.arange(start, stop, dtype=np.int64)
                if np.any(covered[rows]):
                    raise ValueError(f"OGB episode intervals overlap at episode {episode_idx}")
                covered[rows] = 1
                trajectories.append(rows)
            if row_count and not np.all(covered):
                missing = np.flatnonzero(covered == 0)
                raise ValueError(
                    "OGB episode intervals do not cover every row; "
                    f"first missing row={int(missing[0])}"
                )
            return trajectories
        for key in ("ep_idx", "episode_idx", "episode", "trajectory"):
            if key not in handle:
                continue
            values = np.asarray(handle[key][:row_count], dtype=np.int64)
            if values.size == 0:
                return []
            cuts = np.flatnonzero(values[1:] != values[:-1]) + 1
            return [
                rows
                for rows in np.split(np.arange(values.size, dtype=np.int64), cuts)
                if rows.size > 0
            ]
        raise KeyError(
            "OGB core HDF5 requires ep_offset/ep_len or an episode/trajectory field"
        )

    @classmethod
    def _raw_identity(cls, handle: h5py.File, row_count: int) -> tuple[np.ndarray, np.ndarray]:
        episode_values: np.ndarray | None = None
        for key in ("ep_idx", "episode_idx", "episode", "trajectory"):
            if key in handle:
                episode_values = np.asarray(handle[key][:row_count], dtype=np.int64)
                break
        trajectories = cls._episode_rows(handle, row_count)
        if episode_values is None:
            episode_values = np.full(row_count, -1, dtype=np.int64)
            for local_episode, rows in enumerate(trajectories):
                episode_values[rows] = int(local_episode)
            if np.any(episode_values < 0):
                raise ValueError("OGB episode metadata did not assign every row")

        step_values: np.ndarray | None = None
        for key in ("step_idx", "frame_idx", "timestep"):
            if key in handle:
                step_values = np.asarray(handle[key][:row_count], dtype=np.int64)
                break
        if step_values is None:
            step_values = np.full(row_count, -1, dtype=np.int64)
            for rows in trajectories:
                step_values[rows] = np.arange(rows.size, dtype=np.int64)
            if np.any(step_values < 0):
                raise ValueError("OGB step metadata did not assign every row")
        return episode_values, step_values

    def _build_refs(self) -> np.ndarray:
        by_name: dict[str, dict[str, int]] = {}
        for shard_idx, shard in enumerate(self._shards):
            by_name.setdefault(str(shard["source_shard"]), {})[
                str(shard["source_variant"])
            ] = int(shard_idx)

        grouped_refs: list[np.ndarray] = []
        group_cursor = 0
        for shard_name in sorted(by_name, key=_shard_sort_key):
            variant_map = by_name[shard_name]
            missing = [variant for variant in self.source_variants if variant not in variant_map]
            if missing:
                raise ValueError(
                    f"OGB aligned shard {shard_name} is missing variants: {missing}"
                )
            canonical_idx = int(variant_map[self.source_variants[0]])
            canonical = self._shards[canonical_idx]
            with h5py.File(canonical["source_path"], "r") as handle:
                row_count = int(handle["pixels"].shape[0])
                trajectories = self._episode_rows(handle, row_count)
                canonical_episode, canonical_step = self._raw_identity(handle, row_count)
            for variant in self.source_variants[1:]:
                with h5py.File(self._shards[variant_map[variant]]["source_path"], "r") as handle:
                    if int(handle["pixels"].shape[0]) != row_count:
                        raise ValueError(
                            f"OGB aligned row count mismatch for {shard_name}: {variant}"
                        )
                    variant_episode, variant_step = self._raw_identity(handle, row_count)
                if not np.array_equal(canonical_episode, variant_episode) or not np.array_equal(
                    canonical_step, variant_step
                ):
                    raise ValueError(
                        f"OGB variant {variant} is not aligned by episode/raw frame in {shard_name}"
                    )
            with h5py.File(canonical["target_path"], "r") as target_h:
                if int(target_h["pixels"].shape[0]) != row_count:
                    raise ValueError(f"OGB D0 target row count mismatch in {shard_name}")
                target_episode, target_step = self._raw_identity(target_h, row_count)
            if not np.array_equal(canonical_episode, target_episode) or not np.array_equal(
                canonical_step, target_step
            ):
                raise ValueError(f"OGB D0 target is not frame-aligned in {shard_name}")

            windows = build_trajectory_windows(
                trajectories,
                offsets=self.temporal_offsets,
                center_stride=self.temporal_center_stride,
            )
            assignments = round_robin_variant_indices(
                int(windows.shape[0]), len(self.source_variants), start=group_cursor
            )
            for window, variant_slot in zip(windows, assignments, strict=False):
                variant = self.source_variants[int(variant_slot)]
                shard_idx = int(variant_map[variant])
                grouped_refs.append(
                    np.column_stack(
                        [
                            np.full(window.shape, shard_idx, dtype=np.int64),
                            window,
                        ]
                    )
                )
            group_cursor += int(windows.shape[0])

        if not grouped_refs:
            return np.empty((0, 2), dtype=np.int64)
        groups = np.stack(grouped_refs, axis=0).astype(np.int64, copy=False)
        if self.shuffle_refs:
            rng = np.random.default_rng(self.balance_seed)
            groups = groups[rng.permutation(groups.shape[0])]
        self._temporal_group_count = int(groups.shape[0])
        return groups.reshape(-1, 2)

    def __getitem__(self, index: int) -> Sample:
        sample = super().__getitem__(index)
        meta = dict(sample.meta)
        slot = int(index) % self._temporal_group_size
        meta.update(
            {
                "core_temporal_group_idx": int(index) // self._temporal_group_size,
                "core_temporal_slot": slot,
                "core_temporal_group_size": self._temporal_group_size,
                "core_temporal_offset": int(self.temporal_offsets[slot]),
                "core_temporal_center_stride_raw": int(self.temporal_center_stride),
                "core_temporal_variant_mode": "one_variant_round_robin_per_window",
                "pair_is_core_clean": True,
                "main_loss_weight": 0.0,
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


class OGBCubePushTStyleAnchoredQuery9CleanBankInfoNCEMixHDF5Dataset(
    Dataset[Sample]
):
    """458 temporal core rows plus six anchored query9 groups per bank."""

    def __init__(
        self,
        *,
        core_data_root: str | Path | dict[str, Any],
        query9_data_dir: str | Path | dict[str, Any],
        core_source_variants: list[str] | tuple[str, ...],
        core_target_variant: str = "d0",
        core_query: str = "arm",
        core_query_template: str = "{color} arm",
        core_query_require_metadata: bool = True,
        core_temporal_offsets: list[int] | tuple[int, ...] = (-2, -1, 0, 1, 2),
        core_temporal_center_stride: int = 5,
        core_split_manifest_path: str | Path | None = None,
        core_shuffle_groups: bool = True,
        core_balance_seed: int = 6274,
        query_source_variants: list[str] | tuple[str, ...] = ("d8",),
        query_frame_stride: int = 1,
        query_group_size: int = 9,
        query_require_visible: bool = True,
        query_shuffle_groups: bool = False,
        query_anchor_seed: int = 20260621,
        query_anchor_mode: str = "bank_core_color",
        bank_size: int = 512,
        query_ratio: float = 0.10,
        max_supergroups: int | None = None,
        contrast_supergroup_merge_factor: int = 1,
        max_open_files: int = 16,
        **_unused: Any,
    ) -> None:
        self.query_group_size = int(query_group_size)
        (
            self.cleanbank_size,
            self.core_rows_per_bank,
            self.query_rows_per_bank,
        ) = resolve_cleanbank_rows(bank_size, query_ratio, self.query_group_size)
        self.query_anchor_seed = int(query_anchor_seed)
        self.query_anchor_mode = str(query_anchor_mode)
        self.contrast_supergroup_merge_factor = max(
            1, int(contrast_supergroup_merge_factor)
        )
        if self.query_anchor_mode != "bank_core_color":
            raise ValueError(f"unsupported query_anchor_mode: {self.query_anchor_mode!r}")
        if self.core_rows_per_bank + self.query_rows_per_bank != self.cleanbank_size:
            raise ValueError("core_rows_per_bank + query_rows_per_bank must equal cleanbank_size")
        if self.query_rows_per_bank % self.query_group_size != 0:
            raise ValueError("query_rows_per_bank must preserve complete query groups")

        self.core = OGBCubePushTStyleTemporalCoreCleanHDF5Dataset(
            core_data_root,
            source_variants=core_source_variants,
            target_variant=core_target_variant,
            temporal_offsets=core_temporal_offsets,
            temporal_center_stride=core_temporal_center_stride,
            query=core_query,
            query_template=core_query_template,
            require_query_metadata=core_query_require_metadata,
            split_manifest_path=core_split_manifest_path,
            shuffle_refs=core_shuffle_groups,
            balance_seed=core_balance_seed,
            max_open_files=max_open_files,
        )
        self.query = OGBCubeSameSourceQuery4HDF5Dataset(
            query9_data_dir,
            frame_stride=query_frame_stride,
            query_group_size=self.query_group_size,
            source_variants=query_source_variants,
            require_visible=query_require_visible,
            shuffle_groups=query_shuffle_groups,
            shuffle_seed=core_balance_seed,
            max_open_files=max_open_files,
        )
        self.query_groups_per_bank = self.query_rows_per_bank // self.query_group_size
        self._query_units_by_color = self._build_query_units_by_core_color()
        core_colors = self._core_colors()
        missing = sorted(color for color in core_colors if color not in self._query_units_by_color)
        if missing:
            raise ValueError(f"query9 anchor colors missing from query support: {missing}")

        base_length = min(
            len(self.core) // self.core_rows_per_bank,
            len(self.query) // self.query_rows_per_bank,
        )
        if max_supergroups is not None:
            base_length = min(base_length, int(max_supergroups))
        if base_length <= 0:
            raise ValueError("no complete OGB Push-T-style clean banks are available")
        self._base_length = int(base_length)
        self._group_size = self.cleanbank_size
        self._length = self._base_length * self._group_size

    @staticmethod
    def _decode_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if hasattr(value, "decode"):
            return value.decode("utf-8")
        return str(value)

    def _core_colors(self) -> set[str]:
        colors: set[str] = set()
        for maps in self.core._metadata_by_shard.values():
            for metadata_map in maps:
                for record in metadata_map.values():
                    color = str(record.get("core_color", "")).strip()
                    if color:
                        colors.add(color)
        return colors

    def _build_query_units_by_core_color(self) -> dict[str, np.ndarray]:
        groups: dict[str, list[int]] = {}
        colors_by_shard: dict[int, np.ndarray] = {}
        for source_unit, (shard_idx_raw, group_idx_raw) in enumerate(self.query._group_refs):
            shard_idx = int(shard_idx_raw)
            group_idx = int(group_idx_raw)
            if shard_idx not in colors_by_shard:
                with h5py.File(self.query._path_from_shard(shard_idx), "r") as handle:
                    if "scene_core_color" not in handle:
                        raise KeyError("OGB query9 HDF5 must contain scene_core_color")
                    colors_by_shard[shard_idx] = np.asarray(
                        [self._decode_text(value) for value in handle["scene_core_color"][:]],
                        dtype=object,
                    )
            color = str(colors_by_shard[shard_idx][group_idx])
            groups.setdefault(color, []).append(int(source_unit))
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

    def _bank_core_color(self, super_idx: int, query_group_offset: int) -> str:
        anchor_slot = int(
            int(query_group_offset) * self.core_rows_per_bank
            // max(1, self.query_groups_per_bank)
        )
        anchor_slot = min(max(anchor_slot, 0), self.core_rows_per_bank - 1)
        core_index = int(super_idx) * self.core_rows_per_bank + anchor_slot
        shard_idx, row_idx = (int(value) for value in self.core._refs[core_index])
        _, color, _ = self.core._query_for(self.core._shards[shard_idx], row_idx)
        if not color:
            raise ValueError(f"core row {core_index} has no dynamic query color")
        return color

    def _query_child_index(self, super_idx: int, local_slot: int) -> tuple[int, str]:
        group_offset = int(local_slot) // self.query_group_size
        query_slot = int(local_slot) % self.query_group_size
        color = self._bank_core_color(super_idx, group_offset)
        units = self._query_units_by_color[color]
        position = (
            int(super_idx) * 104729
            + int(group_offset) * 9176
            + self.query_anchor_seed
        ) % int(units.shape[0])
        return int(units[int(position)]) * self.query_group_size + query_slot, color

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError("OGB Push-T-style clean-bank index out of range")
        super_idx = int(index) // self._group_size
        slot = int(index) % self._group_size
        if slot < self.core_rows_per_bank:
            child_index = super_idx * self.core_rows_per_bank + slot
            sample = self.core[child_index]
            meta = dict(sample.meta)
            positive_key = str(meta["contrast_positive_key"])
            family = "ogbcube_core_temporal_exactbank"
            anchor_color = str(meta.get("query_color", ""))
        else:
            local_slot = slot - self.core_rows_per_bank
            child_index, anchor_color = self._query_child_index(super_idx, local_slot)
            sample = self.query[child_index]
            meta = dict(sample.meta)
            positive_key = str(meta["contrast_positive_key"])
            family = "ogbcube_query9_exactbank_anchored"

        contrast_super_idx = super_idx // self.contrast_supergroup_merge_factor
        meta.update(
            {
                "real_cleanbank_shard_idx": meta.get("shard_idx"),
                "real_cleanbank_source_local_index": meta.get("source_local_index"),
                "exact_cleanbank_child_index": int(child_index),
                "mix_family": family,
                "dataset_task": "ogbcube_pusht_style_anchored_query9_cleanbank",
                "cleanbank_super_idx": super_idx,
                "cleanbank_slot": slot,
                "cleanbank_group_size": self._group_size,
                "cleanbank_core_rows": self.core_rows_per_bank,
                "cleanbank_query_rows": self.query_rows_per_bank,
                "cleanbank_contrast_super_idx": contrast_super_idx,
                "cleanbank_contrast_merge_factor": self.contrast_supergroup_merge_factor,
                "query_anchor_color": anchor_color,
                "query_anchor_mode": self.query_anchor_mode,
                "task": "pusht_source5_query",
                "shard_idx": f"ogbcube_pusht_style_cleanbank{self._group_size}",
                "source_local_index": contrast_super_idx,
                "contrast_positive_key": positive_key,
                "main_loss_weight": 0.0,
            }
        )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map
            if sample.target_label_map is not None
            else _empty_label_map(),
            label_map=sample.label_map
            if sample.label_map is not None
            else torch.zeros((1, 1, 1), dtype=torch.long),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


__all__ = [
    "OGBCubePushTStyleAnchoredQuery9CleanBankInfoNCEMixHDF5Dataset",
    "OGBCubePushTStyleTemporalCoreCleanHDF5Dataset",
]
