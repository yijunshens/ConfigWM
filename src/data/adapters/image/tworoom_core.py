"""Color-balanced core-query TwoRoom HDF5 adapter.

This view is for the main Configurable Encoder denoising route:

    source(clean/Dn) + scene core-color query -> rendered core-clean target

Rows are restricted by an optional scene-level split file and grouped so each
training group contains one sample per core color.  The generic trainer already
preserves ``inbatch_group_size`` groups during sharding, so batches see a
stable color mix without changing trainer logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from ...schema import Sample
from .temporal import BoundedH5FileCache


COLORS = ("blue", "green", "orange", "purple", "red")
SUPPORTED_DATASETS = {
    "tworoom_native_gaussian_core_clean_replaced_from_d8": "clean_replaced_h5",
    "tworoom_native_gaussian_core_d8_color_query_full_v2": "color_query_h5",
    "tworoom_native_gaussian_core_d1_from_d8_nested": "d1_h5",
    "tworoom_native_gaussian_core_d2_from_d8_nested": "d2_h5",
    "tworoom_native_gaussian_core_d3_from_d8_nested": "d3_h5",
    "tworoom_native_gaussian_core_d4_from_d8_nested": "d4_h5",
    "tworoom_native_gaussian_core_d5_from_d8_nested": "d5_h5",
    "tworoom_native_gaussian_core_d6_from_d8_nested": "d6_h5",
    "tworoom_native_gaussian_core_d7_from_d8_nested": "d7_h5",
}


class TwoRoomNativeGaussianCoreQueryBalancedHDF5Dataset(Dataset[Sample]):
    """Core-query-only TwoRoom dataset with scene split and color balance."""

    def __init__(
        self,
        data_dir: str | Path | dict[str, Any],
        *,
        num_frames: int = 1,
        frame_stride: int = 1,
        query_group_size: int = 5,
        scene_split_path: str | Path | None = None,
        scene_split_name: str = "train",
        balance_core_colors: bool = True,
        balance_seed: int = 3072,
        shuffle_refs: bool = True,
        max_open_files: int = 16,
    ) -> None:
        self.data_dir = self._resolve_data_dir(data_dir)
        self.num_frames = int(num_frames)
        if self.num_frames != 1:
            raise ValueError(
                "TwoRoomNativeGaussianCoreQueryBalancedHDF5Dataset supports "
                "num_frames=1 only"
            )
        self.frame_stride = max(1, int(frame_stride))
        self.scene_split_path = (
            None if scene_split_path is None else Path(scene_split_path).expanduser().resolve()
        )
        self.scene_split_name = str(scene_split_name)
        self.balance_core_colors = bool(balance_core_colors)
        self.balance_seed = int(balance_seed)
        self.shuffle_refs = bool(shuffle_refs)

        self._manifest = self._load_manifest(self.data_dir)
        manifest_colors = self._manifest.get("colors") or COLORS
        self.colors = tuple(str(color) for color in manifest_colors)
        if not self.colors:
            raise ValueError(f"{self.data_dir} manifest has no colors")
        self.query_group_size = int(query_group_size)
        if self.query_group_size != len(self.colors):
            raise ValueError(
                f"query_group_size must match manifest colors ({len(self.colors)}), "
                f"got {self.query_group_size}"
            )

        dataset_name = self._manifest.get("dataset") or self._manifest.get("format")
        h5_path_key = SUPPORTED_DATASETS.get(dataset_name)
        if (
            h5_path_key is None
            and str(dataset_name).startswith("tworoom_9color_native_gaussian_core_")
            and str(dataset_name).endswith("_color_query_full")
        ):
            h5_path_key = "color_query_h5"
        if h5_path_key is None:
            raise ValueError(
                "Expected one of "
                f"{sorted(SUPPORTED_DATASETS)} or tworoom_9color_native_gaussian_core_d*_color_query_full, "
                f"got {dataset_name!r}"
            )
        self._dataset_name = str(dataset_name)
        self._h5_path_key = h5_path_key
        self._shards = list(self._manifest.get("shards", []))
        if not self._shards:
            raise ValueError(f"No shards found in {self.data_dir / 'manifest.json'}")

        self._file_cache = BoundedH5FileCache(max_open_files=max_open_files)
        self._indices: list[dict[str, np.ndarray] | None] = [None] * len(self._shards)
        self._selected_scene_ids = self._load_selected_scene_ids()
        self._refs, self._base_length, self._group_size, self._color_counts = (
            self._build_refs()
        )
        if self._refs.size == 0:
            raise ValueError(
                "No source frames selected for "
                f"{self._dataset_name} split={self.scene_split_name!r}"
            )

    @staticmethod
    def _resolve_data_dir(data_dir: str | Path | dict[str, Any]) -> Path:
        if isinstance(data_dir, dict):
            for key in ("local", "path", "root"):
                value = data_dir.get(key)
                if value:
                    return Path(value).expanduser().resolve()
            raise ValueError(
                f"data_dir mapping must contain one of local/path/root: {data_dir}"
            )
        return Path(data_dir).expanduser().resolve()

    @staticmethod
    def _load_manifest(root: Path) -> dict[str, Any]:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    @staticmethod
    def _decode_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if hasattr(value, "decode"):
            return value.decode("utf-8")
        return str(value)

    def _load_selected_scene_ids(self) -> set[int] | None:
        if self.scene_split_path is None:
            return None
        split = json.loads(self.scene_split_path.read_text(encoding="utf-8"))
        key_candidates = (
            f"{self.scene_split_name}_scene_ids",
            f"{self.scene_split_name}_scenes",
            self.scene_split_name,
        )
        for key in key_candidates:
            if key in split:
                return {int(scene_id) for scene_id in split[key]}
        raise KeyError(
            f"Split file {self.scene_split_path} does not contain any of "
            f"{key_candidates}"
        )

    def _path_from_shard(self, shard_idx: int, key: str) -> Path:
        raw_path = self._shards[shard_idx]["paths"][key]
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.data_dir / path
        elif not path.exists():
            candidate = self.data_dir / path.parent.name / path.name
            if candidate.exists():
                path = candidate
        return path

    def _open_file(self, shard_idx: int) -> h5py.File:
        return self._file_cache.open(
            int(shard_idx), self._path_from_shard(shard_idx, self._h5_path_key)
        )

    def _load_index(self, shard_idx: int) -> dict[str, np.ndarray]:
        cached = self._indices[shard_idx]
        if cached is not None:
            return cached

        handle = self._open_file(shard_idx)
        scene_source_offset = np.asarray(handle["scene_source_offset"], dtype=np.int64)
        scene_ep_len = np.asarray(handle["scene_ep_len"], dtype=np.int64)
        scene_global_idx = np.asarray(handle["scene_global_idx"], dtype=np.int64)
        scene_original_episode_idx = np.asarray(
            handle["scene_original_episode_idx"], dtype=np.int64
        )
        scene_core_color = np.asarray(
            [self._decode_text(value) for value in handle["scene_core_color"][:]],
            dtype=object,
        )
        scene_core_color_idx = np.asarray(handle["scene_core_color_idx"], dtype=np.int64)

        pair_scene_idx = np.asarray(handle["pair_scene_idx"], dtype=np.int64)
        pair_query_idx = np.asarray(handle["pair_query_idx"], dtype=np.int64)
        pair_source_offset = np.asarray(handle["pair_source_offset"], dtype=np.int64)
        pair_target_offset = np.asarray(handle["pair_target_offset"], dtype=np.int64)
        pair_ep_len = np.asarray(handle["pair_ep_len"], dtype=np.int64)
        pair_is_core_clean = np.asarray(handle["pair_is_core_clean"], dtype=np.uint8)
        query = np.asarray(
            [self._decode_text(value) for value in handle["query"][:]],
            dtype=object,
        )
        target_color = np.asarray(
            [self._decode_text(value) for value in handle["target_color"][:]],
            dtype=object,
        )

        scene_pair_indices = np.full(
            (scene_source_offset.shape[0], self.query_group_size),
            -1,
            dtype=np.int64,
        )
        for pair_idx, (scene_idx, query_idx) in enumerate(zip(pair_scene_idx, pair_query_idx)):
            if scene_idx < 0 or scene_idx >= scene_pair_indices.shape[0]:
                raise ValueError(f"Shard {shard_idx} has out-of-range pair_scene_idx={scene_idx}")
            if query_idx < 0 or query_idx >= self.query_group_size:
                raise ValueError(f"Shard {shard_idx} has out-of-range pair_query_idx={query_idx}")
            if scene_pair_indices[int(scene_idx), int(query_idx)] >= 0:
                raise ValueError(
                    f"Shard {shard_idx} repeats query slot {query_idx} for scene {scene_idx}"
                )
            expected_color = str(self.colors[int(query_idx)])
            actual_color = str(target_color[pair_idx])
            actual_query = str(query[pair_idx])
            if actual_color != expected_color or actual_query != f"{actual_color} ball":
                raise ValueError(
                    "TwoRoom core/query cache has inconsistent query/color semantics: "
                    f"shard={shard_idx}, scene={scene_idx}, slot={query_idx}, "
                    f"query={actual_query!r}, target_color={actual_color!r}, "
                    f"expected_color={expected_color!r}"
                )
            scene_pair_indices[int(scene_idx), int(query_idx)] = int(pair_idx)
        if np.any(scene_pair_indices < 0):
            raise ValueError(
                f"Shard {shard_idx} does not have exactly {self.query_group_size} "
                "query pairs for every scene"
            )

        index = {
            "scene_source_offset": scene_source_offset,
            "scene_ep_len": scene_ep_len,
            "scene_global_idx": scene_global_idx,
            "scene_original_episode_idx": scene_original_episode_idx,
            "scene_core_color": scene_core_color,
            "scene_core_color_idx": scene_core_color_idx,
            "scene_pair_indices": scene_pair_indices,
            "pair_source_offset": pair_source_offset,
            "pair_target_offset": pair_target_offset,
            "pair_ep_len": pair_ep_len,
            "pair_query_idx": pair_query_idx,
            "pair_is_core_clean": pair_is_core_clean,
            "query": query,
            "target_color": target_color,
        }
        self._indices[shard_idx] = index
        return index

    def _metadata_for_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        path = self._path_from_shard(shard_idx, self._h5_path_key)
        with h5py.File(path, "r") as handle:
            return {
                "scene_source_offset": np.asarray(
                    handle["scene_source_offset"], dtype=np.int64
                ),
                "scene_ep_len": np.asarray(handle["scene_ep_len"], dtype=np.int64),
                "scene_global_idx": np.asarray(
                    handle["scene_global_idx"], dtype=np.int64
                ),
                "scene_core_color_idx": np.asarray(
                    handle["scene_core_color_idx"], dtype=np.int64
                ),
            }

    def _build_refs(self) -> tuple[np.ndarray, int, int, dict[str, int]]:
        buckets: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(self.colors))}
        selected = self._selected_scene_ids
        for shard_idx in range(len(self._shards)):
            meta = self._metadata_for_shard(shard_idx)
            offsets = meta["scene_source_offset"]
            lengths = meta["scene_ep_len"]
            global_ids = meta["scene_global_idx"]
            core_color_indices = meta["scene_core_color_idx"]
            for local_scene in range(offsets.shape[0]):
                global_scene = int(global_ids[local_scene])
                if selected is not None and global_scene not in selected:
                    continue
                start = int(offsets[local_scene])
                length = int(lengths[local_scene])
                if length <= 0:
                    continue
                source_indices = np.arange(
                    start,
                    start + length,
                    self.frame_stride,
                    dtype=np.int64,
                )
                frame_deltas = source_indices - start
                color_idx = int(core_color_indices[local_scene])
                refs = np.column_stack(
                    [
                        np.full(source_indices.shape, shard_idx, dtype=np.int64),
                        np.full(source_indices.shape, local_scene, dtype=np.int64),
                        source_indices,
                        frame_deltas,
                    ]
                )
                buckets[color_idx].append(refs)

        color_arrays: dict[int, np.ndarray] = {}
        for color_idx in range(len(self.colors)):
            parts = buckets[color_idx]
            color_arrays[color_idx] = (
                np.concatenate(parts, axis=0)
                if parts
                else np.empty((0, 4), dtype=np.int64)
            )

        rng = np.random.default_rng(self.balance_seed)
        for color_idx, refs in color_arrays.items():
            if refs.shape[0] <= 0:
                continue
            if self.shuffle_refs:
                color_arrays[color_idx] = refs[rng.permutation(refs.shape[0])]
            else:
                order = np.lexsort((refs[:, 2], refs[:, 0]))
                color_arrays[color_idx] = refs[order]

        if not self.balance_core_colors:
            refs = np.concatenate(
                [color_arrays[idx] for idx in range(len(self.colors))], axis=0
            )
            if self.shuffle_refs:
                refs = refs[rng.permutation(refs.shape[0])]
            else:
                refs = refs[np.lexsort((refs[:, 2], refs[:, 0]))]
            color_counts = {
                self.colors[idx]: int(color_arrays[idx].shape[0]) for idx in range(len(self.colors))
            }
            return refs.astype(np.int64, copy=False), int(refs.shape[0]), 1, color_counts

        group_count = min(int(color_arrays[idx].shape[0]) for idx in range(len(self.colors)))
        if group_count <= 0:
            raise ValueError(
                f"Cannot color-balance {self._dataset_name}: at least one core color has no frames"
            )
        refs = np.empty((group_count * len(self.colors), 4), dtype=np.int64)
        for group_idx in range(group_count):
            # Rotate color order to avoid binding one color to a fixed slot.
            order = (np.arange(len(self.colors)) + group_idx) % len(self.colors)
            for slot, color_idx in enumerate(order.tolist()):
                refs[group_idx * len(self.colors) + slot] = color_arrays[color_idx][group_idx]
        color_counts = {
            self.colors[idx]: int(color_arrays[idx].shape[0]) for idx in range(len(self.colors))
        }
        return refs, group_count, len(self.colors), color_counts

    def __len__(self) -> int:
        return int(self._refs.shape[0])

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
                "TwoRoomNativeGaussianCoreQueryBalancedHDF5Dataset index out of range"
            )

        shard_idx, scene_idx, source_local_index, frame_delta = (
            int(value) for value in self._refs[index]
        )
        handle = self._open_file(shard_idx)
        shard_index = self._load_index(shard_idx)

        core_query_slot = int(shard_index["scene_core_color_idx"][scene_idx])
        pair_idx = int(shard_index["scene_pair_indices"][scene_idx, core_query_slot])
        pair_source_offset = int(shard_index["pair_source_offset"][pair_idx])
        pair_len = int(shard_index["pair_ep_len"][pair_idx])
        if pair_source_offset != int(shard_index["scene_source_offset"][scene_idx]):
            raise ValueError(
                "Pair source offset does not match scene source offset: "
                f"pair_idx={pair_idx}, scene_idx={scene_idx}"
            )
        if frame_delta < 0 or frame_delta >= pair_len:
            raise IndexError(
                "Resolved source frame outside pair bounds: "
                f"frame_delta={frame_delta}, pair_idx={pair_idx}, pair_len={pair_len}"
            )

        target_local_index = int(shard_index["pair_target_offset"][pair_idx] + frame_delta)
        query = str(shard_index["query"][pair_idx])
        target_color = str(shard_index["target_color"][pair_idx])
        core_color = str(shard_index["scene_core_color"][scene_idx])
        pair_is_core_clean = bool(shard_index["pair_is_core_clean"][pair_idx])
        if not pair_is_core_clean or target_color != core_color:
            raise ValueError(
                "Core-query balanced sample resolved to a non-core-clean target: "
                f"pair_idx={pair_idx}, query={query!r}, target_color={target_color!r}, "
                f"core_color={core_color!r}, pair_is_core_clean={pair_is_core_clean}"
            )

        source_frame = Image.fromarray(
            handle["source_pixels"][source_local_index]
        ).convert("RGB")
        target_frame = Image.fromarray(
            handle["target_pixels"][target_local_index]
        ).convert("RGB")
        target_mask = None
        if "target_masks" in handle:
            target_mask = torch.from_numpy(
                np.asarray(handle["target_masks"][target_local_index], dtype=np.uint8)
            ).unsqueeze(0)

        return Sample(
            frames=[source_frame],
            target_frames=[target_frame],
            target_label_map=target_mask,
            label_map=torch.zeros((1, 1, 1), dtype=torch.long),
            targets=[],
            meta={
                "id": f"CB:{shard_idx}:{source_local_index}:{core_query_slot}",
                "task": "core_query_balanced",
                "shard_idx": shard_idx,
                "source_local_index": source_local_index,
                "target_local_index": target_local_index,
                "scene_idx": scene_idx,
                "scene_global_idx": int(shard_index["scene_global_idx"][scene_idx]),
                "original_episode_idx": int(
                    shard_index["scene_original_episode_idx"][scene_idx]
                ),
                "frame_delta": frame_delta,
                "pair_idx": pair_idx,
                "pair_query_idx": int(shard_index["pair_query_idx"][pair_idx]),
                "query_slot": int(core_query_slot),
                "query": query,
                "target_color": target_color,
                "core_color": core_color,
                "pair_is_core_clean": pair_is_core_clean,
                "source_dataset": self._dataset_name,
                "scene_split_path": (
                    None if self.scene_split_path is None else str(self.scene_split_path)
                ),
                "scene_split_name": self.scene_split_name,
                "balance_core_colors": self.balance_core_colors,
                "shuffle_refs": self.shuffle_refs,
            },
            query=query,
        )


__all__ = ["TwoRoomNativeGaussianCoreQueryBalancedHDF5Dataset"]
