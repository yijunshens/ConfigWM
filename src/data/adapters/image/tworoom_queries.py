"""Unified-source native Gaussian TwoRoom HDF5 adapter.

This reads the unified language-configuration cache:

- one source frame per scene/time step;
- five query-target pairs per source, one for each candidate color;
- target for ``"{color} ball"`` keeps only objects matching that color.

Rows are exposed as single-frame paired Configurable Encoder samples.  The adapter also
reports ``inbatch_group_size=5`` so the training sharder keeps the five queries
for the same source frame adjacent when it builds worker-local shuffled views.
"""

from __future__ import annotations

import bisect
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
    "tworoom_native_gaussian_unified_source_5query_full_v1": "unified_h5",
    "tworoom_native_gaussian_core_d8_color_query_full_v2": "color_query_h5",
    "tworoom_native_gaussian_core_d1_from_d8_nested": "d1_h5",
    "tworoom_native_gaussian_core_d2_from_d8_nested": "d2_h5",
    "tworoom_native_gaussian_core_d3_from_d8_nested": "d3_h5",
    "tworoom_native_gaussian_core_d4_from_d8_nested": "d4_h5",
    "tworoom_native_gaussian_core_d5_from_d8_nested": "d5_h5",
    "tworoom_9color_source9_stride5": "output_h5_path",
}


class TwoRoomNativeGaussianUnifiedSource5QueryHDF5Dataset(Dataset[Sample]):
    """Read unified-source TwoRoom query-conditioned shards."""

    def __init__(
        self,
        data_dir: str | Path | dict[str, Any],
        *,
        num_frames: int = 1,
        frame_stride: int = 1,
        query_group_size: int = 5,
        scene_split_path: str | Path | None = None,
        scene_split_name: str = "train",
        max_open_files: int = 16,
    ) -> None:
        self.data_dir = self._resolve_data_dir(data_dir)
        self.num_frames = int(num_frames)
        if self.num_frames != 1:
            raise ValueError(
                "TwoRoomNativeGaussianUnifiedSource5QueryHDF5Dataset supports "
                "num_frames=1 only"
            )
        self.frame_stride = max(1, int(frame_stride))
        self.scene_split_path = (
            None if scene_split_path is None else Path(scene_split_path).expanduser().resolve()
        )
        self.scene_split_name = str(scene_split_name)

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
        self._h5_path_key = h5_path_key

        self._shards = list(self._manifest.get("shards", []))
        if not self._shards:
            raise ValueError(f"No shards found in {self.data_dir / 'manifest.json'}")

        self._source_row_starts: list[int] = []
        self._source_rows_by_shard: list[int] = []
        total_source_rows = 0
        for shard in self._shards:
            source_rows = int(shard.get("total_frames", shard.get("source_frame_count", shard.get("frame_count", 0))))
            self._source_row_starts.append(total_source_rows)
            self._source_rows_by_shard.append(source_rows)
            total_source_rows += source_rows

        self._source_rows = total_source_rows
        self._file_cache = BoundedH5FileCache(max_open_files=max_open_files)
        self._indices: list[dict[str, np.ndarray] | None] = [None] * len(self._shards)
        self._selected_scene_ids = self._load_selected_scene_ids()
        self._source_refs = self._build_source_refs()
        self._source_units = int(self._source_refs.shape[0])
        self._total_rows = self._source_units * self.query_group_size

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
            f"Split file {self.scene_split_path} does not contain any of {key_candidates}"
        )

    def _metadata_for_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        path = self._path_from_shard(shard_idx, self._h5_path_key)
        with h5py.File(path, "r") as handle:
            return {
                "scene_source_offset": np.asarray(handle["scene_source_offset"], dtype=np.int64),
                "scene_ep_len": np.asarray(handle["scene_ep_len"], dtype=np.int64),
                "scene_global_idx": np.asarray(handle["scene_global_idx"], dtype=np.int64),
            }

    def _build_source_refs(self) -> np.ndarray:
        selected = self._selected_scene_ids
        chunks: list[np.ndarray] = []
        for shard_idx in range(len(self._shards)):
            meta = self._metadata_for_shard(shard_idx)
            offsets = meta["scene_source_offset"]
            lengths = meta["scene_ep_len"]
            global_ids = meta["scene_global_idx"]
            for local_scene in range(offsets.shape[0]):
                global_scene = int(global_ids[local_scene])
                if selected is not None and global_scene not in selected:
                    continue
                start = int(offsets[local_scene])
                length = int(lengths[local_scene])
                if length <= 0:
                    continue
                source_indices = np.arange(start, start + length, self.frame_stride, dtype=np.int64)
                frame_deltas = source_indices - start
                chunks.append(
                    np.column_stack(
                        [
                            np.full(source_indices.shape, shard_idx, dtype=np.int64),
                            np.full(source_indices.shape, local_scene, dtype=np.int64),
                            source_indices,
                            frame_deltas,
                        ]
                    )
                )
        if not chunks:
            raise ValueError(
                f"No source frames selected for split={self.scene_split_name!r} "
                f"in {self.data_dir}"
            )
        return np.concatenate(chunks, axis=0).astype(np.int64, copy=False)

    def _path_from_shard(self, shard_idx: int, key: str) -> Path:
        shard = self._shards[shard_idx]
        if "paths" in shard and key in shard["paths"]:
            raw_path = shard["paths"][key]
        elif key in shard:
            raw_path = shard[key]
        else:
            raise KeyError(
                f"Shard {shard_idx} has no path for key {key!r}; "
                f"available top-level keys={sorted(shard.keys())}"
            )
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
        if "scene_original_episode_idx" in handle:
            scene_original_episode_idx = np.asarray(
                handle["scene_original_episode_idx"], dtype=np.int64
            )
        else:
            scene_original_episode_idx = scene_global_idx.copy()
        if "scene_core_color" in handle:
            scene_core_color = np.asarray(
                [self._decode_text(value) for value in handle["scene_core_color"][:]],
                dtype=object,
            )
        else:
            scene_core_color = np.asarray([""] * int(scene_source_offset.shape[0]), dtype=object)
        if "scene_core_color_idx" in handle:
            scene_core_color_idx = np.asarray(handle["scene_core_color_idx"], dtype=np.int64)
        else:
            scene_core_color_idx = np.zeros(int(scene_source_offset.shape[0]), dtype=np.int64)

        pair_scene_idx = np.asarray(handle["pair_scene_idx"], dtype=np.int64)
        pair_query_idx = np.asarray(handle["pair_query_idx"], dtype=np.int64)
        pair_source_offset = np.asarray(handle["pair_source_offset"], dtype=np.int64)
        pair_target_offset = np.asarray(handle["pair_target_offset"], dtype=np.int64)
        pair_ep_len = np.asarray(handle["pair_ep_len"], dtype=np.int64)
        if "pair_is_core_clean" in handle:
            pair_is_core_clean = np.asarray(handle["pair_is_core_clean"], dtype=np.uint8)
        else:
            pair_is_core_clean = np.zeros(int(pair_query_idx.shape[0]), dtype=np.uint8)
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
                    "TwoRoom query group has inconsistent query/color semantics: "
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

    def __len__(self) -> int:
        return self._total_rows

    @property
    def base_length(self) -> int:
        return self._source_units

    @property
    def inbatch_group_size(self) -> int:
        return self.query_group_size

    @staticmethod
    def _global_to_shard_local(starts: list[int], index: int) -> tuple[int, int]:
        shard_idx = bisect.bisect_right(starts, index) - 1
        if shard_idx < 0:
            raise IndexError("TwoRoom unified-source index out of range")
        return shard_idx, index - starts[shard_idx]

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError(
                "TwoRoomNativeGaussianUnifiedSource5QueryHDF5Dataset index out of range"
            )

        source_unit = index // self.query_group_size
        query_slot = index % self.query_group_size
        shard_idx, scene_idx, source_local_index, frame_delta = (
            int(value) for value in self._source_refs[source_unit]
        )
        handle = self._open_file(shard_idx)
        shard_index = self._load_index(shard_idx)

        scene_len = int(shard_index["scene_ep_len"][scene_idx])
        if frame_delta < 0 or frame_delta >= scene_len:
            raise IndexError(
                "Resolved source frame outside scene bounds: "
                f"source_local_index={source_local_index}, scene_idx={scene_idx}"
            )

        pair_idx = int(shard_index["scene_pair_indices"][scene_idx, query_slot])
        pair_source_offset = int(shard_index["pair_source_offset"][pair_idx])
        pair_len = int(shard_index["pair_ep_len"][pair_idx])
        if pair_source_offset != int(shard_index["scene_source_offset"][scene_idx]):
            raise ValueError(
                "Pair source offset does not match scene source offset: "
                f"pair_idx={pair_idx}, scene_idx={scene_idx}"
            )
        if frame_delta >= pair_len:
            raise IndexError(
                "Resolved source frame outside pair bounds: "
                f"frame_delta={frame_delta}, pair_idx={pair_idx}, pair_len={pair_len}"
            )
        target_local_index = int(shard_index["pair_target_offset"][pair_idx] + frame_delta)
        query = str(shard_index["query"][pair_idx])
        target_color = str(shard_index["target_color"][pair_idx])
        core_color = str(shard_index["scene_core_color"][scene_idx])

        source_frame = Image.fromarray(
            handle["source_pixels"][source_local_index]
        ).convert("RGB")
        target_frame = Image.fromarray(
            handle["target_pixels"][target_local_index]
        ).convert("RGB")
        if "target_masks" in handle:
            target_mask = torch.from_numpy(
                np.asarray(handle["target_masks"][target_local_index], dtype=np.uint8)
            ).unsqueeze(0)
        else:
            # Match PushT source-query caches: query-support rows do not carry
            # optional target masks unless the H5 explicitly provides them.
            target_mask = None

        return Sample(
            frames=[source_frame],
            target_frames=[target_frame],
            target_label_map=target_mask,
            label_map=torch.zeros((1, 1, 1), dtype=torch.long),
            targets=[],
            meta={
                "id": f"U:{shard_idx}:{source_local_index}:{query_slot}",
                "task": "unified_source_5query",
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
                "query_slot": int(query_slot),
                "query": query,
                "target_color": target_color,
                "core_color": core_color,
                "pair_is_core_clean": bool(shard_index["pair_is_core_clean"][pair_idx]),
                "source_dataset": self._manifest.get("dataset"),
                "scene_split_path": (
                    None if self.scene_split_path is None else str(self.scene_split_path)
                ),
                "scene_split_name": self.scene_split_name,
            },
            query=query,
        )


__all__ = ["TwoRoomNativeGaussianUnifiedSource5QueryHDF5Dataset"]
