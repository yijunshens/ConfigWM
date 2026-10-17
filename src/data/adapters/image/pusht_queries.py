"""PushT source5 multi-query HDF5 adapter."""

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


PUSHT_SOURCE5_COLORS = ("gray", "yellow", "pink", "brown", "cyan")
SUPPORTED_DATASET_NAMES = {
    "pusht_source5_query_direct_h5",
    "pusht_source5_query_hdf5",
    "pusht_source9_query_hdf5",
    "pusht_9color_far_source9_query_h5",
    "pusht_9color_far_source9_query_h5_diffexact_v2",
}
SOURCE9_FORMAT = "pusht_source9_query_hdf5"


class PushTSource5QueryHDF5Dataset(Dataset[Sample]):
    """Read source5 samples where one source frame has five query targets."""

    def __init__(
        self,
        data_dir: str | Path | dict[str, Any],
        *,
        num_frames: int = 1,
        frame_stride: int = 1,
        query_group_size: int = 5,
        max_open_files: int = 16,
    ) -> None:
        self.data_dir = self._resolve_data_dir(data_dir)
        self.num_frames = int(num_frames)
        if self.num_frames != 1:
            raise ValueError("PushTSource5QueryHDF5Dataset supports num_frames=1 only")
        self.frame_stride = max(1, int(frame_stride))
        self._manifest = self._load_manifest(self.data_dir)
        manifest_colors = self._manifest.get("colors")
        self.colors = tuple(
            str(color) for color in (manifest_colors or PUSHT_SOURCE5_COLORS)
        )
        if not self.colors or len(set(self.colors)) != len(self.colors):
            raise ValueError("PushT query manifest colors must be non-empty and unique")

        self.query_group_size = int(query_group_size)
        if self.query_group_size != len(self.colors):
            raise ValueError(
                f"query_group_size must match the {len(self.colors)} manifest colors, "
                f"got {self.query_group_size}"
            )

        dataset_names = {
            str(value)
            for value in (self._manifest.get("dataset"), self._manifest.get("format"))
            if value is not None
        }
        if not dataset_names.intersection(SUPPORTED_DATASET_NAMES):
            raise ValueError(f"Unexpected PushT query dataset: {sorted(dataset_names)!r}")
        if self._manifest.get("format") == SOURCE9_FORMAT:
            expected = {
                "source_variant": "d8",
                "query_template": "{color} T block",
                "frame_stride": 5,
            }
            mismatches = {
                key: (self._manifest.get(key), value)
                for key, value in expected.items()
                if self._manifest.get(key) != value
            }
            if mismatches:
                raise ValueError(f"Invalid PushT source9 manifest contract: {mismatches}")

        self._shards = list(self._manifest.get("shards", []))
        if not self._shards:
            raise ValueError(f"No shards found in {self.data_dir / 'manifest.json'}")

        self._source_row_starts: list[int] = []
        total_source_rows = 0
        for shard in self._shards:
            self._source_row_starts.append(total_source_rows)
            total_source_rows += int(shard["total_frames"])
        self._source_rows = total_source_rows
        self._files = BoundedH5FileCache(max_open_files=max_open_files)
        self._indices: list[dict[str, np.ndarray] | None] = [None] * len(self._shards)
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
            raise ValueError(f"data_dir mapping must contain local/path/root: {data_dir}")
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

    def _path_from_shard(self, shard_idx: int) -> Path:
        shard = self._shards[shard_idx]
        paths = shard.get("paths")
        if isinstance(paths, dict):
            raw_path = next(
                (
                    paths.get(key)
                    for key in (
                        "pusht_source5_query_h5",
                        "pusht_source9_query_h5",
                        "pusht_9color_far_source9_query_h5",
                        "pusht_9color_far_source9_query_h5_diffexact_v2",
                        "unified_h5",
                    )
                    if paths.get(key)
                ),
                None,
            )
        else:
            raw_path = shard.get("path")
        if not raw_path:
            raise KeyError("PushT query-support shard must define a HDF5 path")
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            return self.data_dir / path
        if path.exists():
            return path
        for index in range(len(path.parts) - 1, -1, -1):
            if path.parts[index].startswith("shard_"):
                return self.data_dir.joinpath(*path.parts[index:])
        return path

    def _metadata_for_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        with h5py.File(self._path_from_shard(shard_idx), "r") as handle:
            return {
                "scene_source_offset": np.asarray(handle["scene_source_offset"], dtype=np.int64),
                "scene_ep_len": np.asarray(handle["scene_ep_len"], dtype=np.int64),
                "scene_global_idx": np.asarray(handle["scene_global_idx"], dtype=np.int64),
            }

    def _build_source_refs(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for shard_idx in range(len(self._shards)):
            meta = self._metadata_for_shard(shard_idx)
            offsets = meta["scene_source_offset"]
            lengths = meta["scene_ep_len"]
            global_ids = meta["scene_global_idx"]
            for local_scene in range(offsets.shape[0]):
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
                chunks.append(
                    np.column_stack(
                        [
                            np.full(source_indices.shape, shard_idx, dtype=np.int64),
                            np.full(source_indices.shape, local_scene, dtype=np.int64),
                            source_indices,
                            frame_deltas,
                            np.full(source_indices.shape, int(global_ids[local_scene]), dtype=np.int64),
                        ]
                    )
                )
        if not chunks:
            raise ValueError(f"No source frames selected in {self.data_dir}")
        return np.concatenate(chunks, axis=0).astype(np.int64, copy=False)

    def _open_file(self, shard_idx: int) -> h5py.File:
        return self._files.open(int(shard_idx), self._path_from_shard(shard_idx))

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
            color = str(target_color[pair_idx])
            if str(query[pair_idx]) != f"{color} T block":
                raise ValueError(
                    f"PushT query/color mismatch in shard {shard_idx}, pair {pair_idx}"
                )
            expected_color = self.colors[int(query_idx)]
            if color != expected_color:
                raise ValueError(
                    f"PushT query slot {query_idx} in shard {shard_idx} has color "
                    f"{color!r}, expected {expected_color!r} from manifest order"
                )
            scene_pair_indices[int(scene_idx), int(query_idx)] = int(pair_idx)
        if np.any(scene_pair_indices < 0):
            raise ValueError(
                f"Shard {shard_idx} does not have {self.query_group_size} query pairs "
                "for every source scene"
            )
        for scene_idx in range(scene_pair_indices.shape[0]):
            colors = {
                str(target_color[int(pair_idx)])
                for pair_idx in scene_pair_indices[scene_idx]
            }
            if colors != set(self.colors):
                raise ValueError(
                    f"Shard {shard_idx} scene {scene_idx} target colors do not match manifest"
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

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError("PushTSource5QueryHDF5Dataset index out of range")
        source_unit = index // self.query_group_size
        query_slot = index % self.query_group_size
        shard_idx, scene_idx, source_local_index, frame_delta, scene_global = (
            int(value) for value in self._source_refs[source_unit]
        )
        handle = self._open_file(shard_idx)
        shard_index = self._load_index(shard_idx)

        pair_idx = int(shard_index["scene_pair_indices"][scene_idx, query_slot])
        pair_source_offset = int(shard_index["pair_source_offset"][pair_idx])
        pair_len = int(shard_index["pair_ep_len"][pair_idx])
        if pair_source_offset != int(shard_index["scene_source_offset"][scene_idx]):
            raise ValueError("Pair source offset does not match scene source offset")
        if frame_delta < 0 or frame_delta >= pair_len:
            raise IndexError("Resolved source frame outside pair bounds")
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

        return Sample(
            frames=[source_frame],
            target_frames=[target_frame],
            target_label_map=None,
            label_map=torch.zeros((1, 1, 1), dtype=torch.long),
            targets=[],
            meta={
                "id": f"PushTSource5:{shard_idx}:{source_local_index}:{query_slot}",
                "task": "pusht_source5_query",
                "shard_idx": shard_idx,
                "source_local_index": source_local_index,
                "target_local_index": target_local_index,
                "scene_idx": scene_idx,
                "scene_global_idx": scene_global,
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
                "query_group_size": self.query_group_size,
                "query_colors": self.colors,
            },
            query=query,
        )


__all__ = ["PushTSource5QueryHDF5Dataset", "PUSHT_SOURCE5_COLORS"]
