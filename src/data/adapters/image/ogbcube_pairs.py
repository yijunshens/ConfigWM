"""OGB-cube clean-bank InfoNCE mix for Configurable Encoder training.

This adapter mirrors the PushT/TwoRoom anchored clean-bank recipe while keeping
OGB-cube specifics local to data loading:

* core rows align d1/d2/d3 source frames to the matching d0 clean frame;
* query rows use same-source D3 query4 groups, where all four query colors are
  visible and therefore suitable for query-conditioned separation;
* every 512-row bank is emitted through the existing ``pusht_source5_query``
  collator path so the trainer uses the same pure InfoNCE branch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401 - registers compressed HDF5 filters
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from ...schema import Sample
from .temporal import BoundedH5FileCache


DEFAULT_OGBCUBE_QUERY_COLORS = ("silver", "green", "yellow", "cyan")


def _resolve_path(value: str | Path | dict[str, Any]) -> Path:
    if isinstance(value, dict):
        for key in ("local", "path", "root"):
            raw = value.get(key)
            if raw:
                return Path(raw).expanduser().resolve()
        raise ValueError(f"Path mapping must contain local/path/root: {value}")
    return Path(value).expanduser().resolve()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def _image_from_array(array: Any) -> Image.Image:
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return Image.fromarray(arr, mode="RGB")


def _empty_label_map() -> torch.Tensor:
    return torch.zeros((1, 1, 1), dtype=torch.long)


def _shard_sort_key(name: str) -> tuple[str, int]:
    stem = str(name).rstrip("/").split("/")[-1]
    try:
        return (stem.rsplit("_", 1)[0], int(stem.rsplit("_", 1)[1]))
    except Exception:
        return (stem, -1)


class OGBCubeCoreCleanHDF5Dataset(Dataset[Sample]):
    """Read OGB d1/d2/d3 source frames paired with aligned d0 clean frames."""

    def __init__(
        self,
        data_root: str | Path | dict[str, Any],
        *,
        source_variants: list[str] | tuple[str, ...] | None = None,
        target_variant: str = "d0",
        num_frames: int = 1,
        frame_stride: int = 5,
        temporal_offsets: list[int] | tuple[int, ...] | None = None,
        temporal_center_stride: int = 1,
        query: str = "core arm",
        query4_data_dir: str | Path | dict[str, Any] | None = None,
        query_template: str = "{color} arm",
        require_query_metadata: bool = True,
        split_manifest_path: str | Path | None = None,
        shuffle_refs: bool = True,
        balance_seed: int = 6274,
        max_open_files: int = 16,
    ) -> None:
        self.data_root = _resolve_path(data_root)
        self.source_variants = list(source_variants or ("d1", "d2", "d3"))
        self.target_variant = str(target_variant)
        self.num_frames = int(num_frames)
        if self.num_frames != 1:
            raise ValueError("OGBCubeCoreCleanHDF5Dataset supports num_frames=1 only")
        self.frame_stride = max(1, int(frame_stride))
        self.temporal_offsets = None if temporal_offsets is None else tuple(int(v) for v in temporal_offsets)
        if self.temporal_offsets is not None:
            if len(self.temporal_offsets) < 2:
                raise ValueError("temporal_offsets must contain at least two offsets")
            if len(set(self.temporal_offsets)) != len(self.temporal_offsets):
                raise ValueError(f"Duplicate temporal_offsets are not allowed: {self.temporal_offsets}")
            if 0 not in self.temporal_offsets:
                raise ValueError("temporal_offsets must include 0 for the anchor frame")
        self.temporal_center_stride = max(1, int(temporal_center_stride))
        self.query = str(query)
        self.query_template = str(query_template)
        self.require_query_metadata = bool(require_query_metadata)
        self.shuffle_refs = bool(shuffle_refs)
        self.balance_seed = int(balance_seed)
        self.split_manifest_path = None if split_manifest_path is None else Path(split_manifest_path)

        self._shards = self._collect_shards()
        (
            self._query_color_by_ref,
            self._metadata_refs_by_key,
            self._core_metadata_by_ref,
        ) = self._load_core_query_colors(query4_data_dir)
        if not self._shards:
            raise ValueError("No OGB-cube core-clean shards selected")
        self._source_files = BoundedH5FileCache(max_open_files=max_open_files)
        self._target_files = BoundedH5FileCache(max_open_files=max_open_files)
        self._refs = self._build_refs()
        if self._refs.size == 0:
            raise ValueError("No OGB-cube core-clean frames selected")

    def _collect_shards(self) -> list[dict[str, Any]]:
        variant_to_names: dict[str, set[str]] = {variant: set() for variant in self.source_variants}
        if self.split_manifest_path is not None:
            manifest = _load_json(self.split_manifest_path)
            for shard in manifest.get("shards", []):
                variant = str(shard.get("source_variant", ""))
                source_shard = str(shard.get("source_shard", ""))
                if variant in variant_to_names and source_shard:
                    variant_to_names[variant].add(source_shard)
            selected_shards = set().union(*(names for names in variant_to_names.values()))
            # Query-support manifests may only list the hardest source variant
            # (e.g. d8). Reuse that shard split for the other aligned core
            # variants so D0-DK -> D0 keeps the same train/eval partition.
            for variant in list(variant_to_names):
                if not variant_to_names[variant]:
                    variant_to_names[variant] = {
                        name
                        for name in selected_shards
                        if (self.data_root / variant / name).is_dir()
                    }
            if self.target_variant in variant_to_names and not variant_to_names[self.target_variant]:
                variant_to_names[self.target_variant] = {
                    name
                    for name in selected_shards
                    if (self.data_root / self.target_variant / name).is_dir()
                }
        else:
            for variant in self.source_variants:
                variant_dir = self.data_root / variant
                variant_to_names[variant] = {
                    path.name for path in variant_dir.glob("shard_*") if path.is_dir()
                }

        shards: list[dict[str, Any]] = []
        for variant in self.source_variants:
            names = sorted(variant_to_names.get(variant, set()), key=_shard_sort_key)
            if not names:
                raise ValueError(f"No shards selected for OGB-cube variant {variant!r}")
            for name in names:
                source_path = self.data_root / variant / name / "dataset.h5"
                target_path = self.data_root / self.target_variant / name / "dataset.h5"
                if not source_path.exists():
                    raise FileNotFoundError(f"Missing source HDF5: {source_path}")
                if not target_path.exists():
                    raise FileNotFoundError(f"Missing target HDF5: {target_path}")
                shards.append(
                    {
                        "source_variant": variant,
                        "target_variant": self.target_variant,
                        "source_shard": name,
                        "source_path": source_path,
                        "target_path": target_path,
                    }
                )
        return shards

    def _path_from_query4_shard(self, data_dir: Path, shard: dict[str, Any]) -> Path:
        paths = shard.get("paths")
        raw_path = None
        if isinstance(paths, dict):
            raw_path = (
                paths.get("ogbcube_same_source_query9_h5")
                or paths.get("ogbcube_same_source_query4_h5")
                or paths.get("pusht_source5_query_h5")
                or paths.get("unified_h5")
            )
        if not raw_path:
            raw_path = shard.get("path")
        if not raw_path:
            raise KeyError("OGB-cube query4 shard must define a HDF5 path")
        path = Path(str(raw_path))
        if not path.is_absolute():
            split_candidate = data_dir / path
            root_candidate = data_dir.parent / path
            path = split_candidate if split_candidate.exists() else root_candidate
        return path

    def _load_core_query_colors(
        self, query4_data_dir: str | Path | dict[str, Any] | None
    ) -> tuple[
        dict[tuple[str, str, int], str],
        dict[tuple[str, str], list[int]],
        dict[tuple[str, str, int], dict[str, Any]],
    ]:
        # Core-clean queries come from the source shard episode metadata. The
        # query4 dataset is still used by the mix dataset for the 10% same-source
        # query support, but it should not define the 90% core-clean rows.
        del query4_data_dir
        mapping: dict[tuple[str, str, int], str] = {}
        refs_by_key: dict[tuple[str, str], list[int]] = {}
        metadata_by_ref: dict[tuple[str, str, int], dict[str, Any]] = {}
        missing: list[tuple[str, str]] = []

        for shard in self._shards:
            variant = str(shard["source_variant"])
            source_shard = str(shard["source_shard"])
            meta_path = self.data_root / variant / source_shard / "episode_metadata.jsonl"
            records = _load_jsonl(meta_path)
            by_episode: dict[int, dict[str, Any]] = {}
            by_source_episode: dict[int, dict[str, Any]] = {}
            by_local_episode: dict[int, dict[str, Any]] = {}
            for rec in records:
                if "episode" in rec:
                    by_episode[int(rec["episode"])] = rec
                if "source_episode" in rec:
                    by_source_episode[int(rec["source_episode"])] = rec
                if "local_episode_idx" in rec:
                    by_local_episode[int(rec["local_episode_idx"])] = rec

            with h5py.File(shard["source_path"], "r") as handle:
                n = int(handle["pixels"].shape[0])
                rows = np.arange(0, n, self.frame_stride, dtype=np.int64)
                ep_values: np.ndarray | None = None
                local_values: np.ndarray | None = None
                if "ep_idx" in handle:
                    ep_values = np.asarray(handle["ep_idx"][rows], dtype=np.int64)
                if "ep_offset" in handle:
                    offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
                    local_values = np.searchsorted(offsets, rows, side="right") - 1

            key = (variant, source_shard)
            for i, row_val in enumerate(rows):
                rec: dict[str, Any] | None = None
                if ep_values is not None:
                    ep = int(ep_values[i])
                    rec = by_episode.get(ep) or by_source_episode.get(ep)
                if rec is None and local_values is not None:
                    rec = by_local_episode.get(int(local_values[i]))
                if rec is None:
                    continue
                color = str(rec.get("core_color", "")).strip()
                if not color:
                    continue
                row = int(row_val)
                ref_key = (variant, source_shard, row)
                mapping[ref_key] = color
                refs_by_key.setdefault(key, []).append(row)
                metadata_by_ref[ref_key] = rec

            if self.require_query_metadata and not refs_by_key.get(key):
                missing.append(key)
                if len(missing) >= 5:
                    break

        if missing:
            raise ValueError(f"Missing dynamic core query rows from episode metadata for shards: {missing}")
        for key, rows in refs_by_key.items():
            refs_by_key[key] = sorted(set(int(x) for x in rows))
        return mapping, refs_by_key, metadata_by_ref

    def _query_for(self, shard: dict[str, Any], row_idx: int) -> tuple[str, str, dict[str, Any]]:
        key = (str(shard["source_variant"]), str(shard["source_shard"]), int(row_idx))
        color = self._query_color_by_ref.get(key, "")
        if color:
            return self.query_template.format(color=color), color, self._core_metadata_by_ref.get(key, {})
        if self.require_query_metadata:
            raise KeyError(f"Missing dynamic core query color for {key}")
        return self.query, "", {}

    def _build_refs(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for shard_idx, shard in enumerate(self._shards):
            with h5py.File(shard["source_path"], "r") as source_h, h5py.File(
                shard["target_path"], "r"
            ) as target_h:
                source_n = int(source_h["pixels"].shape[0])
                target_n = int(target_h["pixels"].shape[0])
            n = min(source_n, target_n)
            if n <= 0:
                continue
            key = (str(shard["source_variant"]), str(shard["source_shard"]))
            if self.require_query_metadata:
                rows = np.asarray(
                    [row for row in self._metadata_refs_by_key.get(key, []) if 0 <= int(row) < n],
                    dtype=np.int64,
                )
            else:
                rows = np.arange(0, n, self.frame_stride, dtype=np.int64)
            if rows.size == 0:
                continue
            if self.temporal_offsets is None:
                expanded_rows = rows
            else:
                # Rows are already sampled at frame_stride. Offsets are in stride
                # units, e.g. [-2,-1,0,1,2] with frame_stride=5 means raw-frame
                # offsets [-10,-5,0,5,10].
                left = max(0, -min(self.temporal_offsets))
                right = max(0, max(self.temporal_offsets))
                windows: list[int] = []
                if rows.size > left + right:
                    for center_pos in range(left, int(rows.size) - right, self.temporal_center_stride):
                        windows.extend(int(rows[center_pos + offset]) for offset in self.temporal_offsets)
                expanded_rows = np.asarray(windows, dtype=np.int64)
            if expanded_rows.size == 0:
                continue
            chunks.append(
                np.column_stack(
                    [
                        np.full(expanded_rows.shape, shard_idx, dtype=np.int64),
                        expanded_rows,
                    ]
                )
            )
        if not chunks:
            return np.zeros((0, 2), dtype=np.int64)
        refs = np.concatenate(chunks, axis=0).astype(np.int64, copy=False)
        if self.shuffle_refs:
            rng = np.random.default_rng(self.balance_seed)
            if self.temporal_offsets is None:
                rng.shuffle(refs, axis=0)
            else:
                group_size = len(self.temporal_offsets)
                group_count = refs.shape[0] // group_size
                refs = refs[: group_count * group_size]
                groups = refs.reshape(group_count, group_size, 2)
                groups = groups[rng.permutation(group_count)]
                refs = groups.reshape(group_count * group_size, 2)
        return refs

    def __len__(self) -> int:
        return int(self._refs.shape[0])

    def _open_source(self, shard_idx: int) -> h5py.File:
        return self._source_files.open(
            int(shard_idx), self._shards[shard_idx]["source_path"]
        )

    def _open_target(self, shard_idx: int) -> h5py.File:
        return self._target_files.open(
            int(shard_idx), self._shards[shard_idx]["target_path"]
        )

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError("OGBCubeCoreCleanHDF5Dataset index out of range")
        shard_idx, row_idx = [int(x) for x in self._refs[index]]
        shard = self._shards[shard_idx]
        source_h = self._open_source(shard_idx)
        target_h = self._open_target(shard_idx)
        frame = _image_from_array(source_h["pixels"][row_idx])
        target = _image_from_array(target_h["pixels"][row_idx])
        query, query_color, core_meta = self._query_for(shard, row_idx)
        positive_key = f"ogbcube_coreclean:{shard['source_shard']}:{row_idx}:{query}"
        core_cell_raw = core_meta.get("core_cell", -1)
        core_cell = -1 if core_cell_raw in (None, "") else int(core_cell_raw)
        meta = {
            "id": f"OGBCubeCoreClean:{shard['source_variant']}:{shard['source_shard']}:{row_idx}",
            "task": "ogbcube_core_clean",
            "source_variant": shard["source_variant"],
            "target_variant": shard["target_variant"],
            "source_shard": shard["source_shard"],
            "shard_idx": f"{shard['source_variant']}:{shard['source_shard']}",
            "source_local_index": int(row_idx),
            "target_local_index": int(row_idx),
            "source_frame_index": int(row_idx),
            "target_frame_index": int(row_idx),
            "query": query,
            "query_color": query_color,
            "core_color_token": str(core_meta.get("core_color_token", "")),
            "core_cell": core_cell,
            "core_episode": int(core_meta["episode"]) if "episode" in core_meta else -1,
            "core_source_episode": int(core_meta["source_episode"]) if "source_episode" in core_meta else -1,
            "core_query_dynamic": bool(query_color),
            "frame_stride": int(self.frame_stride),
            "contrast_positive_key": positive_key,
        }
        return Sample(
            frames=[frame],
            target_frames=[target],
            target_label_map=_empty_label_map(),
            label_map=_empty_label_map(),
            targets=[],
            meta=meta,
            query=query,
        )


class OGBCubeSameSourceQuery4HDF5Dataset(Dataset[Sample]):
    """Read OGB same-source four-query rows from the compat query4 dataset."""

    def __init__(
        self,
        data_dir: str | Path | dict[str, Any],
        *,
        num_frames: int = 1,
        frame_stride: int = 1,
        query_group_size: int = 4,
        source_variants: list[str] | tuple[str, ...] | None = None,
        require_visible: bool = True,
        shuffle_groups: bool = False,
        shuffle_seed: int = 20260627,
        max_open_files: int = 16,
    ) -> None:
        self.data_dir = _resolve_path(data_dir)
        self.num_frames = int(num_frames)
        if self.num_frames != 1:
            raise ValueError("OGBCubeSameSourceQuery4HDF5Dataset supports num_frames=1 only")
        self.frame_stride = max(1, int(frame_stride))
        self.query_group_size = int(query_group_size)
        if self.query_group_size <= 0:
            raise ValueError(f"query_group_size must be positive, got {self.query_group_size}")
        self.source_variants = None if source_variants is None else {str(v) for v in source_variants}
        self.require_visible = bool(require_visible)
        self.shuffle_groups = bool(shuffle_groups)
        self.shuffle_seed = int(shuffle_seed)

        self._manifest = _load_json(self.data_dir / "manifest.json")
        dataset_name = self._manifest.get("dataset") or self._manifest.get("format")
        if dataset_name not in {
            "ogbcube_2x2_same_source_query4_d1d3_to_d0",
            "ogbcube_same_source_query4_compat",
            "ogbcube_same_source_query9_compat",
            "ogbcube_2x2_same_source_query9_d1d8_to_d0",
            "ogbcube_d8_same_source_query9_to_d0",
        }:
            raise ValueError(f"Unexpected OGB-cube query4 dataset: {dataset_name!r}")
        self._shards = self._select_shards()
        if not self._shards:
            raise ValueError(f"No OGB-cube query4 shards selected from {self.data_dir}")
        self._files = BoundedH5FileCache(max_open_files=max_open_files)
        self._group_refs = self._build_group_refs()
        self._source_units = int(self._group_refs.shape[0])
        self._total_rows = self._source_units * self.query_group_size

    def _select_shards(self) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for shard in self._manifest.get("shards", []):
            variant = str(shard.get("source_variant", ""))
            if self.source_variants is not None and variant not in self.source_variants:
                continue
            selected.append(dict(shard))
        return selected

    def _path_from_shard(self, shard_idx: int) -> Path:
        shard = self._shards[shard_idx]
        paths = shard.get("paths")
        raw_path = None
        if isinstance(paths, dict):
            raw_path = (
                paths.get("ogbcube_same_source_query9_h5")
                or paths.get("ogbcube_same_source_query4_h5")
                or paths.get("pusht_source5_query_h5")
                or paths.get("unified_h5")
            )
        if not raw_path:
            raw_path = shard.get("path")
        if not raw_path:
            raise KeyError("OGB-cube query4 shard must define a HDF5 path")
        path = Path(str(raw_path))
        if not path.is_absolute():
            split_candidate = self.data_dir / path
            root_candidate = self.data_dir.parent / path
            path = split_candidate if split_candidate.exists() else root_candidate
        return path

    def _build_group_refs(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for shard_idx, shard in enumerate(self._shards):
            num_groups = int(shard.get("num_groups", 0))
            with h5py.File(self._path_from_shard(shard_idx), "r") as handle:
                if num_groups <= 0:
                    num_groups = int(handle["source_pixels"].shape[0])
                self._validate_query_groups(handle, shard_idx, num_groups)
            rows = np.arange(0, num_groups, self.frame_stride, dtype=np.int64)
            chunks.append(
                np.column_stack(
                    [
                        np.full(rows.shape, shard_idx, dtype=np.int64),
                        rows,
                    ]
                )
            )
        if not chunks:
            raise ValueError("No OGB-cube query4 source groups selected")
        refs = np.concatenate(chunks, axis=0).astype(np.int64, copy=False)
        if self.shuffle_groups:
            rng = np.random.default_rng(self.shuffle_seed)
            rng.shuffle(refs, axis=0)
        return refs

    def _validate_query_groups(
        self, handle: h5py.File, shard_idx: int, num_groups: int
    ) -> None:
        required = (
            "source_pixels",
            "target_pixels",
            "pair_query_idx",
            "pair_scene_idx",
            "query",
            "query_color",
            "target_color",
            "source_key",
            "target_key",
            "target_path",
            "target_row_index",
        )
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"OGB query shard {shard_idx} is missing fields: {missing}")
        expected_pairs = int(num_groups) * self.query_group_size
        if int(handle["source_pixels"].shape[0]) != int(num_groups):
            raise ValueError(f"OGB query shard {shard_idx} source group count mismatch")
        if int(handle["target_pixels"].shape[0]) != expected_pairs:
            raise ValueError(f"OGB query shard {shard_idx} target row count mismatch")

        query_indices = np.asarray(handle["pair_query_idx"][:], dtype=np.int64)
        scene_indices = np.asarray(handle["pair_scene_idx"][:], dtype=np.int64)
        expected_indices = np.tile(
            np.arange(self.query_group_size, dtype=np.int64), int(num_groups)
        )
        expected_scenes = np.repeat(np.arange(int(num_groups), dtype=np.int64), self.query_group_size)
        if query_indices.shape != (expected_pairs,) or not np.array_equal(
            query_indices, expected_indices
        ):
            raise ValueError(
                f"OGB query shard {shard_idx} does not contain one ordered complete query group"
            )
        if scene_indices.shape != (expected_pairs,) or not np.array_equal(
            scene_indices, expected_scenes
        ):
            raise ValueError(f"OGB query shard {shard_idx} pair_scene_idx is not group-aligned")

        query = [_decode_text(value) for value in handle["query"][:]]
        query_color = [_decode_text(value) for value in handle["query_color"][:]]
        target_color = [_decode_text(value) for value in handle["target_color"][:]]
        source_key = [_decode_text(value) for value in handle["source_key"][:]]
        target_key = [_decode_text(value) for value in handle["target_key"][:]]
        target_path = [_decode_text(value) for value in handle["target_path"][:]]
        target_row_index = np.asarray(handle["target_row_index"][:], dtype=np.int64)
        if target_row_index.shape != (expected_pairs,):
            raise ValueError(
                f"OGB query shard {shard_idx} target_row_index shape mismatch"
            )
        for group_idx in range(int(num_groups)):
            start = group_idx * self.query_group_size
            stop = start + self.query_group_size
            colors = query_color[start:stop]
            if len(set(colors)) != self.query_group_size:
                raise ValueError(
                    f"OGB query shard {shard_idx} group {group_idx} repeats query colors"
                )
            if len(set(source_key[start:stop])) != 1:
                raise ValueError(
                    f"OGB query shard {shard_idx} group {group_idx} mixes source frames"
                )
            for pair_idx in range(start, stop):
                if not target_key[pair_idx]:
                    raise ValueError(
                        f"OGB query shard {shard_idx} pair {pair_idx} has empty target_key"
                    )
                expected_target_key = (
                    f"{target_path[pair_idx]}::pixels[{int(target_row_index[pair_idx])}]"
                )
                if target_key[pair_idx] != expected_target_key:
                    raise ValueError(
                        "OGB target_key does not uniquely identify target_path/row: "
                        f"shard={shard_idx}, pair={pair_idx}"
                    )
                if target_color[pair_idx] != query_color[pair_idx] or query[pair_idx] != (
                    f"{query_color[pair_idx]} arm"
                ):
                    raise ValueError(
                        "OGB query/color/target semantics are inconsistent: "
                        f"shard={shard_idx}, pair={pair_idx}"
                    )

    def __len__(self) -> int:
        return int(self._total_rows)

    @property
    def inbatch_group_size(self) -> int:
        return self.query_group_size

    def _open_file(self, shard_idx: int) -> h5py.File:
        return self._files.open(int(shard_idx), self._path_from_shard(shard_idx))

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError("OGBCubeSameSourceQuery4HDF5Dataset index out of range")
        source_unit = int(index // self.query_group_size)
        query_slot = int(index % self.query_group_size)
        shard_idx, group_idx = [int(x) for x in self._group_refs[source_unit]]
        shard = self._shards[shard_idx]
        pair_idx = group_idx * self.query_group_size + query_slot
        handle = self._open_file(shard_idx)

        visible = int(handle["pair_query_visible"][pair_idx]) if "pair_query_visible" in handle else 1
        if self.require_visible and visible != 1:
            raise ValueError(
                "Invisible query row encountered in OGB-cube query support: "
                f"shard={shard.get('source_variant')}:{shard.get('source_shard')} "
                f"group={group_idx} slot={query_slot}"
            )

        frame = _image_from_array(handle["source_pixels"][group_idx])
        target = _image_from_array(handle["target_pixels"][pair_idx])
        raw_query = _decode_text(handle["query"][pair_idx])
        query_color = _decode_text(handle["query_color"][pair_idx]) if "query_color" in handle else ""
        target_color = _decode_text(handle["target_color"][pair_idx]) if "target_color" in handle else query_color
        query = f"{query_color} arm" if query_color else raw_query
        pair_role = _decode_text(handle["pair_role"][pair_idx]) if "pair_role" in handle else ""
        source_key = _decode_text(handle["source_key"][pair_idx]) if "source_key" in handle else ""
        target_key = _decode_text(handle["target_key"][pair_idx]) if "target_key" in handle else ""
        if not target_key:
            raise ValueError(
                f"OGB query target key is empty: shard={shard_idx}, pair={pair_idx}"
            )
        positive_key = f"ogbcube_querytarget:{target_key}:{target_color}"
        meta = {
            "id": f"OGBCubeQuery4:{shard.get('source_variant')}:{shard.get('source_shard')}:{group_idx}:{query_slot}",
            "task": "ogbcube_same_source_query4",
            "source_variant": shard.get("source_variant"),
            "source_shard": shard.get("source_shard"),
            "shard_idx": f"{shard.get('source_variant')}:{shard.get('source_shard')}",
            "source_local_index": int(group_idx),
            "target_local_index": int(pair_idx),
            "source_group_index": int(group_idx),
            "pair_index": int(pair_idx),
            "pair_query_idx": int(query_slot),
            "pair_query_visible": int(visible),
            "pair_role": pair_role,
            "query": query,
            "raw_query": raw_query,
            "query_color": query_color,
            "target_color": target_color,
            "source_key": source_key,
            "target_key": target_key,
            "contrast_positive_key": positive_key,
        }
        return Sample(
            frames=[frame],
            target_frames=[target],
            target_label_map=_empty_label_map(),
            label_map=_empty_label_map(),
            targets=[],
            meta=meta,
            query=query,
        )


class OGBCubeCleanBankInfoNCEMixHDF5Dataset(Dataset[Sample]):
    """Pool OGB core-clean rows and D3 same-source query4 rows into banks."""

    def __init__(
        self,
        *,
        core_data_root: str | Path | dict[str, Any],
        query4_data_dir: str | Path | dict[str, Any] | None = None,
        core_source_variants: list[str] | tuple[str, ...] | None = None,
        core_target_variant: str = "d0",
        core_query: str = "core arm",
        core_query4_data_dir: str | Path | dict[str, Any] | None = None,
        core_query_template: str = "{color} arm",
        core_query_require_metadata: bool = True,
        core_frame_stride: int = 5,
        core_temporal_offsets: list[int] | tuple[int, ...] | None = None,
        core_temporal_center_stride: int = 1,
        core_split_manifest_path: str | Path | None = None,
        core_shuffle_groups: bool = True,
        core_balance_seed: int = 6274,
        query_source_variants: list[str] | tuple[str, ...] | None = None,
        query_frame_stride: int = 1,
        query_group_size: int = 4,
        query_require_visible: bool = True,
        query_shuffle_groups: bool = False,
        cleanbank_size: int = 512,
        core_rows_per_bank: int = 460,
        core_bank_mode: str = "sequential",
        query_rows_per_bank: int | None = None,
        source5_rows_per_bank: int | None = None,
        max_supergroups: int | None = None,
        contrast_supergroup_merge_factor: int = 1,
        num_frames: int = 1,
        require_complete_query_groups: bool = True,
        max_open_files: int = 16,
        **_unused: Any,
    ) -> None:
        if query_rows_per_bank is None:
            query_rows_per_bank = 0 if source5_rows_per_bank is None else int(source5_rows_per_bank)
        self.cleanbank_size = int(cleanbank_size)
        self.core_rows_per_bank = int(core_rows_per_bank)
        self.core_bank_mode = str(core_bank_mode)
        self.query_rows_per_bank = int(query_rows_per_bank)
        self.contrast_supergroup_merge_factor = max(1, int(contrast_supergroup_merge_factor))
        self.query_group_size = int(query_group_size)

        if self.cleanbank_size <= 1:
            raise ValueError("cleanbank_size must be at least 2")
        if self.core_rows_per_bank <= 0:
            raise ValueError("core_rows_per_bank must be positive")
        if self.query_rows_per_bank < 0:
            raise ValueError("query_rows_per_bank must be non-negative")
        if self.core_rows_per_bank + self.query_rows_per_bank != self.cleanbank_size:
            raise ValueError("core_rows_per_bank + query_rows_per_bank must equal cleanbank_size")
        if require_complete_query_groups and self.query_rows_per_bank:
            if self.query_rows_per_bank % self.query_group_size != 0:
                raise ValueError("query_rows_per_bank must preserve complete same-source query groups")

        self.core = OGBCubeCoreCleanHDF5Dataset(
            core_data_root,
            source_variants=core_source_variants,
            target_variant=core_target_variant,
            num_frames=num_frames,
            frame_stride=core_frame_stride,
            temporal_offsets=core_temporal_offsets,
            temporal_center_stride=core_temporal_center_stride,
            query=core_query,
            query4_data_dir=core_query4_data_dir,
            query_template=core_query_template,
            require_query_metadata=core_query_require_metadata,
            split_manifest_path=core_split_manifest_path,
            shuffle_refs=core_shuffle_groups,
            balance_seed=core_balance_seed,
            max_open_files=max_open_files,
        )
        self._bound_core_groups: list[list[int]] | None = None
        self._bound_bundle_size = 0
        if self.core_bank_mode in {"bound_by_positive_key", "bound_by_row"}:
            self._bound_core_groups = self._build_bound_core_groups(core_balance_seed)
            if not self._bound_core_groups:
                raise ValueError("No complete bound OGB-cube core groups available")
            self._bound_bundle_size = len(self._bound_core_groups[0])
            if self.core_rows_per_bank % self._bound_bundle_size != 0:
                raise ValueError(
                    "core_rows_per_bank must be divisible by the bound core bundle size "
                    f"({self._bound_bundle_size})"
                )
        elif self.core_bank_mode != "sequential":
            raise ValueError(f"Unsupported OGB-cube core_bank_mode: {self.core_bank_mode!r}")
        self.query = None
        if self.query_rows_per_bank > 0:
            if query4_data_dir is None:
                raise ValueError("query4_data_dir is required when query rows are used")
            self.query = OGBCubeSameSourceQuery4HDF5Dataset(
                query4_data_dir,
                num_frames=num_frames,
                frame_stride=query_frame_stride,
                query_group_size=self.query_group_size,
                source_variants=query_source_variants,
                require_visible=query_require_visible,
                shuffle_groups=query_shuffle_groups,
                shuffle_seed=core_balance_seed,
                max_open_files=max_open_files,
            )

        if self._bound_core_groups is not None:
            self._bound_bundles_per_bank = self.core_rows_per_bank // self._bound_bundle_size
            base_length = len(self._bound_core_groups) // self._bound_bundles_per_bank
        else:
            self._bound_bundles_per_bank = 0
            base_length = len(self.core) // self.core_rows_per_bank
        if self.query_rows_per_bank > 0:
            if self.query is None:
                raise ValueError("query dataset was not initialized")
            base_length = min(base_length, len(self.query) // self.query_rows_per_bank)
        if max_supergroups is not None:
            base_length = min(base_length, int(max_supergroups))
        if base_length <= 0:
            raise ValueError("No complete OGB-cube clean-bank groups available")
        self._base_length = int(base_length)
        self._group_size = int(self.cleanbank_size)
        self._length = self._base_length * self._group_size

    def _build_bound_core_groups(self, seed: int) -> list[list[int]]:
        refs = getattr(self.core, "_refs")
        shards = getattr(self.core, "_shards")
        expected_variants = list(getattr(self.core, "source_variants", ()))
        groups: dict[str, dict[str, int]] = {}
        for core_index, (shard_idx_raw, row_idx_raw) in enumerate(refs):
            shard_idx = int(shard_idx_raw)
            row_idx = int(row_idx_raw)
            shard = shards[shard_idx]
            query, _, _ = self.core._query_for(shard, row_idx)
            key = f"ogbcube_coreclean:{shard['source_shard']}:{row_idx}:{query}"
            variant = str(shard["source_variant"])
            groups.setdefault(key, {})[variant] = int(core_index)

        ordered_groups: list[list[int]] = []
        for key in sorted(groups):
            by_variant = groups[key]
            if all(variant in by_variant for variant in expected_variants):
                ordered_groups.append([by_variant[variant] for variant in expected_variants])
        if self.core.shuffle_refs:
            rng = np.random.default_rng(int(seed))
            rng.shuffle(ordered_groups)
        return ordered_groups

    def __len__(self) -> int:
        return self._length

    @property
    def base_length(self) -> int:
        return self._base_length

    @property
    def inbatch_group_size(self) -> int:
        return self._group_size

    def _core_positive_key(self, meta: dict[str, Any]) -> str:
        return str(meta.get("contrast_positive_key") or meta.get("id") or "ogbcube_core_unknown")

    def _query_positive_key(self, meta: dict[str, Any]) -> str:
        return str(meta.get("contrast_positive_key") or meta.get("target_key") or meta.get("id") or "ogbcube_query_unknown")

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError("OGBCubeCleanBankInfoNCEMixHDF5Dataset index out of range")
        super_idx = index // self._group_size
        slot = index % self._group_size
        if slot < self.core_rows_per_bank:
            if self._bound_core_groups is not None:
                bundle_idx = super_idx * self._bound_bundles_per_bank + (slot // self._bound_bundle_size)
                bundle_slot = slot % self._bound_bundle_size
                child_index = self._bound_core_groups[bundle_idx][bundle_slot]
            else:
                bundle_idx = -1
                bundle_slot = -1
                child_index = super_idx * self.core_rows_per_bank + slot
            sample = self.core[child_index]
            meta = dict(sample.meta)
            positive_key = self._core_positive_key(meta)
            family = "ogbcube_core_clean_exactbank"
        else:
            if self.query is None:
                raise RuntimeError("query rows requested but query dataset is absent")
            local_slot = slot - self.core_rows_per_bank
            child_index = super_idx * self.query_rows_per_bank + local_slot
            sample = self.query[child_index]
            meta = dict(sample.meta)
            positive_key = self._query_positive_key(meta)
            family = "ogbcube_same_source_query_exactbank"

        contrast_super_idx = int(super_idx) // int(self.contrast_supergroup_merge_factor)
        meta.update(
            {
                "real_cleanbank_shard_idx": meta.get("shard_idx"),
                "real_cleanbank_source_local_index": meta.get("source_local_index"),
                "exact_cleanbank_child_index": int(child_index),
                "mix_family": family,
                "dataset_task": "ogbcube_cleanbank_infonce_mix",
                "cleanbank_super_idx": int(super_idx),
                "cleanbank_slot": int(slot),
                "cleanbank_group_size": int(self._group_size),
                "cleanbank_core_rows": int(self.core_rows_per_bank),
                "cleanbank_query_rows": int(self.query_rows_per_bank),
                "cleanbank_source5_rows": int(self.query_rows_per_bank),
                "cleanbank_core_bank_mode": self.core_bank_mode,
                "cleanbank_core_temporal_offsets": list(self.core.temporal_offsets)
                if self.core.temporal_offsets is not None
                else [],
                "cleanbank_core_temporal_center_stride": int(self.core.temporal_center_stride),
                "cleanbank_bound_bundle_idx": int(bundle_idx) if slot < self.core_rows_per_bank else -1,
                "cleanbank_bound_bundle_slot": int(bundle_slot) if slot < self.core_rows_per_bank else -1,
                "cleanbank_bound_bundle_size": int(self._bound_bundle_size),
                "cleanbank_contrast_super_idx": int(contrast_super_idx),
                "cleanbank_contrast_merge_factor": int(self.contrast_supergroup_merge_factor),
                # Reuse the established contrastive collator/trainer branch.
                "task": "pusht_source5_query",
                "shard_idx": f"ogbcube_cleanbank{int(self._group_size)}",
                "source_local_index": int(contrast_super_idx),
                "contrast_positive_key": positive_key,
                "main_loss_weight": 0.0,
            }
        )
        return Sample(
            frames=sample.frames,
            target_frames=sample.target_frames,
            target_label_map=sample.target_label_map,
            label_map=sample.label_map if sample.label_map is not None else _empty_label_map(),
            targets=sample.targets,
            meta=meta,
            query=sample.query,
            text_input=sample.text_input,
        )


__all__ = [
    "OGBCubeCoreCleanHDF5Dataset",
    "OGBCubeSameSourceQuery4HDF5Dataset",
    "OGBCubeCleanBankInfoNCEMixHDF5Dataset",
]
