"""Paired HDF5 adapter for aligned PushT distractor/clean caches."""

from __future__ import annotations

import bisect
import csv
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
from PIL import Image
import torch
from torch.utils.data import Dataset

from ...schema import Sample
from .temporal import BoundedH5FileCache


class PushTPairedHDF5Dataset(Dataset[Sample]):
    """Read aligned PushT variant/clean frames as paired single-image samples."""

    def __init__(
        self,
        data_dir: str | Path,
        target_data_dir: str | Path,
        *,
        query: str = "focus on the colored T block",
        query_metadata_path: str | Path | None = None,
        query_template: str = "focus on the {color} T block",
        num_frames: int = 1,
        num_hard_negatives: int = 0,
        max_open_files: int = 16,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.target_data_dir = Path(target_data_dir).expanduser().resolve()
        self.query = str(query)
        self.query_template = str(query_template)
        self.query_metadata_path = (
            None
            if query_metadata_path is None
            else Path(query_metadata_path).expanduser().resolve()
        )
        self._query_color_by_episode = self._load_query_metadata(
            self.query_metadata_path
        )
        self.num_frames = int(num_frames)
        self.num_hard_negatives = int(num_hard_negatives)
        self.inbatch_group_size = 1
        if self.num_frames != 1:
            raise ValueError("PushTPairedHDF5Dataset supports num_frames=1 only")
        if self.num_hard_negatives != 0:
            raise ValueError(
                "PushTPairedHDF5Dataset does not support hard negatives yet"
            )

        self._source_manifest = self._load_manifest(self.data_dir)
        self._target_manifest = self._load_manifest(self.target_data_dir)
        self._validate_manifests()

        self._step_starts: list[int] = []
        total = 0
        for shard in self._source_manifest["shards"]:
            self._step_starts.append(total)
            total += self._shard_steps(shard)
        self._total_steps = total
        self._source_files = BoundedH5FileCache(max_open_files=max_open_files)
        self._target_files = BoundedH5FileCache(max_open_files=max_open_files)

    @staticmethod
    def _extract_color_from_record(record: dict) -> str | None:
        for key in ("core_color", "color", "target_color"):
            if key in record and record[key] is not None:
                return str(record[key])
        return None

    def _load_query_metadata(self, metadata_path: Path | None) -> dict[int, str]:
        if metadata_path is None:
            return {}
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing query metadata: {metadata_path}")

        color_by_episode: dict[int, str] = {}
        if metadata_path.is_dir():
            metadata_files = sorted(
                metadata_path.glob("shard_*/episode_metadata.jsonl")
            )
            if not metadata_files:
                raise FileNotFoundError(
                    f"No shard_*/episode_metadata.jsonl files under {metadata_path}"
                )
        else:
            metadata_files = [metadata_path]

        require_global_episode_idx = metadata_path.is_dir()

        def add_record(record: dict, source: Path) -> None:
            if require_global_episode_idx and "global_episode_idx" not in record:
                raise KeyError(
                    f"Directory metadata must use global_episode_idx: {source}"
                )
            episode_idx = self._record_episode_idx(record)
            color = self._extract_color_from_record(record)
            if color is None:
                raise ValueError(
                    f"Missing core color in {source}, episode_idx={episode_idx}"
                )
            previous = color_by_episode.get(episode_idx)
            if previous is not None:
                raise ValueError(
                    f"Duplicate metadata for global episode_idx={episode_idx}: "
                    f"{previous!r} and {color!r} in {source}"
                )
            color_by_episode[episode_idx] = color

        for source in metadata_files:
            if source.suffix.lower() == ".csv":
                with source.open("r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        add_record(row, source)
            else:
                with source.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            add_record(json.loads(line), source)
        if not color_by_episode:
            raise ValueError(f"Empty query metadata: {metadata_path}")
        return color_by_episode

    @staticmethod
    def _record_episode_idx(record: dict) -> int:
        for key in ("episode_idx", "global_episode_idx", "local_episode_idx"):
            if key in record:
                return int(record[key])
        raise KeyError("metadata record must contain an episode index")

    @staticmethod
    def _load_manifest(root: Path) -> dict:
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    @staticmethod
    def _shard_path(shard: dict) -> str:
        if "path" in shard:
            return str(shard["path"])
        paths = shard.get("paths")
        if isinstance(paths, dict) and "clean" in paths:
            return str(paths["clean"])
        raise KeyError("manifest shard must define path")

    @classmethod
    def _manifest_shard_file(cls, root: Path, shard: dict) -> Path:
        path = Path(cls._shard_path(shard)).expanduser()
        if not path.is_absolute():
            return root / path
        if path.exists():
            return path
        for index in range(len(path.parts) - 1, -1, -1):
            if path.parts[index].startswith("shard_"):
                return root.joinpath(*path.parts[index:])
        return path

    @classmethod
    def _shard_key(cls, shard: dict) -> str:
        path = Path(cls._shard_path(shard))
        parts = path.parts
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return path.name

    @staticmethod
    def _shard_episodes(shard: dict) -> int:
        if "num_episodes" in shard:
            return int(shard["num_episodes"])
        episodes = shard.get("episodes")
        if episodes is not None:
            if isinstance(episodes, list):
                return len(episodes)
            return int(episodes)
        raise KeyError("manifest shard must define episode count")

    @staticmethod
    def _shard_steps(shard: dict) -> int:
        for key in ("steps", "total_frames", "frames"):
            if key in shard:
                return int(shard[key])
        raise KeyError("manifest shard must define frame/step count")

    def _validate_manifests(self) -> None:
        source_shards = self._source_manifest.get("shards", [])
        target_shards = self._target_manifest.get("shards", [])
        if len(source_shards) != len(target_shards):
            raise ValueError("Source/target shard counts do not match")

        for source_entry, target_entry in zip(source_shards, target_shards):
            if self._shard_key(source_entry) != self._shard_key(target_entry):
                raise ValueError("Source/target shard paths do not match")
            if self._shard_episodes(source_entry) != self._shard_episodes(target_entry):
                raise ValueError("Source/target episode counts do not match")
            if self._shard_steps(source_entry) != self._shard_steps(target_entry):
                raise ValueError("Source/target step counts do not match")

    def _open_file(
        self,
        root: Path,
        handles: BoundedH5FileCache,
        manifest: dict,
        shard_idx: int,
    ) -> h5py.File:
        shard_path = self._manifest_shard_file(
            root, manifest["shards"][shard_idx]
        )
        return handles.open(int(shard_idx), shard_path)

    def __len__(self) -> int:
        return self._total_steps

    @property
    def base_length(self) -> int:
        return self._total_steps

    def _global_to_shard_local(self, index: int) -> tuple[int, int]:
        shard_idx = bisect.bisect_right(self._step_starts, index) - 1
        local_index = index - self._step_starts[shard_idx]
        return shard_idx, local_index

    def _query_for_episode(self, episode_idx: int) -> str:
        if not self._query_color_by_episode:
            return self.query
        try:
            color = self._query_color_by_episode[int(episode_idx)]
        except KeyError as exc:
            raise KeyError(
                f"Missing query color for episode_idx={episode_idx} "
                f"in {self.query_metadata_path}"
            ) from exc
        return self.query_template.format(color=color)

    @staticmethod
    def _read_episode_idx(handle: h5py.File, local_index: int) -> int:
        if "episode_idx" in handle:
            return int(handle["episode_idx"][local_index])
        if "ep_idx" in handle:
            return int(handle["ep_idx"][local_index])
        raise KeyError("HDF5 file must contain episode_idx or ep_idx")

    @staticmethod
    def _dataset_name(manifest: dict) -> str | None:
        value = manifest.get("dataset_name", manifest.get("dataset"))
        return None if value is None else str(value)

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self):
            raise IndexError("PushTPairedHDF5Dataset index out of range")

        shard_idx, local_index = self._global_to_shard_local(index)
        source_file = self._open_file(
            self.data_dir, self._source_files, self._source_manifest, shard_idx
        )
        target_file = self._open_file(
            self.target_data_dir,
            self._target_files,
            self._target_manifest,
            shard_idx,
        )

        source_frame = Image.fromarray(source_file["pixels"][local_index]).convert("RGB")
        target_frame = Image.fromarray(target_file["pixels"][local_index]).convert("RGB")

        step_idx = int(source_file["step_idx"][local_index])
        episode_idx = self._read_episode_idx(source_file, local_index)
        target_step_idx = int(target_file["step_idx"][local_index])
        target_episode_idx = self._read_episode_idx(target_file, local_index)
        if step_idx != target_step_idx or episode_idx != target_episode_idx:
            raise ValueError(
                "Source/target row alignment mismatch at "
                f"shard={shard_idx}, local_index={local_index}"
            )

        query = self._query_for_episode(episode_idx)
        return Sample(
            frames=[source_frame],
            target_frames=[target_frame],
            label_map=torch.zeros((1, 1, 1), dtype=torch.long),
            targets=[],
            meta={
                "id": f"{shard_idx}:{local_index}",
                "shard_idx": shard_idx,
                "local_index": local_index,
                "episode_idx": episode_idx,
                "step_idx": step_idx,
                "source_dataset": self._dataset_name(self._source_manifest),
                "target_dataset": self._dataset_name(self._target_manifest),
                "query": query,
                "query_metadata_path": (
                    None
                    if self.query_metadata_path is None
                    else str(self.query_metadata_path)
                ),
            },
            query=query,
        )


__all__ = ["PushTPairedHDF5Dataset"]
