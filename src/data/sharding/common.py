"""Common helpers shared by manifest-backed sharding implementations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import logging
import math
import os
from pathlib import Path
import random
import re
from typing import Any, TypeVar

from src.data._infra import (
    cache_lock as _manifest_lock,
    expand_user_strict as _expand_user_strict,
)
from src.data._infra import (
    iter_shuffle_buffer,
    stable_cache_key,
)
import torch.distributed as dist
from torch.utils.data import get_worker_info

T = TypeVar("T")

MANIFEST_VERSION = 1
DEFAULT_JSONL_CHUNK_LINES = 512
DEFAULT_PARQUET_SAMPLES_PER_CHUNK = 8_192
DEFAULT_TAR_SAMPLES_PER_CHUNK = 512
DEFAULT_MANIFEST_CACHE_ROOT: Path | None = None
DEFAULT_STREAM_SHUFFLE_BUFFER = 2_048
MAX_RECORD_ERROR_LOGS = 10


logger = logging.getLogger(__name__)
_GZIP_SCAN_FALLBACK_WARNED: set[tuple[str, ...]] = set()
_JSONL_LOGICAL_SPLIT_WARNED: set[tuple[tuple[str, ...], int]] = set()
_PARQUET_LOGICAL_SPLIT_WARNED: set[tuple[tuple[str, ...], int]] = set()
_TAR_LOGICAL_SPLIT_WARNED: set[tuple[tuple[str, ...], int]] = set()
_REQUIRED_PREBUILT_PATHS: dict[Path, RequiredPrebuiltPath] = {}


@dataclass(frozen=True)
class ShardInfo:
    """Distributed rank plus DataLoader worker identity."""

    rank: int
    world_size: int
    worker_id: int
    num_workers: int

    @property
    def slot_id(self) -> int:
        return self.rank * self.num_workers + self.worker_id

    @property
    def total_slots(self) -> int:
        return self.world_size * self.num_workers

    @property
    def id(self) -> int:
        return self.slot_id

    @property
    def total(self) -> int:
        return self.total_slots


@dataclass(frozen=True)
class ChunkInfo:
    """Format-agnostic physical manifest chunk metadata."""

    file_index: int
    file_path: Path
    start_record_index: int
    num_items: int


@dataclass(frozen=True)
class CachedManifestDescriptor:
    """Resolved on-disk location for one cached manifest payload."""

    kind: str
    files: tuple[Path, ...]
    params: dict[str, Any]
    cache_key: str
    manifest_path: Path


@dataclass(frozen=True)
class ManifestSummary:
    kind: str
    file_count: int
    chunk_count: int
    total_items: int
    samples_per_chunk: int | None


@dataclass(frozen=True)
class RequiredPrebuiltPath:
    """Path that training must load from disk instead of building at runtime."""

    path: Path
    dataset_label: str
    description: str


class ManifestBackedShardSource:
    """Typed wrapper around one manifest-backed sharding source."""

    def __init__(
        self,
        *,
        kind: str,
        raw_manifest: Mapping[str, Any],
        total_items_key: str,
    ) -> None:
        self._kind = str(kind)
        self._raw_manifest = dict(raw_manifest)
        self._files = tuple(_normalize_paths(self._raw_manifest["files"]))
        self._total_items = int(self._raw_manifest[total_items_key])

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def files(self) -> tuple[Path, ...]:
        return self._files

    @property
    def raw_manifest(self) -> dict[str, Any]:
        return self._raw_manifest

    @property
    def total_items(self) -> int:
        return self._total_items

    def __len__(self) -> int:
        return self.total_items

    def manifest_summary(self) -> ManifestSummary:
        raw_value = self.raw_manifest.get("samples_per_chunk")
        samples_per_chunk = int(raw_value) if raw_value is not None else None
        return ManifestSummary(
            kind=self.kind,
            file_count=len(self.files),
            chunk_count=len(self.raw_manifest.get("chunk_file_indices", ())),
            total_items=self.total_items,
            samples_per_chunk=samples_per_chunk,
        )

    def iter_chunks(self) -> Iterator[ChunkInfo]:
        raise NotImplementedError


def get_shard_info() -> ShardInfo:
    """Return current distributed/data-loader sharding context."""
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # Spawned DataLoader workers inherit torchrun's rank, not its process group.
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    worker_info = get_worker_info()
    if worker_info is None:
        worker_id = 0
        num_workers = 1
    else:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

    return ShardInfo(
        rank=rank,
        world_size=world_size,
        worker_id=worker_id,
        num_workers=num_workers,
    )


def set_default_manifest_cache_root(path: str | Path | None) -> Path | None:
    """Set the default manifest cache root shared by sharding helpers."""
    global DEFAULT_MANIFEST_CACHE_ROOT

    resolved = None if path is None else _expand_user_strict(path).resolve()
    DEFAULT_MANIFEST_CACHE_ROOT = resolved
    return resolved


def set_required_prebuilt_paths(
    requirements: Sequence[RequiredPrebuiltPath] | None,
) -> tuple[RequiredPrebuiltPath, ...]:
    """Register paths that must already exist before dataset construction."""
    global _REQUIRED_PREBUILT_PATHS

    previous = tuple(_REQUIRED_PREBUILT_PATHS.values())
    resolved_requirements: dict[Path, RequiredPrebuiltPath] = {}
    for requirement in requirements or ():
        resolved_path = _expand_user_strict(requirement.path).resolve()
        resolved_requirements[resolved_path] = RequiredPrebuiltPath(
            path=resolved_path,
            dataset_label=str(requirement.dataset_label),
            description=str(requirement.description),
        )

    _REQUIRED_PREBUILT_PATHS = resolved_requirements
    return previous


def _normalize_paths(paths: Sequence[str | Path]) -> list[Path]:
    return [_expand_user_strict(path).resolve() for path in paths]


def _format_files_identity(files: Sequence[str | Path]) -> str:
    resolved = [str(Path(path)) for path in files]
    if not resolved:
        return "<unknown>"
    if len(resolved) == 1:
        return resolved[0]
    return f"{resolved[0]} (+{len(resolved) - 1} more)"


def _default_cache_root(cache_root: str | Path | None) -> Path:
    if cache_root is None:
        if DEFAULT_MANIFEST_CACHE_ROOT is None:
            raise ValueError(
                "manifest_cache_root is not configured. Set top-level manifest_cache_root "
                "in data.yaml before building manifest-backed datasets."
            )
        root = DEFAULT_MANIFEST_CACHE_ROOT
    else:
        root = cache_root
    root = _expand_user_strict(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _manifest_cache_key(
    kind: str,
    files: Sequence[Path],
    *,
    params: Mapping[str, Any],
) -> str:
    return stable_cache_key(
        kind,
        files,
        params=params,
        version=MANIFEST_VERSION,
    )


def _manifest_cache_label(files: Sequence[Path]) -> str:
    if not files:
        return "manifest"

    if len(files) == 1:
        path = files[0]
        parent_name = path.parent.name or "root"
        label = f"{parent_name}-{path.stem}"
    else:
        common_parent = Path(
            os.path.commonpath([str(path.parent) for path in files])
        )
        parts = [
            part
            for part in common_parent.parts
            if part not in {common_parent.anchor, ""}
        ]
        tail = (
            parts[-2:]
            if len(parts) >= 2
            else parts[-1:]
            if parts
            else [files[0].parent.name or files[0].stem]
        )
        label = "-".join([*tail, f"{len(files)}files"])

    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("._-")
    if not safe_label:
        return "manifest"
    if len(safe_label) > 80:
        safe_label = safe_label[:80].rstrip("._-")
    return safe_label


def _manifest_cache_paths(
    kind: str,
    files: Sequence[Path],
    cache_key: str,
    *,
    cache_root: str | Path | None,
) -> Path:
    cache_dir = _default_cache_root(cache_root)
    label = _manifest_cache_label(files)
    return cache_dir / f"{kind}-{label}-{cache_key}.json"


def describe_cached_manifest(
    *,
    kind: str,
    files: Sequence[str | Path],
    params: Mapping[str, Any],
    cache_root: str | Path | None,
) -> CachedManifestDescriptor:
    """Resolve the deterministic cache key and path for one manifest."""
    normalized_files = tuple(_normalize_paths(files))
    cache_key = _manifest_cache_key(
        kind,
        normalized_files,
        params=params,
    )
    manifest_path = _manifest_cache_paths(
        kind,
        normalized_files,
        cache_key,
        cache_root=cache_root,
    )
    return CachedManifestDescriptor(
        kind=str(kind),
        files=normalized_files,
        params=dict(params),
        cache_key=cache_key,
        manifest_path=manifest_path,
    )


def _raise_for_required_prebuilt_paths(paths: Sequence[str | Path]) -> None:
    matches: list[RequiredPrebuiltPath] = []
    for path in _normalize_paths(paths):
        requirement = _REQUIRED_PREBUILT_PATHS.get(path)
        if requirement is not None:
            matches.append(requirement)

    if not matches:
        return

    culprit = next(
        (item for item in matches if not item.path.exists()), matches[0]
    )
    state = (
        "is missing"
        if not culprit.path.exists()
        else "exists but is invalid or incomplete"
    )
    raise FileNotFoundError(
        f"Dataset '{culprit.dataset_label}' requires {culprit.description} at "
        f"{culprit.path}, but it {state}. Run `python -m src.data.manifest.build ...` "
        "before training; runtime prebuild is disabled for this dataset."
    )


def _plan_logical_subchunks(
    *,
    physical_chunk_sizes: Sequence[int],
    total_items: int,
    total_slots: int,
) -> list[list[tuple[int, int]]] | None:
    physical_chunk_count = len(physical_chunk_sizes)
    if physical_chunk_count >= total_slots or total_items <= 0:
        return None

    target_chunk_count = min(total_slots, total_items)
    max_items_per_logical_chunk = max(
        1, math.ceil(total_items / target_chunk_count)
    )

    plan: list[list[tuple[int, int]]] = []
    for raw_chunk_size in physical_chunk_sizes:
        chunk_size = int(raw_chunk_size)
        if chunk_size <= 0:
            plan.append([])
            continue

        logical_chunk_count = max(
            1, math.ceil(chunk_size / max_items_per_logical_chunk)
        )
        base_size = chunk_size // logical_chunk_count
        extra_items = chunk_size % logical_chunk_count

        chunk_plan: list[tuple[int, int]] = []
        offset = 0
        for logical_index in range(logical_chunk_count):
            logical_size = base_size + (1 if logical_index < extra_items else 0)
            if logical_size <= 0:
                continue
            chunk_plan.append((offset, logical_size))
            offset += logical_size
        plan.append(chunk_plan)

    return plan


def _owned_chunk_indices(
    num_chunks: int,
    *,
    seed: int,
    epoch: int,
    info: ShardInfo,
    shuffle: bool = True,
) -> list[int]:
    indices = list(range(num_chunks))
    if shuffle:
        random.Random(seed + epoch * 1_000_003).shuffle(indices)
    rank_indices = indices[info.rank :: info.world_size]
    return rank_indices[info.worker_id :: info.num_workers]


def _owns_stream_record(record_index: int, info: ShardInfo) -> bool:
    if record_index % info.world_size != info.rank:
        return False
    rank_local_index = record_index // info.world_size
    return rank_local_index % info.num_workers == info.worker_id


def _iter_shuffle_buffer(
    records: Iterator[T],
    *,
    shuffle: bool,
    shuffle_buffer_size: int,
    seed: int,
    epoch: int,
    info: ShardInfo,
) -> Iterator[T]:
    yield from iter_shuffle_buffer(
        records,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        epoch=epoch,
        slot_id=info.slot_id,
    )


def _maybe_shuffle_paths(
    paths: Sequence[Path],
    *,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> list[Path]:
    ordered_paths = list(paths)
    if shuffle and len(ordered_paths) > 1:
        random.Random(seed + epoch * 1_000_003).shuffle(ordered_paths)
    return ordered_paths


def _warn_gzip_scan_fallback_once(
    *,
    files: Sequence[Path],
    info: ShardInfo,
) -> None:
    if info.rank != 0 or info.worker_id != 0:
        return

    gzip_paths = tuple(str(path) for path in files if path.suffix == ".gz")
    if not gzip_paths or gzip_paths in _GZIP_SCAN_FALLBACK_WARNED:
        return

    _GZIP_SCAN_FALLBACK_WARNED.add(gzip_paths)
    logger.warning(
        "Falling back to scan-based sharding for gzip JSONL files from %s. "
        "Manifest-based sharding currently supports plain .jsonl only.",
        _format_files_identity(gzip_paths),
    )


def _warn_logical_split_once(
    *,
    kind_label: str,
    manifest: Mapping[str, Any],
    info: ShardInfo,
    logical_chunk_count: int,
    warned_keys: set[tuple[tuple[str, ...], int]],
) -> None:
    if info.rank != 0 or info.worker_id != 0:
        return

    files_key = tuple(str(path) for path in manifest["files"])
    key = (files_key, info.total_slots)
    if key in warned_keys:
        return

    warned_keys.add(key)
    logger.warning(
        "%s manifest for %s has fewer physical chunks (%d) than total rank/worker slots (%d). "
        "Using logical sub-chunk splitting to avoid empty workers; generated %d logical chunks.",
        kind_label,
        _format_files_identity(manifest["files"]),
        len(manifest["chunk_file_indices"]),
        info.total_slots,
        logical_chunk_count,
    )


def _warn_parquet_logical_split_once(
    *,
    manifest: Mapping[str, Any],
    info: ShardInfo,
    logical_chunk_count: int,
) -> None:
    _warn_logical_split_once(
        kind_label="Parquet",
        manifest=manifest,
        info=info,
        logical_chunk_count=logical_chunk_count,
        warned_keys=_PARQUET_LOGICAL_SPLIT_WARNED,
    )


def _warn_jsonl_logical_split_once(
    *,
    manifest: Mapping[str, Any],
    info: ShardInfo,
    logical_chunk_count: int,
) -> None:
    _warn_logical_split_once(
        kind_label="JSONL",
        manifest=manifest,
        info=info,
        logical_chunk_count=logical_chunk_count,
        warned_keys=_JSONL_LOGICAL_SPLIT_WARNED,
    )


def _warn_tar_logical_split_once(
    *,
    manifest: Mapping[str, Any],
    info: ShardInfo,
    logical_chunk_count: int,
) -> None:
    _warn_logical_split_once(
        kind_label="Tar",
        manifest=manifest,
        info=info,
        logical_chunk_count=logical_chunk_count,
        warned_keys=_TAR_LOGICAL_SPLIT_WARNED,
    )


def _log_record_error(
    *,
    context: str,
    path: Path,
    record_index: int,
    exc: Exception,
    log_state: dict[str, int],
    file_line: int | None = None,
    snippet: str | None = None,
) -> None:
    log_state["count"] += 1
    if log_state["count"] > MAX_RECORD_ERROR_LOGS:
        return

    location = f"path={path}"
    if file_line is not None:
        location += f", file_line={file_line}"
    message = (
        f"{context}: skipping malformed record at {location}, record_index={record_index}: "
        f"{type(exc).__name__}: {exc}"
    )
    if snippet:
        message += f" | snippet={snippet!r}"
    logger.warning(message)

    if log_state["count"] == MAX_RECORD_ERROR_LOGS:
        logger.warning(
            "%s: suppressing further malformed-record warnings on this worker after %d records.",
            context,
            MAX_RECORD_ERROR_LOGS,
        )


def _stable_int(value: object) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def make_record_rng(seed: int, epoch: int, record_id: object) -> random.Random:
    """Deterministic RNG for one source record in one epoch."""
    mixed = seed + epoch * 1_000_003 + _stable_int(record_id)
    return random.Random(mixed)
