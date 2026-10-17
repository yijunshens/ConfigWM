"""Canonical sharding utilities used across ``src.data``."""

from __future__ import annotations

from . import common as _common
from .common import (
    DEFAULT_JSONL_CHUNK_LINES,
    DEFAULT_MANIFEST_CACHE_ROOT,
    DEFAULT_PARQUET_SAMPLES_PER_CHUNK,
    DEFAULT_STREAM_SHUFFLE_BUFFER,
    DEFAULT_TAR_SAMPLES_PER_CHUNK,
    MANIFEST_VERSION,
    CachedManifestDescriptor,
    ChunkInfo,
    ManifestBackedShardSource,
    ManifestSummary,
    RequiredPrebuiltPath,
    ShardInfo,
    describe_cached_manifest,
    get_shard_info,
    make_record_rng,
)
from .common import (
    set_default_manifest_cache_root as _set_default_manifest_cache_root,
)
from .common import (
    set_required_prebuilt_paths as _set_required_prebuilt_paths,
)
from .json import normalize_jsonl_source, normalize_jsonl_sources
from .jsonl import (
    JsonlShardSource,
    ensure_jsonl_meta,
    iter_jsonl_chunks,
)
from .parquet import (
    ParquetShardSource,
    ensure_parquet_meta,
    iter_parquet_chunks,
    resolve_parquet_files,
)
from .tar import (
    TarShardSource,
    ensure_offsets,
    iter_tar_samples,
)


def set_default_manifest_cache_root(path):
    resolved = _set_default_manifest_cache_root(path)
    globals()["DEFAULT_MANIFEST_CACHE_ROOT"] = (
        _common.DEFAULT_MANIFEST_CACHE_ROOT
    )
    return resolved


def set_required_prebuilt_paths(requirements):
    return _set_required_prebuilt_paths(requirements)


__all__ = [
    "DEFAULT_JSONL_CHUNK_LINES",
    "DEFAULT_MANIFEST_CACHE_ROOT",
    "DEFAULT_PARQUET_SAMPLES_PER_CHUNK",
    "DEFAULT_STREAM_SHUFFLE_BUFFER",
    "DEFAULT_TAR_SAMPLES_PER_CHUNK",
    "MANIFEST_VERSION",
    "CachedManifestDescriptor",
    "ChunkInfo",
    "JsonlShardSource",
    "ManifestBackedShardSource",
    "ManifestSummary",
    "ParquetShardSource",
    "RequiredPrebuiltPath",
    "ShardInfo",
    "TarShardSource",
    "describe_cached_manifest",
    "get_shard_info",
    "iter_jsonl_chunks",
    "iter_parquet_chunks",
    "iter_tar_samples",
    "make_record_rng",
    "normalize_jsonl_source",
    "normalize_jsonl_sources",
    "resolve_parquet_files",
    "set_default_manifest_cache_root",
    "set_required_prebuilt_paths",
]
