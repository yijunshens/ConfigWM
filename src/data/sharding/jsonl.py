"""JSONL manifesting and iteration helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import gzip
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

from src.data._infra import atomic_write_bytes as _atomic_write_bytes

from .common import (
    DEFAULT_JSONL_CHUNK_LINES,
    DEFAULT_STREAM_SHUFFLE_BUFFER,
    MANIFEST_VERSION,
    ChunkInfo,
    ManifestBackedShardSource,
    ShardInfo,
    _iter_shuffle_buffer,
    _log_record_error,
    _maybe_shuffle_paths,
    _normalize_paths,
    _owned_chunk_indices,
    _owns_stream_record,
    _plan_logical_subchunks,
    _warn_gzip_scan_fallback_once,
    _warn_jsonl_logical_split_once,
    get_shard_info,
)

T = TypeVar("T")


logger = logging.getLogger(__name__)


def _iter_jsonl_scan_fallback(
    paths: Sequence[Path],
    *,
    parser: Callable[[str], T],
    info: ShardInfo,
    seed: int,
    epoch: int,
    shuffle: bool,
    shuffle_buffer_size: int,
) -> Iterator[tuple[int, Path, T]]:
    log_state = {"count": 0}

    def iter_owned_records() -> Iterator[tuple[int, Path, T]]:
        record_index = 0
        for path in _maybe_shuffle_paths(
            paths, shuffle=shuffle, seed=seed, epoch=epoch
        ):
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(
                path, "rt", encoding="utf-8", errors="ignore"
            ) as handle:
                for file_line, line in enumerate(handle, start=1):
                    owned = _owns_stream_record(record_index, info)
                    current_index = record_index
                    record_index += 1
                    if not owned:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield current_index, path, parser(line)
                    except Exception as exc:
                        _log_record_error(
                            context="_iter_jsonl_scan_fallback",
                            path=path,
                            file_line=file_line,
                            record_index=current_index,
                            exc=exc,
                            log_state=log_state,
                            snippet=line[:200],
                        )

    yield from _iter_shuffle_buffer(
        iter_owned_records(),
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        epoch=epoch,
        info=info,
    )


def _scan_jsonl_line_count(path: Path) -> int:
    """Count lines in a plain JSONL file."""
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


def _default_jsonl_meta_dir(files: Sequence[Path]) -> Path:
    common = Path(os.path.commonpath([str(f.parent) for f in files]))
    return common / ".jsonl_meta"


def ensure_jsonl_meta(
    paths: Sequence[str | Path],
    *,
    meta_dir: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[Path], list[int]]:
    """Ensure per-file JSONL line-count sidecars exist.

    Each sidecar is a plain text file containing a single integer (line count).

    Returns:
        (files, line_counts) — parallel lists.
    """
    files = _normalize_paths(paths)
    gzip_files = [path for path in files if path.suffix == ".gz"]
    if gzip_files:
        raise ValueError(
            "JSONL meta only supports plain .jsonl files. "
            f"Found gzip files: {[str(path) for path in gzip_files]}"
        )

    if meta_dir is not None:
        resolved_meta_dir = Path(meta_dir).expanduser().resolve()
    else:
        preferred = _default_jsonl_meta_dir(files)
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            resolved_meta_dir = preferred
        except OSError:
            from .common import DEFAULT_MANIFEST_CACHE_ROOT

            if DEFAULT_MANIFEST_CACHE_ROOT is None:
                raise PermissionError(
                    f"Cannot create meta dir at {preferred} (read-only) "
                    "and no manifest_cache_root is configured as fallback."
                ) from None
            label = preferred.parent.name or "dataset"
            resolved_meta_dir = (
                DEFAULT_MANIFEST_CACHE_ROOT / f"{label}.jsonl_meta"
            )
            logger.info(
                "Data dir is read-only, falling back to %s", resolved_meta_dir
            )

    resolved_meta_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(resolved_meta_dir, 0o775)

    readme_path = resolved_meta_dir / "README"
    if not readme_path.exists():
        with contextlib.suppress(OSError):
            readme_path.write_text(
                "Line-count metadata sidecars for the JSONL files in the parent directory.\n"
                "\n"
                "Format\n"
                "  NNNNNN.linecount  Plain-text file containing a single integer (line count).\n"
                "\n"
                "File naming: NNNNNN is a zero-padded index into the alphabetically\n"
                "sorted list of JSONL files in the parent directory.\n"
                "\n"
                "Generated by: python -m src.data.manifest.build\n"
                "Config-independent. Shared across all users and training configs.\n"
                "Do not delete unless the JSONL files themselves have changed.\n"
            )

    def emit_progress(event: dict[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(event)

    line_counts: list[int] = []

    for file_index, path in enumerate(files):
        started_at = perf_counter()
        meta_path = resolved_meta_dir / f"{file_index:06d}.linecount"

        if meta_path.exists():
            count = int(meta_path.read_text().strip())
        else:
            count = _scan_jsonl_line_count(path)
            _atomic_write_bytes(meta_path, str(count).encode())

        line_counts.append(count)
        emit_progress(
            {
                "event": "file_scanned",
                "file_index": file_index + 1,
                "num_files": len(files),
                "path": str(path),
                "record_count": count,
                "running_total_records": sum(line_counts),
                "elapsed_sec": perf_counter() - started_at,
            }
        )

    total = sum(line_counts)
    logger.info(
        "JSONL meta ready: files=%d records=%s meta_dir=%s",
        len(files),
        f"{total:,}",
        resolved_meta_dir,
    )
    return files, line_counts


def _build_jsonl_manifest_dict(
    paths: Sequence[str | Path],
    *,
    chunk_num_lines: int = DEFAULT_JSONL_CHUNK_LINES,
    meta_dir: str | Path | None = None,
    cache_root: str | Path | None = None,  # Deprecated, ignored.
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build a JSONL manifest from pre-scanned line-count metadata.

    JSONL chunks use byte offsets computed by accumulating line lengths.
    Since we only store line counts (not byte offsets) in the sidecar,
    we re-scan byte offsets at manifest build time — this is fast because
    we only seek through files without parsing JSON.
    """
    if chunk_num_lines < 1:
        raise ValueError("chunk_num_lines must be >= 1")

    files = _normalize_paths(paths)
    gzip_files = [path for path in files if path.suffix == ".gz"]
    if gzip_files:
        raise ValueError(
            "JSONL manifests only support plain .jsonl files. "
            f"Found gzip JSONL files: {[str(path) for path in gzip_files]}"
        )

    def emit_progress(event: dict[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(event)

    chunk_file_indices: list[int] = []
    chunk_start_bytes: list[int] = []
    chunk_num_lines_list: list[int] = []
    chunk_start_record_indices: list[int] = []
    total_records = 0

    for file_index, path in enumerate(files):
        file_started_at = perf_counter()
        chunk_count = 0
        file_records = 0
        with path.open("rb") as handle:
            while True:
                start_byte = handle.tell()
                line_count = 0

                while line_count < chunk_num_lines:
                    line = handle.readline()
                    if not line:
                        break
                    line_count += 1

                if line_count == 0:
                    break

                chunk_file_indices.append(file_index)
                chunk_start_bytes.append(start_byte)
                chunk_num_lines_list.append(line_count)
                chunk_start_record_indices.append(total_records)
                total_records += line_count
                chunk_count += 1
                file_records += line_count

        emit_progress(
            {
                "event": "file_scanned",
                "file_index": file_index + 1,
                "num_files": len(files),
                "path": str(path),
                "file_size_bytes": path.stat().st_size,
                "record_count": file_records,
                "chunk_count": chunk_count,
                "running_total_records": total_records,
                "elapsed_sec": perf_counter() - file_started_at,
            }
        )

    manifest = {
        "kind": "jsonl",
        "files": [str(path) for path in files],
        "chunk_file_indices": chunk_file_indices,
        "chunk_start_bytes": chunk_start_bytes,
        "chunk_num_lines": chunk_num_lines_list,
        "chunk_start_record_indices": chunk_start_record_indices,
        "total_records": total_records,
        "manifest_version": MANIFEST_VERSION,
    }

    logger.info(
        "JSONL manifest ready: files=%d records=%s chunks=%d",
        len(files),
        f"{total_records:,}",
        len(chunk_file_indices),
    )
    return manifest


class JsonlShardSource(ManifestBackedShardSource):
    """Typed JSONL manifest-backed sharding source."""

    def __init__(self, raw_manifest: Mapping[str, Any]) -> None:
        super().__init__(
            kind="jsonl",
            raw_manifest=raw_manifest,
            total_items_key="total_records",
        )

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> JsonlShardSource:
        return cls(manifest)

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | Path],
        *,
        chunk_num_lines: int = DEFAULT_JSONL_CHUNK_LINES,
        meta_dir: str | Path | None = None,
        cache_root: str | Path | None = None,  # Deprecated, ignored.
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> JsonlShardSource:
        return cls(
            _build_jsonl_manifest_dict(
                paths,
                chunk_num_lines=chunk_num_lines,
                meta_dir=meta_dir,
                progress_callback=progress_callback,
            )
        )

    def iter_chunks(self) -> Iterator[ChunkInfo]:
        for chunk_index, file_index in enumerate(
            self.raw_manifest["chunk_file_indices"]
        ):
            yield ChunkInfo(
                file_index=int(file_index),
                file_path=self.files[int(file_index)],
                start_record_index=int(
                    self.raw_manifest["chunk_start_record_indices"][chunk_index]
                ),
                num_items=int(
                    self.raw_manifest["chunk_num_lines"][chunk_index]
                ),
            )

    def iter_records(
        self,
        *,
        parser: Callable[[str], T] = json.loads,
        seed: int = 0,
        epoch: int = 0,
        info: ShardInfo | None = None,
        shuffle: bool = True,
        shuffle_buffer_size: int = DEFAULT_STREAM_SHUFFLE_BUFFER,
    ) -> Iterator[tuple[int, Path, T]]:
        info = info or get_shard_info()
        physical_manifest = dict(self.raw_manifest)
        physical_chunk_count = len(physical_manifest["chunk_file_indices"])
        manifest = _logical_jsonl_manifest(physical_manifest, info=info)
        if len(manifest["chunk_file_indices"]) > physical_chunk_count:
            _warn_jsonl_logical_split_once(
                manifest=physical_manifest,
                info=info,
                logical_chunk_count=len(manifest["chunk_file_indices"]),
            )

        owned_indices = _owned_chunk_indices(
            len(manifest["chunk_file_indices"]),
            seed=seed,
            epoch=epoch,
            info=info,
            shuffle=shuffle,
        )
        log_state = {"count": 0}

        def iter_owned_records() -> Iterator[tuple[int, Path, T]]:
            for chunk_index in owned_indices:
                file_path = self.files[
                    manifest["chunk_file_indices"][chunk_index]
                ]
                start_byte = manifest["chunk_start_bytes"][chunk_index]
                skip_lines = (
                    manifest["chunk_skip_lines"][chunk_index]
                    if "chunk_skip_lines" in manifest
                    else 0
                )
                num_lines = manifest["chunk_num_lines"][chunk_index]
                start_record_index = manifest["chunk_start_record_indices"][
                    chunk_index
                ]

                with file_path.open("rb") as handle:
                    handle.seek(start_byte)
                    skip_truncated = False
                    for _ in range(skip_lines):
                        if not handle.readline():
                            skip_truncated = True
                            break
                    if skip_truncated:
                        continue
                    for line_offset in range(num_lines):
                        line = handle.readline()
                        if not line:
                            break
                        record_index = start_record_index + line_offset
                        text = line.decode("utf-8", errors="ignore").strip()
                        if not text:
                            continue
                        try:
                            yield record_index, file_path, parser(text)
                        except Exception as exc:
                            _log_record_error(
                                context="iter_jsonl_chunks",
                                path=file_path,
                                record_index=record_index,
                                exc=exc,
                                log_state=log_state,
                                snippet=text[:200],
                            )

        yield from _iter_shuffle_buffer(
            iter_owned_records(),
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            epoch=epoch,
            info=info,
        )


def _logical_jsonl_manifest(
    manifest: Mapping[str, Any],
    *,
    info: ShardInfo,
) -> dict[str, Any]:
    logical_plan = _plan_logical_subchunks(
        physical_chunk_sizes=manifest["chunk_num_lines"],
        total_items=int(manifest["total_records"]),
        total_slots=info.total_slots,
    )
    if logical_plan is None:
        return dict(manifest)

    chunk_file_indices: list[int] = []
    chunk_start_bytes: list[int] = []
    chunk_skip_lines: list[int] = []
    chunk_num_lines: list[int] = []
    chunk_start_record_indices: list[int] = []

    for chunk_index, file_index in enumerate(manifest["chunk_file_indices"]):
        start_byte = manifest["chunk_start_bytes"][chunk_index]
        record_index = manifest["chunk_start_record_indices"][chunk_index]
        for skip_lines, logical_lines in logical_plan[chunk_index]:
            chunk_file_indices.append(file_index)
            chunk_start_bytes.append(start_byte)
            chunk_skip_lines.append(skip_lines)
            chunk_num_lines.append(logical_lines)
            chunk_start_record_indices.append(record_index + skip_lines)

    logical_manifest = dict(manifest)
    logical_manifest["chunk_file_indices"] = chunk_file_indices
    logical_manifest["chunk_start_bytes"] = chunk_start_bytes
    logical_manifest["chunk_skip_lines"] = chunk_skip_lines
    logical_manifest["chunk_num_lines"] = chunk_num_lines
    logical_manifest["chunk_start_record_indices"] = chunk_start_record_indices
    return logical_manifest


def iter_jsonl_chunks(
    manifest_or_paths: JsonlShardSource
    | Mapping[str, Any]
    | Sequence[str | Path],
    *,
    parser: Callable[[str], T] = json.loads,
    seed: int = 0,
    epoch: int = 0,
    info: ShardInfo | None = None,
    chunk_num_lines: int = DEFAULT_JSONL_CHUNK_LINES,
    shuffle: bool = True,
    shuffle_buffer_size: int = DEFAULT_STREAM_SHUFFLE_BUFFER,
) -> Iterator[tuple[int, Path, T]]:
    """Iterate owned JSONL chunks with byte-range seeks."""
    info = info or get_shard_info()

    if isinstance(manifest_or_paths, JsonlShardSource):
        source = manifest_or_paths
    elif isinstance(manifest_or_paths, Mapping):
        source = JsonlShardSource.from_manifest(manifest_or_paths)
    else:
        files = _normalize_paths(manifest_or_paths)
        if any(path.suffix == ".gz" for path in files):
            _warn_gzip_scan_fallback_once(files=files, info=info)
            yield from _iter_jsonl_scan_fallback(
                files,
                parser=parser,
                info=info,
                seed=seed,
                epoch=epoch,
                shuffle=shuffle,
                shuffle_buffer_size=shuffle_buffer_size,
            )
            return
        source = JsonlShardSource.from_files(
            files,
            chunk_num_lines=chunk_num_lines,
        )
    yield from source.iter_records(
        parser=parser,
        seed=seed,
        epoch=epoch,
        info=info,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
    )
