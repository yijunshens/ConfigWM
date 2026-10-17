"""Tar manifest helpers for local sample-indexed sharding."""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Any

from src.data._infra import atomic_write_bytes as _atomic_write_bytes

from .common import (
    DEFAULT_STREAM_SHUFFLE_BUFFER,
    DEFAULT_TAR_SAMPLES_PER_CHUNK,
    MANIFEST_VERSION,
    ChunkInfo,
    ManifestBackedShardSource,
    ShardInfo,
    _iter_shuffle_buffer,
    _log_record_error,
    _normalize_paths,
    _owned_chunk_indices,
    _plan_logical_subchunks,
    _warn_tar_logical_split_once,
    get_shard_info,
)

logger = logging.getLogger(__name__)

_TAR_BLOCK_SIZE = 512
_ZERO_BLOCK = b"\0" * _TAR_BLOCK_SIZE
_REGULAR_TAR_TYPES = {b"", b"\0", b"0", b"7"}


@dataclass(frozen=True)
class _TarHeader:
    header_offset: int
    data_offset: int
    next_offset: int
    typeflag: bytes
    name: str
    size: int


@dataclass(frozen=True)
class _TarScanResult:
    file_index: int
    path: str
    offsets_path: str
    file_size_bytes: int
    sample_count: int
    elapsed_sec: float


def _validate_tar_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.suffix != ".tar":
            raise ValueError(
                f"Only local uncompressed .tar files are supported by tar manifests, got {path}"
            )
        if not path.is_file():
            raise ValueError(f"Tar file does not exist: {path}")


def _decode_tar_string(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8", errors="ignore")


def _parse_tar_octal(raw: bytes) -> int:
    field = raw.rstrip(b"\0 ").lstrip(b" ")
    if not field:
        return 0
    if raw[:1] and raw[0] & 0x80:
        raise ValueError("Base-256 tar numeric fields are not supported")
    return int(field, 8)


def _tar_header_name(header: bytes) -> str:
    name = _decode_tar_string(header[0:100])
    prefix = _decode_tar_string(header[345:500])
    if prefix and name:
        return f"{prefix}/{name}"
    return prefix or name


def _tar_base_plus_ext(name: str) -> tuple[str | None, str | None]:
    normalized = name.rstrip("/")
    if not normalized:
        return None, None
    basename = normalized.rsplit("/", 1)[-1]
    if "." not in basename:
        return None, None
    stem, suffix = normalized.rsplit(".", 1)
    if not stem or not suffix:
        return None, None
    return stem, suffix


def _resolve_sample_member(
    prefix: str,
    suffix: str,
    *,
    current_key: str | None,
    seen_suffixes: set[str],
) -> tuple[str, str]:
    """Fold compound sidecars like ``0.metadata.json`` into sample ``0``."""
    normalized_suffix = suffix.lower()
    if current_key is None:
        return prefix, normalized_suffix

    nested_prefix = f"{current_key}."
    if prefix.startswith(nested_prefix):
        nested_suffix = f"{prefix[len(nested_prefix) :]}.{normalized_suffix}"
        if nested_suffix not in seen_suffixes:
            return current_key, nested_suffix

    return prefix, normalized_suffix


def _read_tar_header(handle, header_offset: int) -> _TarHeader | None:
    handle.seek(header_offset)
    header = handle.read(_TAR_BLOCK_SIZE)
    if not header:
        return None
    if len(header) < _TAR_BLOCK_SIZE:
        raise ValueError("Truncated tar header")
    if header == _ZERO_BLOCK:
        return None

    size = _parse_tar_octal(header[124:136])
    data_offset = header_offset + _TAR_BLOCK_SIZE
    data_blocks = (size + _TAR_BLOCK_SIZE - 1) // _TAR_BLOCK_SIZE
    typeflag = header[156:157]
    return _TarHeader(
        header_offset=header_offset,
        data_offset=data_offset,
        next_offset=data_offset + data_blocks * _TAR_BLOCK_SIZE,
        typeflag=typeflag,
        name=_tar_header_name(header),
        size=size,
    )


def _read_tar_payload(handle, header: _TarHeader) -> bytes:
    handle.seek(header.data_offset)
    payload = handle.read(header.size)
    if len(payload) != header.size:
        raise ValueError("Truncated tar member payload")
    return payload


def _parse_pax_records(payload: bytes) -> dict[str, str]:
    records: dict[str, str] = {}
    cursor = 0
    while cursor < len(payload):
        separator = payload.find(b" ", cursor)
        if separator < 0:
            break
        length = int(payload[cursor:separator])
        record = payload[separator + 1 : cursor + length]
        if not record.endswith(b"\n"):
            break
        body = record[:-1]
        key, _, value = body.partition(b"=")
        if key:
            records[key.decode("utf-8", errors="ignore")] = value.decode(
                "utf-8", errors="ignore"
            )
        cursor += length
    return records


def _scan_tar_sample_offsets(path: Path) -> list[int]:
    sample_offsets: list[int] = []
    current_sample_start: int | None = None
    current_key: str | None = None
    seen_suffixes: set[str] = set()
    file_end_offset = 0

    pending_pax: dict[str, str] | None = None
    pending_longname: str | None = None
    pending_member_start: int | None = None

    with path.open("rb") as handle:
        offset = 0
        while True:
            try:
                header = _read_tar_header(handle, offset)
            except Exception as exc:
                logger.warning(
                    "tar manifest scan: stopping early while scanning %s at offset %d: %s",
                    path,
                    offset,
                    exc,
                )
                file_end_offset = offset
                break
            if header is None:
                file_end_offset = offset
                break
            offset = header.next_offset
            file_end_offset = header.next_offset

            if header.typeflag == b"x":
                try:
                    pending_pax = _parse_pax_records(
                        _read_tar_payload(handle, header)
                    )
                except Exception as exc:
                    logger.warning(
                        "tar manifest scan: stopping early while reading pax header from %s at offset %d: %s",
                        path,
                        header.header_offset,
                        exc,
                    )
                    file_end_offset = header.header_offset
                    break
                if pending_member_start is None:
                    pending_member_start = header.header_offset
                continue
            if header.typeflag == b"L":
                try:
                    pending_longname = (
                        _read_tar_payload(handle, header)
                        .split(b"\0", 1)[0]
                        .decode("utf-8", errors="ignore")
                    )
                except Exception as exc:
                    logger.warning(
                        "tar manifest scan: stopping early while reading GNU longname from %s at offset %d: %s",
                        path,
                        header.header_offset,
                        exc,
                    )
                    file_end_offset = header.header_offset
                    break
                if pending_member_start is None:
                    pending_member_start = header.header_offset
                continue
            if header.typeflag not in _REGULAR_TAR_TYPES:
                pending_pax = None
                pending_longname = None
                pending_member_start = None
                continue

            member_name = (
                (pending_pax or {}).get("path")
                or pending_longname
                or header.name
            )
            member_start = (
                header.header_offset
                if pending_member_start is None
                else pending_member_start
            )
            pending_pax = None
            pending_longname = None
            pending_member_start = None

            prefix, suffix = _tar_base_plus_ext(member_name)
            if prefix is None or suffix is None:
                continue
            effective_key, effective_suffix = _resolve_sample_member(
                prefix,
                suffix,
                current_key=current_key,
                seen_suffixes=seen_suffixes,
            )

            if (
                current_sample_start is None
                or effective_key != current_key
                or effective_suffix in seen_suffixes
            ):
                if current_sample_start is not None:
                    sample_offsets.append(current_sample_start)
                current_sample_start = member_start
                current_key = effective_key
                seen_suffixes = {effective_suffix}
                continue

            seen_suffixes.add(effective_suffix)

    if current_sample_start is not None:
        sample_offsets.append(current_sample_start)
    sample_offsets.append(file_end_offset or path.stat().st_size)
    return sample_offsets


def _encode_offsets(offsets: list[int]) -> bytes:
    encoded = array("Q", offsets)
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


def _load_offsets(path: Path) -> array:
    encoded = array("Q")
    encoded.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded


def _scan_tar_file_to_sidecar(
    *,
    file_index: int,
    path_str: str,
    offsets_path_str: str,
) -> _TarScanResult:
    """Scan a tar file and write its sample offsets to a sidecar file."""
    path = Path(path_str)
    offsets_path = Path(offsets_path_str)
    started_at = perf_counter()
    if offsets_path.exists():
        # Already scanned — derive sample count from file size (8 bytes per entry).
        sample_count = max(0, offsets_path.stat().st_size // 8 - 1)
    else:
        offsets = _scan_tar_sample_offsets(path)
        sample_count = max(0, len(offsets) - 1)
        _atomic_write_bytes(offsets_path, _encode_offsets(offsets))
        with contextlib.suppress(OSError):
            os.chmod(offsets_path, 0o664)
    return _TarScanResult(
        file_index=file_index,
        path=str(path),
        offsets_path=str(offsets_path),
        file_size_bytes=path.stat().st_size,
        sample_count=sample_count,
        elapsed_sec=perf_counter() - started_at,
    )


def _default_offsets_dir(files: Sequence[Path]) -> Path:
    """Infer the offsets directory from the common parent of tar files."""
    common = Path(os.path.commonpath([str(f.parent) for f in files]))
    return common / ".offsets"


def _load_summary(summary_path: Path, num_files: int) -> list[int] | None:
    """Load the SUMMARY file if it exists and matches the expected file count."""
    if not summary_path.exists():
        return None
    data = summary_path.read_bytes()
    if len(data) != num_files * 4:
        return None
    counts = array("I")  # unsigned 32-bit
    counts.frombytes(data)
    if sys.byteorder != "little":
        counts.byteswap()
    return list(counts)


def _write_summary(summary_path: Path, sample_counts: list[int]) -> None:
    """Write a SUMMARY file with one uint32 sample count per tar file."""
    encoded = array("I", sample_counts)
    if sys.byteorder != "little":
        encoded.byteswap()
    _atomic_write_bytes(summary_path, encoded.tobytes())


def ensure_offsets(
    paths: Sequence[str | Path],
    *,
    offsets_dir: str | Path | None = None,
    num_workers: int = 1,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[Path], list[str], list[int]]:
    """Ensure sample-offset sidecar files exist for all tar files.

    Scans only the files that don't already have offsets (incremental).
    Stores offsets in ``offsets_dir`` (default: ``{data_dir}/.offsets/``).

    Returns:
        (files, offsets_paths, sample_counts) — parallel lists.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    files = _normalize_paths(paths)

    if offsets_dir is not None:
        resolved_offsets_dir = Path(offsets_dir).expanduser().resolve()
    else:
        preferred = _default_offsets_dir(files)
        label = preferred.parent.name or "dataset"
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=preferred,
                prefix=".write_probe_",
                suffix=".tmp",
                delete=True,
            ):
                pass
            resolved_offsets_dir = preferred
        except OSError:
            from .common import DEFAULT_MANIFEST_CACHE_ROOT

            if DEFAULT_MANIFEST_CACHE_ROOT is None:
                raise PermissionError(
                    f"Cannot create offsets dir at {preferred} (read-only) "
                    "and no manifest_cache_root is configured as fallback."
                ) from None
            resolved_offsets_dir = (
                DEFAULT_MANIFEST_CACHE_ROOT / f"{label}.offsets"
            )
            logger.info(
                "Data dir is read-only, falling back to %s",
                resolved_offsets_dir,
            )

    # Fast path: if SUMMARY exists, skip expensive per-file validation.
    summary_path = resolved_offsets_dir / "SUMMARY"
    cached_counts = _load_summary(summary_path, len(files))
    if cached_counts is not None:
        offsets_paths = [
            str(resolved_offsets_dir / f"{i:06d}.offsets")
            for i in range(len(files))
        ]
        logger.info(
            "Offsets ready (from SUMMARY): files=%d samples=%s offsets_dir=%s",
            len(files),
            f"{sum(cached_counts):,}",
            resolved_offsets_dir,
        )
        return files, offsets_paths, cached_counts

    # Full path: validate files and scan missing offsets.
    _validate_tar_paths(files)

    resolved_offsets_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(resolved_offsets_dir, 0o775)

    # Write a README so others understand what these files are.
    readme_path = resolved_offsets_dir / "README"
    if not readme_path.exists():
        with contextlib.suppress(OSError):
            readme_path.write_text(
                "Sample-offset sidecars for the tar archives in the parent directory.\n"
                "\n"
                "Format\n"
                "  NNNNNN.offsets  Binary array of little-endian uint64 byte offsets.\n"
                "                  One entry per sample boundary, plus a final end-of-file\n"
                "                  entry (N+1 entries for N samples).\n"
                "  SUMMARY         Binary array of little-endian uint32 sample counts,\n"
                "                  one per tar file. Used for fast manifest generation.\n"
                "\n"
                "File naming: NNNNNN is a zero-padded index into the alphabetically\n"
                "sorted list of tar files in the parent directory.\n"
                "\n"
                "Generated by: python -m src.data.manifest.build\n"
                "Config-independent. Shared across all users and training configs.\n"
                "Do not delete unless the tar files themselves have changed.\n"
            )

    def emit_progress(event: dict[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(event)

    emit_progress(
        {
            "event": "ensure_offsets_start",
            "offsets_dir": str(resolved_offsets_dir),
            "num_files": len(files),
            "num_workers": num_workers,
        }
    )

    file_results: list[_TarScanResult | None] = [None] * len(files)
    completed_samples = 0

    if num_workers == 1 or len(files) <= 1:
        for file_index, path in enumerate(files):
            result = _scan_tar_file_to_sidecar(
                file_index=file_index,
                path_str=str(path),
                offsets_path_str=str(
                    resolved_offsets_dir / f"{file_index:06d}.offsets"
                ),
            )
            file_results[file_index] = result
            completed_samples += result.sample_count
            emit_progress(
                {
                    "event": "file_scanned",
                    "file_index": file_index + 1,
                    "num_files": len(files),
                    "path": result.path,
                    "sample_count": result.sample_count,
                    "offsets_path": result.offsets_path,
                    "running_total_samples": completed_samples,
                    "elapsed_sec": result.elapsed_sec,
                }
            )
    else:
        max_workers = min(num_workers, len(files))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    _scan_tar_file_to_sidecar,
                    file_index=file_index,
                    path_str=str(path),
                    offsets_path_str=str(
                        resolved_offsets_dir / f"{file_index:06d}.offsets"
                    ),
                ): file_index
                for file_index, path in enumerate(files)
            }
            for future in as_completed(future_to_index):
                result = future.result()
                file_results[result.file_index] = result
                completed_samples += result.sample_count
                emit_progress(
                    {
                        "event": "file_scanned",
                        "file_index": result.file_index + 1,
                        "num_files": len(files),
                        "path": result.path,
                        "sample_count": result.sample_count,
                        "offsets_path": result.offsets_path,
                        "running_total_samples": completed_samples,
                        "elapsed_sec": result.elapsed_sec,
                    }
                )

    offsets_paths: list[str] = []
    sample_counts: list[int] = []
    for file_index, result in enumerate(file_results):
        if result is None:
            raise RuntimeError(
                f"Tar offset scan did not produce a result for file index {file_index}"
            )
        offsets_paths.append(result.offsets_path)
        sample_counts.append(result.sample_count)

    # Write SUMMARY for fast loading next time.
    _write_summary(summary_path, sample_counts)

    logger.info(
        "Offsets ready: files=%d samples=%s offsets_dir=%s",
        len(files),
        f"{sum(sample_counts):,}",
        resolved_offsets_dir,
    )
    emit_progress(
        {
            "event": "ensure_offsets_complete",
            "offsets_dir": str(resolved_offsets_dir),
            "num_files": len(files),
            "total_samples": sum(sample_counts),
        }
    )

    return files, offsets_paths, sample_counts


def _build_tar_manifest_dict(
    paths: Sequence[str | Path],
    *,
    samples_per_chunk: int = DEFAULT_TAR_SAMPLES_PER_CHUNK,
    offsets_dir: str | Path | None = None,
    num_workers: int = 1,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build a tar manifest by ensuring offsets exist, then chunking in-memory.

    Offsets are stored in ``offsets_dir`` (default: ``{data_dir}/.offsets/``).
    The manifest dict is generated on-the-fly — no JSON caching needed.
    """
    if samples_per_chunk < 1:
        raise ValueError("samples_per_chunk must be >= 1")

    started_at = perf_counter()
    files, offsets_paths, sample_counts = ensure_offsets(
        paths,
        offsets_dir=offsets_dir,
        num_workers=num_workers,
        progress_callback=progress_callback,
    )

    # Fast in-memory chunk computation from offsets.
    chunk_file_indices: list[int] = []
    chunk_start_sample_indices: list[int] = []
    chunk_num_samples: list[int] = []
    file_num_samples: list[int] = []
    file_start_record_indices: list[int] = []
    total_samples = 0

    for file_index, sample_count in enumerate(sample_counts):
        file_num_samples.append(sample_count)
        file_start_record_indices.append(total_samples)

        for start in range(0, sample_count, samples_per_chunk):
            chunk_file_indices.append(file_index)
            chunk_start_sample_indices.append(start)
            chunk_num_samples.append(
                min(samples_per_chunk, sample_count - start)
            )

        total_samples += sample_count

    manifest = {
        "kind": "tar-samples",
        "files": [str(path) for path in files],
        "sample_offsets_files": offsets_paths,
        "samples_per_chunk": samples_per_chunk,
        "chunk_file_indices": chunk_file_indices,
        "chunk_start_sample_indices": chunk_start_sample_indices,
        "chunk_num_samples": chunk_num_samples,
        "file_num_samples": file_num_samples,
        "file_start_record_indices": file_start_record_indices,
        "total_samples": total_samples,
        "manifest_version": MANIFEST_VERSION,
    }

    elapsed = perf_counter() - started_at
    logger.info(
        "Tar manifest ready: files=%d samples=%s chunks=%d (%.1fs)",
        len(files),
        f"{total_samples:,}",
        len(chunk_file_indices),
        elapsed,
    )
    return manifest


class TarShardSource(ManifestBackedShardSource):
    """Typed tar manifest-backed sharding source."""

    def __init__(self, raw_manifest: Mapping[str, Any]) -> None:
        super().__init__(
            kind="tar-samples",
            raw_manifest=raw_manifest,
            total_items_key="total_samples",
        )

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> TarShardSource:
        return cls(manifest)

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | Path],
        *,
        samples_per_chunk: int = DEFAULT_TAR_SAMPLES_PER_CHUNK,
        offsets_dir: str | Path | None = None,
        cache_root: str | Path | None = None,  # Deprecated, ignored.
        num_workers: int = 1,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> TarShardSource:
        return cls(
            _build_tar_manifest_dict(
                paths,
                samples_per_chunk=samples_per_chunk,
                offsets_dir=offsets_dir,
                num_workers=num_workers,
                progress_callback=progress_callback,
            )
        )

    def iter_chunks(self) -> Iterator[ChunkInfo]:
        record_bases = self.raw_manifest["file_start_record_indices"]
        for chunk_index, file_index in enumerate(
            self.raw_manifest["chunk_file_indices"]
        ):
            start_sample = int(
                self.raw_manifest["chunk_start_sample_indices"][chunk_index]
            )
            yield ChunkInfo(
                file_index=int(file_index),
                file_path=self.files[int(file_index)],
                start_record_index=int(record_bases[file_index]) + start_sample,
                num_items=int(
                    self.raw_manifest["chunk_num_samples"][chunk_index]
                ),
            )

    def iter_samples(
        self,
        *,
        seed: int = 0,
        epoch: int = 0,
        info: ShardInfo | None = None,
        shuffle: bool = True,
        shuffle_buffer_size: int = DEFAULT_STREAM_SHUFFLE_BUFFER,
    ) -> Iterator[tuple[int, Path, dict[str, Any]]]:
        info = info or get_shard_info()
        physical_manifest = dict(self.raw_manifest)
        physical_chunk_count = len(physical_manifest["chunk_file_indices"])
        manifest = _logical_tar_manifest(physical_manifest, info=info)
        if len(manifest["chunk_file_indices"]) > physical_chunk_count:
            _warn_tar_logical_split_once(
                manifest=physical_manifest,
                info=info,
                logical_chunk_count=len(manifest["chunk_file_indices"]),
            )
        offset_files = [
            Path(path) for path in self.raw_manifest["sample_offsets_files"]
        ]
        owned_indices = _owned_chunk_indices(
            len(manifest["chunk_file_indices"]),
            seed=seed,
            epoch=epoch,
            info=info,
            shuffle=shuffle,
        )
        log_state = {"count": 0}

        def iter_owned_samples() -> Iterator[tuple[int, Path, dict[str, Any]]]:
            file_handles: dict[Path, Any] = {}
            offsets_cache: dict[int, array] = {}
            try:
                for chunk_index in owned_indices:
                    file_index = manifest["chunk_file_indices"][chunk_index]
                    file_path = self.files[file_index]
                    handle = file_handles.get(file_path)
                    if handle is None:
                        handle = file_path.open("rb")
                        file_handles[file_path] = handle

                    offsets = offsets_cache.get(file_index)
                    if offsets is None:
                        offsets = _load_offsets(offset_files[file_index])
                        offsets_cache[file_index] = offsets

                    start_sample = manifest["chunk_start_sample_indices"][
                        chunk_index
                    ]
                    num_samples = manifest["chunk_num_samples"][chunk_index]
                    record_base = manifest["file_start_record_indices"][
                        file_index
                    ]

                    for sample_index in range(
                        start_sample, start_sample + num_samples
                    ):
                        start_offset = int(offsets[sample_index])
                        end_offset = int(offsets[sample_index + 1])
                        record_index = record_base + sample_index
                        try:
                            sample = _read_tar_sample(
                                handle,
                                start_offset=start_offset,
                                end_offset=end_offset,
                                file_path=file_path,
                            )
                        except Exception as exc:
                            _log_record_error(
                                context="iter_tar_samples",
                                path=file_path,
                                record_index=record_index,
                                exc=exc,
                                log_state=log_state,
                                snippet=f"offsets={start_offset}:{end_offset}",
                            )
                            continue
                        if sample is not None:
                            yield record_index, file_path, sample
            finally:
                for handle in file_handles.values():
                    handle.close()

        yield from _iter_shuffle_buffer(
            iter_owned_samples(),
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            epoch=epoch,
            info=info,
        )


def _logical_tar_manifest(
    manifest: Mapping[str, Any],
    *,
    info: ShardInfo,
) -> dict[str, Any]:
    logical_plan = _plan_logical_subchunks(
        physical_chunk_sizes=manifest["chunk_num_samples"],
        total_items=int(manifest["total_samples"]),
        total_slots=info.total_slots,
    )
    if logical_plan is None:
        return dict(manifest)

    chunk_file_indices: list[int] = []
    chunk_start_sample_indices: list[int] = []
    chunk_num_samples: list[int] = []

    for chunk_index, file_index in enumerate(manifest["chunk_file_indices"]):
        start_sample = manifest["chunk_start_sample_indices"][chunk_index]
        for sample_offset, logical_samples in logical_plan[chunk_index]:
            chunk_file_indices.append(file_index)
            chunk_start_sample_indices.append(start_sample + sample_offset)
            chunk_num_samples.append(logical_samples)

    logical_manifest = dict(manifest)
    logical_manifest["chunk_file_indices"] = chunk_file_indices
    logical_manifest["chunk_start_sample_indices"] = chunk_start_sample_indices
    logical_manifest["chunk_num_samples"] = chunk_num_samples
    return logical_manifest


def _read_tar_sample(
    handle,
    *,
    start_offset: int,
    end_offset: int,
    file_path: Path,
) -> dict[str, Any] | None:
    sample: dict[str, Any] = {}
    sample_key: str | None = None
    seen_suffixes: set[str] = set()

    pending_pax: dict[str, str] | None = None
    pending_longname: str | None = None

    offset = start_offset
    while offset < end_offset:
        header = _read_tar_header(handle, offset)
        if header is None:
            break
        if header.next_offset > end_offset:
            raise ValueError("Tar sample extends beyond indexed end offset")
        offset = header.next_offset

        if header.typeflag == b"x":
            pending_pax = _parse_pax_records(_read_tar_payload(handle, header))
            continue
        if header.typeflag == b"L":
            pending_longname = (
                _read_tar_payload(handle, header)
                .split(b"\0", 1)[0]
                .decode("utf-8", errors="ignore")
            )
            continue
        if header.typeflag not in _REGULAR_TAR_TYPES:
            pending_pax = None
            pending_longname = None
            continue

        member_name = (
            (pending_pax or {}).get("path") or pending_longname or header.name
        )
        pending_pax = None
        pending_longname = None

        prefix, suffix = _tar_base_plus_ext(member_name)
        if prefix is None or suffix is None:
            continue
        effective_key, effective_suffix = _resolve_sample_member(
            prefix,
            suffix,
            current_key=sample_key,
            seen_suffixes=seen_suffixes,
        )

        if sample_key is None:
            sample_key = effective_key
            sample["__key__"] = effective_key
            sample["__url__"] = str(file_path)
        elif effective_key != sample_key:
            raise ValueError(
                "Encountered a second sample key inside one indexed tar sample"
            )

        if effective_suffix in seen_suffixes:
            raise ValueError(
                f"Encountered duplicate suffix '{effective_suffix}' inside one indexed tar sample"
            )
        sample[effective_suffix] = _read_tar_payload(handle, header)
        seen_suffixes.add(effective_suffix)

    return sample if sample_key is not None else None


def iter_tar_samples(
    manifest_or_paths: TarShardSource
    | Mapping[str, Any]
    | Sequence[str | Path],
    *,
    seed: int = 0,
    epoch: int = 0,
    info: ShardInfo | None = None,
    samples_per_chunk: int = DEFAULT_TAR_SAMPLES_PER_CHUNK,
    offsets_dir: str | Path | None = None,
    shuffle: bool = True,
    shuffle_buffer_size: int = DEFAULT_STREAM_SHUFFLE_BUFFER,
) -> Iterator[tuple[int, Path, dict[str, Any]]]:
    """Iterate sharded tar samples using sample-offset sidecars."""
    if isinstance(manifest_or_paths, TarShardSource):
        source = manifest_or_paths
    elif isinstance(manifest_or_paths, Mapping):
        source = TarShardSource.from_manifest(manifest_or_paths)
    else:
        source = TarShardSource.from_files(
            manifest_or_paths,
            samples_per_chunk=samples_per_chunk,
            offsets_dir=offsets_dir,
        )
    yield from source.iter_samples(
        seed=seed,
        epoch=epoch,
        info=info,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
    )
