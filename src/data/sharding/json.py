"""JSON-family normalization helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import gzip
import json as std_json
import logging
from pathlib import Path
from time import perf_counter
from typing import TextIO

from src.data._infra import atomic_write_lines as _atomic_write_lines

from .common import (
    _manifest_lock,
    _normalize_paths,
    _raise_for_required_prebuilt_paths,
)

NORMALIZE_JSONL_PROGRESS_EVERY_RECORDS = 10_000
NORMALIZE_JSONL_PROGRESS_INTERVAL_SEC = 10.0


logger = logging.getLogger(__name__)


def _normalized_jsonl_output_path(
    source_path: Path,
    *,
    cache_root: str | Path | None,
) -> Path:
    del cache_root
    if source_path.suffix == ".json":
        return source_path.with_suffix(".jsonl")
    if source_path.suffix == ".gz" and source_path.name.endswith(".jsonl.gz"):
        return source_path.with_suffix("")
    raise ValueError(f"Unsupported JSON-family source: {source_path}")


def _peek_first_non_whitespace_char(handle: TextIO) -> str | None:
    while True:
        chunk = handle.read(4096)
        if not chunk:
            return None
        stripped = chunk.lstrip()
        if stripped:
            return stripped[0]


def _iter_line_delimited_json_records(handle: TextIO) -> Iterator[str]:
    for line in handle:
        text = line.strip()
        if not text:
            continue
        std_json.loads(text)
        yield text


def _iter_json_array_records(handle: TextIO) -> Iterator[str]:
    decoder = std_json.JSONDecoder()
    buffer = ""
    started = False
    finished = False
    eof = False
    needs_more_input = True

    while True:
        if not eof and (needs_more_input or not buffer):
            chunk = handle.read(1 << 20)
            if chunk:
                buffer += chunk
                needs_more_input = False
            else:
                eof = True

        cursor = 0
        while True:
            while cursor < len(buffer) and buffer[cursor].isspace():
                cursor += 1

            if not started:
                if cursor >= len(buffer):
                    break
                if buffer[cursor] != "[":
                    raise ValueError(
                        "JSON source is neither line-delimited JSON nor a JSON array"
                    )
                started = True
                cursor += 1
                continue

            while cursor < len(buffer) and (
                buffer[cursor].isspace() or buffer[cursor] == ","
            ):
                cursor += 1

            if cursor >= len(buffer):
                break

            if buffer[cursor] == "]":
                finished = True
                cursor += 1
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor < len(buffer):
                    raise ValueError(
                        "Trailing non-whitespace content after JSON array"
                    )
                buffer = ""
                break

            try:
                record, next_cursor = decoder.raw_decode(buffer, cursor)
            except std_json.JSONDecodeError:
                if eof:
                    raise
                needs_more_input = True
                break

            yield std_json.dumps(record, separators=(",", ":"))
            cursor = next_cursor

        if finished:
            return

        buffer = buffer[cursor:]
        if eof:
            if buffer.strip():
                raise ValueError(
                    "Unterminated JSON array while normalizing source"
                )
            raise ValueError("JSON array source ended before closing bracket")


def normalize_jsonl_source(
    path: str | Path,
    *,
    cache_root: str | Path | None = None,
    allow_build: bool = True,
) -> Path:
    """Normalize a local JSON-family source into a stable plain-JSONL sidecar."""
    source_path = _normalize_paths([path])[0]
    if source_path.suffix == ".jsonl":
        return source_path

    if source_path.suffix == ".gz" and not str(source_path).endswith(
        ".jsonl.gz"
    ):
        raise ValueError(f"Unsupported gzip JSON source: {source_path}")
    if source_path.suffix not in {".json", ".gz"}:
        raise ValueError(f"Unsupported JSON-family source: {source_path}")

    output_path = _normalized_jsonl_output_path(
        source_path, cache_root=cache_root
    )
    lock_path = output_path.with_suffix(".lock")
    if output_path.exists():
        logger.info(
            "Using cached normalized JSONL: source=%s output=%s",
            source_path,
            output_path,
        )
        return output_path

    if not allow_build:
        return output_path

    _raise_for_required_prebuilt_paths([output_path])

    logger.info(
        "Preparing normalized JSONL sidecar: source=%s output=%s input_bytes=%d",
        source_path,
        output_path,
        source_path.stat().st_size,
    )

    with _manifest_lock(lock_path):
        if output_path.exists():
            logger.info(
                "Using cached normalized JSONL after waiting on lock: source=%s output=%s",
                source_path,
                output_path,
            )
            return output_path

        first_char: str | None = None
        source_kind = "jsonl.gz" if source_path.suffix == ".gz" else "json"
        if source_path.suffix == ".gz":

            def iter_lines() -> Iterator[str]:
                with gzip.open(
                    source_path, "rt", encoding="utf-8", errors="ignore"
                ) as handle:
                    for line in handle:
                        text = line.strip()
                        if text:
                            yield text

        else:
            with source_path.open("r", encoding="utf-8") as handle:
                first_char = _peek_first_non_whitespace_char(handle)
            source_kind = (
                "json-array" if first_char == "[" else "line-delimited-json"
            )

            def iter_lines() -> Iterator[str]:
                with source_path.open("r", encoding="utf-8") as handle:
                    if first_char == "[":
                        yield from _iter_json_array_records(handle)
                    else:
                        yield from _iter_line_delimited_json_records(handle)

        logger.info(
            "Normalizing JSON-family source -> JSONL: source=%s kind=%s output=%s",
            source_path,
            source_kind,
            output_path,
        )
        progress_state = {
            "records": 0,
            "started_at": perf_counter(),
            "last_logged_at": perf_counter(),
            "last_logged_records": 0,
        }

        def iter_lines_with_progress() -> Iterator[str]:
            for line in iter_lines():
                progress_state["records"] += 1
                now = perf_counter()
                records = progress_state["records"]
                if (
                    records == 1
                    or records - progress_state["last_logged_records"]
                    >= NORMALIZE_JSONL_PROGRESS_EVERY_RECORDS
                    or now - progress_state["last_logged_at"]
                    >= NORMALIZE_JSONL_PROGRESS_INTERVAL_SEC
                ):
                    logger.info(
                        "Normalizing JSON-family source progress: source=%s kind=%s records=%d elapsed=%.1fs",
                        source_path,
                        source_kind,
                        records,
                        now - progress_state["started_at"],
                    )
                    progress_state["last_logged_at"] = now
                    progress_state["last_logged_records"] = records
                yield line

        _atomic_write_lines(output_path, iter_lines_with_progress())
        elapsed_sec = perf_counter() - progress_state["started_at"]
        logger.info(
            "Normalized JSON-family source complete: source=%s kind=%s output=%s records=%d output_bytes=%d elapsed=%.1fs",
            source_path,
            source_kind,
            output_path,
            progress_state["records"],
            output_path.stat().st_size,
            elapsed_sec,
        )
        return output_path


def normalize_jsonl_sources(
    paths: Sequence[str | Path],
    *,
    cache_root: str | Path | None = None,
    allow_build: bool = True,
) -> list[Path]:
    """Normalize JSON-family sources into plain `.jsonl` files."""
    return [
        normalize_jsonl_source(
            path,
            cache_root=cache_root,
            allow_build=allow_build,
        )
        for path in paths
    ]
