from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, TypeVar

T = TypeVar("T")


def expand_user_strict(path: str | Path) -> Path:
    text = str(path)
    if text == "~" or text.startswith("~/"):
        home = os.environ.get("HOME")
        if not home:
            raise ValueError(f"Path uses '~' but HOME is not set: {text}")
        if text == "~":
            return Path(home)
        return Path(home) / text[2:]
    if text.startswith("~"):
        raise ValueError(f"Unsupported home path syntax: {text}")
    return Path(text)


def resolve_cache_root(
    cache_root: str | Path | None,
    *,
    default_root: str | Path | None = None,
) -> Path:
    root = default_root if cache_root is None else cache_root
    if root is None:
        raise ValueError("cache_root must be provided or have a default")
    resolved = expand_user_strict(root).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def stable_cache_key(
    kind: str,
    dependency_paths: Sequence[str | Path],
    *,
    params: Mapping[str, Any],
    version: int | str | None = None,
) -> str:
    payload = {
        "kind": str(kind),
        "params": dict(params),
        "files": [
            _file_signature(expand_user_strict(path).resolve())
            for path in dependency_paths
        ],
    }
    if version is not None:
        payload["version"] = version
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


@contextmanager
def cache_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_or_build_cached_value(
    *,
    lock_path: Path,
    load_cached: Callable[[], T | None],
    build: Callable[[], T],
) -> T:
    cached = load_cached()
    if cached is not None:
        return cached

    with cache_lock(lock_path):
        cached = load_cached()
        if cached is not None:
            return cached
        return build()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=path.stem,
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)
    with suppress(OSError):
        os.chmod(path, 0o664)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
    )


def atomic_write_lines(path: Path, lines: Iterator[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.stem,
        suffix=".tmp",
        delete=False,
    ) as handle:
        for line in lines:
            handle.write(line)
            if not line.endswith("\n"):
                handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def iter_shuffle_buffer(
    records: Iterator[T],
    *,
    shuffle: bool,
    shuffle_buffer_size: int,
    seed: int,
    epoch: int,
    slot_id: int,
) -> Iterator[T]:
    if not shuffle or shuffle_buffer_size <= 1:
        yield from records
        return

    rng = random.Random(seed + epoch * 1_000_003 + slot_id * 10_007)
    buffer: list[T] = []
    for record in records:
        if len(buffer) < shuffle_buffer_size:
            buffer.append(record)
            continue

        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = record

    rng.shuffle(buffer)
    yield from buffer


def resolve_candidate_path(name: str, candidates: Sequence[str | Path]) -> Path:
    resolved_candidates = [
        expand_user_strict(candidate).resolve() for candidate in candidates
    ]
    for candidate in resolved_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve {name}. Tried: {', '.join(str(candidate) for candidate in resolved_candidates)}"
    )


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_lines",
    "cache_lock",
    "expand_user_strict",
    "iter_shuffle_buffer",
    "load_or_build_cached_value",
    "resolve_cache_root",
    "resolve_candidate_path",
    "stable_cache_key",
]
