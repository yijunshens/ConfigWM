from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from src.data._infra import iter_shuffle_buffer
from src.data.sharding import (
    DEFAULT_JSONL_CHUNK_LINES,
    DEFAULT_PARQUET_SAMPLES_PER_CHUNK,
    DEFAULT_STREAM_SHUFFLE_BUFFER,
    DEFAULT_TAR_SAMPLES_PER_CHUNK,
    normalize_jsonl_sources,
    resolve_parquet_files,
)
from torch.utils.data import IterableDataset

T = TypeVar("T")


@dataclass(frozen=True)
class OfflineManifestSource:
    """Adapter-owned description of files needed for offline manifest prebuild."""

    kind: str
    data_dir: Path
    scan_root: Path
    files: list[Path]
    chunk_arg_name: str
    default_chunk_size: int
    prebuilt_artifacts: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ResolvedJsonFamilySource:
    """Canonical JSON-family source resolution shared by runtime and prebuild."""

    data_dir: Path
    scan_root: Path
    source_files: list[Path]
    normalized_files: list[Path]
    prebuilt_artifacts: tuple[Path, ...] = ()


def _is_normalized_jsonl_sidecar(path: Path) -> bool:
    if path.suffix != ".jsonl":
        return False

    name = path.name
    if name.startswith("."):
        name = name[1:]
    if not name.endswith(".jsonl"):
        return False

    base_name, _, hash_part = name[: -len(".jsonl")].rpartition(".")
    if len(hash_part) != 32 or any(
        char not in "0123456789abcdef" for char in hash_part
    ):
        return False
    return base_name.endswith((".json", ".jsonl.gz"))


def _json_family_kind(path: Path) -> str:
    if path.name.endswith(".jsonl.gz"):
        return "jsonl.gz"
    if path.suffix == ".jsonl":
        return "jsonl"
    if path.suffix == ".json":
        return "json"
    raise ValueError(f"Unsupported JSON-family path: {path}")


def _json_family_logical_name(path: Path) -> str:
    kind = _json_family_kind(path)
    if kind == "jsonl.gz":
        return path.name[: -len(".gz")]
    if kind == "json":
        return f"{path.stem}.jsonl"
    return path.name


def _prefer_plain_jsonl_files(paths: list[Path]) -> list[Path]:
    priority = {
        "jsonl": 0,
        "jsonl.gz": 1,
        "json": 2,
    }
    selected: dict[str, Path] = {}

    for path in paths:
        if _is_normalized_jsonl_sidecar(path):
            continue
        logical_name = _json_family_logical_name(path)
        current = selected.get(logical_name)
        if (
            current is None
            or priority[_json_family_kind(path)]
            < priority[_json_family_kind(current)]
        ):
            selected[logical_name] = path

    return [selected[key] for key in sorted(selected)]


class ShardedIterableDataset(IterableDataset):
    """Internal base for sharded local iterable adapters."""

    OFFLINE_MANIFEST_KIND: str | None = None
    OFFLINE_MANIFEST_SKIP_REASON: str | None = None
    OFFLINE_MANIFEST_PREFERRED_JSON_FILES: tuple[str, ...] = ()
    OFFLINE_MANIFEST_TAR_RECURSIVE: bool = False

    def __init__(
        self,
        *,
        seed: int = 42,
        shuffle: bool = True,
        shuffle_buffer_size: int = DEFAULT_STREAM_SHUFFLE_BUFFER,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def manifest_summary(self):
        for attribute_name in (
            "_jsonl_source",
            "_parquet_source",
            "_tar_source",
        ):
            source = getattr(self, attribute_name, None)
            summary = getattr(source, "manifest_summary", None)
            if callable(summary):
                return summary()
        return None

    @classmethod
    def build_offline_parquet_source(
        cls,
        *,
        data_dir: Path,
        scan_root: Path,
        files: list[Path],
        prebuilt_artifacts: tuple[Path, ...] = (),
    ) -> OfflineManifestSource:
        return OfflineManifestSource(
            kind="parquet",
            data_dir=data_dir,
            scan_root=scan_root,
            files=files,
            chunk_arg_name="samples_per_chunk",
            default_chunk_size=DEFAULT_PARQUET_SAMPLES_PER_CHUNK,
            prebuilt_artifacts=prebuilt_artifacts,
        )

    @classmethod
    def build_offline_jsonl_source(
        cls,
        *,
        data_dir: Path,
        scan_root: Path,
        files: list[Path],
        prebuilt_artifacts: tuple[Path, ...] = (),
    ) -> OfflineManifestSource:
        return OfflineManifestSource(
            kind="jsonl",
            data_dir=data_dir,
            scan_root=scan_root,
            files=files,
            chunk_arg_name="samples_per_chunk",
            default_chunk_size=DEFAULT_JSONL_CHUNK_LINES,
            prebuilt_artifacts=prebuilt_artifacts,
        )

    @classmethod
    def build_offline_tar_source(
        cls,
        *,
        data_dir: Path,
        scan_root: Path,
        files: list[Path],
        prebuilt_artifacts: tuple[Path, ...] = (),
    ) -> OfflineManifestSource:
        return OfflineManifestSource(
            kind="tar",
            data_dir=data_dir,
            scan_root=scan_root,
            files=files,
            chunk_arg_name="samples_per_chunk",
            default_chunk_size=DEFAULT_TAR_SAMPLES_PER_CHUNK,
            prebuilt_artifacts=prebuilt_artifacts,
        )

    @classmethod
    def build_offline_jsonl_source_from_resolved(
        cls,
        resolved: ResolvedJsonFamilySource,
    ) -> OfflineManifestSource:
        return cls.build_offline_jsonl_source(
            data_dir=resolved.data_dir,
            scan_root=resolved.scan_root,
            files=resolved.normalized_files,
            prebuilt_artifacts=resolved.prebuilt_artifacts,
        )

    @classmethod
    def resolve_offline_manifest_source(
        cls,
        spec: Mapping[str, Any],
        *,
        cache_root: str | Path | None = None,
        workers: int = 1,
        allow_build: bool = True,
    ) -> OfflineManifestSource | None:
        del workers

        data_dir = Path(spec["data_dir"]).expanduser().resolve()
        kind = cls.OFFLINE_MANIFEST_KIND

        if kind == "parquet":
            split = str(spec.get("split", "train"))
            files = resolve_parquet_files(data_dir, split=split)
            return cls.build_offline_parquet_source(
                data_dir=data_dir,
                scan_root=data_dir,
                files=files,
            )

        if kind == "jsonl":
            return cls._resolve_default_jsonl_offline_manifest_source(
                spec,
                cache_root=cache_root,
                allow_build=allow_build,
            )

        if kind == "tar":
            scan_root = cls._resolve_default_tar_scan_root(data_dir, spec)
            if cls.OFFLINE_MANIFEST_TAR_RECURSIVE:
                files = sorted(scan_root.rglob("*.tar"))
            else:
                files = sorted(scan_root.glob("*.tar"))
            return cls.build_offline_tar_source(
                data_dir=data_dir,
                scan_root=scan_root,
                files=files,
            )

        return None

    @classmethod
    def _resolve_default_jsonl_offline_manifest_source(
        cls,
        spec: Mapping[str, Any],
        *,
        cache_root: str | Path | None = None,
        allow_build: bool = True,
    ) -> OfflineManifestSource:
        split = str(spec.get("split", "train"))
        resolved = cls.resolve_default_json_family_source(
            spec["data_dir"],
            split=split,
            cache_root=cache_root,
            allow_build=allow_build,
        )
        return cls.build_offline_jsonl_source_from_resolved(resolved)

    @classmethod
    def resolve_default_json_family_source(
        cls,
        data_dir: str | Path,
        *,
        split: str = "train",
        cache_root: str | Path | None = None,
        required: bool = True,
        allow_build: bool = True,
    ) -> ResolvedJsonFamilySource | None:
        resolved_data_dir = Path(data_dir).expanduser().resolve()

        for scan_root, files in cls._iter_json_family_source_candidates(
            resolved_data_dir,
            split=split,
        ):
            if not files:
                continue
            return cls._build_resolved_json_family_source(
                data_dir=resolved_data_dir,
                scan_root=scan_root,
                source_files=files,
                cache_root=cache_root,
                allow_build=allow_build,
            )

        if required:
            raise ValueError(
                f"No JSON, JSONL, or gzip JSONL files found under {resolved_data_dir}"
            )
        return None

    @classmethod
    def _build_resolved_json_family_source(
        cls,
        *,
        data_dir: Path,
        scan_root: Path,
        source_files: list[Path],
        cache_root: str | Path | None = None,
        allow_build: bool = True,
    ) -> ResolvedJsonFamilySource:
        normalized_files = normalize_jsonl_sources(
            source_files,
            cache_root=cache_root,
            allow_build=allow_build,
        )
        prebuilt_artifacts = tuple(
            normalized_path
            for source_path, normalized_path in zip(
                source_files, normalized_files, strict=False
            )
            if normalized_path != source_path
        )
        return ResolvedJsonFamilySource(
            data_dir=data_dir,
            scan_root=scan_root,
            source_files=source_files,
            normalized_files=normalized_files,
            prebuilt_artifacts=prebuilt_artifacts,
        )

    @classmethod
    def resolve_samples_per_chunk(
        cls,
        *,
        samples_per_chunk: Any | None = None,
        default_value: int = DEFAULT_JSONL_CHUNK_LINES,
    ) -> int:
        raw_value = (
            default_value if samples_per_chunk is None else samples_per_chunk
        )
        resolved_value = int(raw_value)
        if resolved_value < 1:
            raise ValueError("samples_per_chunk must be >= 1")
        return resolved_value

    @classmethod
    def _iter_json_family_source_candidates(
        cls,
        data_dir: Path,
        *,
        split: str,
    ) -> list[tuple[Path, list[Path]]]:
        suffixes = (".jsonl", ".jsonl.gz", ".json")
        roots: list[Path] = []
        for candidate in (
            data_dir / split,
            data_dir / "data" / split,
            data_dir / "data",
            data_dir,
        ):
            if candidate.exists() and candidate not in roots:
                roots.append(candidate)

        candidates: list[tuple[Path, list[Path]]] = []
        preferred_names = tuple(cls.OFFLINE_MANIFEST_PREFERRED_JSON_FILES)
        for root in roots:
            if preferred_names:
                preferred_files = _prefer_plain_jsonl_files(
                    [
                        root / name
                        for name in preferred_names
                        if (root / name).exists()
                    ]
                )
                if preferred_files:
                    candidates.append((root, preferred_files))

            direct_files = _prefer_plain_jsonl_files(
                [
                    root / f"{split}{suffix}"
                    for suffix in suffixes
                    if (root / f"{split}{suffix}").exists()
                ]
            )
            if direct_files:
                candidates.append((root, direct_files))

            prefixed_files: list[Path] = []
            for pattern in (
                f"{split}-*.jsonl",
                f"{split}-*.jsonl.gz",
                f"{split}-*.json",
            ):
                prefixed_files.extend(sorted(root.glob(pattern)))
            prefixed_files = _prefer_plain_jsonl_files(prefixed_files)
            if prefixed_files:
                candidates.append((root, prefixed_files))

            any_files: list[Path] = []
            for pattern in ("*.jsonl", "*.jsonl.gz", "*.json"):
                any_files.extend(sorted(root.glob(pattern)))
            any_files = _prefer_plain_jsonl_files(any_files)
            if any_files:
                candidates.append((root, any_files))

        return candidates

    @classmethod
    def _resolve_default_tar_scan_root(
        cls,
        data_dir: Path,
        spec: Mapping[str, Any],
    ) -> Path:
        for field_name in ("stage", "subset", "split"):
            field_value = spec.get(field_name)
            if field_value is None:
                continue
            candidate = data_dir / str(field_value)
            if candidate.exists():
                return candidate
        return data_dir


__all__ = [
    "OfflineManifestSource",
    "ResolvedJsonFamilySource",
    "ShardedIterableDataset",
    "iter_shuffle_buffer",
]
