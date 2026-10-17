from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from src.data.adapters import get_adapter
from src.data.adapters.utils.iterable import OfflineManifestSource
from src.data.config import (
    DATASET_SECTION_KEYS,
    validate_dataset_sections,
)
from src.data.config import (
    resolve_data_config as _resolve_data_config,
)
from src.data.sharding.common import (
    RequiredPrebuiltPath,
)
from src.data.specs import dataset_label

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManifestBuildTarget:
    dataset_id: str
    label: str
    kind: str
    data_dir: Path
    scan_root: Path
    files: list[Path]
    cache_root: Path | None
    chunk_arg_name: str
    chunk_arg_value: int
    prebuilt_artifacts: tuple[Path, ...] = ()
    build_workers: int = 1


@dataclass(frozen=True)
class ManifestBuildSummary:
    label: str
    dataset_id: str
    kind: str
    num_files: int
    input_bytes: int
    total_items: int
    num_chunks: int
    manifest_path: Path
    manifest_bytes: int
    extra_cache_path: Path | None
    extra_cache_bytes: int
    elapsed_sec: float
    status: str


@dataclass(frozen=True)
class SkippedDataset:
    label: str
    dataset_id: str
    reason: str


def resolve_data_config(
    config_path: str | Path, where_am_i: str | None
) -> dict[str, Any]:
    return _resolve_data_config(config_path, where_am_i)


def target_label(spec: dict[str, Any]) -> str:
    return dataset_label(spec)


def apply_max_shards(files: list[Path], spec: dict[str, Any]) -> list[Path]:
    max_shards = spec.get("max_shards")
    if max_shards is None:
        return files
    return files[: int(max_shards)]


def resolve_chunk_size(
    spec: dict[str, Any],
    *,
    key: str,
    cli_override: int | None,
    default_value: int,
) -> int:
    if cli_override is not None:
        value = cli_override
    elif key in spec:
        value = spec[key]
    else:
        value = default_value
    resolved_value = int(value)
    if resolved_value < 1:
        raise ValueError(
            f"{key} must be >= 1 for dataset '{target_label(spec)}', got {resolved_value}"
        )
    return resolved_value


def _resolve_cli_chunk_override(
    chunk_arg_name: str,
    *,
    samples_per_chunk: int | None,
    rows_per_chunk: int | None = None,
) -> int | None:
    if chunk_arg_name == "samples_per_chunk":
        return samples_per_chunk
    if chunk_arg_name == "rows_per_chunk":
        return rows_per_chunk
    raise ValueError(
        f"Unsupported offline manifest chunk arg '{chunk_arg_name}'"
    )


def _resolve_adapter_source(
    adapter_cls: type[Any],
    spec: dict[str, Any],
    *,
    cache_root: Path | None,
    workers: int,
    allow_build: bool,
) -> OfflineManifestSource | None:
    resolver = getattr(adapter_cls, "resolve_offline_manifest_source", None)
    if not callable(resolver):
        return None
    return resolver(
        spec,
        cache_root=cache_root,
        workers=workers,
        allow_build=allow_build,
    )


def collect_targets(
    config: dict[str, Any],
    *,
    dataset_ids: set[str] | None,
    cache_root: Path | None,
    samples_per_chunk: int | None,
    workers: int,
    rows_per_chunk: int | None = None,
    allow_build: bool = True,
) -> tuple[list[ManifestBuildTarget], list[SkippedDataset]]:
    targets: list[ManifestBuildTarget] = []
    skipped: list[SkippedDataset] = []
    seen_keys: set[tuple[str, tuple[str, ...]]] = set()

    validate_dataset_sections(config)

    for datasets_key in DATASET_SECTION_KEYS:
        for spec in config.get(datasets_key, []) or []:
            dataset_id = str(spec.get("id", ""))
            label = target_label(spec)
            if dataset_ids and dataset_id not in dataset_ids:
                continue

            logger.info(
                "Collecting offline manifest target for dataset '%s' (id=%s)",
                label,
                dataset_id,
            )

            try:
                adapter_cls = get_adapter(dataset_id)
            except KeyError:
                skipped.append(
                    SkippedDataset(
                        label=label,
                        dataset_id=dataset_id,
                        reason="adapter is not registered",
                    )
                )
                continue

            skip_reason = getattr(
                adapter_cls, "OFFLINE_MANIFEST_SKIP_REASON", None
            )
            if skip_reason is not None:
                skipped.append(
                    SkippedDataset(
                        label=label,
                        dataset_id=dataset_id,
                        reason=str(skip_reason),
                    )
                )
                continue

            source = _resolve_adapter_source(
                adapter_cls,
                spec,
                cache_root=cache_root,
                workers=workers,
                allow_build=allow_build,
            )
            if source is None:
                skipped.append(
                    SkippedDataset(
                        label=label,
                        dataset_id=dataset_id,
                        reason="adapter does not declare offline manifest prebuild support",
                    )
                )
                continue

            files = apply_max_shards(source.files, spec)
            if not files:
                raise ValueError(
                    f"No {source.kind} source files found for dataset '{label}' under {source.scan_root}"
                )

            legacy_chunk_keys = [
                key
                for key in (
                    "chunk_num_lines",
                    "lines_per_chunk",
                    "rows_per_chunk",
                )
                if key in spec
            ]
            if legacy_chunk_keys:
                legacy_keys = ", ".join(repr(key) for key in legacy_chunk_keys)
                raise ValueError(
                    f"Dataset '{label}' no longer accepts chunk keys {legacy_keys}. "
                    "Use 'samples_per_chunk' instead."
                )

            chunk_arg_value = resolve_chunk_size(
                spec,
                key=source.chunk_arg_name,
                cli_override=_resolve_cli_chunk_override(
                    source.chunk_arg_name,
                    samples_per_chunk=samples_per_chunk,
                    rows_per_chunk=rows_per_chunk,
                ),
                default_value=source.default_chunk_size,
            )
            prebuilt_artifacts = tuple(
                path
                for path in source.prebuilt_artifacts
                if path not in source.files or path in files
            )
            target = ManifestBuildTarget(
                dataset_id=dataset_id,
                label=label,
                kind=source.kind,
                data_dir=source.data_dir,
                scan_root=source.scan_root,
                files=files,
                cache_root=cache_root,
                chunk_arg_name=source.chunk_arg_name,
                chunk_arg_value=chunk_arg_value,
                prebuilt_artifacts=prebuilt_artifacts,
                build_workers=workers,
            )

            dedupe_key = (
                target.kind,
                tuple(str(path) for path in target.files),
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            targets.append(target)

    return targets, skipped


def collect_prebuilt_runtime_requirements(
    config: dict[str, Any],
    *,
    dataset_ids: set[str] | None,
    cache_root: Path | None,
    samples_per_chunk: int | None,
    workers: int,
) -> tuple[list[RequiredPrebuiltPath], list[SkippedDataset]]:
    targets, skipped = collect_targets(
        config,
        dataset_ids=dataset_ids,
        cache_root=cache_root,
        samples_per_chunk=samples_per_chunk,
        workers=workers,
        allow_build=False,
    )

    requirements_by_path: dict[Path, RequiredPrebuiltPath] = {}
    for target in targets:
        for artifact_path in target.prebuilt_artifacts:
            resolved_artifact = artifact_path.expanduser().resolve()
            requirements_by_path.setdefault(
                resolved_artifact,
                RequiredPrebuiltPath(
                    path=resolved_artifact,
                    dataset_label=target.label,
                    description=f"prebuilt {target.kind} source artifact",
                ),
            )

        if not all(path.exists() for path in target.files):
            continue

        # All formats now store metadata alongside data and generate
        # manifests on-the-fly — no prebuilt manifest JSON to check for.

    return list(requirements_by_path.values()), skipped
