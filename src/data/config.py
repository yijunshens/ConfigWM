"""Helpers for strict environment-routed data config loading."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from pathlib import Path
from typing import Any

import yaml

DATASET_SECTION_KEYS = ("train_datasets", "eval_datasets")
PATH_ROUTED_FIELDS = (
    "data_dir",
    "image_dir",
    "video_dir",
    "target_data_dir",
)
LEGACY_DATASET_SECTION_KEY = "datasets"


def load_data_config(
    config_or_path: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Load a data config from YAML or clone an in-memory mapping."""
    if isinstance(config_or_path, (str, Path)):
        path = Path(config_or_path).expanduser().resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = copy.deepcopy(dict(config_or_path))

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected data config to be a mapping, got {type(data).__name__}"
        )
    validate_dataset_sections(data)
    return data


def validate_dataset_sections(data: Mapping[str, Any]) -> None:
    """Reject legacy top-level dataset sections."""
    if LEGACY_DATASET_SECTION_KEY not in data:
        return
    raise ValueError(
        "Top-level field 'datasets' is no longer supported. "
        "Use 'train_datasets' and/or 'eval_datasets' instead."
    )


def _location_suffix(location: str, env_key: str | None = None) -> str:
    if env_key is None:
        return location
    return f"{location}[{env_key!r}]"


def _ensure_non_empty_pathlike(value: Any, *, location: str) -> str:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{location} must be a non-empty path string.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{location} must be a non-empty path string.")
    return text


def require_env_mapping(value: Any, *, location: str) -> Mapping[str, Any]:
    """Require a config value to be written as a where_am_i -> value mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{location} must be a mapping from where_am_i to value. "
            "Direct values are no longer supported."
        )
    if not value:
        raise ValueError(
            f"{location} must define at least one where_am_i entry."
        )
    return value


def resolve_required_env_mapping(
    value: Any,
    where_am_i: str | None,
    *,
    location: str,
) -> Any:
    """Resolve a where_am_i-routed value, requiring mapping syntax."""
    mapping = require_env_mapping(value, location=location)
    if where_am_i is None:
        raise ValueError(
            f"{location} is routed by environment. Pass --where_am_i to resolve it."
        )
    if where_am_i not in mapping:
        available = sorted(str(key) for key in mapping)
        raise ValueError(
            f"Unknown where_am_i '{where_am_i}' for {location}. Available: {available}"
        )
    return mapping[where_am_i]


def resolve_optional_env_mapping(
    value: Any,
    where_am_i: str | None,
    *,
    location: str,
) -> Any:
    """Resolve a value only when it is written as a where_am_i mapping."""
    if not isinstance(value, Mapping):
        return value
    return resolve_required_env_mapping(value, where_am_i, location=location)


def validate_path_routed_value(
    value: Any,
    *,
    dataset_id: str,
    field_name: str,
) -> Mapping[str, Any]:
    """Validate that one dataset path field uses strict where_am_i routing."""
    location = f"dataset '{dataset_id}' field '{field_name}'"
    mapping = require_env_mapping(value, location=location)
    for env_key, routed_value in mapping.items():
        _ensure_non_empty_pathlike(
            routed_value,
            location=_location_suffix(location, str(env_key)),
        )
    return mapping


def resolve_path_routed_value(
    value: Any,
    where_am_i: str | None,
    *,
    dataset_id: str,
    field_name: str,
) -> str:
    """Resolve one dataset path field from strict where_am_i routing."""
    location = f"dataset '{dataset_id}' field '{field_name}'"
    mapping = validate_path_routed_value(
        value,
        dataset_id=dataset_id,
        field_name=field_name,
    )
    resolved_value = resolve_required_env_mapping(
        mapping,
        where_am_i,
        location=location,
    )
    return _ensure_non_empty_pathlike(
        resolved_value,
        location=_location_suffix(location, where_am_i),
    )


def resolve_data_config(
    config_or_path: str | Path | Mapping[str, Any],
    where_am_i: str | None,
) -> dict[str, Any]:
    """Load and resolve a training-style data config."""
    config = load_data_config(config_or_path)

    manifest_cache_root = config.get("manifest_cache_root")
    if manifest_cache_root is not None:
        resolved_cache_root = resolve_optional_env_mapping(
            manifest_cache_root,
            where_am_i,
            location="top-level field 'manifest_cache_root'",
        )
        config["manifest_cache_root"] = _ensure_non_empty_pathlike(
            resolved_cache_root,
            location="top-level field 'manifest_cache_root'",
        )

    for datasets_key in DATASET_SECTION_KEYS:
        dataset_specs = config.get(datasets_key) or []
        for index, spec in enumerate(dataset_specs):
            if not isinstance(spec, dict):
                raise ValueError(
                    f"Expected {datasets_key}[{index}] to be a mapping, got {type(spec).__name__}"
                )
            dataset_id = str(spec.get("id", "unknown"))
            for field_name in PATH_ROUTED_FIELDS:
                value = spec.get(field_name)
                if value is None:
                    continue
                spec[field_name] = resolve_path_routed_value(
                    value,
                    where_am_i,
                    dataset_id=dataset_id,
                    field_name=field_name,
                )

    return config


__all__ = [
    "DATASET_SECTION_KEYS",
    "LEGACY_DATASET_SECTION_KEY",
    "PATH_ROUTED_FIELDS",
    "load_data_config",
    "require_env_mapping",
    "resolve_data_config",
    "resolve_optional_env_mapping",
    "resolve_path_routed_value",
    "resolve_required_env_mapping",
    "validate_dataset_sections",
    "validate_path_routed_value",
]
