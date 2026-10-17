from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import math
from typing import Any

from .adapters.registry import get_adapter_family

_TRAIN_ONLY_KEYS = frozenset({"id", "p_sample_sentence_chunk", "weight"})
_LEGACY_KEYS = {
    "chunk_num_lines": "samples_per_chunk",
    "lines_per_chunk": "samples_per_chunk",
    "rows_per_chunk": "samples_per_chunk",
    "min_image_size": "min_shortest_size",
    "min_short_side": "min_shortest_size",
    "n_frames": "num_frames",
}


@dataclass(frozen=True)
class ResolvedDatasetSpec:
    dataset_id: str
    family: str
    label: str
    weight: float
    spec: dict[str, Any]
    adapter_kwargs: dict[str, Any]


def dataset_label(spec: Mapping[str, Any]) -> str:
    parts = [str(spec.get("id", "<unknown>"))]
    for key in ("source", "subset", "stage", "split"):
        value = spec.get(key)
        if value:
            parts.append(str(value))
    return ":".join(parts)


def resolve_dataset_spec(
    spec: Mapping[str, Any],
    data_config: Mapping[str, Any] | None = None,
) -> ResolvedDatasetSpec:
    resolved_spec = copy.deepcopy(dict(spec))
    _reject_legacy_keys(resolved_spec)

    dataset_id = str(resolved_spec.get("id", "")).strip()
    if not dataset_id:
        raise ValueError("dataset spec must define a non-empty 'id'")

    family = get_adapter_family(dataset_id)
    resolved_spec = _apply_global_defaults(resolved_spec, data_config or {})
    weight = _resolve_weight(resolved_spec, dataset_id=dataset_id)

    adapter_kwargs = {
        key: value
        for key, value in resolved_spec.items()
        if key not in _TRAIN_ONLY_KEYS
    }
    if family == "text":
        adapter_kwargs.pop("num_frames", None)
        adapter_kwargs.pop("min_shortest_size", None)
        adapter_kwargs.pop("video_decode_threads", None)
    elif family == "image":
        adapter_kwargs.pop("video_decode_threads", None)

    return ResolvedDatasetSpec(
        dataset_id=dataset_id,
        family=family,
        label=dataset_label(resolved_spec),
        weight=weight,
        spec=resolved_spec,
        adapter_kwargs=adapter_kwargs,
    )


def _apply_global_defaults(
    spec: dict[str, Any],
    data_config: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = dict(spec)
    for key, min_value in (
        ("num_frames", 1),
        ("min_shortest_size", 1),
        ("video_decode_threads", 0),
        ("shuffle_buffer_size", 0),
        ("samples_per_chunk", 1),
    ):
        if key in data_config:
            resolved[key] = _coerce_int(
                data_config[key], field_name=key, min_value=min_value
            )
        elif key in resolved:
            resolved[key] = _coerce_int(
                resolved[key], field_name=key, min_value=min_value
            )
    return resolved


def _resolve_weight(spec: Mapping[str, Any], *, dataset_id: str) -> float:
    try:
        weight = float(spec.get("weight", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Dataset '{dataset_id}' has invalid weight={spec.get('weight')!r}"
        ) from exc
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"Dataset '{dataset_id}' must set weight >= 0")
    return weight


def _coerce_int(value: Any, *, field_name: str, min_value: int) -> int:
    parsed = int(value)
    if parsed < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    return parsed


def _reject_legacy_keys(spec: Mapping[str, Any]) -> None:
    for legacy_key, canonical_key in _LEGACY_KEYS.items():
        if legacy_key in spec:
            raise ValueError(
                f"'{legacy_key}' is no longer supported. Use '{canonical_key}' instead."
            )


__all__ = [
    "ResolvedDatasetSpec",
    "dataset_label",
    "resolve_dataset_spec",
]
