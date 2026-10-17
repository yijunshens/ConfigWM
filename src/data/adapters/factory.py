"""Adapter construction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.data.specs import ResolvedDatasetSpec, resolve_dataset_spec
from torch.utils.data import Dataset

from .registry import get_adapter
from .utils.iterable import ShardedIterableDataset


def create_adapter(spec: Mapping[str, Any] | ResolvedDatasetSpec) -> Dataset:
    """Instantiate one registered adapter from a resolved dataset spec."""
    resolved = (
        spec
        if isinstance(spec, ResolvedDatasetSpec)
        else resolve_dataset_spec(spec)
    )
    adapter_cls = get_adapter(resolved.dataset_id)
    adapter_kwargs = dict(resolved.adapter_kwargs)

    if resolved.family == "text":
        adapter_kwargs.pop("shuffle_buffer_size", None)

    if not issubclass(adapter_cls, ShardedIterableDataset):
        adapter_kwargs.pop("shuffle", None)
        adapter_kwargs.pop("shuffle_buffer_size", None)

    return adapter_cls(**adapter_kwargs)


__all__ = ["create_adapter"]
