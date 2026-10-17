"""Shared data-only contract for Push-T-style Configurable Encoder temporal banks."""

from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
from typing import Hashable, Iterable, Sequence

import h5py
import numpy as np


def validate_temporal_contract(
    offsets: Sequence[int], center_stride: int
) -> tuple[tuple[int, ...], int]:
    """Validate raw-frame offsets and center stride used by all three domains."""
    normalized = tuple(int(value) for value in offsets)
    if len(normalized) < 2:
        raise ValueError("temporal offsets must contain at least two values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate temporal offsets are not allowed: {normalized}")
    if 0 not in normalized:
        raise ValueError("temporal offsets must include 0")
    stride = int(center_stride)
    if stride <= 0:
        raise ValueError("temporal center stride must be positive")
    return normalized, stride


def build_trajectory_windows(
    trajectories: Iterable[Sequence[int] | np.ndarray],
    *,
    offsets: Sequence[int],
    center_stride: int,
) -> np.ndarray:
    """Build raw-frame windows independently inside each trajectory.

    Each input sequence is one trajectory ordered by raw frame index. Keeping
    trajectories separate makes it impossible for a temporal hard-negative
    window to cross an episode boundary.
    """
    normalized, stride = validate_temporal_contract(offsets, center_stride)
    left = max(0, -min(normalized))
    right = max(0, max(normalized))
    windows: list[np.ndarray] = []
    for trajectory in trajectories:
        rows = np.asarray(trajectory, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError(f"trajectory rows must be one-dimensional, got {rows.shape}")
        if rows.size <= left + right:
            continue
        for center_pos in range(left, int(rows.size) - right, stride):
            windows.append(rows[np.asarray(normalized, dtype=np.int64) + center_pos])
    if not windows:
        return np.empty((0, len(normalized)), dtype=np.int64)
    return np.stack(windows, axis=0).astype(np.int64, copy=False)


def round_robin_variant_indices(
    count: int, variant_count: int, *, start: int = 0
) -> np.ndarray:
    """Assign exactly one source variant to each temporal window."""
    count = int(count)
    variant_count = int(variant_count)
    if count < 0:
        raise ValueError("count must be non-negative")
    if variant_count <= 0:
        raise ValueError("variant_count must be positive")
    return (np.arange(count, dtype=np.int64) + int(start)) % variant_count


class BoundedH5FileCache:
    """Per-process LRU cache that prevents HDF5 descriptor accumulation."""

    def __init__(self, max_open_files: int = 16) -> None:
        self.max_open_files = int(max_open_files)
        if self.max_open_files <= 0:
            raise ValueError("max_open_files must be positive")
        self._pid = os.getpid()
        self._handles: OrderedDict[Hashable, h5py.File] = OrderedDict()

    def _reset_after_fork(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._pid:
            return
        self._handles = OrderedDict()
        self._pid = current_pid

    def open(self, key: Hashable, path: str | Path) -> h5py.File:
        self._reset_after_fork()
        handle = self._handles.pop(key, None)
        if handle is not None:
            if handle.id.valid:
                self._handles[key] = handle
                return handle
            handle.close()
        handle = h5py.File(Path(path), "r")
        self._handles[key] = handle
        while len(self._handles) > self.max_open_files:
            _, oldest = self._handles.popitem(last=False)
            oldest.close()
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __getstate__(self) -> dict[str, int]:
        return {"max_open_files": self.max_open_files}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(max_open_files=int(state["max_open_files"]))

    def __del__(self) -> None:
        self.close()


__all__ = [
    "BoundedH5FileCache",
    "build_trajectory_windows",
    "round_robin_variant_indices",
    "validate_temporal_contract",
]
