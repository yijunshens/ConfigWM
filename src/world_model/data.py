"""TwoRoom trajectory dataset for TS-style latent dynamics training."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.preprocessor.dinov2 import DINOv2Preprocessor


def _open_h5_read(path: Path) -> h5py.File:
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def _preprocess_pixels_fast(
    pixels: np.ndarray,
    *,
    image_mean: tuple[float, float, float] | list[float],
    image_std: tuple[float, float, float] | list[float],
) -> torch.Tensor:
    pixel_values = np.asarray(pixels, dtype=np.float32)
    if pixel_values.ndim == 3:
        pixel_values = pixel_values[None, ...]
    if pixel_values.ndim != 4 or pixel_values.shape[-1] != 3:
        raise ValueError(
            f"Expected pixel batch with shape (N, H, W, 3), got {pixel_values.shape}"
        )
    pixel_values = pixel_values / 255.0
    pixel_values = np.transpose(pixel_values, (0, 3, 1, 2))
    mean = np.asarray(image_mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(image_std, dtype=np.float32).reshape(1, 3, 1, 1)
    normalized = (pixel_values - mean) / std
    return torch.from_numpy(normalized)


def _read_target_pos_like(handle: h5py.File, index: int) -> np.ndarray:
    """Return a 2-D position-like target for diagnostics/cache compatibility."""
    for key in (
        "pos_agent",
        "proprio",
        "state",
        "privileged_block_0_pos",
        "privileged_target_block_pos",
        "observation",
        "qpos",
    ):
        if key not in handle:
            continue
        value = np.asarray(handle[key][index], dtype=np.float32).reshape(-1)
        if value.size < 2:
            raise ValueError(
                f"Dataset {key!r} must have at least two values, got shape {value.shape}"
            )
        return value[:2].astype(np.float32, copy=False)
    raise KeyError("Expected one of pos_agent, proprio, or state in transition HDF5")


def _read_action_context(
    handle: h5py.File,
    hist_indices: list[int],
    action_steps: int,
) -> np.ndarray:
    """Return per-history action blocks aligned with LeWM AR conditioning."""
    return np.stack(
        [
            np.asarray(handle["action"][idx : idx + action_steps], dtype=np.float32)
            for idx in hist_indices
        ],
        axis=0,
    )


def _finalize_action_stats(
    action_sum: np.ndarray,
    action_sumsq: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        raise ValueError("Cannot fit action normalizer from an empty action array")
    mean = action_sum / float(count)
    var = np.maximum(action_sumsq / float(count) - np.square(mean), 1.0e-12)
    std = np.sqrt(var)
    std = np.where(std < 1.0e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _load_query_metadata(path: str | Path | None) -> dict[int, str]:
    if path is None:
        return {}
    metadata_path = Path(path).expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing query metadata: {metadata_path}")
    query_by_episode: dict[int, str] = {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item: dict[str, Any] = json.loads(line)
            episode = item.get("episode_idx")
            if episode is None:
                episode = item.get("scene_idx")
            query = item.get("query")
            if query is None:
                color = item.get("core_color") or item.get("target_color")
                if color is not None:
                    query = f"{color} ball"
            if episode is None or query is None:
                continue
            query_by_episode[int(episode)] = str(query)
    if not query_by_episode:
        raise ValueError(f"No episode query entries found in {metadata_path}")
    return query_by_episode


class TwoRoomTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    """Return history frames, action chunks, and future frames from d8 HDF5."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        dino_model_name: str,
        seed: int,
        resolution: int = 224,
        num_hist: int = 3,
        frameskip: int = 5,
        target_offset: int = 5,
        max_samples: int | None = None,
        sample_mode: str = "prefix",
        normalize_actions: bool = False,
        query_metadata_path: str | Path | None = None,
    ) -> None:
        if num_hist < 1:
            raise ValueError("num_hist must be >= 1")
        if frameskip < 1:
            raise ValueError("frameskip must be >= 1")
        if target_offset < 1:
            raise ValueError("target_offset must be >= 1")

        self.data_dir = Path(data_dir).expanduser().resolve()
        self.num_hist = int(num_hist)
        self.frameskip = int(frameskip)
        self.target_offset = int(target_offset)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.sample_mode = str(sample_mode)
        self.seed = int(seed)
        self.normalize_actions = bool(normalize_actions)
        self.resolution = int(resolution)

        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.release_root = None
        if "field_map" in self.manifest:
            from src.data_layout import dataset_root, resolve_file
            self.release_root = dataset_root(self.data_dir)
            if query_metadata_path is None and "query_metadata" in self.manifest:
                query_metadata_path = resolve_file(self.release_root, self.manifest["query_metadata"])
        self.query_by_episode = _load_query_metadata(query_metadata_path)
        self.shards = list(self.manifest.get("shards", []))
        if not self.shards:
            raise ValueError(f"No shards found in {manifest_path}")

        self.preprocessor = DINOv2Preprocessor(
            model_name=dino_model_name,
            resolution=resolution,
            use_fast=True,
        )
        self.image_mean = tuple(float(x) for x in self.preprocessor._processor.image_mean)
        self.image_std = tuple(float(x) for x in self.preprocessor._processor.image_std)
        self.action_mean: np.ndarray | None = None
        self.action_std: np.ndarray | None = None
        if self.normalize_actions:
            self.action_mean, self.action_std = self._compute_action_stats()

        self._pixel_handles: list[h5py.File | None] = [None] * len(self.shards)
        self._base_handles: list[h5py.File | None] = [None] * len(self.shards)
        self._index: list[tuple[int, int, int | None]] = self._build_index()
        if self.max_samples is not None:
            self._index = self._sample_index(self._index)

    def _resolve_shard_path(self, path: str | Path) -> Path:
        shard_path = Path(path)
        if not shard_path.is_absolute():
            shard_path = self.data_dir / shard_path
        return shard_path

    def _shard_paths(self, shard: dict[str, object]) -> tuple[Path, Path]:
        if self.release_root is not None:
            from src.data_layout import resolve_file
            pixel = resolve_file(self.release_root, str(shard["frames"]))
            base = resolve_file(self.release_root, str(shard.get("dynamics", shard["frames"])))
            return pixel, base
        if "path" in shard:
            path = self._resolve_shard_path(str(shard["path"]))
            return path, path
        paths = shard.get("paths")
        if not isinstance(paths, dict):
            raise KeyError("Expected shard to contain either 'path' or 'paths'")
        variant = str(self.manifest.get("variant", "")).lower()
        variant_key = f"{variant}_h5" if variant else None
        pixel_path = (
            (paths.get(variant_key) if variant_key else None)
            or paths.get("color_query_h5")
            or paths.get("h5")
            or paths.get("path")
        )
        if pixel_path is None:
            d_variant_paths = [
                value for key, value in sorted(paths.items()) if str(key).endswith("_h5")
            ]
            if len(d_variant_paths) == 1:
                pixel_path = d_variant_paths[0]
        base_path = paths.get("base_trajectories_h5")
        if pixel_path is None or base_path is None:
            if pixel_path is None:
                raise KeyError(f"Could not infer pixel H5 path from shard paths: {paths}")
            pixel_resolved = self._resolve_shard_path(str(pixel_path))
            sibling_base = pixel_resolved.parent / "base_trajectories.h5"
            if sibling_base.exists():
                base_path = sibling_base
            else:
                base_path = pixel_path
        return self._resolve_shard_path(str(pixel_path)), self._resolve_shard_path(str(base_path))

    def _open_files(self, shard_idx: int) -> tuple[h5py.File, h5py.File]:
        pixel_handle = self._pixel_handles[shard_idx]
        base_handle = self._base_handles[shard_idx]
        if pixel_handle is None or base_handle is None:
            pixel_path, base_path = self._shard_paths(self.shards[shard_idx])
            pixel_handle = _open_h5_read(pixel_path)
            base_handle = pixel_handle if pixel_path == base_path else _open_h5_read(base_path)
            self._pixel_handles[shard_idx] = pixel_handle
            self._base_handles[shard_idx] = base_handle
        return pixel_handle, base_handle

    def _compute_action_stats(self) -> tuple[np.ndarray, np.ndarray]:
        action_sum: np.ndarray | None = None
        action_sumsq: np.ndarray | None = None
        count = 0
        chunk_size = 1_000_000
        for shard in self.shards:
            _, base_path = self._shard_paths(shard)
            with _open_h5_read(base_path) as base_handle:
                if "action" not in base_handle:
                    raise KeyError(f"Missing 'action' in {base_path}")
                action_ds = base_handle["action"]
                action_dim = int(action_ds.shape[-1])
                for start in range(0, int(action_ds.shape[0]), chunk_size):
                    stop = min(start + chunk_size, int(action_ds.shape[0]))
                    chunk = np.asarray(action_ds[start:stop], dtype=np.float64).reshape(-1, action_dim)
                    chunk = np.nan_to_num(chunk, copy=False)
                    if action_sum is None:
                        action_sum = np.zeros(action_dim, dtype=np.float64)
                        action_sumsq = np.zeros(action_dim, dtype=np.float64)
                    action_sum += chunk.sum(axis=0)
                    action_sumsq += np.square(chunk).sum(axis=0)
                    count += int(chunk.shape[0])
        if action_sum is None or action_sumsq is None:
            raise ValueError(f"No action rows found in {self.data_dir}")
        return _finalize_action_stats(action_sum, action_sumsq, count)

    def _prepare_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.nan_to_num(actions, copy=False)
        if not self.normalize_actions:
            return actions
        if self.action_mean is None or self.action_std is None:
            raise RuntimeError("Action normalizer was not initialized")
        return ((actions - self.action_mean) / self.action_std).astype(np.float32, copy=False)

    def _sample_index(self, index: list[tuple[int, int, int | None]]) -> list[tuple[int, int, int | None]]:
        if self.max_samples is None:
            return index
        sample_count = min(self.max_samples, len(index))
        if self.sample_mode == "prefix":
            return index[:sample_count]
        if self.sample_mode == "random":
            rng = np.random.default_rng(self.seed)
            selected = np.sort(rng.choice(len(index), size=sample_count, replace=False))
            return [index[int(i)] for i in selected]
        raise ValueError(f"Unsupported sample_mode: {self.sample_mode!r}")

    def _build_index(self) -> list[tuple[int, int, int | None]]:
        valid: list[tuple[int, int, int | None]] = []
        back_span = self.frameskip * (self.num_hist - 1)
        global_episode_idx = 0
        for shard_idx, shard in enumerate(self.shards):
            pixel_path, base_path = self._shard_paths(shard)
            with _open_h5_read(pixel_path) as pixel_handle:
                if pixel_path == base_path:
                    base_context = pixel_handle
                    close_base = False
                else:
                    base_context = _open_h5_read(base_path)
                    close_base = True
                try:
                    step_idx = np.asarray(base_context["step_idx"], dtype=np.int64)
                    if "ep_idx" in base_context:
                        ep_idx = np.asarray(base_context["ep_idx"], dtype=np.int64)
                        n_rows = ep_idx.shape[0]
                        episode_ranges = []
                        start = 0
                        while start < n_rows:
                            episode = ep_idx[start]
                            end = start + 1
                            while end < n_rows and ep_idx[end] == episode:
                                end += 1
                            episode_ranges.append((start, end - start))
                            start = end
                    elif "ep_offset" in base_context and "ep_len" in base_context:
                        episode_ranges = [
                            (int(offset), int(length))
                            for offset, length in zip(
                                np.asarray(base_context["ep_offset"], dtype=np.int64),
                                np.asarray(base_context["ep_len"], dtype=np.int64),
                                strict=True,
                            )
                        ]
                    elif "scene_source_offset" in pixel_handle and "scene_ep_len" in pixel_handle:
                        episode_ranges = [
                            (int(offset), int(length))
                            for offset, length in zip(
                                np.asarray(pixel_handle["scene_source_offset"], dtype=np.int64),
                                np.asarray(pixel_handle["scene_ep_len"], dtype=np.int64),
                                strict=True,
                            )
                        ]
                    else:
                        raise KeyError(
                            f"Could not infer episode ranges from {base_path} / {pixel_path}"
                        )

                    for episode_start, episode_len in episode_ranges:
                        query_episode_idx = global_episode_idx
                        episode_end = episode_start + episode_len
                        for anchor in range(
                            episode_start + back_span,
                            episode_end - self.target_offset,
                        ):
                            hist_start = anchor - back_span
                            hist_indices = range(hist_start, anchor + 1, self.frameskip)
                            future_index = anchor + self.target_offset

                            expected_steps = [
                                step_idx[hist_start] + i * self.frameskip
                                for i in range(self.num_hist)
                            ]
                            actual_steps = [step_idx[idx] for idx in hist_indices]
                            if actual_steps != expected_steps:
                                continue

                            expected_window = list(
                                range(step_idx[anchor], step_idx[anchor] + self.target_offset + 1)
                            )
                            actual_window = step_idx[anchor : future_index + 1].tolist()
                            if actual_window != expected_window:
                                continue

                            valid.append((shard_idx, anchor, query_episode_idx))
                        global_episode_idx += 1
                finally:
                    if close_base:
                        base_context.close()
        if not valid:
            raise ValueError(
                f"No valid transitions found in {self.data_dir} "
                f"for num_hist={self.num_hist}, frameskip={self.frameskip}, target_offset={self.target_offset}"
            )
        return valid

    def __len__(self) -> int:
        return len(self._index)

    def _preprocess_pixels(self, pixels: np.ndarray) -> torch.Tensor:
        if pixels.ndim == 3:
            pixels = pixels[None, ...]
        if pixels.shape[1:3] == (self.resolution, self.resolution):
            return _preprocess_pixels_fast(
                pixels,
                image_mean=self.image_mean,
                image_std=self.image_std,
            )
        frames = [Image.fromarray(frame).convert("RGB") for frame in pixels]
        return self.preprocessor(frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_idx, anchor, query_episode_idx = self._index[index]
        pixel_handle, base_handle = self._open_files(shard_idx)

        back_span = self.frameskip * (self.num_hist - 1)
        hist_start = anchor - back_span
        hist_indices = list(range(hist_start, anchor + 1, self.frameskip))
        future_index = anchor + self.target_offset

        pixel_key = "pixels" if "pixels" in pixel_handle else "source_pixels"
        history_pixels = pixel_handle[pixel_key][hist_indices]
        future_pixels = pixel_handle[pixel_key][future_index]
        action_chunk = np.asarray(
            base_handle["action"][anchor : future_index],
            dtype=np.float32,
        )
        action_context = _read_action_context(base_handle, hist_indices, self.target_offset)
        action_chunk = self._prepare_actions(action_chunk)
        action_context = self._prepare_actions(action_context)
        pos_agent = _read_target_pos_like(base_handle, future_index)

        history_tensor = self._preprocess_pixels(history_pixels)
        future_tensor = self._preprocess_pixels(future_pixels).squeeze(0)

        item = {
            "history_pixel_values": history_tensor,
            "future_pixel_values": future_tensor,
            "action_chunk": torch.from_numpy(action_chunk),
            "action_context": torch.from_numpy(action_context),
            "target_pos_agent": torch.from_numpy(pos_agent),
        }
        if query_episode_idx is not None and self.query_by_episode:
            try:
                item["queries"] = self.query_by_episode[int(query_episode_idx)]
            except KeyError as exc:
                raise KeyError(
                    f"Missing query for episode_idx={query_episode_idx} in "
                    f"{self.data_dir}"
                ) from exc
        return item


class FeatureCacheTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    """Read precomputed frozen encoder features for dynamics-only training."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        max_samples: int | None = None,
        normalize_actions: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing feature-cache manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest.get("schema_version", 0)) != 1:
            raise ValueError(
                f"Unsupported feature-cache schema_version in {manifest_path}: "
                f"{self.manifest.get('schema_version')}"
            )
        self.shards = list(self.manifest.get("shards", []))
        if not self.shards:
            raise ValueError(f"No feature-cache shards found in {manifest_path}")

        self.feature_dim = int(self.manifest["feature_dim"])
        self.feature_shape = tuple(
            int(x) for x in self.manifest.get("feature_shape", [self.feature_dim])
        )
        if not self.feature_shape or self.feature_shape[-1] != self.feature_dim:
            raise ValueError(
                "Feature-cache manifest feature_shape must end with feature_dim: "
                f"feature_shape={self.feature_shape}, feature_dim={self.feature_dim}"
            )
        self.num_hist = int(self.manifest["num_hist"])
        self.target_offset = int(self.manifest["target_offset"])
        self.frameskip = int(self.manifest["frameskip"])
        self.normalize_actions = bool(normalize_actions)
        self.action_mean: np.ndarray | None = None
        self.action_std: np.ndarray | None = None

        lengths = [int(shard["num_samples"]) for shard in self.shards]
        self._validate_shards(lengths)
        self._cum_lengths = np.cumsum(lengths, dtype=np.int64).tolist()
        total = int(self._cum_lengths[-1])
        self._length = total if max_samples is None else min(int(max_samples), total)
        self._handles: list[h5py.File | None] = [None] * len(self.shards)
        if self.normalize_actions:
            self.action_mean, self.action_std = self._compute_action_stats()

    def _validate_shards(self, lengths: list[int]) -> None:
        required = {
            "history_states": (self.num_hist, *self.feature_shape),
            "future_state": self.feature_shape,
            "action_chunk": (
                self.target_offset,
                int(self.manifest.get("action_dim", 2)),
            ),
            "action_context": (
                self.num_hist,
                self.target_offset,
                int(self.manifest.get("action_dim", 2)),
            ),
            "target_pos_agent": (2,),
            "source_index": (),
        }
        for shard, expected_len in zip(self.shards, lengths, strict=True):
            shard_path = self.cache_dir / shard["path"]
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing feature-cache shard: {shard_path}")
            with _open_h5_read(shard_path) as handle:
                for name, trailing_shape in required.items():
                    if name not in handle:
                        raise KeyError(f"Missing dataset {name!r} in {shard_path}")
                    dataset = handle[name]
                    expected_shape = (expected_len, *trailing_shape)
                    if tuple(dataset.shape) != expected_shape:
                        raise ValueError(
                            f"Bad shape for {name!r} in {shard_path}: "
                            f"got {tuple(dataset.shape)}, expected {expected_shape}"
                        )

    def _open_file(self, shard_idx: int) -> h5py.File:
        handle = self._handles[shard_idx]
        if handle is None:
            shard_path = self.cache_dir / self.shards[shard_idx]["path"]
            handle = _open_h5_read(shard_path)
            self._handles[shard_idx] = handle
        return handle

    def _compute_action_stats(self) -> tuple[np.ndarray, np.ndarray]:
        action_sum: np.ndarray | None = None
        action_sumsq: np.ndarray | None = None
        count = 0
        for shard_idx, shard in enumerate(self.shards):
            shard_path = self.cache_dir / shard["path"]
            with _open_h5_read(shard_path) as handle:
                action_ds = handle["action_chunk"]
                action_dim = int(action_ds.shape[-1])
                for start in range(0, int(action_ds.shape[0]), 1_000_000):
                    stop = min(start + 1_000_000, int(action_ds.shape[0]))
                    chunk = np.asarray(action_ds[start:stop], dtype=np.float64).reshape(-1, action_dim)
                    chunk = np.nan_to_num(chunk, copy=False)
                    if action_sum is None:
                        action_sum = np.zeros(action_dim, dtype=np.float64)
                        action_sumsq = np.zeros(action_dim, dtype=np.float64)
                    action_sum += chunk.sum(axis=0)
                    action_sumsq += np.square(chunk).sum(axis=0)
                    count += int(chunk.shape[0])
        if action_sum is None or action_sumsq is None:
            raise ValueError(f"No cached action rows found in {self.cache_dir}")
        return _finalize_action_stats(action_sum, action_sumsq, count)

    def _prepare_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.nan_to_num(actions, copy=False)
        if not self.normalize_actions:
            return actions
        if self.action_mean is None or self.action_std is None:
            raise RuntimeError("Action normalizer was not initialized")
        return ((actions - self.action_mean) / self.action_std).astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self._cum_lengths, index)
        prev = 0 if shard_idx == 0 else int(self._cum_lengths[shard_idx - 1])
        local_idx = int(index - prev)
        handle = self._open_file(shard_idx)
        action_chunk = self._prepare_actions(np.asarray(handle["action_chunk"][local_idx], dtype=np.float32))
        action_context = self._prepare_actions(np.asarray(handle["action_context"][local_idx], dtype=np.float32))
        return {
            "history_states": torch.from_numpy(np.asarray(handle["history_states"][local_idx])),
            "future_state": torch.from_numpy(np.asarray(handle["future_state"][local_idx])),
            "action_chunk": torch.from_numpy(action_chunk),
            "action_context": torch.from_numpy(action_context),
            "target_pos_agent": torch.from_numpy(np.asarray(handle["target_pos_agent"][local_idx])),
        }


class FrameFeatureCacheTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    """Read per-frame frozen encoder features and assemble transition windows."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        max_samples: int | None = None,
        normalize_actions: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing frame-feature-cache manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest.get("schema_version", 0)) != 2:
            raise ValueError(
                f"Unsupported frame-feature-cache schema_version in {manifest_path}: "
                f"{self.manifest.get('schema_version')}"
            )
        if self.manifest.get("cache_layout") != "frame_features":
            raise ValueError(f"Expected cache_layout='frame_features' in {manifest_path}")

        self.shards = list(self.manifest.get("shards", []))
        if not self.shards:
            raise ValueError(f"No frame-feature-cache shards found in {manifest_path}")

        self.feature_dim = int(self.manifest["feature_dim"])
        self.feature_shape = tuple(int(x) for x in self.manifest["feature_shape"])
        if not self.feature_shape or self.feature_shape[-1] != self.feature_dim:
            raise ValueError(
                "Frame-feature-cache manifest feature_shape must end with feature_dim: "
                f"feature_shape={self.feature_shape}, feature_dim={self.feature_dim}"
            )
        self.num_hist = int(self.manifest["num_hist"])
        self.target_offset = int(self.manifest["target_offset"])
        self.frameskip = int(self.manifest["frameskip"])
        self.normalize_actions = bool(normalize_actions)
        self.action_mean: np.ndarray | None = None
        self.action_std: np.ndarray | None = None
        self._feature_handles: list[h5py.File | None] = [None] * len(self.shards)
        self._base_handles: list[h5py.File | None] = [None] * len(self.shards)
        if self.normalize_actions:
            self.action_mean, self.action_std = self._compute_action_stats()
        self._index: list[tuple[int, int]] = self._build_index()
        if max_samples is not None:
            self._index = self._index[: int(max_samples)]

    def _resolve_cache_path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.cache_dir / path
        return path

    def _shard_paths(self, shard: dict[str, object]) -> tuple[Path, Path]:
        feature_path = self._resolve_cache_path(str(shard["feature_path"]))
        base_path = Path(str(shard["base_trajectories_h5"])).expanduser().resolve()
        return feature_path, base_path

    def _open_files(self, shard_idx: int) -> tuple[h5py.File, h5py.File]:
        feature_handle = self._feature_handles[shard_idx]
        base_handle = self._base_handles[shard_idx]
        if feature_handle is None or base_handle is None:
            feature_path, base_path = self._shard_paths(self.shards[shard_idx])
            feature_handle = _open_h5_read(feature_path)
            base_handle = _open_h5_read(base_path)
            self._feature_handles[shard_idx] = feature_handle
            self._base_handles[shard_idx] = base_handle
        return feature_handle, base_handle

    def _compute_action_stats(self) -> tuple[np.ndarray, np.ndarray]:
        action_sum: np.ndarray | None = None
        action_sumsq: np.ndarray | None = None
        count = 0
        chunk_size = 1_000_000
        for shard in self.shards:
            _, base_path = self._shard_paths(shard)
            with _open_h5_read(base_path) as base_handle:
                if "action" not in base_handle:
                    raise KeyError(f"Missing 'action' in {base_path}")
                action_ds = base_handle["action"]
                action_dim = int(action_ds.shape[-1])
                for start in range(0, int(action_ds.shape[0]), chunk_size):
                    stop = min(start + chunk_size, int(action_ds.shape[0]))
                    chunk = np.asarray(action_ds[start:stop], dtype=np.float64).reshape(-1, action_dim)
                    chunk = np.nan_to_num(chunk, copy=False)
                    if action_sum is None:
                        action_sum = np.zeros(action_dim, dtype=np.float64)
                        action_sumsq = np.zeros(action_dim, dtype=np.float64)
                    action_sum += chunk.sum(axis=0)
                    action_sumsq += np.square(chunk).sum(axis=0)
                    count += int(chunk.shape[0])
        if action_sum is None or action_sumsq is None:
            raise ValueError(f"No action rows found in frame cache {self.cache_dir}")
        return _finalize_action_stats(action_sum, action_sumsq, count)

    def _prepare_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.nan_to_num(actions, copy=False)
        if not self.normalize_actions:
            return actions
        if self.action_mean is None or self.action_std is None:
            raise RuntimeError("Action normalizer was not initialized")
        return ((actions - self.action_mean) / self.action_std).astype(np.float32, copy=False)

    def _build_index(self) -> list[tuple[int, int]]:
        valid: list[tuple[int, int]] = []
        back_span = self.frameskip * (self.num_hist - 1)
        for shard_idx, shard in enumerate(self.shards):
            feature_path, base_path = self._shard_paths(shard)
            with _open_h5_read(feature_path) as feature_handle, _open_h5_read(base_path) as base_handle:
                if "frame_features" not in feature_handle:
                    raise KeyError(f"Missing 'frame_features' in {feature_path}")
                num_frames = int(feature_handle["frame_features"].shape[0])
                if tuple(feature_handle["frame_features"].shape[1:]) != self.feature_shape:
                    raise ValueError(
                        f"Bad frame_features shape in {feature_path}: "
                        f"{tuple(feature_handle['frame_features'].shape)}"
                    )
                step_idx = np.asarray(base_handle["step_idx"], dtype=np.int64)
                if step_idx.shape[0] != num_frames:
                    raise ValueError(
                        f"Frame feature/base length mismatch for shard {shard_idx}: "
                        f"features={num_frames}, step_idx={step_idx.shape[0]}"
                    )
                if "ep_idx" in base_handle:
                    ep_idx = np.asarray(base_handle["ep_idx"], dtype=np.int64)
                    n_rows = ep_idx.shape[0]
                    episode_ranges = []
                    start = 0
                    while start < n_rows:
                        episode = ep_idx[start]
                        end = start + 1
                        while end < n_rows and ep_idx[end] == episode:
                            end += 1
                        episode_ranges.append((start, end - start))
                        start = end
                elif "ep_offset" in base_handle and "ep_len" in base_handle:
                    episode_ranges = [
                        (int(offset), int(length))
                        for offset, length in zip(
                            np.asarray(base_handle["ep_offset"], dtype=np.int64),
                            np.asarray(base_handle["ep_len"], dtype=np.int64),
                            strict=True,
                        )
                    ]
                else:
                    raise KeyError(f"Could not infer episode ranges from {base_path}")

                for episode_start, episode_len in episode_ranges:
                    episode_end = episode_start + episode_len
                    for anchor in range(
                        episode_start + back_span,
                        episode_end - self.target_offset,
                    ):
                        hist_start = anchor - back_span
                        hist_indices = range(hist_start, anchor + 1, self.frameskip)
                        future_index = anchor + self.target_offset
                        expected_steps = [
                            step_idx[hist_start] + i * self.frameskip
                            for i in range(self.num_hist)
                        ]
                        actual_steps = [step_idx[idx] for idx in hist_indices]
                        if actual_steps != expected_steps:
                            continue
                        expected_window = list(
                            range(step_idx[anchor], step_idx[anchor] + self.target_offset + 1)
                        )
                        actual_window = step_idx[anchor : future_index + 1].tolist()
                        if actual_window != expected_window:
                            continue
                        valid.append((shard_idx, anchor))
        if not valid:
            raise ValueError(
                f"No valid transitions found in frame feature cache {self.cache_dir} "
                f"for num_hist={self.num_hist}, frameskip={self.frameskip}, target_offset={self.target_offset}"
            )
        return valid

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self._index):
            raise IndexError(index)
        shard_idx, anchor = self._index[index]
        feature_handle, base_handle = self._open_files(shard_idx)
        back_span = self.frameskip * (self.num_hist - 1)
        hist_start = anchor - back_span
        hist_indices = list(range(hist_start, anchor + 1, self.frameskip))
        future_index = anchor + self.target_offset
        action_context = _read_action_context(base_handle, hist_indices, self.target_offset)
        action_chunk = self._prepare_actions(
            np.asarray(base_handle["action"][anchor:future_index], dtype=np.float32)
        )
        action_context = self._prepare_actions(action_context)
        return {
            "history_states": torch.from_numpy(np.asarray(feature_handle["frame_features"][hist_indices])),
            "future_state": torch.from_numpy(np.asarray(feature_handle["frame_features"][future_index])),
            "action_chunk": torch.from_numpy(action_chunk),
            "action_context": torch.from_numpy(action_context),
            "target_pos_agent": torch.from_numpy(
                _read_target_pos_like(base_handle, future_index)
            ),
        }


__all__ = [
    "FeatureCacheTransitionDataset",
    "FrameFeatureCacheTransitionDataset",
    "TransitionDataset",
    "TwoRoomTransitionDataset",
]

TransitionDataset = TwoRoomTransitionDataset

