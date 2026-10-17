"""Compose TwoRoom dynamic distractors from prebuilt donor tracking assets."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL import ImageDraw

from .hdf5_static_copy import (
    _composite_patch_rgba,
    _decontaminate_patch_rgba,
    _json_default,
    _resolve_pixel_key,
    _sanitize_patch_rgba,
    _write_preview_video,
    _write_metadata_files,
)
from .patch_rgba import (
    normalize_visible_luminance as _normalize_visible_luminance_shared,
    patch_quality_metrics as _patch_quality_metrics_shared,
    recolor_patch_rgba_preserve_material as _recolor_patch_rgba_preserve_material_shared,
    select_best_valid_patch_index as _select_best_valid_patch_index_shared,
    scale_patch_rgba as _scale_patch_rgba_premultiplied,
    zero_rgb_where_alpha_is_zero as _zero_rgb_where_alpha_is_zero_shared,
)
from .registry import COLOR_NAME_TO_RGB

_PREVIEW_VIDEO_KWARGS = {
    "codec": "libx264",
    "quality": 10,
    "macro_block_size": 1,
    "ffmpeg_params": ["-crf", "8", "-preset", "slow", "-pix_fmt", "yuv444p"],
}


_OKLAB_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float32,
)
_OKLAB_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float32,
)
_OKLAB_M1_INV = np.linalg.inv(_OKLAB_M1).astype(np.float32)
_OKLAB_M2_INV = np.linalg.inv(_OKLAB_M2).astype(np.float32)


@dataclass
class TwoRoomDynamicAugmentConfig:
    input_h5: Path
    donor_root: Path
    output_dir: Path
    seed: int = 0
    scale_min: float = 0.5
    scale_max: float = 1.5
    max_scale_ratio: float = 0.25


@dataclass
class TwoRoomDynamicFanoutConfig:
    input_h5: Path
    donor_root: Path
    output_root: Path
    seed: int = 0
    scale_min: float = 0.5
    scale_max: float = 1.5
    max_scale_ratio: float = 0.25
    episode_start: int | None = None
    episode_end: int | None = None
    donor_counts: tuple[int, ...] = (1, 2, 3)
    resume_from_episode: int | None = None
    reset_tail_to_clean: bool = False


@dataclass
class DonorEntry:
    episode_idx: int
    ep_len: int
    donor_summary_path: Path
    donor_assets_path: Path
    segmentation_score: float
    rank: int


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default),
        encoding="utf-8",
    )


def _normalize_donor_counts(donor_counts: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    normalized = tuple(int(count) for count in donor_counts)
    if not normalized:
        raise ValueError("donor_counts must not be empty")
    if any(count <= 0 for count in normalized):
        raise ValueError("donor_counts must be positive")
    if tuple(sorted(normalized)) != normalized:
        raise ValueError("donor_counts must be in ascending order")
    if len(set(normalized)) != len(normalized):
        raise ValueError("donor_counts must not contain duplicates")
    return normalized


def _load_episode_frames(input_h5: Path, episode_idx: int) -> dict[str, Any]:
    with h5py.File(input_h5, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        pixel_key = _resolve_pixel_key(f)
        start = int(ep_offset[episode_idx])
        end = int(start + ep_len[episode_idx])
        return {
            "frames": f[pixel_key][start:end],
            "ep_len": int(ep_len[episode_idx]),
            "pos_agent": f["pos_agent"][start:end] if "pos_agent" in f else None,
        }


def load_successful_donor_entries(donor_root: Path) -> list[DonorEntry]:
    index_path = donor_root / "index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    entries: list[DonorEntry] = []
    for row in rows:
        if row.get("status") != "success":
            continue
        summary_path = Path(row["donor_summary_path"])
        if not summary_path.exists():
            continue
        assets_path = summary_path.with_name("donor_assets.npz")
        if not assets_path.exists():
            summary = json.loads(summary_path.read_text())
            assets_path = Path(summary["assets_path"])
        if not assets_path.exists():
            continue
        entries.append(
            DonorEntry(
                episode_idx=int(row["episode_idx"]),
                ep_len=int(row["ep_len"]),
                donor_summary_path=summary_path,
                donor_assets_path=assets_path,
                segmentation_score=float(row.get("segmentation_score") or 0.0),
                rank=int(row.get("rank") or 0),
            )
        )
    return entries


def _load_donor_assets(entry: DonorEntry) -> dict[str, Any]:
    npz = np.load(entry.donor_assets_path, allow_pickle=True)
    patches = [
        _sanitize_patch_rgba(np.asarray(patch, dtype=np.uint8))
        for patch in npz["step_patch_rgba"].tolist()
    ]
    best_patch_index = None
    if "best_patch_index" in npz.files:
        raw_idx = int(np.asarray(npz["best_patch_index"]).item())
        if raw_idx >= 0:
            best_patch_index = raw_idx
    return {
        "step_valid": np.asarray(npz["step_valid"], dtype=bool),
        "step_centers_xy": np.asarray(npz["step_centers_xy"], dtype=np.float32),
        "step_bboxes_xyxy": np.asarray(npz["step_bboxes_xyxy"], dtype=np.int32),
        "step_masks": np.asarray(npz["step_masks"], dtype=bool),
        "step_patch_rgba": patches,
        "best_patch_index": best_patch_index,
    }


def _zero_rgb_where_alpha_is_zero(patch_rgba: np.ndarray) -> np.ndarray:
    return _zero_rgb_where_alpha_is_zero_shared(patch_rgba)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    rgb = np.clip(rgb, 0.0, None)
    return np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * (rgb ** (1.0 / 2.4)) - 0.055,
    ).astype(np.float32)


def _linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = rgb @ _OKLAB_M1.T
    lms_cuberoot = np.cbrt(lms)
    return (lms_cuberoot @ _OKLAB_M2.T).astype(np.float32)


def _oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lms_cuberoot = lab @ _OKLAB_M2_INV.T
    lms = lms_cuberoot ** 3
    return (lms @ _OKLAB_M1_INV.T).astype(np.float32)


def _compute_target_hue_rad(target_rgb: np.ndarray) -> float:
    rgb = np.asarray(target_rgb, dtype=np.float32).reshape(-1)[:3] / 255.0
    lab = _linear_to_oklab(_srgb_to_linear(rgb[None, :]))[0]
    return float(np.arctan2(lab[2], lab[1]))


_TARGET_HUE_RAD_BY_RGB_KEY = {
    tuple(int(v) for v in rgb.tolist()): _compute_target_hue_rad(rgb)
    for rgb in COLOR_NAME_TO_RGB.values()
}


def _target_hue_rad(target_rgb: np.ndarray) -> float:
    arr = np.asarray(target_rgb, dtype=np.uint8).reshape(-1)[:3]
    key = tuple(int(v) for v in arr.tolist())
    return _TARGET_HUE_RAD_BY_RGB_KEY.get(key, _compute_target_hue_rad(arr))


def _recolor_patch_rgba(
    patch_rgba: np.ndarray,
    target_rgb: np.ndarray,
    *,
    return_clip_ratio: bool = False,
    low_chroma_thresh: float = 0.01,
) -> np.ndarray | tuple[np.ndarray, float]:
    sanitized = _sanitize_patch_rgba(_zero_rgb_where_alpha_is_zero(patch_rgba))
    return _recolor_patch_rgba_preserve_material_shared(
        sanitized,
        target_rgb,
        return_clip_ratio=return_clip_ratio,
        low_chroma_thresh=low_chroma_thresh,
    )


def _scale_patch_rgba(patch_rgba: np.ndarray, scale_factor: float) -> np.ndarray:
    return _scale_patch_rgba_premultiplied(patch_rgba, scale_factor)


def _allowed_translation_ranges(
    image_shape: tuple[int, int],
    centers_xy: np.ndarray,
    patches: list[np.ndarray],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    height, width = image_shape
    tx_min, tx_max = -10**9, 10**9
    ty_min, ty_max = -10**9, 10**9
    for center_xy, patch_rgba in zip(centers_xy, patches):
        patch_h, patch_w = patch_rgba.shape[:2]
        half_w = patch_w / 2.0
        half_h = patch_h / 2.0
        cx, cy = float(center_xy[0]), float(center_xy[1])
        frame_tx_min = int(np.ceil(half_w - cx))
        frame_tx_max = int(np.floor((width - 1 - half_w) - cx))
        frame_ty_min = int(np.ceil(half_h - cy))
        frame_ty_max = int(np.floor((height - 1 - half_h) - cy))
        tx_min = max(tx_min, frame_tx_min)
        tx_max = min(tx_max, frame_tx_max)
        ty_min = max(ty_min, frame_ty_min)
        ty_max = min(ty_max, frame_ty_max)
    if tx_min > tx_max or ty_min > ty_max:
        return None
    return (tx_min, tx_max), (ty_min, ty_max)


def _sample_donor_transform(
    *,
    rng: np.random.Generator,
    donor_assets: dict[str, Any],
    image_shape: tuple[int, int],
    scale_min: float,
    scale_max: float,
    max_scale_ratio: float,
    used_color_names: set[str],
) -> dict[str, Any]:
    valid_indices = np.flatnonzero(donor_assets["step_valid"])
    if valid_indices.size == 0:
        raise RuntimeError("donor_has_no_valid_steps")
    best_patch_idx = _resolve_best_patch_index(donor_assets)
    if best_patch_idx is None:
        raise RuntimeError("donor_has_no_valid_patch")

    palette = [name for name in COLOR_NAME_TO_RGB if name != "red" and name not in used_color_names]
    if not palette:
        palette = [name for name in COLOR_NAME_TO_RGB if name != "red"]
    color_name = palette[int(rng.integers(0, len(palette)))]
    color_rgb = COLOR_NAME_TO_RGB[color_name]

    height, width = image_shape
    short_side = min(height, width)
    raw_best_patch = donor_assets["step_patch_rgba"][best_patch_idx]
    raw_centers = donor_assets["step_centers_xy"][valid_indices]

    for _ in range(128):
        scale_factor = float(rng.uniform(scale_min, scale_max))
        scaled_patch = _scale_patch_rgba(_recolor_patch_rgba(raw_best_patch, color_rgb), scale_factor)
        if max(scaled_patch.shape[:2]) > short_side * max_scale_ratio:
            continue
        allowed = _allowed_translation_ranges(
            image_shape,
            raw_centers,
            [scaled_patch for _ in range(len(raw_centers))],
        )
        if allowed is None:
            continue
        (tx_min, tx_max), (ty_min, ty_max) = allowed
        tx = int(rng.integers(tx_min, tx_max + 1))
        ty = int(rng.integers(ty_min, ty_max + 1))
        return {
            "color": color_name,
            "color_rgb": color_rgb,
            "scale_factor": scale_factor,
            "translation_xy": [tx, ty],
            "best_patch_index": int(best_patch_idx),
        }
    raise RuntimeError("transform_sampling_failed")


def _valid_source_index(step_valid: np.ndarray, frame_idx: int) -> int | None:
    source_idx = min(frame_idx, len(step_valid) - 1)
    while source_idx >= 0 and not step_valid[source_idx]:
        source_idx -= 1
    if source_idx < 0:
        return None
    return int(source_idx)


def _patch_crop_edge_contact_fraction(alpha_mask: np.ndarray) -> float:
    if not np.any(alpha_mask):
        return 1.0
    edge_mask = np.zeros_like(alpha_mask, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    return float(np.count_nonzero(alpha_mask & edge_mask) / max(np.count_nonzero(alpha_mask), 1))


def _patch_core_mask(alpha: np.ndarray) -> np.ndarray:
    visible = alpha > 0
    if not np.any(visible):
        return visible
    visible_alpha = alpha[visible].astype(np.float32)
    if visible_alpha.max() >= 250:
        threshold = 250.0
    else:
        threshold = float(np.percentile(visible_alpha, 75))
    return visible & (alpha.astype(np.float32) >= threshold)


def _patch_directional_asymmetry(core_mask: np.ndarray, luminance: np.ndarray) -> float:
    if not np.any(core_mask):
        return 1.0
    ys, xs = np.nonzero(core_mask)
    center_x = float(xs.mean())
    center_y = float(ys.mean())

    def _masked_mean(mask: np.ndarray) -> float:
        return float(luminance[mask].mean()) if np.any(mask) else float(luminance[core_mask].mean())

    left = core_mask & (np.arange(core_mask.shape[1])[None, :] <= center_x)
    right = core_mask & (np.arange(core_mask.shape[1])[None, :] > center_x)
    top = core_mask & (np.arange(core_mask.shape[0])[:, None] <= center_y)
    bottom = core_mask & (np.arange(core_mask.shape[0])[:, None] > center_y)

    core_median = float(np.median(luminance[core_mask]))
    denom = max(25.0, core_median)
    lr = abs(_masked_mean(left) - _masked_mean(right)) / denom
    tb = abs(_masked_mean(top) - _masked_mean(bottom)) / denom
    return float(max(lr, tb))


def _patch_relative_dark_fraction(core_mask: np.ndarray, luminance: np.ndarray) -> float:
    if not np.any(core_mask):
        return 1.0
    core_luma = luminance[core_mask]
    threshold = max(8.0, 0.5 * float(np.median(core_luma)))
    return float(np.count_nonzero(core_luma < threshold) / max(core_luma.size, 1))


def _patch_quality_metrics(patch_rgba: np.ndarray) -> dict[str, float | int]:
    return _patch_quality_metrics_shared(patch_rgba)


def _select_best_valid_patch_index(step_valid: np.ndarray, patches: list[np.ndarray]) -> int | None:
    return _select_best_valid_patch_index_shared(step_valid, patches)


def _resolve_best_patch_index(donor_assets: dict[str, Any], patches: list[np.ndarray] | None = None) -> int | None:
    step_valid = np.asarray(donor_assets["step_valid"], dtype=bool)
    preferred_idx = donor_assets.get("best_patch_index")
    if preferred_idx is not None:
        preferred_idx = int(preferred_idx)
        if 0 <= preferred_idx < len(step_valid) and bool(step_valid[preferred_idx]):
            return preferred_idx
    patch_list = donor_assets["step_patch_rgba"] if patches is None else patches
    return _select_best_valid_patch_index(step_valid, patch_list)


def _sample_center_in_bounds(
    *,
    rng: np.random.Generator,
    image_shape: tuple[int, int],
    patch_rgba: np.ndarray,
) -> np.ndarray:
    height, width = image_shape
    patch_h, patch_w = patch_rgba.shape[:2]
    half_w = patch_w / 2.0
    half_h = patch_h / 2.0
    min_x = float(np.ceil(half_w))
    max_x = float(np.floor((width - 1) - half_w))
    min_y = float(np.ceil(half_h))
    max_y = float(np.floor((height - 1) - half_h))
    cx = float(rng.uniform(min_x, max_x)) if max_x > min_x else min_x
    cy = float(rng.uniform(min_y, max_y)) if max_y > min_y else min_y
    return np.asarray([cx, cy], dtype=np.float32)


def sample_iid_random_centers(
    *,
    rng: np.random.Generator,
    num_frames: int,
    image_shape: tuple[int, int],
    patch_rgba: np.ndarray,
) -> np.ndarray:
    centers = [
        _sample_center_in_bounds(rng=rng, image_shape=image_shape, patch_rgba=patch_rgba)
        for _ in range(num_frames)
    ]
    return np.asarray(centers, dtype=np.float32)


def sample_smooth_random_centers(
    *,
    rng: np.random.Generator,
    num_frames: int,
    image_shape: tuple[int, int],
    patch_rgba: np.ndarray,
    velocity_scale: float = 3.0,
    noise_scale: float = 0.6,
    alpha: float = 0.9,
) -> np.ndarray:
    height, width = image_shape
    patch_h, patch_w = patch_rgba.shape[:2]
    half_w = patch_w / 2.0
    half_h = patch_h / 2.0
    min_x = float(np.ceil(half_w))
    max_x = float(np.floor((width - 1) - half_w))
    min_y = float(np.ceil(half_h))
    max_y = float(np.floor((height - 1) - half_h))

    center = _sample_center_in_bounds(rng=rng, image_shape=image_shape, patch_rgba=patch_rgba)
    velocity = rng.normal(0.0, velocity_scale, size=2).astype(np.float32)
    centers = [center.copy()]
    for _ in range(1, num_frames):
        velocity = (alpha * velocity + rng.normal(0.0, noise_scale, size=2)).astype(np.float32)
        candidate = center + velocity
        if candidate[0] < min_x or candidate[0] > max_x:
            velocity[0] *= -1.0
            candidate[0] = np.clip(center[0] + velocity[0], min_x, max_x)
        if candidate[1] < min_y or candidate[1] > max_y:
            velocity[1] *= -1.0
            candidate[1] = np.clip(center[1] + velocity[1], min_y, max_y)
        center = candidate.astype(np.float32)
        centers.append(center.copy())
    return np.asarray(centers, dtype=np.float32)


def _sample_random_trajectory_transform(
    *,
    rng: np.random.Generator,
    donor_assets: dict[str, Any],
    image_shape: tuple[int, int],
    scale_min: float,
    scale_max: float,
    max_scale_ratio: float,
    used_color_names: set[str],
) -> dict[str, Any]:
    best_patch_idx = _resolve_best_patch_index(donor_assets)
    if best_patch_idx is None:
        raise RuntimeError("donor_has_no_valid_patch")

    palette = [name for name in COLOR_NAME_TO_RGB if name != "red" and name not in used_color_names]
    if not palette:
        palette = [name for name in COLOR_NAME_TO_RGB if name != "red"]

    height, width = image_shape
    short_side = min(height, width)
    raw_best_patch = donor_assets["step_patch_rgba"][best_patch_idx]
    for _ in range(128):
        color_name = palette[int(rng.integers(0, len(palette)))]
        color_rgb = COLOR_NAME_TO_RGB[color_name]
        scale_factor = float(rng.uniform(scale_min, scale_max))
        scaled_patch = _scale_patch_rgba(_recolor_patch_rgba(raw_best_patch, color_rgb), scale_factor)
        if max(scaled_patch.shape[:2]) > short_side * max_scale_ratio:
            continue
        return {
            "color": color_name,
            "color_rgb": color_rgb,
            "scale_factor": scale_factor,
            "best_patch_index": int(best_patch_idx),
        }
    raise RuntimeError("random_trajectory_transform_sampling_failed")


def _compose_single_donor_allow_hold(
    base_frames: np.ndarray,
    donor_assets: dict[str, Any],
    transform: dict[str, Any],
) -> np.ndarray:
    out = base_frames.copy()
    scale_factor = float(transform["scale_factor"])
    tx, ty = [int(v) for v in transform["translation_xy"]]
    color_rgb = np.asarray(transform["color_rgb"], dtype=np.uint8)
    best_patch_idx = int(
        transform.get(
            "best_patch_index",
            _resolve_best_patch_index(donor_assets),
        )
    )
    if best_patch_idx < 0:
        raise RuntimeError("donor_has_no_valid_patch")
    transformed_patch = _scale_patch_rgba(
        _recolor_patch_rgba(donor_assets["step_patch_rgba"][best_patch_idx], color_rgb),
        scale_factor,
    )
    for frame_idx in range(len(out)):
        source_idx = _valid_source_index(donor_assets["step_valid"], frame_idx)
        if source_idx is None:
            continue
        center = donor_assets["step_centers_xy"][source_idx] + np.array([tx, ty], dtype=np.float32)
        out[frame_idx] = _composite_patch_rgba(
            out[frame_idx],
            transformed_patch,
            center_xy=(float(center[0]), float(center[1])),
        )
    return out


def _compose_single_donor_from_explicit_transform(
    *,
    base_frames: np.ndarray,
    donor_assets: dict[str, Any],
    color_name: str,
    scale_factor: float,
    translation_xy: tuple[int, int] | list[int],
) -> np.ndarray:
    return _compose_single_donor_allow_hold(
        base_frames,
        donor_assets,
        {
            "color": color_name,
            "color_rgb": COLOR_NAME_TO_RGB[color_name],
            "scale_factor": float(scale_factor),
            "translation_xy": [int(translation_xy[0]), int(translation_xy[1])],
        },
    )


def compose_episode_with_same_trajectory_donor(
    *,
    config: TwoRoomDynamicAugmentConfig,
    target_episode_idx: int,
    donor_entry: DonorEntry,
    color_name: str,
    scale_factor: float,
    translation_xy: tuple[int, int] | list[int],
) -> dict[str, Any]:
    episode = _load_episode_frames(config.input_h5, target_episode_idx)
    frames_before = episode["frames"]
    pos_agent = episode["pos_agent"]
    if pos_agent is None:
        raise RuntimeError("target_episode_missing_pos_agent")

    donor_assets = _load_donor_assets(donor_entry)
    color_rgb = COLOR_NAME_TO_RGB[color_name]
    tx, ty = int(translation_xy[0]), int(translation_xy[1])
    transformed_patches = [
        _scale_patch_rgba(_recolor_patch_rgba(patch, color_rgb), scale_factor)
        for patch in donor_assets["step_patch_rgba"]
    ]
    best_patch_idx = _resolve_best_patch_index(donor_assets, transformed_patches)
    if best_patch_idx is None:
        raise RuntimeError("donor_has_no_valid_patch")
    best_patch = transformed_patches[best_patch_idx]

    out = frames_before.copy()
    for frame_idx in range(len(out)):
        center = pos_agent[frame_idx] + np.array([tx, ty], dtype=np.float32)
        out[frame_idx] = _composite_patch_rgba(
            out[frame_idx],
            best_patch,
            center_xy=(float(center[0]), float(center[1])),
        )

    return {
        "frames_before": frames_before,
        "frames_after": out,
        "target_episode_idx": target_episode_idx,
        "target_ep_len": int(episode["ep_len"]),
        "donor_episode_idx": donor_entry.episode_idx,
        "donor_ep_len": donor_entry.ep_len,
        "color": color_name,
        "scale_factor": float(scale_factor),
        "translation_xy": [tx, ty],
        "copy_mode": "same_trajectory_offset_from_cache",
        "patch_source_mode": "best_valid_patch",
        "best_patch_index": int(best_patch_idx),
    }


def compose_episode_with_random_trajectory_donor(
    *,
    config: TwoRoomDynamicAugmentConfig,
    target_episode_idx: int,
    donor_entry: DonorEntry,
    color_name: str,
    scale_factor: float,
    motion_mode: str,
    seed: int | None = None,
) -> dict[str, Any]:
    episode = _load_episode_frames(config.input_h5, target_episode_idx)
    frames_before = episode["frames"]
    donor_assets = _load_donor_assets(donor_entry)
    color_rgb = COLOR_NAME_TO_RGB[color_name]
    transformed_patches = [
        _scale_patch_rgba(_recolor_patch_rgba(patch, color_rgb), scale_factor)
        for patch in donor_assets["step_patch_rgba"]
    ]
    best_patch_idx = _resolve_best_patch_index(donor_assets, transformed_patches)
    if best_patch_idx is None:
        raise RuntimeError("donor_has_no_valid_patch")
    best_patch = transformed_patches[best_patch_idx]

    rng_seed = config.seed + target_episode_idx * 1009 + 97 if seed is None else int(seed)
    rng = np.random.default_rng(rng_seed)
    if motion_mode == "iid_jump":
        centers = sample_iid_random_centers(
            rng=rng,
            num_frames=len(frames_before),
            image_shape=frames_before[0].shape[:2],
            patch_rgba=best_patch,
        )
    elif motion_mode == "smooth_random":
        centers = sample_smooth_random_centers(
            rng=rng,
            num_frames=len(frames_before),
            image_shape=frames_before[0].shape[:2],
            patch_rgba=best_patch,
        )
    else:
        raise ValueError(f"unknown motion_mode: {motion_mode}")

    out = frames_before.copy()
    for frame_idx in range(len(out)):
        center = centers[frame_idx]
        out[frame_idx] = _composite_patch_rgba(
            out[frame_idx],
            best_patch,
            center_xy=(float(center[0]), float(center[1])),
        )

    return {
        "frames_before": frames_before,
        "frames_after": out,
        "target_episode_idx": target_episode_idx,
        "target_ep_len": int(episode["ep_len"]),
        "donor_episode_idx": donor_entry.episode_idx,
        "donor_ep_len": donor_entry.ep_len,
        "color": color_name,
        "scale_factor": float(scale_factor),
        "copy_mode": "random_trajectory_from_cache",
        "motion_mode": motion_mode,
        "patch_source_mode": "best_valid_patch",
        "best_patch_index": int(best_patch_idx),
        "generated_centers_xy": centers.tolist(),
    }


def _sample_same_trajectory_transform(
    *,
    rng: np.random.Generator,
    donor_assets: dict[str, Any],
    target_centers_xy: np.ndarray,
    image_shape: tuple[int, int],
    scale_min: float,
    scale_max: float,
    max_scale_ratio: float,
    used_color_names: set[str],
) -> dict[str, Any]:
    best_patch_idx = _resolve_best_patch_index(donor_assets)
    if best_patch_idx is None:
        raise RuntimeError("donor_has_no_valid_patch")

    palette = [name for name in COLOR_NAME_TO_RGB if name != "red" and name not in used_color_names]
    if not palette:
        palette = [name for name in COLOR_NAME_TO_RGB if name != "red"]

    height, width = image_shape
    short_side = min(height, width)
    raw_best_patch = donor_assets["step_patch_rgba"][best_patch_idx]

    for _ in range(128):
        color_name = palette[int(rng.integers(0, len(palette)))]
        color_rgb = COLOR_NAME_TO_RGB[color_name]
        scale_factor = float(rng.uniform(scale_min, scale_max))
        scaled_patch = _scale_patch_rgba(_recolor_patch_rgba(raw_best_patch, color_rgb), scale_factor)
        if max(scaled_patch.shape[:2]) > short_side * max_scale_ratio:
            continue
        allowed = _allowed_translation_ranges(
            image_shape,
            target_centers_xy,
            [scaled_patch for _ in range(len(target_centers_xy))],
        )
        if allowed is None:
            continue
        (tx_min, tx_max), (ty_min, ty_max) = allowed
        tx = int(rng.integers(tx_min, tx_max + 1))
        ty = int(rng.integers(ty_min, ty_max + 1))
        return {
            "color": color_name,
            "color_rgb": color_rgb,
            "scale_factor": scale_factor,
            "translation_xy": [tx, ty],
            "best_patch_index": int(best_patch_idx),
        }
    raise RuntimeError("same_trajectory_transform_sampling_failed")


def _compose_same_trajectory_dataset_episode(
    *,
    config: TwoRoomDynamicAugmentConfig,
    target_episode_idx: int,
    donor_entry: DonorEntry,
) -> dict[str, Any]:
    episode = _load_episode_frames(config.input_h5, target_episode_idx)
    frames_before = episode["frames"]
    pos_agent = episode["pos_agent"]
    if pos_agent is None:
        raise RuntimeError("target_episode_missing_pos_agent")

    donor_assets = _load_donor_assets(donor_entry)
    rng = np.random.default_rng(config.seed + target_episode_idx * 1009 + 31)
    transform = _sample_same_trajectory_transform(
        rng=rng,
        donor_assets=donor_assets,
        target_centers_xy=pos_agent,
        image_shape=frames_before[0].shape[:2],
        scale_min=config.scale_min,
        scale_max=config.scale_max,
        max_scale_ratio=config.max_scale_ratio,
        used_color_names=set(),
    )
    result = compose_episode_with_same_trajectory_donor(
        config=config,
        target_episode_idx=target_episode_idx,
        donor_entry=donor_entry,
        color_name=transform["color"],
        scale_factor=float(transform["scale_factor"]),
        translation_xy=transform["translation_xy"],
    )
    result["best_patch_index"] = int(transform["best_patch_index"])
    return result


def select_donor_entries_for_target(
    donor_entries: list[DonorEntry],
    target_episode_idx: int,
    target_ep_len: int,
    donor_count: int,
    seed: int,
) -> list[DonorEntry]:
    candidates = [entry for entry in donor_entries if entry.episode_idx != target_episode_idx]
    if len(candidates) < donor_count:
        raise RuntimeError("not_enough_donors")
    candidates = sorted(candidates, key=lambda e: (-e.ep_len, e.rank, e.episode_idx))
    gte = [entry for entry in candidates if entry.ep_len >= target_ep_len]
    lt = [entry for entry in candidates if entry.ep_len < target_ep_len]

    selected: list[DonorEntry] = []
    if gte:
        offset = (target_episode_idx + seed) % len(gte)
        ordered_gte = gte[offset:] + gte[:offset]
        selected.extend(ordered_gte[:donor_count])
    if len(selected) < donor_count:
        need = donor_count - len(selected)
        fallback_pool = [entry for entry in lt if entry not in selected]
        selected.extend(fallback_pool[:need])
    if len(selected) < donor_count:
        raise RuntimeError("not_enough_donors_after_fallback")
    return selected[:donor_count]


def _compose_with_selected_donors(
    *,
    config: TwoRoomDynamicAugmentConfig,
    target_episode_idx: int,
    selected_entries: list[DonorEntry],
) -> dict[str, Any]:
    episode = _load_episode_frames(config.input_h5, target_episode_idx)
    frames_before = episode["frames"]
    target_ep_len = int(episode["ep_len"])
    rng = np.random.default_rng(config.seed + target_episode_idx * 1009 + len(selected_entries) * 17)

    transforms = []
    used_colors: set[str] = set()
    composed_frames = frames_before.copy()
    cumulative_frames = []
    for entry in selected_entries:
        donor_assets = _load_donor_assets(entry)
        transform = _sample_donor_transform(
            rng=rng,
            donor_assets=donor_assets,
            image_shape=frames_before[0].shape[:2],
            scale_min=config.scale_min,
            scale_max=config.scale_max,
            max_scale_ratio=config.max_scale_ratio,
            used_color_names=used_colors,
        )
        used_colors.add(transform["color"])
        composed_frames = _compose_single_donor_allow_hold(composed_frames, donor_assets, transform)
        transforms.append(
            {
                "donor_episode_idx": entry.episode_idx,
                "donor_ep_len": entry.ep_len,
                "rank": entry.rank,
                "color": transform["color"],
                "scale_factor": float(transform["scale_factor"]),
                "translation_xy": transform["translation_xy"],
                "length_match_mode": "gte_match" if entry.ep_len >= target_ep_len else "fallback_hold_last",
                "best_patch_index": int(transform["best_patch_index"]),
                "patch_source_mode": "best_valid_patch",
            }
        )
        cumulative_frames.append(composed_frames.copy())

    frames_by_count = {
        count: cumulative_frames[count - 1].copy()
        for count in range(1, len(cumulative_frames) + 1)
    }

    return {
        "frames_before": frames_before,
        "frames_by_count": frames_by_count,
        "frames_d1": cumulative_frames[0] if len(cumulative_frames) >= 1 else frames_before.copy(),
        "frames_d2": cumulative_frames[1] if len(cumulative_frames) >= 2 else (cumulative_frames[0] if cumulative_frames else frames_before.copy()),
        "frames_d3": cumulative_frames[2] if len(cumulative_frames) >= 3 else (cumulative_frames[-1] if cumulative_frames else frames_before.copy()),
        "donor_transforms": transforms,
        "target_episode_idx": target_episode_idx,
        "target_ep_len": target_ep_len,
    }


def _plan_dynamic_episode_rows(
    *,
    config: TwoRoomDynamicAugmentConfig,
    donor_entries: list[DonorEntry],
    target_episode_idx: int,
    target_ep_len: int,
    image_shape: tuple[int, int],
    donor_counts: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    donor_counts = _normalize_donor_counts(donor_counts)
    selected_entries = select_donor_entries_for_target(
        donor_entries=donor_entries,
        target_episode_idx=target_episode_idx,
        target_ep_len=target_ep_len,
        donor_count=max(donor_counts),
        seed=config.seed,
    )
    rng = np.random.default_rng(config.seed + target_episode_idx * 1009 + len(selected_entries) * 17)
    used_colors: set[str] = set()
    planned_transforms: list[dict[str, Any]] = []
    for entry in selected_entries:
        donor_assets = _load_donor_assets(entry)
        transform = _sample_donor_transform(
            rng=rng,
            donor_assets=donor_assets,
            image_shape=image_shape,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
            max_scale_ratio=config.max_scale_ratio,
            used_color_names=used_colors,
        )
        used_colors.add(transform["color"])
        planned_transforms.append(
            {
                "donor_episode_idx": entry.episode_idx,
                "donor_ep_len": entry.ep_len,
                "rank": entry.rank,
                "color": transform["color"],
                "scale_factor": float(transform["scale_factor"]),
                "translation_xy": transform["translation_xy"],
                "length_match_mode": "gte_match" if entry.ep_len >= target_ep_len else "fallback_hold_last",
                "best_patch_index": int(transform["best_patch_index"]),
                "patch_source_mode": "best_valid_patch",
            }
        )

    return {
        count: {
            "episode_idx": int(target_episode_idx),
            "num_donors": int(count),
            "augmentation_success": True,
            "failure_reason": None,
            "donor_transforms": planned_transforms[:count],
            "inserted_objects": planned_transforms[:count],
            "target_ep_len": int(target_ep_len),
            "copy_mode": "dynamic_replay_from_cache",
            "count_coupling": "prefix_control",
            "color_policy": "non_red_only_non_red_reuse_allowed",
            "seed": int(config.seed),
        }
        for count in donor_counts
    }


def compose_episode_with_dynamic_donors(
    *,
    config: TwoRoomDynamicAugmentConfig,
    target_episode_idx: int,
    donor_entries: list[DonorEntry],
    donor_count: int,
) -> dict[str, Any]:
    selected_entries = donor_entries[:donor_count]
    return _compose_with_selected_donors(
        config=config,
        target_episode_idx=target_episode_idx,
        selected_entries=selected_entries,
    )


def _build_labeled_contact(frames: list[np.ndarray], labels: list[str], output_path: Path) -> None:
    label_h = 20
    frame_h = max(frame.shape[0] for frame in frames)
    frame_w = max(frame.shape[1] for frame in frames)
    canvas = Image.new("RGB", (frame_w * len(frames), frame_h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (frame, label) in enumerate(zip(frames, labels)):
        x0 = idx * frame_w
        canvas.paste(Image.fromarray(frame.astype(np.uint8)).convert("RGB"), (x0, label_h))
        draw.text((x0 + 4, 2), label, fill=(0, 0, 0))
    canvas.save(output_path)


def _write_labeled_preview_video(
    path: Path,
    frames_by_label: list[np.ndarray],
    labels: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_h = 20
    stacked_frames = []
    for frame_group in zip(*frames_by_label):
        row = np.concatenate(list(frame_group), axis=1)
        canvas = Image.new("RGB", (row.shape[1], row.shape[0] + label_h), (255, 255, 255))
        canvas.paste(Image.fromarray(row.astype(np.uint8)).convert("RGB"), (0, label_h))
        draw = ImageDraw.Draw(canvas)
        panel_w = frame_group[0].shape[1]
        for idx, label in enumerate(labels):
            draw.text((idx * panel_w + 4, 2), label, fill=(0, 0, 0))
        stacked_frames.append(np.asarray(canvas, dtype=np.uint8))
    imageio.mimsave(path, stacked_frames, fps=10, **_PREVIEW_VIDEO_KWARGS)


def build_dynamic_preview_set(
    *,
    config: TwoRoomDynamicAugmentConfig,
    target_episode_indices: list[int],
    donor_counts: tuple[int, ...] | list[int],
) -> dict[str, Any]:
    donor_counts = _normalize_donor_counts(donor_counts)
    max_donor_count = max(donor_counts)
    donor_entries = load_successful_donor_entries(config.donor_root)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode_idx in target_episode_indices:
        target_ep_len = int(_load_episode_frames(config.input_h5, episode_idx)["ep_len"])
        selected = select_donor_entries_for_target(
            donor_entries=donor_entries,
            target_episode_idx=episode_idx,
            target_ep_len=target_ep_len,
            donor_count=max_donor_count,
            seed=config.seed,
        )
        result = _compose_with_selected_donors(
            config=config,
            target_episode_idx=episode_idx,
            selected_entries=selected,
        )
        case_dir = config.output_dir / f"episode_{episode_idx:05d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        labels = ["d0"] + [f"d{count}" for count in donor_counts]
        _write_labeled_preview_video(
            case_dir / "preview.mp4",
            [result["frames_before"], *[result["frames_by_count"][count] for count in donor_counts]],
            labels,
        )
        first_frame_contact = case_dir / "first_frame_contact.png"
        _build_labeled_contact(
            [
                result["frames_before"][0],
                *[result["frames_by_count"][count][0] for count in donor_counts],
            ],
            labels,
            first_frame_contact,
        )
        shutil.copy2(first_frame_contact, case_dir / "first_frame.png")
        case_summary = {
            "target_episode_idx": episode_idx,
            "target_ep_len": result["target_ep_len"],
            "donor_counts": list(donor_counts),
            "donor_transforms": result["donor_transforms"],
            "preview_video": str(case_dir / "preview.mp4"),
            "first_frame_contact": str(first_frame_contact),
            "first_frame": str(case_dir / "first_frame.png"),
        }
        _write_json(case_dir / "summary.json", case_summary)
        rows.append(case_summary)

    summary = {
        "num_cases": len(rows),
        "target_episode_indices": target_episode_indices,
        "donor_counts": list(donor_counts),
        "cases": rows,
        "config": asdict(config),
    }
    _write_json(config.output_dir / "summary.json", summary)
    return summary


def build_d0_d3_preview_set(
    *,
    config: TwoRoomDynamicAugmentConfig,
    target_episode_indices: list[int],
) -> dict[str, Any]:
    return build_dynamic_preview_set(
        config=config,
        target_episode_indices=target_episode_indices,
        donor_counts=(1, 2, 3),
    )


def _resolve_episode_indices(
    num_episodes: int,
    episode_start: int | None,
    episode_end: int | None,
) -> list[int]:
    start = 0 if episode_start is None else int(episode_start)
    end = num_episodes if episode_end is None else int(episode_end)
    if start < 0 or end < 0 or start > end or end > num_episodes:
        raise IndexError("invalid episode range")
    return list(range(start, end))


def _load_metadata_rows(metadata_jsonl: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in metadata_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _filter_metadata_rows(
    rows: list[dict[str, Any]],
    episode_indices: set[int],
) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["episode_idx"]) in episode_indices]


def augment_tworoom_dynamic_dataset_fanout(
    config: TwoRoomDynamicFanoutConfig,
) -> dict[str, Any]:
    donor_counts = _normalize_donor_counts(config.donor_counts)
    print(
        f"[fanout] start donor_counts={list(donor_counts)} resume_from_episode={config.resume_from_episode} reset_tail_to_clean={config.reset_tail_to_clean}",
        flush=True,
    )
    split_names = {count: f"d{count}" for count in donor_counts}
    max_donor_count = max(donor_counts)
    config.output_root.mkdir(parents=True, exist_ok=True)
    output_dirs = {count: config.output_root / split_names[count] for count in donor_counts}
    output_h5s = {name: output_dirs[name] / "augmented.h5" for name in output_dirs}
    for count, out_dir in output_dirs.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        if config.resume_from_episode is not None and output_h5s[count].exists():
            continue
        shutil.copy2(config.input_h5, output_h5s[count])

    print("[fanout] loading donor entries", flush=True)
    donor_entries = load_successful_donor_entries(config.donor_root)
    print(f"[fanout] loaded {len(donor_entries)} donor entries", flush=True)
    compose_cfg = TwoRoomDynamicAugmentConfig(
        input_h5=config.input_h5,
        donor_root=config.donor_root,
        output_dir=config.output_root,
        seed=config.seed,
        scale_min=config.scale_min,
        scale_max=config.scale_max,
        max_scale_ratio=config.max_scale_ratio,
    )

    with h5py.File(config.input_h5, "r") as src:
        print("[fanout] opened source h5", flush=True)
        ep_len = src["ep_len"][:]
        ep_offset = src["ep_offset"][:]
        pixel_key = _resolve_pixel_key(src)
        image_shape = tuple(int(v) for v in src[pixel_key].shape[1:3])
        episode_indices = _resolve_episode_indices(len(ep_len), config.episode_start, config.episode_end)
        resume_from_episode = (
            None
            if config.resume_from_episode is None
            else max(int(config.resume_from_episode), episode_indices[0])
        )
        rows_by_name = {count: [] for count in donor_counts}
        if resume_from_episode is not None and resume_from_episode > episode_indices[0]:
            prefix_episode_indices = [episode_idx for episode_idx in episode_indices if episode_idx < resume_from_episode]
            print(
                f"[fanout] rebuilding metadata rows for prefix episodes < {resume_from_episode} (count={len(prefix_episode_indices)})",
                flush=True,
            )
            for episode_idx in prefix_episode_indices:
                target_ep_len = int(ep_len[episode_idx])
                planned_rows = _plan_dynamic_episode_rows(
                    config=compose_cfg,
                    donor_entries=donor_entries,
                    target_episode_idx=int(episode_idx),
                    target_ep_len=target_ep_len,
                    image_shape=image_shape,
                    donor_counts=donor_counts,
                )
                for count in donor_counts:
                    rows_by_name[count].append(planned_rows[count])
        dst_handles = {count: h5py.File(output_h5s[count], "r+") for count in donor_counts}
        try:
            print("[fanout] opened destination h5 files", flush=True)
            dst_map = {count: dst_handles[count][pixel_key] for count in donor_counts}
            if resume_from_episode is not None and config.reset_tail_to_clean:
                resume_start = int(ep_offset[resume_from_episode])
                resume_end = int(ep_offset[episode_indices[-1]] + ep_len[episode_indices[-1]])
                print(
                    f"[fanout] resetting tail to clean for frames [{resume_start}:{resume_end}]",
                    flush=True,
                )
                for count in donor_counts:
                    dst_map[count][resume_start:resume_end] = src[pixel_key][resume_start:resume_end]
                print("[fanout] tail reset complete", flush=True)

            run_episode_indices = [
                episode_idx
                for episode_idx in episode_indices
                if resume_from_episode is None or episode_idx >= resume_from_episode
            ]
            if resume_from_episode is not None:
                print(
                    f"[fanout-resume] rebuilding prefix rows for {len(rows_by_name[donor_counts[0]])} episodes; resuming pixel writes from episode {resume_from_episode}",
                    flush=True,
                )
            for loop_idx, episode_idx in enumerate(run_episode_indices, start=1):
                if loop_idx == 1 or loop_idx % 100 == 0 or loop_idx == len(run_episode_indices):
                    print(
                        f"[fanout] processing episode {episode_idx} ({loop_idx}/{len(run_episode_indices)})",
                        flush=True,
                    )
                target_ep_len = int(ep_len[episode_idx])
                selected = select_donor_entries_for_target(
                    donor_entries=donor_entries,
                    target_episode_idx=int(episode_idx),
                    target_ep_len=target_ep_len,
                    donor_count=max_donor_count,
                    seed=config.seed,
                )
                result = _compose_with_selected_donors(
                    config=compose_cfg,
                    target_episode_idx=int(episode_idx),
                    selected_entries=selected,
                )
                start = int(ep_offset[episode_idx])
                end = int(start + ep_len[episode_idx])
                for count in donor_counts:
                    dst_map[count][start:end] = result["frames_by_count"][count]
                    donor_transforms = result["donor_transforms"][:count]
                    rows_by_name[count].append(
                        {
                            "episode_idx": int(episode_idx),
                            "num_donors": count,
                            "augmentation_success": True,
                            "failure_reason": None,
                            "donor_transforms": donor_transforms,
                            "inserted_objects": donor_transforms,
                            "target_ep_len": target_ep_len,
                            "copy_mode": "dynamic_replay_from_cache",
                            "count_coupling": "prefix_control",
                            "color_policy": "non_red_only_non_red_reuse_allowed",
                            "seed": int(config.seed),
                        }
                    )
        finally:
            for handle in dst_handles.values():
                handle.close()

    for count, rows in rows_by_name.items():
        _write_metadata_files(
            rows,
            output_dirs[count],
            {
                "num_episodes": len(rows),
                "num_successes": len(rows),
                "copy_mode": "dynamic_replay_from_cache",
                "donor_count": int(count),
                "count_coupling": "prefix_control",
                "color_policy": "non_red_only_non_red_reuse_allowed",
                "seed": int(config.seed),
            },
        )
    return {
        "num_episodes": len(rows_by_name[donor_counts[0]]),
        "output_root": str(config.output_root),
        "donor_pool_size": len(donor_entries),
        "donor_counts": list(donor_counts),
    }


def augment_tworoom_dynamic_dataset_from_metadata(
    *,
    config: TwoRoomDynamicFanoutConfig,
    metadata_jsonl: Path,
    forced_color_name: str,
) -> dict[str, Any]:
    if forced_color_name not in COLOR_NAME_TO_RGB:
        raise KeyError(f"unknown color: {forced_color_name}")

    config.output_root.mkdir(parents=True, exist_ok=True)
    output_h5 = config.output_root / "augmented.h5"
    shutil.copy2(config.input_h5, output_h5)

    donor_entries = load_successful_donor_entries(config.donor_root)
    donor_by_episode = {int(entry.episode_idx): entry for entry in donor_entries}

    with h5py.File(config.input_h5, "r") as src:
        ep_len = src["ep_len"][:]
        ep_offset = src["ep_offset"][:]
        pixel_key = _resolve_pixel_key(src)
        episode_indices = set(
            _resolve_episode_indices(len(ep_len), config.episode_start, config.episode_end)
        )

    source_rows = _filter_metadata_rows(_load_metadata_rows(metadata_jsonl), episode_indices)

    with h5py.File(output_h5, "r+") as dst:
        dst_pixels = dst[pixel_key]
        output_rows = []
        for row in source_rows:
            episode_idx = int(row["episode_idx"])
            target_ep_len = int(row["target_ep_len"])
            donor_transforms = row.get("donor_transforms") or []
            if len(donor_transforms) != 1:
                raise RuntimeError("expected_single_donor_row")
            donor_transform = dict(donor_transforms[0])
            donor_episode_idx = int(donor_transform["donor_episode_idx"])
            donor_entry = donor_by_episode.get(donor_episode_idx)
            if donor_entry is None:
                raise RuntimeError(f"missing_donor_entry:{donor_episode_idx}")
            donor_assets = _load_donor_assets(donor_entry)
            episode = _load_episode_frames(config.input_h5, episode_idx)
            frames_after = _compose_single_donor_from_explicit_transform(
                base_frames=episode["frames"],
                donor_assets=donor_assets,
                color_name=forced_color_name,
                scale_factor=float(donor_transform["scale_factor"]),
                translation_xy=donor_transform["translation_xy"],
            )
            start = int(ep_offset[episode_idx])
            end = int(start + ep_len[episode_idx])
            dst_pixels[start:end] = frames_after

            original_color = donor_transform.get("color")
            donor_transform["original_color"] = original_color
            donor_transform["color"] = forced_color_name
            inserted_object = dict((row.get("inserted_objects") or donor_transforms)[0])
            inserted_object["original_color"] = inserted_object.get("color")
            inserted_object["color"] = forced_color_name
            output_rows.append(
                {
                    "episode_idx": episode_idx,
                    "num_donors": 1,
                    "augmentation_success": True,
                    "failure_reason": None,
                    "donor_transforms": [donor_transform],
                    "inserted_objects": [inserted_object],
                    "target_ep_len": target_ep_len,
                    "copy_mode": "dynamic_replay_from_cache_color_override",
                }
            )

    _write_metadata_files(
        output_rows,
        config.output_root,
        {
            "num_episodes": len(output_rows),
            "num_successes": len(output_rows),
            "copy_mode": "dynamic_replay_from_cache_color_override",
            "forced_color_name": forced_color_name,
            "source_metadata_jsonl": metadata_jsonl,
        },
    )
    return {
        "num_episodes": len(output_rows),
        "output_root": str(config.output_root),
        "donor_pool_size": len(donor_entries),
        "forced_color_name": forced_color_name,
        "copy_mode": "dynamic_replay_from_cache_color_override",
    }


def augment_tworoom_same_trajectory_dataset(
    config: TwoRoomDynamicFanoutConfig,
) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    output_h5 = config.output_root / "augmented.h5"
    shutil.copy2(config.input_h5, output_h5)

    donor_entries = load_successful_donor_entries(config.donor_root)
    compose_cfg = TwoRoomDynamicAugmentConfig(
        input_h5=config.input_h5,
        donor_root=config.donor_root,
        output_dir=config.output_root,
        seed=config.seed,
        scale_min=config.scale_min,
        scale_max=config.scale_max,
        max_scale_ratio=config.max_scale_ratio,
    )

    with h5py.File(config.input_h5, "r") as src, h5py.File(output_h5, "r+") as dst:
        ep_len = src["ep_len"][:]
        ep_offset = src["ep_offset"][:]
        pixel_key = _resolve_pixel_key(src)
        episode_indices = _resolve_episode_indices(len(ep_len), config.episode_start, config.episode_end)
        dst_pixels = dst[pixel_key]
        rows = []
        for episode_idx in episode_indices:
            target_ep_len = int(ep_len[episode_idx])
            donor_entry = select_donor_entries_for_target(
                donor_entries=donor_entries,
                target_episode_idx=int(episode_idx),
                target_ep_len=target_ep_len,
                donor_count=1,
                seed=config.seed,
            )[0]
            result = _compose_same_trajectory_dataset_episode(
                config=compose_cfg,
                target_episode_idx=int(episode_idx),
                donor_entry=donor_entry,
            )
            start = int(ep_offset[episode_idx])
            end = int(start + ep_len[episode_idx])
            dst_pixels[start:end] = result["frames_after"]
            rows.append(
                {
                    "episode_idx": int(episode_idx),
                    "num_donors": 1,
                    "augmentation_success": True,
                    "failure_reason": None,
                    "donor_transforms": [
                        {
                            "donor_episode_idx": int(result["donor_episode_idx"]),
                            "donor_ep_len": int(result["donor_ep_len"]),
                            "rank": int(donor_entry.rank),
                            "color": result["color"],
                            "scale_factor": float(result["scale_factor"]),
                            "translation_xy": result["translation_xy"],
                            "best_patch_index": int(result["best_patch_index"]),
                            "patch_source_mode": result["patch_source_mode"],
                            "length_match_mode": "same_trajectory_target_motion",
                        }
                    ],
                    "inserted_objects": [
                        {
                            "donor_episode_idx": int(result["donor_episode_idx"]),
                            "donor_ep_len": int(result["donor_ep_len"]),
                            "color": result["color"],
                            "scale_factor": float(result["scale_factor"]),
                            "translation_xy": result["translation_xy"],
                            "best_patch_index": int(result["best_patch_index"]),
                            "patch_source_mode": result["patch_source_mode"],
                        }
                    ],
                    "target_ep_len": target_ep_len,
                    "copy_mode": "same_trajectory_offset_from_cache",
                }
            )

    _write_metadata_files(
        rows,
        config.output_root,
        {
            "num_episodes": len(rows),
            "num_successes": len(rows),
            "copy_mode": "same_trajectory_offset_from_cache",
        },
    )
    return {
        "num_episodes": len(rows),
        "output_root": str(config.output_root),
        "donor_pool_size": len(donor_entries),
        "copy_mode": "same_trajectory_offset_from_cache",
    }


def augment_tworoom_random_trajectory_dataset(
    config: TwoRoomDynamicFanoutConfig,
    *,
    motion_mode: str,
) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    output_h5 = config.output_root / "augmented.h5"
    shutil.copy2(config.input_h5, output_h5)

    donor_entries = load_successful_donor_entries(config.donor_root)
    compose_cfg = TwoRoomDynamicAugmentConfig(
        input_h5=config.input_h5,
        donor_root=config.donor_root,
        output_dir=config.output_root,
        seed=config.seed,
        scale_min=config.scale_min,
        scale_max=config.scale_max,
        max_scale_ratio=config.max_scale_ratio,
    )

    with h5py.File(config.input_h5, "r") as src, h5py.File(output_h5, "r+") as dst:
        ep_len = src["ep_len"][:]
        ep_offset = src["ep_offset"][:]
        pixel_key = _resolve_pixel_key(src)
        episode_indices = _resolve_episode_indices(len(ep_len), config.episode_start, config.episode_end)
        dst_pixels = dst[pixel_key]
        rows = []
        for episode_idx in episode_indices:
            target_ep_len = int(ep_len[episode_idx])
            donor_entry = select_donor_entries_for_target(
                donor_entries=donor_entries,
                target_episode_idx=int(episode_idx),
                target_ep_len=target_ep_len,
                donor_count=1,
                seed=config.seed,
            )[0]
            donor_assets = _load_donor_assets(donor_entry)
            rng = np.random.default_rng(config.seed + episode_idx * 1009 + 131)
            transform = _sample_random_trajectory_transform(
                rng=rng,
                donor_assets=donor_assets,
                image_shape=dst_pixels[0].shape[:2],
                scale_min=config.scale_min,
                scale_max=config.scale_max,
                max_scale_ratio=config.max_scale_ratio,
                used_color_names=set(),
            )
            result = compose_episode_with_random_trajectory_donor(
                config=compose_cfg,
                target_episode_idx=int(episode_idx),
                donor_entry=donor_entry,
                color_name=transform["color"],
                scale_factor=float(transform["scale_factor"]),
                motion_mode=motion_mode,
                seed=config.seed + episode_idx * 1009 + 197,
            )
            start = int(ep_offset[episode_idx])
            end = int(start + ep_len[episode_idx])
            dst_pixels[start:end] = result["frames_after"]
            rows.append(
                {
                    "episode_idx": int(episode_idx),
                    "num_donors": 1,
                    "augmentation_success": True,
                    "failure_reason": None,
                    "donor_transforms": [
                        {
                            "donor_episode_idx": int(result["donor_episode_idx"]),
                            "donor_ep_len": int(result["donor_ep_len"]),
                            "rank": int(donor_entry.rank),
                            "color": result["color"],
                            "scale_factor": float(result["scale_factor"]),
                            "best_patch_index": int(result["best_patch_index"]),
                            "patch_source_mode": result["patch_source_mode"],
                            "length_match_mode": motion_mode,
                        }
                    ],
                    "inserted_objects": [
                        {
                            "donor_episode_idx": int(result["donor_episode_idx"]),
                            "donor_ep_len": int(result["donor_ep_len"]),
                            "color": result["color"],
                            "scale_factor": float(result["scale_factor"]),
                            "best_patch_index": int(result["best_patch_index"]),
                            "patch_source_mode": result["patch_source_mode"],
                            "motion_mode": motion_mode,
                        }
                    ],
                    "target_ep_len": target_ep_len,
                    "copy_mode": "random_trajectory_from_cache",
                    "motion_mode": motion_mode,
                    "generated_centers_xy": result["generated_centers_xy"],
                }
            )

    _write_metadata_files(
        rows,
        config.output_root,
        {
            "num_episodes": len(rows),
            "num_successes": len(rows),
            "copy_mode": "random_trajectory_from_cache",
            "motion_mode": motion_mode,
        },
    )
    return {
        "num_episodes": len(rows),
        "output_root": str(config.output_root),
        "donor_pool_size": len(donor_entries),
        "copy_mode": "random_trajectory_from_cache",
        "motion_mode": motion_mode,
    }
