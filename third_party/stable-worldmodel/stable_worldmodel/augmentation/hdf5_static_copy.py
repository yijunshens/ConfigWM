"""Static object-copy augmentation for stable-worldmodel HDF5 datasets."""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL import ImageFilter

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None

from .patch_rgba import normalize_visible_luminance
from .registry import (
    COLOR_NAME_TO_RGB,
    build_prompt_candidates,
    describe_color_name,
    describe_core_object,
    get_object_spec,
)

_PREVIEW_VIDEO_KWARGS = {
    'codec': 'libx264',
    'quality': 10,
    'macro_block_size': 1,
    'ffmpeg_params': ['-crf', '8', '-preset', 'slow', '-pix_fmt', 'yuv444p'],
}


class SegmenterBackend(Protocol):
    def segment_best(
        self,
        image_np: np.ndarray,
        prompts: list[str],
        confidence_threshold: float,
        debug_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass
class ObjectCopyConfig:
    input_h5: Path
    output_h5: Path | None
    env_id: str
    object_name: str
    num: int
    seed: int = 0
    colors: list[str] | None = None
    drop_original: bool = False
    debug_episode_idx: int | None = None
    debug_only: bool = False
    confidence_threshold: float = 0.5
    reference_frames_to_try: int = 5
    episode_start: int | None = None
    episode_end: int | None = None


@dataclass
class FanoutObjectCopyConfig:
    input_h5: Path
    output_root: Path
    env_id: str
    object_name: str
    num_values: list[int]
    seed: int = 0
    colors: list[str] | None = None
    drop_original: bool = False
    confidence_threshold: float = 0.5
    reference_frames_to_try: int = 5
    episode_start: int | None = None
    episode_end: int | None = None


def _resolve_episode_indices(
    num_episodes: int,
    episode_start: int | None,
    episode_end: int | None,
) -> list[int]:
    start = 0 if episode_start is None else int(episode_start)
    end = num_episodes if episode_end is None else int(episode_end)
    if start < 0 or end < 0:
        raise IndexError('episode_start and episode_end must be >= 0')
    if start > end:
        raise IndexError('episode_start must be <= episode_end')
    if end > num_episodes:
        raise IndexError(f'episode_end {end} out of range for {num_episodes} episodes')
    if start >= num_episodes:
        raise IndexError(f'episode_start {start} out of range for {num_episodes} episodes')
    return list(range(start, end))


def _resolve_pixel_key(h5_file: h5py.File) -> str:
    if 'pixels' in h5_file:
        return 'pixels'
    pixel_views = sorted(k for k in h5_file.keys() if k.startswith('pixels.'))
    if pixel_views:
        return pixel_views[0]
    raise KeyError("No pixel dataset found. Expected 'pixels' or a 'pixels.*' dataset.")


def _mask_to_bbox(mask_bool: np.ndarray) -> list[int]:
    ys, xs = np.where(mask_bool)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _boxes_overlap(box_a: list[int], box_b: list[int], pad: int = 0) -> bool:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    return not (
        ax1 + pad < bx0
        or bx1 + pad < ax0
        or ay1 + pad < by0
        or by1 + pad < ay0
    )


def _bbox_from_center(
    center_xy: tuple[int, int],
    patch_shape_hw: tuple[int, int],
) -> list[int]:
    patch_h, patch_w = patch_shape_hw
    cx, cy = center_xy
    x0 = int(round(cx - patch_w / 2))
    y0 = int(round(cy - patch_h / 2))
    x1 = x0 + patch_w - 1
    y1 = y0 + patch_h - 1
    return [x0, y0, x1, y1]


def sample_non_overlapping_centers(
    image_shape_hw: tuple[int, int],
    patch_shape_hw: tuple[int, int],
    num: int,
    original_bbox_xyxy: list[int],
    seed: int,
    pad: int = 4,
    max_tries: int = 4000,
) -> list[tuple[int, int]]:
    height, width = image_shape_hw
    patch_h, patch_w = patch_shape_hw
    half_w = math.ceil(patch_w / 2)
    half_h = math.ceil(patch_h / 2)
    rng = np.random.default_rng(seed)

    centers: list[tuple[int, int]] = []
    boxes: list[list[int]] = []
    for _ in range(max_tries):
        if len(centers) == num:
            break
        cx = int(rng.integers(half_w, max(half_w + 1, width - half_w)))
        cy = int(rng.integers(half_h, max(half_h + 1, height - half_h)))
        cand_box = _bbox_from_center((cx, cy), patch_shape_hw)
        if cand_box[0] < 0 or cand_box[1] < 0 or cand_box[2] >= width or cand_box[3] >= height:
            continue
        if _boxes_overlap(cand_box, original_bbox_xyxy, pad=pad):
            continue
        if any(_boxes_overlap(cand_box, existing, pad=pad) for existing in boxes):
            continue
        centers.append((cx, cy))
        boxes.append(cand_box)

    return centers


def _normalize_rgb(rgb: np.ndarray | list[float] | list[int] | None) -> np.ndarray | None:
    if rgb is None:
        return None
    arr = np.asarray(rgb, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    arr = arr[:3]
    if float(np.max(arr)) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _extract_variation_rgb(
    episode_data: dict[str, Any],
    key: str | None,
    color_index: int | None = None,
) -> np.ndarray | None:
    if not key or key not in episode_data:
        return None
    value = episode_data[key]
    if hasattr(value, 'numpy'):
        value = value.numpy()
    arr = np.asarray(value)
    if arr.ndim >= 2:
        arr = arr[0]
    if color_index is not None and arr.ndim >= 2:
        arr = arr[color_index]
    return _normalize_rgb(arr)


def _infer_mask_rgb(image_np: np.ndarray, mask_bool: np.ndarray) -> np.ndarray | None:
    if not mask_bool.any():
        return None
    mean_rgb = image_np[mask_bool].mean(axis=0)
    return _normalize_rgb(mean_rgb)


def _recolor_rgb_with_mask(
    image_np: np.ndarray,
    mask_bool: np.ndarray,
    target_rgb: np.ndarray,
) -> np.ndarray:
    recolored = image_np.copy().astype(np.float32)
    masked_pixels = recolored[mask_bool]
    if masked_pixels.size == 0:
        return recolored.astype(np.uint8)
    luminance = (
        0.299 * masked_pixels[:, 0]
        + 0.587 * masked_pixels[:, 1]
        + 0.114 * masked_pixels[:, 2]
    ) / 255.0
    luminance = normalize_visible_luminance(
        luminance,
        np.ones_like(luminance, dtype=bool),
        reference_mask=np.ones_like(luminance, dtype=bool),
    )[:, None]
    recolored[mask_bool] = target_rgb[None, :].astype(np.float32) * luminance
    return np.clip(recolored, 0, 255).astype(np.uint8)


def _extract_feathered_patch(
    image_np: np.ndarray,
    mask_bool: np.ndarray,
    blur_radius: float = 1.5,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    x0, y0, x1, y1 = _mask_to_bbox(mask_bool)
    crop_rgb = image_np[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_mask = mask_bool[y0 : y1 + 1, x0 : x1 + 1]
    crop_h, crop_w = crop_mask.shape
    feather_pad = 0 if blur_radius <= 0 else max(1, int(math.ceil(float(blur_radius) * 3.0)))
    canvas_h = crop_h + 2 * feather_pad
    canvas_w = crop_w + 2 * feather_pad
    canvas_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    row_slice = slice(feather_pad, feather_pad + crop_h)
    col_slice = slice(feather_pad, feather_pad + crop_w)
    canvas_mask[row_slice, col_slice] = crop_mask
    canvas_rgb_crop = canvas_rgb[row_slice, col_slice]
    canvas_rgb_crop[crop_mask] = crop_rgb[crop_mask]

    if blur_radius > 0:
        alpha = np.array(
            Image.fromarray((canvas_mask.astype(np.uint8) * 255)).filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            ),
            dtype=np.uint8,
        )
    else:
        alpha = (canvas_mask.astype(np.uint8) * 255)
    alpha[canvas_mask] = 255
    patch_rgba = np.concatenate([canvas_rgb, alpha[..., None]], axis=-1)
    patch_rgba = _sanitize_patch_rgba(patch_rgba, blur_radius=blur_radius)
    return patch_rgba, [x0, y0, x1, y1], canvas_mask


def _erode_binary_mask(mask_bool: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return mask_bool.astype(bool).copy()
    mask = np.asarray(mask_bool, dtype=bool)
    height, width = mask.shape
    padded = np.pad(mask.astype(np.uint8), radius, mode="constant", constant_values=0)
    eroded = np.ones((height, width), dtype=bool)
    size = radius * 2 + 1
    for dy in range(size):
        for dx in range(size):
            eroded &= padded[dy : dy + height, dx : dx + width].astype(bool)
    return eroded


def _binary_connected_components(mask_bool: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask_bool, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            coords.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        comp = np.zeros_like(mask, dtype=bool)
        ys, xs = zip(*coords)
        comp[np.asarray(ys), np.asarray(xs)] = True
        components.append(comp)
    return components


def _sanitize_patch_rgba(
    patch_rgba: np.ndarray,
    blur_radius: float = 1.5,
) -> np.ndarray:
    patch = np.asarray(patch_rgba, dtype=np.uint8).copy()
    alpha_u8 = patch[..., 3]
    visible = alpha_u8 > 0
    if not np.any(visible):
        return patch

    patch[~visible, :3] = 0

    height, width = visible.shape
    erosion_radius = max(1, int(round(min(height, width) * 0.12)))
    core = _erode_binary_mask(visible, erosion_radius)
    if int(core.sum()) < max(9, int(round(0.15 * visible.sum()))):
        core = _erode_binary_mask(visible, 1)
    if not np.any(core):
        return _decontaminate_patch_rgba(patch, blur_radius=blur_radius)

    rgb = patch[..., :3].astype(np.float32)
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    core_luma = luminance[core]
    dark_thresh = max(8.0, 0.55 * float(np.median(core_luma)))
    peripheral = visible & ~core
    suspicious = peripheral & (alpha_u8 >= 224) & (luminance < dark_thresh)

    if np.any(suspicious):
        ys_vis, xs_vis = np.nonzero(visible)
        center_y = float(ys_vis.mean())
        center_x = float(xs_vis.mean())
        remove = np.zeros_like(visible, dtype=bool)
        for comp in _binary_connected_components(suspicious):
            ys, xs = np.nonzero(comp)
            if ys.size == 0:
                continue
            touches_crop_edge = (
                int(ys.min()) == 0
                or int(xs.min()) == 0
                or int(ys.max()) == height - 1
                or int(xs.max()) == width - 1
            )
            centroid_offset = math.hypot(float(xs.mean() - center_x), float(ys.mean() - center_y)) / max(height, width)
            if touches_crop_edge or centroid_offset > 0.18:
                remove |= comp
        if np.any(remove):
            patch[remove, :3] = 0
            cleaned_visible = visible & ~remove
            if np.any(cleaned_visible):
                rebuilt_alpha = np.array(
                    Image.fromarray((cleaned_visible.astype(np.uint8) * 255)).filter(
                        ImageFilter.GaussianBlur(radius=blur_radius)
                    ),
                    dtype=np.uint8,
                )
                rebuilt_alpha[cleaned_visible] = 255
                patch[..., 3] = rebuilt_alpha
                patch[patch[..., 3] == 0, :3] = 0
            else:
                patch[remove, 3] = 0

    return _decontaminate_patch_rgba(patch, blur_radius=blur_radius)


def _decontaminate_patch_rgba(
    patch_rgba: np.ndarray,
    blur_radius: float = 1.5,
) -> np.ndarray:
    patch_rgba = np.asarray(patch_rgba, dtype=np.uint8).copy()
    alpha_u8 = patch_rgba[..., 3]
    rgb = patch_rgba[..., :3].astype(np.float32) / 255.0
    alpha = alpha_u8.astype(np.float32) / 255.0

    alpha_zero = alpha_u8 == 0
    if np.any(alpha_zero):
        patch_rgba[alpha_zero, :3] = 0

    hard_mask = alpha_u8 == 255
    feather_mask = (alpha_u8 > 0) & (alpha_u8 < 255)
    if not np.any(feather_mask) or not np.any(hard_mask):
        return patch_rgba

    premult_hard = rgb * hard_mask[..., None].astype(np.float32)
    blurred_premult = np.zeros_like(premult_hard)
    for channel_idx in range(3):
        blurred_premult[..., channel_idx] = (
            np.array(
                Image.fromarray(
                    np.round(np.clip(premult_hard[..., channel_idx], 0.0, 1.0) * 255.0).astype(np.uint8)
                ).filter(ImageFilter.GaussianBlur(radius=blur_radius)),
                dtype=np.float32,
            )
            / 255.0
        )

    repaired_rgb = rgb.copy()
    safe_alpha = np.maximum(alpha[..., None], 1.0 / 255.0)
    repaired_rgb[feather_mask] = np.clip(
        blurred_premult[feather_mask] / safe_alpha[feather_mask],
        0.0,
        1.0,
    )
    repaired_rgb[hard_mask] = rgb[hard_mask]
    repaired_rgb[alpha_zero] = 0.0

    repaired = np.concatenate([repaired_rgb, alpha[..., None]], axis=-1)
    return np.round(np.clip(repaired, 0.0, 1.0) * 255.0).astype(np.uint8)


def _erase_original_object(
    frame_np: np.ndarray,
    mask_bool: np.ndarray,
) -> np.ndarray:
    out = frame_np.copy()
    background_rgb = np.median(
        np.concatenate(
            [
                out[0, :, :],
                out[-1, :, :],
                out[:, 0, :],
                out[:, -1, :],
            ],
            axis=0,
        ),
        axis=0,
    ).astype(np.uint8)
    out[mask_bool] = background_rgb
    return out


def _composite_patch_rgba(
    canvas_rgb: np.ndarray,
    patch_rgba: np.ndarray,
    center_xy: tuple[int, int],
) -> np.ndarray:
    out = canvas_rgb.astype(np.float32).copy()
    patch_h, patch_w = patch_rgba.shape[:2]
    x0 = int(round(center_xy[0] - patch_w / 2))
    y0 = int(round(center_xy[1] - patch_h / 2))
    x1 = x0 + patch_w
    y1 = y0 + patch_h

    crop_x0 = max(0, -x0)
    crop_y0 = max(0, -y0)
    crop_x1 = patch_w - max(0, x1 - out.shape[1])
    crop_y1 = patch_h - max(0, y1 - out.shape[0])
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = dst_x0 + (crop_x1 - crop_x0)
    dst_y1 = dst_y0 + (crop_y1 - crop_y0)

    if crop_x0 >= crop_x1 or crop_y0 >= crop_y1:
        return canvas_rgb

    patch_crop = patch_rgba[crop_y0:crop_y1, crop_x0:crop_x1]
    alpha = patch_crop[..., 3:4].astype(np.float32) / 255.0
    out[dst_y0:dst_y1, dst_x0:dst_x1] = (
        out[dst_y0:dst_y1, dst_x0:dst_x1] * (1.0 - alpha)
        + patch_crop[..., :3].astype(np.float32) * alpha
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def _build_color_sequence(
    num: int,
    original_color_name: str | None,
    colors: list[str] | None,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    if colors:
        color_names = colors
    else:
        palette = [name for name in COLOR_NAME_TO_RGB if name != original_color_name]
        rng = np.random.default_rng(seed)
        color_names = [palette[int(i)] for i in rng.integers(0, len(palette), size=num)]
    if len(color_names) == 1 and num > 1:
        color_names = color_names * num
    if len(color_names) != num:
        raise ValueError('colors must have length 1 or num')
    return [(name, COLOR_NAME_TO_RGB[name]) for name in color_names]


def _overlay_mask(image_np: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    overlay = image_np.copy().astype(np.float32)
    overlay[mask_bool] = overlay[mask_bool] * 0.55 + np.array([255, 64, 64], dtype=np.float32) * 0.45
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _write_preview_video(path: Path, before_frames: np.ndarray, after_frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.concatenate([before_frames, after_frames], axis=2)
    imageio.mimsave(
        path,
        stacked.astype(np.uint8),
        fps=10,
        **_PREVIEW_VIDEO_KWARGS,
    )


def _save_debug_artifacts(
    debug_dir: Path,
    before_frames: np.ndarray,
    after_frames: np.ndarray,
    mask_bool: np.ndarray,
    patch_rgba: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    before = before_frames[0]
    after = after_frames[0]
    overlay = _overlay_mask(before, mask_bool)
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    Image.fromarray(before).save(debug_dir / 'before.png')
    Image.fromarray(overlay).save(debug_dir / 'overlay.png')
    Image.fromarray(mask_u8).save(debug_dir / 'mask.png')
    Image.fromarray(patch_rgba).save(debug_dir / 'patch.png')
    Image.fromarray(after).save(debug_dir / 'after.png')
    Image.fromarray(np.concatenate([before, after], axis=1)).save(
        debug_dir / 'before_after.png'
    )
    _write_preview_video(debug_dir / 'preview.mp4', before_frames, after_frames)


def _save_failure_debug_artifacts(
    debug_dir: Path,
    before_frames: np.ndarray,
    prompts: list[str],
    failure_reason: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    before = before_frames[0]
    Image.fromarray(before).save(debug_dir / 'before.png')
    Image.fromarray(before).save(debug_dir / 'after.png')
    Image.fromarray(np.concatenate([before, before], axis=1)).save(
        debug_dir / 'before_after.png'
    )
    (debug_dir / 'failure_reason.txt').write_text(failure_reason, encoding='utf-8')
    (debug_dir / 'prompt_candidates.json').write_text(
        json.dumps({'prompts': prompts}, indent=2),
        encoding='utf-8',
    )
    _write_preview_video(debug_dir / 'preview.mp4', before_frames, before_frames)


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f'not JSON serializable: {type(obj)}')


def _write_metadata_files(rows: list[dict[str, Any]], output_dir: Path, summary: dict[str, Any]) -> None:
    jsonl_path = output_dir / 'episode_metadata.jsonl'
    csv_path = output_dir / 'episode_metadata.csv'
    summary_path = output_dir / 'sidecar_summary.json'

    with jsonl_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, default=_json_default) + '\n')

    csv_rows = []
    for row in rows:
        flat = row.copy()
        flat['inserted_objects'] = json.dumps(row['inserted_objects'], default=_json_default)
        csv_rows.append(flat)

    fieldnames = sorted({key for row in csv_rows for key in row.keys()})
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default),
        encoding='utf-8',
    )


def _segment_episode_reference_frames(
    frames: np.ndarray,
    prompts: list[str],
    confidence_threshold: float,
    backend: SegmenterBackend,
    reference_frames_to_try: int,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    best_failure: dict[str, Any] = {
        'success': False,
        'mask': None,
        'score': None,
        'used_prompt': None,
        'used_prompt_variant': None,
        'reference_frame_idx': None,
    }
    for frame_idx in range(min(reference_frames_to_try, len(frames))):
        segmentation = backend.segment_best(
            frames[frame_idx],
            prompts=prompts,
            confidence_threshold=confidence_threshold,
            debug_dir=debug_dir if frame_idx == 0 else None,
        )
        segmentation['reference_frame_idx'] = frame_idx
        if segmentation['success']:
            return segmentation
        score = segmentation.get('score')
        best_score = best_failure.get('score')
        if score is not None and (best_score is None or float(score) > float(best_score)):
            best_failure = segmentation
    return best_failure


def _retry_segmentation_for_position_sampling(
    *,
    frames: np.ndarray,
    prompts: list[str],
    confidence_threshold: float,
    backend: SegmenterBackend,
    reference_frames_to_try: int,
    required_num: int,
    seed: int,
) -> dict[str, Any] | None:
    best_candidate: dict[str, Any] | None = None
    best_key: tuple[int, int, float] | None = None

    for prompt in prompts:
        segmentation = _segment_episode_reference_frames(
            frames,
            prompts=[prompt],
            confidence_threshold=confidence_threshold,
            backend=backend,
            reference_frames_to_try=reference_frames_to_try,
            debug_dir=None,
        )
        if not segmentation['success']:
            continue
        mask_bool = segmentation['mask']
        bbox = _mask_to_bbox(mask_bool)
        patch_rgba, _, crop_mask = _extract_feathered_patch(
            frames[segmentation['reference_frame_idx']],
            mask_bool,
        )
        centers = sample_non_overlapping_centers(
            image_shape_hw=frames[0].shape[:2],
            patch_shape_hw=patch_rgba.shape[:2],
            num=required_num,
            original_bbox_xyxy=bbox,
            seed=seed,
        )
        area = int((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1))
        score = float(segmentation.get('score') or 0.0)
        key = (len(centers), -area, score)
        candidate = {
            'segmentation': segmentation,
            'patch_rgba': patch_rgba,
            'crop_mask': crop_mask,
            'centers': centers,
        }
        if len(centers) >= required_num:
            return candidate
        if best_key is None or key > best_key:
            best_key = key
            best_candidate = candidate

    return best_candidate


def _build_episode_row(
    *,
    config_like: Any,
    spec,
    episode_idx: int,
    core_desc: dict[str, Any],
    used_variation: bool,
    segmentation: dict[str, Any],
    inserted_objects: list[dict[str, Any]],
    failure_reason: str | None,
    debug_dir: Path | None,
    output_h5: Path | None,
) -> dict[str, Any]:
    augmentation_success = bool(inserted_objects)
    summary_text = (
        'no distractor inserted'
        if not inserted_objects
        else 'added '
        + f"{len(inserted_objects)} distractor {spec.shape_hint}s: "
        + ', '.join(
            f"{item['color']} at ({item['position_norm_xy'][0]:.2f}, {item['position_norm_xy'][1]:.2f})"
            for item in inserted_objects
        )
    )
    return {
        'env_id': config_like.env_id,
        'episode_idx': episode_idx,
        'input_h5': str(config_like.input_h5),
        'output_h5': str(output_h5) if output_h5 else None,
        'augmentation_success': augmentation_success,
        'core_object_prompt_base': spec.object_name,
        'core_object_prompt_resolved': segmentation.get('used_prompt'),
        'core_object_source': 'variation' if used_variation else 'fallback_prompt',
        'num_requested': config_like.num,
        'num_inserted': len(inserted_objects),
        'inserted_objects': inserted_objects,
        'perturbation_summary_text': summary_text,
        'used_variation_metadata': used_variation,
        'used_prompt': segmentation.get('used_prompt'),
        'used_prompt_variant': segmentation.get('used_prompt_variant'),
        'segmentation_score': segmentation.get('score'),
        'segmentation_reference_frame_idx': segmentation.get('reference_frame_idx'),
        'failure_reason': failure_reason,
        'debug_dir': str(debug_dir) if debug_dir else None,
        **core_desc,
    }


def augment_hdf5_dataset(
    config: ObjectCopyConfig,
    backend: SegmenterBackend,
) -> dict[str, Any]:
    if config.num < 1:
        raise ValueError('num must be >= 1')
    if config.debug_only and config.output_h5 is None:
        output_dir = config.input_h5.parent / f'{config.input_h5.stem}__debug_aug'
    else:
        if config.output_h5 is None:
            raise ValueError('output_h5 must be set unless debug_only is enabled')
        output_dir = config.output_h5.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        if not config.debug_only:
            shutil.copy2(config.input_h5, config.output_h5)

    debug_idx = 0 if config.debug_only and config.debug_episode_idx is None else config.debug_episode_idx

    rows: list[dict[str, Any]] = []
    spec = get_object_spec(config.env_id, config.object_name)

    with h5py.File(config.input_h5, 'r') as src:
        dst = h5py.File(config.output_h5, 'r+') if (config.output_h5 is not None and not config.debug_only) else None
        try:
            ep_lens = src['ep_len'][:]
            ep_offsets = src['ep_offset'][:]
            pixel_key = _resolve_pixel_key(src)
            pixel_ds_src = src[pixel_key]
            pixel_ds_dst = dst[pixel_key] if dst is not None else None
            if debug_idx is not None and config.debug_only:
                if debug_idx < 0 or debug_idx >= len(ep_lens):
                    raise IndexError(
                        f'debug_episode_idx {debug_idx} out of range for {len(ep_lens)} episodes'
                    )
                episode_indices = [int(debug_idx)]
            else:
                episode_indices = _resolve_episode_indices(
                    num_episodes=len(ep_lens),
                    episode_start=config.episode_start,
                    episode_end=config.episode_end,
                )

            for episode_idx in episode_indices:
                ep_len = ep_lens[episode_idx]
                ep_offset = ep_offsets[episode_idx]
                start = int(ep_offset)
                end = int(ep_offset + ep_len)
                frames = pixel_ds_src[start:end]
                used_variation = False
                failure_reason = None
                inserted_objects: list[dict[str, Any]] = []
                debug_dir = None
                if debug_idx == episode_idx:
                    debug_dir = output_dir / 'debug' / f'episode_{episode_idx}'

                episode_data = {
                    key: src[key][start:end]
                    for key in src.keys()
                    if key.startswith('variation.')
                }
                variation_rgb = _extract_variation_rgb(
                    episode_data,
                    spec.variation_color_key,
                    color_index=spec.variation_color_index,
                )
                used_variation = variation_rgb is not None
                variation_color_name = describe_color_name(variation_rgb)
                prompts = build_prompt_candidates(
                    spec,
                    color_name=variation_color_name,
                    shape_hint=spec.shape_hint,
                )

                segmentation = _segment_episode_reference_frames(
                    frames,
                    prompts=prompts,
                    confidence_threshold=config.confidence_threshold,
                    backend=backend,
                    reference_frames_to_try=config.reference_frames_to_try,
                    debug_dir=debug_dir,
                )

                if not segmentation['success']:
                    failure_reason = 'segmentation_failed'
                    core_rgb = variation_rgb
                    core_desc = describe_core_object(spec, core_rgb, spec.shape_hint)
                    if debug_dir is not None:
                        _save_failure_debug_artifacts(
                            debug_dir,
                            before_frames=frames,
                            prompts=prompts,
                            failure_reason=failure_reason,
                        )
                    row = _build_episode_row(
                        config_like=config,
                        spec=spec,
                        episode_idx=episode_idx,
                        core_desc=core_desc,
                        used_variation=used_variation,
                        segmentation=segmentation,
                        inserted_objects=[],
                        failure_reason=failure_reason,
                        debug_dir=debug_dir,
                        output_h5=config.output_h5,
                    )
                    rows.append(row)
                    continue

                mask_bool = segmentation['mask']
                original_bbox = _mask_to_bbox(mask_bool)
                patch_rgba, _, crop_mask = _extract_feathered_patch(
                    frames[segmentation['reference_frame_idx']],
                    mask_bool,
                )
                patch_shape_hw = patch_rgba.shape[:2]
                centers = sample_non_overlapping_centers(
                    image_shape_hw=frames[0].shape[:2],
                    patch_shape_hw=patch_shape_hw,
                    num=config.num,
                    original_bbox_xyxy=original_bbox,
                    seed=config.seed + episode_idx,
                )
                if len(centers) < config.num and len(prompts) > 1:
                    recovered = _retry_segmentation_for_position_sampling(
                        frames=frames,
                        prompts=prompts,
                        confidence_threshold=config.confidence_threshold,
                        backend=backend,
                        reference_frames_to_try=config.reference_frames_to_try,
                        required_num=config.num,
                        seed=config.seed + episode_idx,
                    )
                    if recovered is not None and len(recovered['centers']) > len(centers):
                        segmentation = recovered['segmentation']
                        mask_bool = segmentation['mask']
                        original_bbox = _mask_to_bbox(mask_bool)
                        patch_rgba = recovered['patch_rgba']
                        crop_mask = recovered['crop_mask']
                        patch_shape_hw = patch_rgba.shape[:2]
                        centers = recovered['centers']

                core_rgb = variation_rgb if variation_rgb is not None else _infer_mask_rgb(
                    frames[segmentation['reference_frame_idx']],
                    mask_bool,
                )
                core_desc = describe_core_object(spec, core_rgb, spec.shape_hint)
                color_sequence = _build_color_sequence(
                    config.num,
                    core_desc['core_object_color'],
                    config.colors,
                    seed=config.seed + episode_idx,
                )
                if len(centers) == 0:
                    failure_reason = 'position_sampling_failed'
                elif len(centers) < config.num:
                    failure_reason = 'partial_position_sampling'

                augmented_frames = frames.copy()
                if config.drop_original:
                    augmented_frames = np.stack(
                        [_erase_original_object(frame, mask_bool) for frame in augmented_frames],
                        axis=0,
                    )

                if centers:
                    for instance_id, ((color_name, color_rgb), center) in enumerate(
                        zip(color_sequence[: len(centers)], centers)
                    ):
                        recolored_rgb = _recolor_rgb_with_mask(
                            patch_rgba[..., :3],
                            crop_mask,
                            color_rgb,
                        )
                        recolored_patch = patch_rgba.copy()
                        recolored_patch[..., :3] = recolored_rgb
                        for frame_idx in range(len(augmented_frames)):
                            augmented_frames[frame_idx] = _composite_patch_rgba(
                                augmented_frames[frame_idx],
                                recolored_patch,
                                center,
                            )
                        bbox = _bbox_from_center(center, patch_shape_hw)
                        inserted_objects.append(
                            {
                                'instance_id': instance_id,
                                'object_name': spec.object_name,
                                'color': color_name,
                                'position_norm_xy': [
                                    center[0] / frames[0].shape[1],
                                    center[1] / frames[0].shape[0],
                                ],
                                'bbox_norm_xyxy': [
                                    bbox[0] / frames[0].shape[1],
                                    bbox[1] / frames[0].shape[0],
                                    bbox[2] / frames[0].shape[1],
                                    bbox[3] / frames[0].shape[0],
                                ],
                                'mask_score': float(segmentation['score']),
                                'copy_mode': 'feathered_mask',
                            }
                        )

                if pixel_ds_dst is not None:
                    pixel_ds_dst[start:end] = augmented_frames

                if debug_dir is not None:
                    _save_debug_artifacts(
                        debug_dir,
                        before_frames=frames,
                        after_frames=augmented_frames,
                        mask_bool=mask_bool,
                        patch_rgba=patch_rgba,
                    )

                rows.append(
                    _build_episode_row(
                        config_like=config,
                        spec=spec,
                        episode_idx=episode_idx,
                        core_desc=core_desc,
                        used_variation=used_variation,
                        segmentation=segmentation,
                        inserted_objects=inserted_objects,
                        failure_reason=failure_reason,
                        debug_dir=debug_dir,
                        output_h5=config.output_h5,
                    )
                )
        finally:
            if dst is not None:
                dst.close()

    summary = {
        'env_id': config.env_id,
        'input_h5': str(config.input_h5),
        'output_h5': str(config.output_h5) if config.output_h5 else None,
        'object_name': config.object_name,
        'num_requested': config.num,
        'episodes_total': len(rows),
        'episodes_succeeded': int(sum(row['augmentation_success'] for row in rows)),
        'config': asdict(config),
    }
    _write_metadata_files(rows, output_dir, summary)
    return summary


def augment_hdf5_dataset_fanout(
    config: FanoutObjectCopyConfig,
    backend: SegmenterBackend,
) -> dict[str, Any]:
    if not config.num_values:
        raise ValueError('num_values must be non-empty')
    normalized_num_values = sorted(set(int(v) for v in config.num_values))
    if normalized_num_values[0] < 1:
        raise ValueError('all num_values must be >= 1')

    spec = get_object_spec(config.env_id, config.object_name)
    config.output_root.mkdir(parents=True, exist_ok=True)

    bundle_by_num: dict[int, dict[str, Any]] = {}
    for num in normalized_num_values:
        output_dir = config.output_root / f'd{num}'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_h5 = output_dir / 'augmented.h5'
        shutil.copy2(config.input_h5, output_h5)
        bundle_by_num[num] = {
            'output_dir': output_dir,
            'output_h5': output_h5,
            'rows': [],
        }

    with h5py.File(config.input_h5, 'r') as src:
        dst_by_num = {
            num: h5py.File(bundle['output_h5'], 'r+')
            for num, bundle in bundle_by_num.items()
        }
        try:
            ep_lens = src['ep_len'][:]
            ep_offsets = src['ep_offset'][:]
            pixel_key = _resolve_pixel_key(src)
            pixel_ds_src = src[pixel_key]
            pixel_ds_dst_by_num = {
                num: dst[pixel_key]
                for num, dst in dst_by_num.items()
            }

            episode_indices = _resolve_episode_indices(
                num_episodes=len(ep_lens),
                episode_start=config.episode_start,
                episode_end=config.episode_end,
            )
            for episode_idx in episode_indices:
                ep_len = ep_lens[episode_idx]
                ep_offset = ep_offsets[episode_idx]
                start = int(ep_offset)
                end = int(ep_offset + ep_len)
                frames = pixel_ds_src[start:end]
                episode_data = {
                    key: src[key][start:end]
                    for key in src.keys()
                    if key.startswith('variation.')
                }
                variation_rgb = _extract_variation_rgb(
                    episode_data,
                    spec.variation_color_key,
                    color_index=spec.variation_color_index,
                )
                used_variation = variation_rgb is not None
                prompts = build_prompt_candidates(
                    spec,
                    color_name=describe_color_name(variation_rgb),
                    shape_hint=spec.shape_hint,
                )
                segmentation = _segment_episode_reference_frames(
                    frames,
                    prompts=prompts,
                    confidence_threshold=config.confidence_threshold,
                    backend=backend,
                    reference_frames_to_try=config.reference_frames_to_try,
                    debug_dir=None,
                )

                if not segmentation['success']:
                    core_desc = describe_core_object(spec, variation_rgb, spec.shape_hint)
                    for num, bundle in bundle_by_num.items():
                        cfg_like = type('Cfg', (), {
                            'env_id': config.env_id,
                            'input_h5': config.input_h5,
                            'num': num,
                        })()
                        bundle['rows'].append(
                            _build_episode_row(
                                config_like=cfg_like,
                                spec=spec,
                                episode_idx=episode_idx,
                                core_desc=core_desc,
                                used_variation=used_variation,
                                segmentation=segmentation,
                                inserted_objects=[],
                                failure_reason='segmentation_failed',
                                debug_dir=None,
                                output_h5=bundle['output_h5'],
                            )
                        )
                    continue

                max_num = normalized_num_values[-1]
                mask_bool = segmentation['mask']
                original_bbox = _mask_to_bbox(mask_bool)
                patch_rgba, _, crop_mask = _extract_feathered_patch(
                    frames[segmentation['reference_frame_idx']],
                    mask_bool,
                )
                patch_shape_hw = patch_rgba.shape[:2]
                centers = sample_non_overlapping_centers(
                    image_shape_hw=frames[0].shape[:2],
                    patch_shape_hw=patch_shape_hw,
                    num=max_num,
                    original_bbox_xyxy=original_bbox,
                    seed=config.seed + episode_idx,
                )
                if len(centers) < max_num and len(prompts) > 1:
                    recovered = _retry_segmentation_for_position_sampling(
                        frames=frames,
                        prompts=prompts,
                        confidence_threshold=config.confidence_threshold,
                        backend=backend,
                        reference_frames_to_try=config.reference_frames_to_try,
                        required_num=max_num,
                        seed=config.seed + episode_idx,
                    )
                    if recovered is not None and len(recovered['centers']) > len(centers):
                        segmentation = recovered['segmentation']
                        mask_bool = segmentation['mask']
                        original_bbox = _mask_to_bbox(mask_bool)
                        patch_rgba = recovered['patch_rgba']
                        crop_mask = recovered['crop_mask']
                        patch_shape_hw = patch_rgba.shape[:2]
                        centers = recovered['centers']

                core_rgb = variation_rgb if variation_rgb is not None else _infer_mask_rgb(
                    frames[segmentation['reference_frame_idx']],
                    mask_bool,
                )
                core_desc = describe_core_object(spec, core_rgb, spec.shape_hint)
                color_sequence = _build_color_sequence(
                    max_num,
                    core_desc['core_object_color'],
                    config.colors,
                    seed=config.seed + episode_idx,
                )

                for num, bundle in bundle_by_num.items():
                    selected_centers = centers[: min(num, len(centers))]
                    inserted_objects: list[dict[str, Any]] = []
                    failure_reason = None
                    if len(selected_centers) == 0:
                        failure_reason = 'position_sampling_failed'
                    elif len(selected_centers) < num:
                        failure_reason = 'partial_position_sampling'

                    augmented_frames = frames.copy()
                    if config.drop_original:
                        augmented_frames = np.stack(
                            [_erase_original_object(frame, mask_bool) for frame in augmented_frames],
                            axis=0,
                        )

                    for instance_id, ((color_name, color_rgb), center) in enumerate(
                        zip(color_sequence[: len(selected_centers)], selected_centers)
                    ):
                        recolored_rgb = _recolor_rgb_with_mask(
                            patch_rgba[..., :3],
                            crop_mask,
                            color_rgb,
                        )
                        recolored_patch = patch_rgba.copy()
                        recolored_patch[..., :3] = recolored_rgb
                        for frame_idx in range(len(augmented_frames)):
                            augmented_frames[frame_idx] = _composite_patch_rgba(
                                augmented_frames[frame_idx],
                                recolored_patch,
                                center,
                            )
                        bbox = _bbox_from_center(center, patch_shape_hw)
                        inserted_objects.append(
                            {
                                'instance_id': instance_id,
                                'object_name': spec.object_name,
                                'color': color_name,
                                'position_norm_xy': [
                                    center[0] / frames[0].shape[1],
                                    center[1] / frames[0].shape[0],
                                ],
                                'bbox_norm_xyxy': [
                                    bbox[0] / frames[0].shape[1],
                                    bbox[1] / frames[0].shape[0],
                                    bbox[2] / frames[0].shape[1],
                                    bbox[3] / frames[0].shape[0],
                                ],
                                'mask_score': float(segmentation['score']),
                                'copy_mode': 'feathered_mask',
                            }
                        )

                    pixel_ds_dst_by_num[num][start:end] = augmented_frames
                    cfg_like = type('Cfg', (), {
                        'env_id': config.env_id,
                        'input_h5': config.input_h5,
                        'num': num,
                    })()
                    bundle['rows'].append(
                        _build_episode_row(
                            config_like=cfg_like,
                            spec=spec,
                            episode_idx=episode_idx,
                            core_desc=core_desc,
                            used_variation=used_variation,
                            segmentation=segmentation,
                            inserted_objects=inserted_objects,
                            failure_reason=failure_reason,
                            debug_dir=None,
                            output_h5=bundle['output_h5'],
                        )
                    )
        finally:
            for dst in dst_by_num.values():
                dst.close()

    summary_by_num: dict[str, Any] = {}
    for num, bundle in bundle_by_num.items():
        summary = {
            'env_id': config.env_id,
            'input_h5': str(config.input_h5),
            'output_h5': str(bundle['output_h5']),
            'object_name': config.object_name,
            'num_requested': num,
            'episodes_total': len(bundle['rows']),
            'episodes_succeeded': int(sum(row['augmentation_success'] for row in bundle['rows'])),
            'config': {
                'input_h5': str(config.input_h5),
                'output_root': str(config.output_root),
                'env_id': config.env_id,
                'object_name': config.object_name,
                'num_values': normalized_num_values,
                'seed': config.seed,
                'colors': config.colors,
                'drop_original': config.drop_original,
                'confidence_threshold': config.confidence_threshold,
                'reference_frames_to_try': config.reference_frames_to_try,
                'episode_start': config.episode_start,
                'episode_end': config.episode_end,
            },
        }
        _write_metadata_files(bundle['rows'], bundle['output_dir'], summary)
        summary_by_num[f'd{num}'] = summary

    return {
        'env_id': config.env_id,
        'input_h5': str(config.input_h5),
        'output_root': str(config.output_root),
        'object_name': config.object_name,
        'num_values': normalized_num_values,
        'summaries': summary_by_num,
    }


def repair_hdf5_dataset_fanout_failures(
    config: FanoutObjectCopyConfig,
    backend: SegmenterBackend,
) -> dict[str, Any]:
    if not config.num_values:
        raise ValueError('num_values must be non-empty')
    normalized_num_values = sorted(set(int(v) for v in config.num_values))
    spec = get_object_spec(config.env_id, config.object_name)

    bundle_by_num: dict[int, dict[str, Any]] = {}
    for num in normalized_num_values:
        output_dir = config.output_root / f'd{num}'
        output_h5 = output_dir / 'augmented.h5'
        metadata_jsonl = output_dir / 'episode_metadata.jsonl'
        if not output_h5.exists():
            raise FileNotFoundError(f'missing augmented.h5 for d{num}: {output_h5}')
        if not metadata_jsonl.exists():
            raise FileNotFoundError(f'missing episode_metadata.jsonl for d{num}: {metadata_jsonl}')
        rows = [json.loads(line) for line in metadata_jsonl.open(encoding='utf-8')]
        bundle_by_num[num] = {
            'output_dir': output_dir,
            'output_h5': output_h5,
            'rows': rows,
        }

    num_episodes = len(next(iter(bundle_by_num.values()))['rows'])
    episode_indices = set(
        _resolve_episode_indices(
            num_episodes=num_episodes,
            episode_start=config.episode_start,
            episode_end=config.episode_end,
        )
    )
    failed_indices = sorted(
        episode_idx
        for episode_idx in episode_indices
        if any(
            not bundle_by_num[num]['rows'][episode_idx].get('augmentation_success', False)
            for num in normalized_num_values
        )
    )

    repaired_counts = {f'd{num}': 0 for num in normalized_num_values}

    with h5py.File(config.input_h5, 'r') as src:
        dst_by_num = {
            num: h5py.File(bundle['output_h5'], 'r+')
            for num, bundle in bundle_by_num.items()
        }
        try:
            ep_lens = src['ep_len'][:]
            ep_offsets = src['ep_offset'][:]
            pixel_key = _resolve_pixel_key(src)
            pixel_ds_src = src[pixel_key]
            pixel_ds_dst_by_num = {
                num: dst[pixel_key]
                for num, dst in dst_by_num.items()
            }

            for episode_idx in failed_indices:
                ep_len = ep_lens[episode_idx]
                ep_offset = ep_offsets[episode_idx]
                start = int(ep_offset)
                end = int(ep_offset + ep_len)
                frames = pixel_ds_src[start:end]
                episode_data = {
                    key: src[key][start:end]
                    for key in src.keys()
                    if key.startswith('variation.')
                }
                variation_rgb = _extract_variation_rgb(
                    episode_data,
                    spec.variation_color_key,
                    color_index=spec.variation_color_index,
                )
                used_variation = variation_rgb is not None
                prompts = build_prompt_candidates(
                    spec,
                    color_name=describe_color_name(variation_rgb),
                    shape_hint=spec.shape_hint,
                )
                segmentation = _segment_episode_reference_frames(
                    frames,
                    prompts=prompts,
                    confidence_threshold=config.confidence_threshold,
                    backend=backend,
                    reference_frames_to_try=config.reference_frames_to_try,
                    debug_dir=None,
                )

                core_desc = describe_core_object(spec, variation_rgb, spec.shape_hint)
                if segmentation['success']:
                    mask_bool = segmentation['mask']
                    original_bbox = _mask_to_bbox(mask_bool)
                    patch_rgba, _, crop_mask = _extract_feathered_patch(
                        frames[segmentation['reference_frame_idx']],
                        mask_bool,
                    )
                    patch_shape_hw = patch_rgba.shape[:2]
                    max_num = normalized_num_values[-1]
                    centers = sample_non_overlapping_centers(
                        image_shape_hw=frames[0].shape[:2],
                        patch_shape_hw=patch_shape_hw,
                        num=max_num,
                        original_bbox_xyxy=original_bbox,
                        seed=config.seed + episode_idx,
                    )
                    if len(centers) < max_num and len(prompts) > 1:
                        recovered = _retry_segmentation_for_position_sampling(
                            frames=frames,
                            prompts=prompts,
                            confidence_threshold=config.confidence_threshold,
                            backend=backend,
                            reference_frames_to_try=config.reference_frames_to_try,
                            required_num=max_num,
                            seed=config.seed + episode_idx,
                        )
                        if recovered is not None and len(recovered['centers']) > len(centers):
                            segmentation = recovered['segmentation']
                            mask_bool = segmentation['mask']
                            original_bbox = _mask_to_bbox(mask_bool)
                            patch_rgba = recovered['patch_rgba']
                            crop_mask = recovered['crop_mask']
                            patch_shape_hw = patch_rgba.shape[:2]
                            centers = recovered['centers']
                    core_rgb = variation_rgb if variation_rgb is not None else _infer_mask_rgb(
                        frames[segmentation['reference_frame_idx']],
                        mask_bool,
                    )
                    core_desc = describe_core_object(spec, core_rgb, spec.shape_hint)
                    color_sequence = _build_color_sequence(
                        max_num,
                        core_desc['core_object_color'],
                        config.colors,
                        seed=config.seed + episode_idx,
                    )
                else:
                    centers = []
                    color_sequence = []

                for num, bundle in bundle_by_num.items():
                    failure_reason = None
                    inserted_objects: list[dict[str, Any]] = []

                    if segmentation['success']:
                        selected_centers = centers[: min(num, len(centers))]
                        augmented_frames = frames.copy()
                        if config.drop_original:
                            augmented_frames = np.stack(
                                [_erase_original_object(frame, mask_bool) for frame in augmented_frames],
                                axis=0,
                            )
                        for instance_id, ((color_name, color_rgb), center) in enumerate(
                            zip(color_sequence[: len(selected_centers)], selected_centers)
                        ):
                            recolored_rgb = _recolor_rgb_with_mask(
                                patch_rgba[..., :3],
                                crop_mask,
                                color_rgb,
                            )
                            recolored_patch = patch_rgba.copy()
                            recolored_patch[..., :3] = recolored_rgb
                            for frame_idx in range(len(augmented_frames)):
                                augmented_frames[frame_idx] = _composite_patch_rgba(
                                    augmented_frames[frame_idx],
                                    recolored_patch,
                                    center,
                                )
                            bbox = _bbox_from_center(center, patch_shape_hw)
                            inserted_objects.append(
                                {
                                    'instance_id': instance_id,
                                    'object_name': spec.object_name,
                                    'color': color_name,
                                    'position_norm_xy': [
                                        center[0] / frames[0].shape[1],
                                        center[1] / frames[0].shape[0],
                                    ],
                                    'bbox_norm_xyxy': [
                                        bbox[0] / frames[0].shape[1],
                                        bbox[1] / frames[0].shape[0],
                                        bbox[2] / frames[0].shape[1],
                                        bbox[3] / frames[0].shape[0],
                                    ],
                                    'mask_score': float(segmentation['score']),
                                    'copy_mode': 'feathered_mask',
                                }
                            )
                        if len(selected_centers) == 0:
                            failure_reason = 'position_sampling_failed'
                        elif len(selected_centers) < num:
                            failure_reason = 'partial_position_sampling'
                        pixel_ds_dst_by_num[num][start:end] = augmented_frames
                    else:
                        failure_reason = 'segmentation_failed'

                    cfg_like = type('Cfg', (), {
                        'env_id': config.env_id,
                        'input_h5': config.input_h5,
                        'num': num,
                    })()
                    row = _build_episode_row(
                        config_like=cfg_like,
                        spec=spec,
                        episode_idx=episode_idx,
                        core_desc=core_desc,
                        used_variation=used_variation,
                        segmentation=segmentation,
                        inserted_objects=inserted_objects,
                        failure_reason=failure_reason,
                        debug_dir=None,
                        output_h5=bundle['output_h5'],
                    )
                    bundle['rows'][episode_idx] = row
                    if row['augmentation_success']:
                        repaired_counts[f'd{num}'] += 1
        finally:
            for dst in dst_by_num.values():
                dst.close()

    summary_by_num: dict[str, Any] = {}
    for num, bundle in bundle_by_num.items():
        summary = {
            'env_id': config.env_id,
            'input_h5': str(config.input_h5),
            'output_h5': str(bundle['output_h5']),
            'object_name': config.object_name,
            'num_requested': num,
            'episodes_total': len(bundle['rows']),
            'episodes_succeeded': int(sum(row['augmentation_success'] for row in bundle['rows'])),
            'config': {
                'input_h5': str(config.input_h5),
                'output_root': str(config.output_root),
                'env_id': config.env_id,
                'object_name': config.object_name,
                'num_values': normalized_num_values,
                'seed': config.seed,
                'colors': config.colors,
                'drop_original': config.drop_original,
                'confidence_threshold': config.confidence_threshold,
                'reference_frames_to_try': config.reference_frames_to_try,
                'episode_start': config.episode_start,
                'episode_end': config.episode_end,
                'repair_mode': True,
            },
        }
        _write_metadata_files(bundle['rows'], bundle['output_dir'], summary)
        summary_by_num[f'd{num}'] = summary

    return {
        'env_id': config.env_id,
        'input_h5': str(config.input_h5),
        'output_root': str(config.output_root),
        'object_name': config.object_name,
        'num_values': normalized_num_values,
        'failed_indices_retried': failed_indices,
        'repaired_counts': repaired_counts,
        'summaries': summary_by_num,
    }
