from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalMaskRepairSummary:
    repaired_frame_indices: list[int]
    suspicious_frame_indices: list[int]


def _mask_centroid(mask_bool: np.ndarray) -> np.ndarray | None:
    if mask_bool is None or not np.asarray(mask_bool).any():
        return None
    ys, xs = np.where(np.asarray(mask_bool).astype(bool))
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def _shift_mask(mask_bool: np.ndarray, dx: int, dy: int) -> np.ndarray:
    mask = np.asarray(mask_bool).astype(bool)
    height, width = mask.shape
    out = np.zeros_like(mask)

    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(width, width - dx) if dx >= 0 else width
    src_y1 = min(height, height - dy) if dy >= 0 else height
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return out
    out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return out


def repair_video_frame_masks(
    frame_masks: dict[int, np.ndarray],
    *,
    area_drop_ratio: float = 0.5,
    min_reference_area: int = 256,
    history_size: int = 8,
) -> tuple[dict[int, np.ndarray], TemporalMaskRepairSummary]:
    repaired: dict[int, np.ndarray] = {}
    repaired_frame_indices: list[int] = []
    suspicious_frame_indices: list[int] = []

    accepted_areas: list[float] = []
    last_good_mask: np.ndarray | None = None
    last_good_centroid: np.ndarray | None = None

    for frame_idx in sorted(frame_masks):
        current = np.asarray(frame_masks[frame_idx]).astype(bool)
        if not current.any():
            repaired[frame_idx] = current
            continue

        current_area = float(current.sum())
        current_centroid = _mask_centroid(current)
        if accepted_areas:
            reference_area = max(float(np.median(accepted_areas[-history_size:])), float(min_reference_area))
        else:
            reference_area = max(current_area, float(min_reference_area))

        suspicious = current_area < reference_area * area_drop_ratio
        if suspicious:
            suspicious_frame_indices.append(int(frame_idx))

        if suspicious and last_good_mask is not None and last_good_centroid is not None:
            if current_centroid is not None:
                delta = current_centroid - last_good_centroid
                dx = int(round(float(delta[0])))
                dy = int(round(float(delta[1])))
            else:
                dx = 0
                dy = 0
            shifted = _shift_mask(last_good_mask, dx=dx, dy=dy)
            candidate = np.logical_or(shifted, current)
            repaired[frame_idx] = candidate
            repaired_frame_indices.append(int(frame_idx))
            accepted_areas.append(float(candidate.sum()))
            last_good_mask = candidate
            last_good_centroid = _mask_centroid(candidate)
        else:
            repaired[frame_idx] = current
            accepted_areas.append(current_area)
            last_good_mask = current
            last_good_centroid = current_centroid

    return repaired, TemporalMaskRepairSummary(
        repaired_frame_indices=repaired_frame_indices,
        suspicious_frame_indices=suspicious_frame_indices,
    )
