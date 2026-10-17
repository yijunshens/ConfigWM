from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RePromptMergeSummary:
    keyframe_indices: list[int]
    replaced_frame_indices: list[int]


def select_keyframe_reprompt_starts(
    suspicious_frame_indices: list[int],
    *,
    min_gap: int = 8,
) -> list[int]:
    if not suspicious_frame_indices:
        return []
    starts: list[int] = []
    last = None
    for idx in sorted(int(v) for v in suspicious_frame_indices):
        if last is None or idx - last >= min_gap:
            starts.append(idx)
            last = idx
    return starts


def merge_reprompt_masks(
    base_masks: dict[int, np.ndarray],
    reprompt_runs: list[tuple[int, dict[int, np.ndarray]]],
    *,
    min_area_gain_ratio: float = 1.15,
) -> tuple[dict[int, np.ndarray], RePromptMergeSummary]:
    merged = {int(k): np.asarray(v).astype(bool) for k, v in base_masks.items()}
    replaced_frame_indices: list[int] = []
    keyframes = [int(k) for k, _ in reprompt_runs]

    for keyframe_idx, masks in reprompt_runs:
        for frame_idx, new_mask in masks.items():
            if frame_idx < keyframe_idx:
                continue
            new_mask = np.asarray(new_mask).astype(bool)
            if not new_mask.any():
                continue
            old_mask = merged.get(int(frame_idx))
            if old_mask is None or not np.asarray(old_mask).any():
                merged[int(frame_idx)] = new_mask
                replaced_frame_indices.append(int(frame_idx))
                continue
            old_area = float(np.asarray(old_mask).sum())
            new_area = float(new_mask.sum())
            if old_area <= 0 or new_area >= old_area * min_area_gain_ratio:
                merged[int(frame_idx)] = new_mask
                replaced_frame_indices.append(int(frame_idx))

    return merged, RePromptMergeSummary(
        keyframe_indices=keyframes,
        replaced_frame_indices=sorted(set(replaced_frame_indices)),
    )
