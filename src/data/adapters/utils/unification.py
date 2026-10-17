from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import warnings

from PIL import Image
import torch

from ...schema import (
    ImageBoxesSample,
    ImageMasksSample,
    ImageTextSample,
    VideoMasksSample,
    Sample,
)


def select_field(
    item: Mapping[str, Any], preferred: str, *fallbacks: str
) -> str | None:
    for key in (preferred, *fallbacks):
        value = item.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
    return None


def _preview_invalid_value(value: Any, *, limit: int = 160) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _warn_invalid_text_only_sample(
    reason: str,
    *,
    targets: list[Any],
    meta: dict[str, Any],
    query: str | None,
    text_input: Any,
) -> None:
    warnings.warn(
        "Skipping invalid text-only sample: "
        f"{reason}; "
        f"text_input={_preview_invalid_value(text_input)}, "
        f"targets={_preview_invalid_value(targets)}, "
        f"query={_preview_invalid_value(query)}, "
        f"meta={_preview_invalid_value(meta)}",
        UserWarning,
        stacklevel=2,
    )


def make_text_only_sample(
    *,
    targets: list[Any],
    meta: dict[str, Any],
    query: str | None = None,
    text_input: Any,
) -> Sample | None:
    """Create a canonical text-only sample.

    Text-only samples use an empty 1-D long tensor for ``label_map`` and rely on
    ``text_input`` to signal that vision validation should be skipped. Invalid
    records emit a warning and return ``None`` so iterable datasets can skip
    them without interrupting training.
    """
    if not isinstance(text_input, str) or not text_input.strip():
        _warn_invalid_text_only_sample(
            "text-only samples require a non-empty text_input",
            targets=targets,
            meta=meta,
            query=query,
            text_input=text_input,
        )
        return None
    if not targets:
        _warn_invalid_text_only_sample(
            "text-only samples require at least one target",
            targets=targets,
            meta=meta,
            query=query,
            text_input=text_input,
        )
        return None

    normalized_targets: list[str] = []
    for target in targets:
        if not isinstance(target, str) or not target.strip():
            _warn_invalid_text_only_sample(
                "text-only samples require non-empty target strings",
                targets=targets,
                meta=meta,
                query=query,
                text_input=text_input,
            )
            return None
        normalized_targets.append(target)

    return Sample(
        frames=[],
        label_map=torch.empty(0, dtype=torch.long),
        targets=normalized_targets,
        meta=meta,
        query=query,
        text_input=text_input,
    )


def sample_frames(
    frames: list[Image.Image],
    label_map: torch.Tensor,
    num_frames: int,
) -> tuple[list[Image.Image], torch.Tensor]:
    """Uniformly sample or duplicate frames to match num_frames.

    Args:
        frames: List of PIL images.
        label_map: Tensor of shape (T, H, W).
        num_frames: Target number of frames.

    Returns:
        (sampled_frames, sampled_label_map) with exactly num_frames frames.
    """
    n = len(frames)
    if n == num_frames:
        return frames, label_map

    # Uniform sampling indices
    indices = torch.linspace(0, n - 1, num_frames).long()
    sampled_frames = [frames[i] for i in indices]
    sampled_label_map = label_map[indices]
    return sampled_frames, sampled_label_map


def from_text(
    sample: ImageTextSample,
    num_frames: int,
) -> Sample:
    """Convert an ``ImageTextSample`` into a ``Sample``.

    The image is duplicated to num_frames copies.
    Caption-style samples use a minimal ``(num_frames, 1, 1)`` label map filled
    with ones because the training path only needs frame tensors plus text.
    """
    label_map = torch.ones(num_frames, 1, 1, dtype=torch.long)

    return Sample(
        frames=[sample.image] * num_frames,
        label_map=label_map,
        targets=[sample.text],
        query=sample.query,
        meta=sample.meta,
    )


def from_boxes(
    sample: ImageBoxesSample,
    num_frames: int,
    overlap_labeling: str = "single",
) -> Sample:
    """Convert an ``ImageBoxesSample`` into a ``Sample`` with a spatial label map.

    The image is duplicated to num_frames copies.

    Parameters
    ----------
    num_frames : int
        Target number of frames (image is duplicated).
    overlap_labeling : str, default ``"single"``
        How to label pixels covered by multiple boxes:

        * ``"single"`` - each pixel gets the label of the *smallest* (by area)
          box that covers it.
        * ``"concat"`` - overlapping regions receive a composite target string
          where individual class labels are joined with ``", "``.
    """
    width, height = sample.image.size
    num_boxes = len(sample.boxes)

    if num_boxes == 0:
        label_map = torch.zeros(num_frames, height, width, dtype=torch.long)
        return Sample(
            frames=[sample.image] * num_frames,
            label_map=label_map,
            targets=[],
            query=sample.query,
            meta=sample.meta,
        )

    # Pre-compute clipped box coordinates
    box_coords: list[tuple[int, int, int, int]] = []
    for bx, by, bw, bh in sample.boxes:
        x0 = max(0, int(bx))
        y0 = max(0, int(by))
        x1 = min(width, int(bx + bw))
        y1 = min(height, int(by + bh))
        box_coords.append((x0, y0, x1, y1))

    if overlap_labeling == "single":
        # Sort boxes by area ascending so that smaller boxes are painted last
        # and overwrite larger ones, giving each pixel the smallest box's label.
        box_areas = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in box_coords]
        order = sorted(range(num_boxes), key=lambda i: -box_areas[i])

        # Paint box indices (largest first, smallest last overwrites)
        box_idx_map = torch.full((height, width), -1, dtype=torch.long)
        for i in order:
            x0, y0, x1, y1 = box_coords[i]
            if x1 > x0 and y1 > y0:
                box_idx_map[y0:y1, x0:x1] = i

        # Collect per-label pixel areas from the final painting
        unique_box_indices = box_idx_map.unique().tolist()
        label_area: dict[str, int] = {}
        for idx in unique_box_indices:
            if idx < 0:
                continue
            label = str(sample.labels[idx])
            area = int((box_idx_map == idx).sum().item())
            label_area[label] = label_area.get(label, 0) + area

        ranked_labels = sorted(
            label_area.keys(), key=lambda lbl: (-label_area[lbl], lbl)
        )
        label_to_target_idx = {
            label: i + 1 for i, label in enumerate(ranked_labels)
        }

        targets: list[str] = ranked_labels
        label_map = torch.zeros(height, width, dtype=torch.long)
        for idx in unique_box_indices:
            if idx < 0:
                continue
            label_map[box_idx_map == idx] = label_to_target_idx[
                str(sample.labels[idx])
            ]

    elif overlap_labeling == "concat":
        if num_boxes > 63:
            warnings.warn(
                f"Too many boxes ({num_boxes}) for bit-packed encoding; ",
                RuntimeWarning,
                stacklevel=2,
            )

        box_masks = torch.zeros(num_boxes, height, width, dtype=torch.bool)
        for i, (x0, y0, x1, y1) in enumerate(box_coords):
            if x1 > x0 and y1 > y0:
                box_masks[i, y0:y1, x0:x1] = True

        weights = 1 << torch.arange(num_boxes, dtype=torch.long)
        combo_map = (box_masks.long() * weights[:, None, None]).sum(dim=0)

        unique_vals = combo_map.unique().tolist()
        combo_to_key: dict[int, tuple[str, ...]] = {}
        area_by_key: dict[tuple[str, ...], int] = {}

        for val in unique_vals:
            if val == 0:
                continue

            area = int((combo_map == val).sum().item())
            box_indices = [i for i in range(num_boxes) if val & (1 << i)]

            unique_labels = sorted({str(sample.labels[i]) for i in box_indices})
            key = tuple(unique_labels)

            combo_to_key[val] = key
            area_by_key[key] = area_by_key.get(key, 0) + area

        ranked_keys = sorted(
            area_by_key.keys(), key=lambda k: (-area_by_key[k], k)
        )
        key_to_label = {
            key: idx for idx, key in enumerate(ranked_keys, start=1)
        }

        targets = [", ".join(key) for key in ranked_keys]
        val_to_label: dict[int, int] = {
            val: key_to_label[key] for val, key in combo_to_key.items()
        }

        label_map = torch.zeros(height, width, dtype=torch.long)
        for val, label_idx in val_to_label.items():
            label_map[combo_map == val] = label_idx

    else:
        raise ValueError(
            f"overlap_labeling must be 'single' or 'concat', got {overlap_labeling!r}"
        )

    # Duplicate to num_frames
    label_map = label_map.unsqueeze(0).expand(num_frames, -1, -1).clone()

    return Sample(
        frames=[sample.image] * num_frames,
        label_map=label_map,
        targets=targets,
        query=sample.query,
        meta=sample.meta,
    )


def from_masks(
    sample: ImageMasksSample | VideoMasksSample,
    num_frames: int,
    overlap_labeling: str = "single",
) -> Sample:
    """Convert an image/video mask sample into a ``Sample`` with a spatial label map.

    For ImageMasksSample, the image is duplicated to num_frames copies.
    For VideoMasksSample, frames are uniformly sampled to num_frames.

    Parameters
    ----------
    num_frames : int
        Target number of frames.
    overlap_labeling : str, default ``"single"``
        How to label pixels covered by multiple masks:

        * ``"single"`` - each pixel gets the label of the *smallest* (by area)
          mask that covers it.
        * ``"concat"`` - overlapping regions receive a composite target string
          where individual class labels are joined with ``", "``.
    """
    is_image = isinstance(sample, ImageMasksSample)
    if is_image:
        frames = [sample.image]
        masks = sample.masks
        labels = sample.labels
        meta = sample.meta
    elif isinstance(sample, VideoMasksSample):
        frames = sample.frames
        masks = sample.masks
        labels = sample.labels
        meta = sample.meta
    else:
        raise TypeError(
            "sample must be ImageMasksSample or VideoMasksSample, "
            f"got {type(sample).__name__}"
        )

    if not frames:
        raise ValueError("Sample contains no frames")

    width, height = frames[0].size
    num_masks = len(masks)

    if num_masks == 0:
        if is_image:
            # Duplicate for images
            label_map = torch.zeros(num_frames, height, width, dtype=torch.long)
            out_frames = frames * num_frames
        else:
            # Sample for videos
            label_map = torch.zeros(
                len(frames), height, width, dtype=torch.long
            )
            out_frames, label_map = sample_frames(frames, label_map, num_frames)
        return Sample(
            frames=out_frames,
            label_map=label_map,
            targets=[],
            query=sample.query,
            meta=meta,
        )

    normalised_masks: list[torch.Tensor] = []
    temporal_rank: int | None = None
    time_steps: int | None = None

    for mask in masks:
        mask_bool = mask.bool()
        if mask_bool.ndim == 2:
            reduced = mask_bool
            current_rank = 2
        elif mask_bool.ndim == 3:
            reduced = mask_bool
            current_rank = 3
        else:
            raise ValueError(
                f"Mask must be 2D or 3D, got shape {tuple(mask_bool.shape)}"
            )

        if temporal_rank is None:
            temporal_rank = current_rank
        elif temporal_rank != current_rank:
            raise ValueError(
                "All masks must have consistent dimensionality (all 2D or all 3D)"
            )

        if reduced.ndim == 2:
            if tuple(reduced.shape) != (height, width):
                raise ValueError(
                    f"Mask shape {tuple(reduced.shape)} does not match frame size {(height, width)}"
                )
        else:
            t, h, w = reduced.shape
            if (h, w) != (height, width):
                raise ValueError(
                    f"Mask shape {tuple(reduced.shape)} does not match frame size {(height, width)}"
                )
            if time_steps is None:
                time_steps = t
            elif time_steps != t:
                raise ValueError(
                    "All temporal masks must have the same number of frames"
                )

        normalised_masks.append(reduced)

    if temporal_rank == 3:
        if time_steps is None:
            raise ValueError(
                "Temporal mask conversion requires a valid time dimension"
            )
        if time_steps != len(frames):
            raise ValueError(
                f"Temporal mask length ({time_steps}) does not match number of frames ({len(frames)})"
            )

    if overlap_labeling == "single":
        # Sort masks by area descending so that smaller masks are painted last
        # and overwrite larger ones, giving each pixel the smallest mask's label.
        mask_areas = [int(m.sum().item()) for m in normalised_masks]
        order = sorted(range(num_masks), key=lambda i: -mask_areas[i])

        if temporal_rank == 3:
            mask_idx_map = torch.full(
                (time_steps, height, width), -1, dtype=torch.long
            )
        else:
            mask_idx_map = torch.full((height, width), -1, dtype=torch.long)

        for i in order:
            mask_idx_map[normalised_masks[i]] = i

        unique_mask_indices = mask_idx_map.unique().tolist()
        label_area: dict[str, int] = {}
        for idx in unique_mask_indices:
            if idx < 0:
                continue
            label = str(labels[idx])
            area = int((mask_idx_map == idx).sum().item())
            label_area[label] = label_area.get(label, 0) + area

        ranked_labels = sorted(
            label_area.keys(), key=lambda lbl: (-label_area[lbl], lbl)
        )
        label_to_target_idx = {
            label: i + 1 for i, label in enumerate(ranked_labels)
        }

        targets: list[str] = ranked_labels
        label_map = torch.zeros_like(mask_idx_map, dtype=torch.long)
        for idx in unique_mask_indices:
            if idx < 0:
                continue
            label_map[mask_idx_map == idx] = label_to_target_idx[
                str(labels[idx])
            ]

    elif overlap_labeling == "concat":
        if num_masks > 63:
            warnings.warn(
                f"Too many masks ({num_masks}) for bit-packed encoding; ",
                RuntimeWarning,
                stacklevel=2,
            )

        mask_stack = torch.stack(normalised_masks, dim=0).bool()

        weights = 1 << torch.arange(num_masks, dtype=torch.long)
        if mask_stack.ndim == 3:
            combo_map = (mask_stack.long() * weights[:, None, None]).sum(dim=0)
        elif mask_stack.ndim == 4:
            combo_map = (mask_stack.long() * weights[:, None, None, None]).sum(
                dim=0
            )
        else:
            raise ValueError(
                f"Unexpected stacked mask shape: {tuple(mask_stack.shape)}"
            )

        unique_vals = combo_map.unique().tolist()
        combo_to_key: dict[int, tuple[str, ...]] = {}
        area_by_key: dict[tuple[str, ...], int] = {}

        for val in unique_vals:
            if val == 0:
                continue

            area = int((combo_map == val).sum().item())
            mask_indices = [i for i in range(num_masks) if val & (1 << i)]

            unique_labels = sorted({str(labels[i]) for i in mask_indices})
            key = tuple(unique_labels)

            combo_to_key[int(val)] = key
            area_by_key[key] = area_by_key.get(key, 0) + area

        ranked_keys = sorted(
            area_by_key.keys(), key=lambda k: (-area_by_key[k], k)
        )
        key_to_label = {
            key: idx for idx, key in enumerate(ranked_keys, start=1)
        }

        targets: list[str] = [", ".join(key) for key in ranked_keys]
        val_to_label: dict[int, int] = {
            val: key_to_label[key] for val, key in combo_to_key.items()
        }

        label_map = torch.zeros_like(combo_map, dtype=torch.long)
        for val, label_idx in val_to_label.items():
            label_map[combo_map == val] = label_idx

    else:
        raise ValueError(
            f"overlap_labeling must be 'single' or 'concat', got {overlap_labeling!r}"
        )

    if label_map.dim() == 2:
        label_map = label_map.unsqueeze(0)

    # Handle frame count adjustment
    if is_image:
        # Duplicate for images
        label_map = label_map.expand(num_frames, -1, -1).clone()
        out_frames = frames * num_frames
    else:
        # Sample for videos
        out_frames, label_map = sample_frames(frames, label_map, num_frames)

    return Sample(
        frames=out_frames,
        label_map=label_map,
        targets=targets,
        query=sample.query,
        meta=meta,
    )


__all__ = [
    "from_boxes",
    "from_masks",
    "from_text",
    "from_text",
    "make_text_only_sample",
    "sample_frames",
    "sample_frames",
    "select_field",
]
