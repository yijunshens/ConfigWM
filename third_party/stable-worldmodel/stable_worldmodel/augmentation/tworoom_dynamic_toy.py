"""TwoRoom dynamic donor tracking toy built on top of SAM3 video propagation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL import ImageDraw

from .hdf5_static_copy import (
    _bbox_from_center,
    _composite_patch_rgba,
    _extract_feathered_patch,
    _json_default,
    _overlay_mask,
    _resolve_pixel_key,
    _sanitize_patch_rgba,
    _write_preview_video,
)
from .patch_rgba import (
    fill_pose_track as _fill_pose_track_shared,
    mask_orientation_rad as _mask_orientation_rad_shared,
    recolor_patch_rgba_preserve_material as _recolor_patch_rgba_preserve_material_shared,
    render_template_patch_rgba as _render_template_patch_rgba_shared,
    render_template_patch_from_masks as _render_template_patch_from_masks_shared,
    select_best_valid_patch_index as _select_best_valid_patch_index_shared,
    scale_patch_rgba as _scale_patch_rgba_premultiplied,
    wrap_angle_delta_rad as _wrap_angle_delta_rad_shared,
)
from .registry import COLOR_NAME_TO_RGB, get_object_spec
from .sam3_backend import Sam3ImageBackend, Sam3VideoBackend

_PREVIEW_VIDEO_KWARGS = {
    'codec': 'libx264',
    'quality': 10,
    'macro_block_size': 1,
    'ffmpeg_params': ['-crf', '8', '-preset', 'slow', '-pix_fmt', 'yuv444p'],
}


@dataclass
class TwoRoomDynamicToyConfig:
    input_h5: Path
    output_dir: Path
    env_id: str
    object_name: str
    donor_episode_idx: int
    target_episode_idx: int
    reference_frames_to_try: int = 5
    color: str = 'blue'
    scale: float = 1.0
    translate_x: int = 0
    translate_y: int = 0
    confidence_threshold: float = 0.05
    max_centroid_distance: float = 20.0
    prefer_video_propagation: bool = True
    compose_mode: str = 'donor_trajectory'


@dataclass
class DonorTrackResult:
    episode_idx: int
    ep_len: int
    reference_frame_idx: int
    used_prompt: str
    segmentation_score: float | None
    step_valid: np.ndarray
    step_centers_xy: np.ndarray
    step_bboxes_xyxy: np.ndarray
    step_masks: np.ndarray
    step_patch_rgba: list[np.ndarray]
    invalid_frame_indices: list[int]
    invalid_reason_counts: dict[str, int]
    frames: np.ndarray
    video_debug: dict[str, Any]


def _mask_centroid(mask_bool: np.ndarray) -> np.ndarray | None:
    if mask_bool is None or not mask_bool.any():
        return None
    ys, xs = np.where(mask_bool)
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def _episode_slice(ep_offsets: np.ndarray, ep_lens: np.ndarray, episode_idx: int) -> tuple[int, int]:
    start = int(ep_offsets[episode_idx])
    end = int(start + ep_lens[episode_idx])
    return start, end


def _load_episode_data(input_h5: Path, episode_idx: int) -> dict[str, Any]:
    with h5py.File(input_h5, 'r') as f:
        ep_lens = f['ep_len'][:]
        ep_offsets = f['ep_offset'][:]
        pixel_key = _resolve_pixel_key(f)
        start, end = _episode_slice(ep_offsets, ep_lens, episode_idx)
        return {
            'frames': f[pixel_key][start:end],
            'pos_agent': f['pos_agent'][start:end],
            'ep_len': int(ep_lens[episode_idx]),
        }


def _clip_bbox_xyxy(
    bbox_xyxy: list[int] | tuple[int, int, int, int],
    image_hw: tuple[int, int],
) -> list[int]:
    height, width = image_hw
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0, min(x1, width - 1))
    y1 = max(y0, min(y1, height - 1))
    return [x0, y0, x1, y1]


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default),
        encoding='utf-8',
    )


def _pick_reference_frame(
    frames: np.ndarray,
    pos_agent: np.ndarray,
    prompts: list[str],
    backend: Sam3ImageBackend,
    reference_frames_to_try: int,
    confidence_threshold: float,
    max_centroid_distance: float,
) -> dict[str, Any]:
    best_failure: dict[str, Any] = {
        'success': False,
        'reference_frame_idx': None,
        'score': None,
        'used_prompt': None,
        'mask': None,
    }
    for frame_idx in range(min(reference_frames_to_try, len(frames))):
        segmentation = backend.segment_best(
            frames[frame_idx],
            prompts=prompts,
            confidence_threshold=confidence_threshold,
        )
        if not segmentation['success']:
            continue
        centroid = _mask_centroid(segmentation['mask'])
        if centroid is None:
            continue
        dist = float(np.linalg.norm(centroid - pos_agent[frame_idx]))
        segmentation['reference_frame_idx'] = frame_idx
        segmentation['centroid_distance'] = dist
        if dist <= max_centroid_distance:
            return segmentation
        if best_failure['score'] is None or float(segmentation.get('score') or 0.0) > float(
            best_failure['score'] or 0.0
        ):
            best_failure = {
                'success': False,
                'reference_frame_idx': frame_idx,
                'score': segmentation.get('score'),
                'used_prompt': segmentation.get('used_prompt'),
                'mask': segmentation.get('mask'),
                'centroid_distance': dist,
                'failure_reason': 'centroid_too_far',
            }
    return best_failure


def _track_per_frame_with_image_sam3(
    frames: np.ndarray,
    pos_agent: np.ndarray,
    init_bbox_xyxy: list[int],
    image_backend: Sam3ImageBackend,
    confidence_threshold: float,
) -> dict[str, Any]:
    patch_h = max(1, int(init_bbox_xyxy[3] - init_bbox_xyxy[1] + 1))
    patch_w = max(1, int(init_bbox_xyxy[2] - init_bbox_xyxy[0] + 1))
    frame_masks: dict[int, np.ndarray] = {}
    frames_seen = 0
    frames_with_masks = 0
    frames_without_masks = 0
    sample_mask_shapes: list[Any] = []

    for frame_idx, (frame, center_xy) in enumerate(zip(frames, pos_agent)):
        frames_seen += 1
        bbox_xyxy = _clip_bbox_xyxy(
            _bbox_from_center(
                (int(round(center_xy[0])), int(round(center_xy[1]))),
                (patch_h, patch_w),
            ),
            frame.shape[:2],
        )
        segmentation = image_backend.segment_box_best(
            frame,
            bbox_xyxy=bbox_xyxy,
            confidence_threshold=confidence_threshold,
        )
        if not segmentation['success'] or segmentation['mask'] is None:
            frames_without_masks += 1
            continue
        frame_masks[frame_idx] = segmentation['mask'].astype(bool)
        frames_with_masks += 1
        if len(sample_mask_shapes) < 5:
            sample_mask_shapes.append(np.asarray(segmentation['mask']).shape)

    return {
        'success': bool(frame_masks),
        'frame_masks': frame_masks,
        'initial_mask': frame_masks.get(0),
        'initial_centroid': _mask_centroid(frame_masks.get(0)) if 0 in frame_masks else None,
        'obj_ids': [0],
        'used_prompt': 'bbox_per_frame_image_sam3',
        'debug': {
            'backend': 'image_sam3_per_frame',
            'frames_seen': frames_seen,
            'frames_with_masks': frames_with_masks,
            'frames_without_masks': frames_without_masks,
            'num_frame_masks': len(frame_masks),
            'sample_mask_shapes': sample_mask_shapes,
        },
    }


def _video_result_needs_fallback(
    video_result: dict[str, Any],
    expected_num_frames: int,
    reference_mask: np.ndarray,
    *,
    expected_centers_xy: np.ndarray | None = None,
    max_centroid_distance: float | None = None,
) -> bool:
    if not video_result.get('success'):
        return True
    frame_masks = video_result.get('frame_masks') or {}
    if not isinstance(frame_masks, dict):
        try:
            frame_masks = dict(frame_masks)
        except Exception:
            return True
    if len(frame_masks) < expected_num_frames:
        return True

    reference_area = int(np.asarray(reference_mask, dtype=bool).sum())
    if reference_area <= 0:
        return True

    # Reject clearly under-covered masks even when the backend claims success.
    min_area = max(1, int(np.ceil(reference_area * 0.5)))
    for frame_idx in range(expected_num_frames):
        mask = frame_masks.get(frame_idx)
        if mask is None:
            return True
        mask_bool = np.asarray(mask, dtype=bool)
        if not mask_bool.any():
            return True
        if int(mask_bool.sum()) < min_area:
            return True
        if expected_centers_xy is not None and max_centroid_distance is not None:
            centroid = _mask_centroid(mask_bool)
            if centroid is None:
                return True
            dist = float(np.linalg.norm(centroid - np.asarray(expected_centers_xy[frame_idx], dtype=np.float32)))
            if dist > float(max_centroid_distance):
                return True
    return False


def _build_track_contact_sheet(
    frames: np.ndarray,
    masks: np.ndarray,
    patches: list[np.ndarray],
    output_path: Path,
    frame_indices: list[int],
) -> None:
    rows = []
    cell_label_h = 18
    for frame_idx in frame_indices:
        frame = frames[frame_idx]
        mask = masks[frame_idx]
        overlay = _overlay_mask(frame, mask)
        mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
        patch = patches[frame_idx]
        patch_rgb = patch[..., :3]
        rows.append(
            [
                ('frame', frame),
                ('overlay', overlay),
                ('mask', mask_rgb),
                ('patch', patch_rgb),
            ]
        )

    cell_w = max(img.shape[1] for row in rows for _, img in row)
    cell_h = max(img.shape[0] for row in rows for _, img in row)
    canvas = Image.new('RGB', (len(rows[0]) * cell_w, len(rows) * (cell_h + cell_label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for row_idx, row in enumerate(rows):
        for col_idx, (label, img) in enumerate(row):
            pil_img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            x0 = col_idx * cell_w
            y0 = row_idx * (cell_h + cell_label_h)
            canvas.paste(pil_img.resize((cell_w, cell_h)), (x0, y0 + cell_label_h))
            draw.text((x0 + 4, y0 + 2), f'f{frame_indices[row_idx]} {label}', fill=(0, 0, 0))
    canvas.save(output_path)


def _write_mask_overlay_video(path: Path, frames: np.ndarray, masks: np.ndarray) -> None:
    overlay_frames = np.stack([_overlay_mask(frame, mask) for frame, mask in zip(frames, masks)], axis=0)
    imageio.mimsave(
        path,
        overlay_frames.astype(np.uint8),
        fps=10,
        **_PREVIEW_VIDEO_KWARGS,
    )


def _write_mask_only_video(path: Path, masks: np.ndarray) -> None:
    mask_frames = np.repeat((np.asarray(masks, dtype=np.uint8) * 255)[..., None], 3, axis=3)
    imageio.mimsave(
        path,
        mask_frames.astype(np.uint8),
        fps=10,
        **_PREVIEW_VIDEO_KWARGS,
    )


def _recolor_patch_rgba(
    patch_rgba: np.ndarray,
    target_rgb: np.ndarray,
    *,
    template_patch_rgba: np.ndarray | None = None,
) -> np.ndarray:
    return _recolor_patch_rgba_preserve_material_shared(
        patch_rgba,
        target_rgb,
        template_patch_rgba=template_patch_rgba,
    )


def _scale_patch_rgba(patch_rgba: np.ndarray, scale_factor: float) -> np.ndarray:
    return _scale_patch_rgba_premultiplied(patch_rgba, scale_factor)


def _build_first_frame_contact(
    donor_frame: np.ndarray,
    donor_patch: np.ndarray,
    target_before: np.ndarray,
    target_after: np.ndarray,
    output_path: Path,
) -> None:
    patch_rgb = donor_patch[..., :3]
    max_h = max(donor_frame.shape[0], patch_rgb.shape[0], target_before.shape[0], target_after.shape[0])
    panels = [donor_frame, patch_rgb, target_before, target_after]
    panel_w = max(panel.shape[1] for panel in panels)
    label_h = 18
    canvas = Image.new('RGB', (panel_w * 4, max_h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    labels = ['donor', 'donor_patch', 'target_before', 'target_after']
    for idx, (label, panel) in enumerate(zip(labels, panels)):
        pil_img = Image.fromarray(panel.astype(np.uint8)).convert('RGB')
        x0 = idx * panel_w
        canvas.paste(pil_img.resize((panel_w, max_h)), (x0, label_h))
        draw.text((x0 + 4, 2), label, fill=(0, 0, 0))
    canvas.save(output_path)


def build_tworoom_dynamic_donor_track(
    config: TwoRoomDynamicToyConfig,
    image_backend: Sam3ImageBackend,
    video_backend: Sam3VideoBackend | None,
) -> DonorTrackResult:
    spec = get_object_spec(config.env_id, config.object_name)
    episode = _load_episode_data(config.input_h5, config.donor_episode_idx)
    frames = episode['frames']
    pos_agent = episode['pos_agent']
    # Video-grounding works much better on TwoRoom when we describe the
    # target as a small red circular object instead of a generic "ball".
    prompts = [
        'red circle',
        'small red circle',
        'red dot',
        'small red dot',
        'red round object',
        'red object',
        'red ball',
        'ball',
    ]
    init = _pick_reference_frame(
        frames=frames,
        pos_agent=pos_agent,
        prompts=prompts,
        backend=image_backend,
        reference_frames_to_try=config.reference_frames_to_try,
        confidence_threshold=config.confidence_threshold,
        max_centroid_distance=config.max_centroid_distance,
    )
    if not init['success']:
        raise RuntimeError('init_segmentation_failed')

    init_bbox = _extract_feathered_patch(
        frames[int(init['reference_frame_idx'])],
        init['mask'],
    )[1]
    if config.prefer_video_propagation and video_backend is not None:
        video_result = video_backend.propagate_text_prompt(
            frames_np=frames,
            frame_index=int(init['reference_frame_idx']),
            prompt=str(init['used_prompt']),
            output_prob_thresh=float(config.confidence_threshold),
        )
        if _video_result_needs_fallback(
            video_result,
            len(frames),
            init['mask'],
            expected_centers_xy=pos_agent,
            max_centroid_distance=float(config.max_centroid_distance),
        ):
            box_prompt_fn = getattr(video_backend, 'propagate_box_prompt', None)
            if callable(box_prompt_fn):
                box_result = box_prompt_fn(
                    frames_np=frames,
                    frame_index=int(init['reference_frame_idx']),
                    bbox_xyxy=init_bbox,
                    output_prob_thresh=float(config.confidence_threshold),
                )
            else:
                box_result = {
                    'success': False,
                    'frame_masks': {},
                    'debug': {'backend': 'box_prompt_unavailable'},
                }
            if _video_result_needs_fallback(
                box_result,
                len(frames),
                init['mask'],
                expected_centers_xy=pos_agent,
                max_centroid_distance=float(config.max_centroid_distance),
            ):
                video_result = _track_per_frame_with_image_sam3(
                    frames=frames,
                    pos_agent=pos_agent,
                    init_bbox_xyxy=init_bbox,
                    image_backend=image_backend,
                    confidence_threshold=float(config.confidence_threshold),
                )
            else:
                video_result = box_result
    else:
        video_result = _track_per_frame_with_image_sam3(
            frames=frames,
            pos_agent=pos_agent,
            init_bbox_xyxy=init_bbox,
            image_backend=image_backend,
            confidence_threshold=float(config.confidence_threshold),
        )
    if not video_result['success']:
        raise RuntimeError('tracking_failed')

    step_valid = np.zeros(len(frames), dtype=bool)
    step_masks = np.zeros((len(frames), frames.shape[1], frames.shape[2]), dtype=bool)
    step_centers = np.zeros((len(frames), 2), dtype=np.float32)
    step_bboxes = np.zeros((len(frames), 4), dtype=np.int32)
    step_patches: list[np.ndarray] = []
    invalid_indices: list[int] = []
    invalid_reason_counts = {
        'missing_mask': 0,
        'empty_mask': 0,
        'empty_centroid': 0,
        'centroid_too_far': 0,
    }

    for frame_idx in range(len(frames)):
        mask = video_result['frame_masks'].get(frame_idx)
        if mask is None:
            invalid_indices.append(frame_idx)
            invalid_reason_counts['missing_mask'] += 1
            step_patches.append(np.zeros((1, 1, 4), dtype=np.uint8))
            continue
        if not mask.any():
            invalid_indices.append(frame_idx)
            invalid_reason_counts['empty_mask'] += 1
            step_patches.append(np.zeros((1, 1, 4), dtype=np.uint8))
            continue
        centroid = _mask_centroid(mask)
        if centroid is None:
            invalid_indices.append(frame_idx)
            invalid_reason_counts['empty_centroid'] += 1
            step_patches.append(np.zeros((1, 1, 4), dtype=np.uint8))
            continue
        dist = float(np.linalg.norm(centroid - pos_agent[frame_idx]))
        if dist > config.max_centroid_distance:
            invalid_indices.append(frame_idx)
            invalid_reason_counts['centroid_too_far'] += 1
            step_patches.append(np.zeros((1, 1, 4), dtype=np.uint8))
            continue
        patch_rgba, bbox, _crop_mask = _extract_feathered_patch(frames[frame_idx], mask)
        step_valid[frame_idx] = True
        step_masks[frame_idx] = mask
        step_centers[frame_idx] = centroid
        step_bboxes[frame_idx] = np.asarray(bbox, dtype=np.int32)
        step_patches.append(patch_rgba)

    return DonorTrackResult(
        episode_idx=config.donor_episode_idx,
        ep_len=int(episode['ep_len']),
        reference_frame_idx=int(init['reference_frame_idx']),
        used_prompt=str(init['used_prompt']),
        segmentation_score=(
            None if init.get('score') is None else float(init['score'])
        ),
        step_valid=step_valid,
        step_centers_xy=step_centers,
        step_bboxes_xyxy=step_bboxes,
        step_masks=step_masks,
        step_patch_rgba=step_patches,
        invalid_frame_indices=invalid_indices,
        invalid_reason_counts=invalid_reason_counts,
        frames=frames,
        video_debug=video_result.get('debug', {}),
    )


def export_tworoom_donor_track_artifacts(
    config: TwoRoomDynamicToyConfig,
    donor_track: DonorTrackResult,
    output_dir: Path,
) -> dict[str, Any]:
    from .tworoom_dynamic_from_cache import _patch_quality_metrics, _select_best_valid_patch_index

    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_video_path = output_dir / 'donor_tracking_preview.mp4'
    mask_only_video_path = output_dir / 'mask_only_preview.mp4'
    contact_path = output_dir / 'donor_assets_contact.png'
    assets_path = output_dir / 'donor_assets.npz'
    summary_path = output_dir / 'donor_summary.json'
    export_patches = [
        _sanitize_patch_rgba(np.asarray(patch, dtype=np.uint8))
        for patch in donor_track.step_patch_rgba
    ]
    best_patch_index = _select_best_valid_patch_index(donor_track.step_valid, export_patches)
    best_patch_metrics = (
        None
        if best_patch_index is None
        else _patch_quality_metrics(export_patches[int(best_patch_index)])
    )

    _write_mask_overlay_video(tracking_video_path, donor_track.frames, donor_track.step_masks)
    _write_mask_only_video(mask_only_video_path, donor_track.step_masks)
    candidate_indices = sorted(
        {
            0,
            donor_track.reference_frame_idx,
            max(0, donor_track.ep_len // 2),
            max(0, donor_track.ep_len - 1),
        }
    )
    _build_track_contact_sheet(
        frames=donor_track.frames,
        masks=donor_track.step_masks,
        patches=export_patches,
        output_path=contact_path,
        frame_indices=candidate_indices,
    )
    np.savez_compressed(
        assets_path,
        step_valid=donor_track.step_valid,
        step_centers_xy=donor_track.step_centers_xy,
        step_bboxes_xyxy=donor_track.step_bboxes_xyxy,
        step_masks=donor_track.step_masks,
        step_patch_rgba=np.asarray(export_patches, dtype=object),
        best_patch_index=np.asarray(-1 if best_patch_index is None else int(best_patch_index), dtype=np.int32),
    )
    summary = {
        'episode_idx': donor_track.episode_idx,
        'ep_len': donor_track.ep_len,
        'reference_frame_idx': donor_track.reference_frame_idx,
        'used_prompt': donor_track.used_prompt,
        'segmentation_score': donor_track.segmentation_score,
        'valid_steps': int(donor_track.step_valid.sum()),
        'total_steps': donor_track.ep_len,
        'tracking_success': bool(donor_track.step_valid.all()),
        'invalid_frame_indices': donor_track.invalid_frame_indices,
        'invalid_reason_counts': donor_track.invalid_reason_counts,
        'video_debug': donor_track.video_debug,
        'tracking_video': str(tracking_video_path),
        'mask_only_video': str(mask_only_video_path),
        'contact_image': str(contact_path),
        'assets_path': str(assets_path),
        'best_patch_index': None if best_patch_index is None else int(best_patch_index),
        'best_patch_metrics': best_patch_metrics,
    }
    _save_json(summary_path, summary)
    return summary


def compose_tworoom_dynamic_donor_on_target(
    config: TwoRoomDynamicToyConfig,
    donor_track: DonorTrackResult,
    output_dir: Path,
) -> dict[str, Any]:
    episode = _load_episode_data(config.input_h5, config.target_episode_idx)
    target_frames = episode['frames']
    composite_frames = target_frames.copy()
    common_len = min(len(target_frames), donor_track.ep_len)
    color_rgb = COLOR_NAME_TO_RGB[config.color]

    if config.compose_mode == 'donor_trajectory':
        centers_xy = donor_track.step_centers_xy[:common_len]
        center_valid = donor_track.step_valid[:common_len]
    elif config.compose_mode == 'same_trajectory_offset':
        centers_xy = (
            episode['pos_agent'][:common_len]
            + np.array([config.translate_x, config.translate_y], dtype=np.float32)[None, :]
        )
        center_valid = np.ones((common_len,), dtype=bool)
    else:
        raise ValueError(f'unsupported compose_mode: {config.compose_mode}')

    best_patch_idx = _select_best_valid_patch_index_shared(
        donor_track.step_valid,
        donor_track.step_patch_rgba,
    )
    if best_patch_idx is None:
        valid_indices = np.flatnonzero(donor_track.step_valid)
        if valid_indices.size == 0:
            raise RuntimeError('donor_has_no_valid_patch')
        best_patch_idx = int(valid_indices[0])
    template_patch_rgba = donor_track.step_patch_rgba[int(best_patch_idx)]
    template_mask = donor_track.step_masks[int(best_patch_idx)]
    recolored_template = _recolor_patch_rgba(template_patch_rgba, color_rgb)
    scaled_template = _render_template_patch_from_masks_shared(
        recolored_template,
        template_mask=template_mask,
        current_mask=template_mask,
        global_scale_factor=float(config.scale),
    )
    template_angle = _mask_orientation_rad_shared(template_mask)
    angle_track = np.full((common_len,), np.nan, dtype=np.float32)
    for frame_idx in range(common_len):
        if not donor_track.step_valid[frame_idx]:
            continue
        angle = _mask_orientation_rad_shared(donor_track.step_masks[frame_idx])
        if angle is not None:
            angle_track[frame_idx] = float(angle)
    pose_centers, pose_angles, pose_valid, filled_frame_indices = _fill_pose_track_shared(
        centers_xy,
        center_valid,
        angles_rad=angle_track,
    )

    for frame_idx in range(common_len):
        if not pose_valid[frame_idx]:
            continue
        center = pose_centers[frame_idx]
        if config.compose_mode == 'donor_trajectory':
            center = center + np.array(
                [config.translate_x, config.translate_y],
                dtype=np.float32,
            )
        if donor_track.step_valid[frame_idx]:
            rendered_patch = _render_template_patch_from_masks_shared(
                recolored_template,
                template_mask=template_mask,
                current_mask=donor_track.step_masks[frame_idx],
                global_scale_factor=float(config.scale),
            )
        else:
            current_angle = None
            if pose_angles is not None and np.isfinite(pose_angles[frame_idx]):
                current_angle = float(pose_angles[frame_idx])
            rotation_delta = (
                0.0
                if template_angle is None or current_angle is None
                else _wrap_angle_delta_rad_shared(current_angle - float(template_angle))
            )
            rendered_patch = _render_template_patch_rgba_shared(
                scaled_template,
                rotation_delta_rad=float(rotation_delta),
            )
        composite_frames[frame_idx] = _composite_patch_rgba(
            composite_frames[frame_idx],
            rendered_patch,
            center_xy=(float(center[0]), float(center[1])),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / 'before_after_preview.mp4'
    first_frame_path = output_dir / 'first_frame_contact.png'
    summary_path = output_dir / 'composition_summary.json'
    _write_preview_video(preview_path, target_frames, composite_frames)

    template_frame_idx = int(best_patch_idx)
    _build_first_frame_contact(
        donor_frame=donor_track.frames[template_frame_idx],
        donor_patch=scaled_template,
        target_before=target_frames[0],
        target_after=composite_frames[0],
        output_path=first_frame_path,
    )

    summary = {
        'donor_episode_idx': config.donor_episode_idx,
        'target_episode_idx': config.target_episode_idx,
        'donor_ep_len': donor_track.ep_len,
        'target_ep_len': int(episode['ep_len']),
        'frames_composited': int(common_len),
        'color': config.color,
        'scale_factor': float(config.scale),
        'translation_xy': [int(config.translate_x), int(config.translate_y)],
        'copy_mode': str(config.compose_mode),
        'template_frame_idx': int(template_frame_idx),
        'reused_template_patch': True,
        'filled_frame_indices': filled_frame_indices,
        'preview_video': str(preview_path),
        'first_frame_contact': str(first_frame_path),
    }
    _save_json(summary_path, summary)
    return summary


def run_tworoom_dynamic_donor_toy(
    config: TwoRoomDynamicToyConfig,
    image_backend: Sam3ImageBackend,
    video_backend: Sam3VideoBackend,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    donor_track = build_tworoom_dynamic_donor_track(
        config=config,
        image_backend=image_backend,
        video_backend=video_backend,
    )
    donor_dir = config.output_dir / f'donor_episode_{config.donor_episode_idx}'
    target_dir = config.output_dir / (
        f'compose_donor_{config.donor_episode_idx}_target_{config.target_episode_idx}'
    )
    donor_summary = export_tworoom_donor_track_artifacts(
        config=config,
        donor_track=donor_track,
        output_dir=donor_dir,
    )
    composition_summary = compose_tworoom_dynamic_donor_on_target(
        config=config,
        donor_track=donor_track,
        output_dir=target_dir,
    )
    result = {
        'status': 'ok',
        'donor_summary': donor_summary,
        'composition_summary': composition_summary,
        'config': asdict(config),
    }
    _save_json(config.output_dir / 'toy_summary.json', result)
    return result
