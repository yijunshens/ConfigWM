"""SAM3-backed adapters for stable-worldmodel augmentation."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


DEFAULT_SAM3_REPO = Path('/data/syj/jepa/sam3')
DEFAULT_SAM3_CHECKPOINT = Path('/data/model/bodhicitta-sam3/sam3.pt')


def _ensure_sam3_importable(repo_path: str | Path | None) -> None:
    try:
        import sam3  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    if repo_path is None:
        raise ModuleNotFoundError(
            'sam3 is not importable. Install it or pass --sam3_repo_path.'
        )
    repo_str = str(Path(repo_path))
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _apply_dtype_compat_patch() -> None:
    from sam3.model import vitdet

    if getattr(vitdet.Mlp.forward, '_stablewm_dtype_compat_patch', False):
        return

    def patched_forward(self, x):
        x = vitdet.addmm_act(type(self.act), self.fc1, x)
        x = self.drop1(x)
        x = self.norm(x)
        x = x.to(self.fc2.weight.dtype)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    patched_forward._stablewm_dtype_compat_patch = True
    vitdet.Mlp.forward = patched_forward


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.to(torch.float32)
        return tensor.cpu().numpy()
    return np.asarray(value)


def _select_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    confidence_threshold: float,
) -> tuple[np.ndarray | None, float | None]:
    if len(scores) == 0:
        return None, None
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    if best_score < confidence_threshold:
        return None, best_score
    best_mask = masks[best_index]
    if best_mask.ndim == 3:
        best_mask = best_mask[0]
    return best_mask > 0, best_score


class Sam3ImageBackend:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        sam3_repo_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        repo_path = sam3_repo_path or DEFAULT_SAM3_REPO
        _ensure_sam3_importable(repo_path)
        _apply_dtype_compat_patch()

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        ckpt = Path(checkpoint_path or DEFAULT_SAM3_CHECKPOINT)
        if not ckpt.exists():
            raise FileNotFoundError(
                f'SAM3 checkpoint not found: {ckpt}. Pass --checkpoint_path.'
            )

        self.model = build_sam3_image_model(
            checkpoint_path=str(ckpt),
            load_from_HF=False,
            compile=False,
        )
        self.processor = Sam3Processor(
            self.model,
            confidence_threshold=confidence_threshold,
        )

    def segment_best(
        self,
        image_np: np.ndarray,
        prompts: list[str],
        confidence_threshold: float,
        debug_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        image = Image.fromarray(image_np.astype(np.uint8))
        best: dict[str, Any] = {
            'success': False,
            'mask': None,
            'score': None,
            'used_prompt': None,
            'used_prompt_variant': None,
        }
        for prompt in prompts:
            state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(state=state, prompt=prompt)
            masks = _to_numpy(output.get('masks', []))
            scores = _to_numpy(output.get('scores', []))
            mask, score = _select_best_mask(
                masks,
                scores,
                confidence_threshold=confidence_threshold,
            )
            if mask is None:
                if best['score'] is None and score is not None:
                    best['score'] = float(score)
                continue
            if best['score'] is None or float(score) > float(best['score']):
                best = {
                    'success': True,
                    'mask': mask.astype(bool),
                    'score': float(score),
                    'used_prompt': prompt,
                    'used_prompt_variant': prompt,
                }
        return best

    def segment_box_best(
        self,
        image_np: np.ndarray,
        bbox_xyxy: list[int] | tuple[int, int, int, int],
        confidence_threshold: float,
    ) -> dict[str, Any]:
        image = Image.fromarray(image_np.astype(np.uint8))
        state = self.processor.set_image(image)
        x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
        width, height = float(image_np.shape[1]), float(image_np.shape[0])
        box_cxcywh = [
            ((x0 + x1) / 2.0) / width,
            ((y0 + y1) / 2.0) / height,
            max(1.0, (x1 - x0 + 1.0)) / width,
            max(1.0, (y1 - y0 + 1.0)) / height,
        ]
        output = self.processor.add_geometric_prompt(
            box=box_cxcywh,
            label=True,
            state=state,
        )
        masks = _to_numpy(output.get('masks', []))
        scores = _to_numpy(output.get('scores', []))
        boxes = _to_numpy(output.get('boxes', []))
        mask, score = _select_best_mask(
            masks,
            scores,
            confidence_threshold=confidence_threshold,
        )
        if mask is None:
            return {
                'success': False,
                'mask': None,
                'score': None if score is None else float(score),
                'bbox': None,
                'used_prompt': 'bbox',
            }
        best_bbox = None
        if len(boxes) > 0:
            best_index = int(np.argmax(scores))
            best_bbox = boxes[best_index].tolist()
        return {
            'success': True,
            'mask': mask.astype(bool),
            'score': float(score) if score is not None else None,
            'bbox': best_bbox,
            'used_prompt': 'bbox',
        }


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    if mask is None:
        return None
    mask_bool = np.asarray(mask).astype(bool)
    if not mask_bool.any():
        return None
    ys, xs = np.where(mask_bool)
    return float(xs.mean()), float(ys.mean())


def _extract_stream_outputs(response: dict[str, Any]) -> dict[str, Any]:
    outputs = response.get('outputs')
    if isinstance(outputs, dict):
        return outputs
    return response


def _write_video_frames_to_dir(frames: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        imageio.imwrite(output_dir / f'{idx:05d}.jpg', frame.astype(np.uint8))


class Sam3VideoBackend:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        sam3_repo_path: str | Path | None = None,
        *,
        version: str = 'sam3',
        use_fa3: bool = True,
        use_rope_real: bool = True,
        async_loading_frames: bool = False,
    ) -> None:
        repo_path = sam3_repo_path or DEFAULT_SAM3_REPO
        _ensure_sam3_importable(repo_path)

        from sam3 import build_sam3_predictor

        ckpt = Path(checkpoint_path or DEFAULT_SAM3_CHECKPOINT)
        if not ckpt.exists():
            raise FileNotFoundError(
                f'SAM3 checkpoint not found: {ckpt}. Pass --checkpoint_path.'
            )

        self.predictor = build_sam3_predictor(
            checkpoint_path=str(ckpt),
            version=version,
            compile=False,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
            async_loading_frames=async_loading_frames,
        )

    def propagate_text_prompt(
        self,
        frames_np: np.ndarray,
        frame_index: int,
        prompt: str,
        output_prob_thresh: float = 0.5,
        frame_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if frame_dir is None:
            tmp_dir = tempfile.TemporaryDirectory(prefix='stablewm_sam3_video_')
            frame_dir_path = Path(tmp_dir.name)
            cleanup = tmp_dir
        else:
            frame_dir_path = Path(frame_dir)
            cleanup = None
        try:
            _write_video_frames_to_dir(frames_np, frame_dir_path)
            session = self.predictor.handle_request(
                {'type': 'start_session', 'resource_path': str(frame_dir_path)}
            )
            session_id = session['session_id']
            init_response = self.predictor.handle_request(
                {
                    'type': 'add_prompt',
                    'session_id': session_id,
                    'frame_index': int(frame_index),
                    'text': prompt,
                    'output_prob_thresh': float(output_prob_thresh),
                }
            )
            init_outputs = init_response.get('outputs', {})
            init_masks = init_outputs.get('out_binary_masks')
            init_obj_ids = init_outputs.get('out_obj_ids')
            if init_masks is None or len(init_masks) == 0:
                self.predictor.handle_request(
                    {'type': 'close_session', 'session_id': session_id}
                )
                return {
                    'success': False,
                    'frame_masks': {},
                    'initial_mask': None,
                    'initial_centroid': None,
                    'obj_ids': [],
                    'used_prompt': prompt,
                }

            frame_masks: dict[int, np.ndarray] = {}
            frames_seen = 0
            frames_with_masks = 0
            frames_without_masks = 0
            sample_mask_shapes: list[Any] = []
            for response in self.predictor.handle_stream_request(
                {
                    'type': 'propagate_in_video',
                    'session_id': session_id,
                    'start_frame_index': int(frame_index),
                    'output_prob_thresh': float(output_prob_thresh),
                }
            ):
                out_frame_index = response.get('frame_index')
                if out_frame_index is None:
                    continue
                frames_seen += 1
                outputs = _extract_stream_outputs(response)
                obj_ids = outputs.get('out_obj_ids', [])
                masks = outputs.get('out_binary_masks')
                if masks is None or len(masks) == 0:
                    frames_without_masks += 1
                    continue
                if isinstance(obj_ids, torch.Tensor):
                    obj_ids = obj_ids.detach().cpu().numpy()
                if isinstance(masks, torch.Tensor):
                    masks = masks.detach().cpu().numpy()
                if len(masks) == 0:
                    frames_without_masks += 1
                    continue
                if len(sample_mask_shapes) < 5:
                    sample_mask_shapes.append(np.asarray(masks).shape)
                best_mask = masks[0]
                if best_mask.ndim == 3:
                    best_mask = best_mask[0]
                frame_masks[int(out_frame_index)] = best_mask.astype(bool)
                frames_with_masks += 1

            self.predictor.handle_request(
                {'type': 'close_session', 'session_id': session_id}
            )

            init_mask = frame_masks.get(int(frame_index))
            if init_mask is None:
                masks = init_masks.detach().cpu().numpy() if torch.is_tensor(init_masks) else np.asarray(init_masks)
                init_mask = masks[0]
                if init_mask.ndim == 3:
                    init_mask = init_mask[0]
                init_mask = init_mask.astype(bool)

            return {
                'success': bool(frame_masks),
                'frame_masks': frame_masks,
                'initial_mask': init_mask,
                'initial_centroid': _mask_centroid(init_mask),
                'obj_ids': (
                    init_obj_ids.detach().cpu().numpy().tolist()
                    if torch.is_tensor(init_obj_ids)
                    else np.asarray(init_obj_ids).tolist()
                ),
                'used_prompt': prompt,
                'debug': {
                    'frames_seen': frames_seen,
                    'frames_with_masks': frames_with_masks,
                    'frames_without_masks': frames_without_masks,
                    'num_frame_masks': len(frame_masks),
                    'sample_mask_shapes': sample_mask_shapes,
                },
            }
        finally:
            if cleanup is not None:
                cleanup.cleanup()

    def propagate_box_prompt(
        self,
        frames_np: np.ndarray,
        frame_index: int,
        bbox_xyxy: list[int],
        output_prob_thresh: float = 0.5,
        frame_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if frame_dir is None:
            tmp_dir = tempfile.TemporaryDirectory(prefix='stablewm_sam3_video_')
            frame_dir_path = Path(tmp_dir.name)
            cleanup = tmp_dir
        else:
            frame_dir_path = Path(frame_dir)
            cleanup = None
        try:
            _write_video_frames_to_dir(frames_np, frame_dir_path)
            session = self.predictor.handle_request(
                {'type': 'start_session', 'resource_path': str(frame_dir_path)}
            )
            session_id = session['session_id']
            x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
            height, width = frames_np.shape[1:3]
            box_xywh = [
                x0 / width,
                y0 / height,
                (x1 - x0 + 1) / width,
                (y1 - y0 + 1) / height,
            ]
            init_response = self.predictor.handle_request(
                {
                    'type': 'add_prompt',
                    'session_id': session_id,
                    'frame_index': int(frame_index),
                    'bounding_boxes': [box_xywh],
                    'bounding_box_labels': [1],
                    'clear_old_boxes': True,
                    'clear_old_points': True,
                    'output_prob_thresh': float(output_prob_thresh),
                }
            )
            init_outputs = init_response.get('outputs', {})
            init_masks = init_outputs.get('out_binary_masks')
            init_obj_ids = init_outputs.get('out_obj_ids')
            if init_masks is None or len(init_masks) == 0:
                self.predictor.handle_request(
                    {'type': 'close_session', 'session_id': session_id}
                )
                return {
                    'success': False,
                    'frame_masks': {},
                    'initial_mask': None,
                    'initial_centroid': None,
                    'obj_ids': [],
                    'used_prompt': 'bbox',
                }

            frame_masks: dict[int, np.ndarray] = {}
            frames_seen = 0
            frames_with_masks = 0
            frames_without_masks = 0
            sample_mask_shapes: list[Any] = []
            for response in self.predictor.handle_stream_request(
                {
                    'type': 'propagate_in_video',
                    'session_id': session_id,
                    'start_frame_index': int(frame_index),
                    'output_prob_thresh': float(output_prob_thresh),
                }
            ):
                out_frame_index = response.get('frame_index')
                if out_frame_index is None:
                    continue
                frames_seen += 1
                outputs = _extract_stream_outputs(response)
                obj_ids = outputs.get('out_obj_ids', [])
                masks = outputs.get('out_binary_masks')
                if masks is None or len(masks) == 0:
                    frames_without_masks += 1
                    continue
                if isinstance(obj_ids, torch.Tensor):
                    obj_ids = obj_ids.detach().cpu().numpy()
                if isinstance(masks, torch.Tensor):
                    masks = masks.detach().cpu().numpy()
                if len(masks) == 0:
                    frames_without_masks += 1
                    continue
                if len(sample_mask_shapes) < 5:
                    sample_mask_shapes.append(np.asarray(masks).shape)
                best_mask = masks[0]
                if best_mask.ndim == 3:
                    best_mask = best_mask[0]
                frame_masks[int(out_frame_index)] = best_mask.astype(bool)
                frames_with_masks += 1

            self.predictor.handle_request(
                {'type': 'close_session', 'session_id': session_id}
            )

            init_mask = frame_masks.get(int(frame_index))
            if init_mask is None:
                masks = init_masks.detach().cpu().numpy() if torch.is_tensor(init_masks) else np.asarray(init_masks)
                init_mask = masks[0]
                if init_mask.ndim == 3:
                    init_mask = init_mask[0]
                init_mask = init_mask.astype(bool)

            return {
                'success': bool(frame_masks),
                'frame_masks': frame_masks,
                'initial_mask': init_mask,
                'initial_centroid': _mask_centroid(init_mask),
                'obj_ids': (
                    init_obj_ids.detach().cpu().numpy().tolist()
                    if torch.is_tensor(init_obj_ids)
                    else np.asarray(init_obj_ids).tolist()
                ),
                'used_prompt': 'bbox',
                'debug': {
                    'frames_seen': frames_seen,
                    'frames_with_masks': frames_with_masks,
                    'frames_without_masks': frames_without_masks,
                    'num_frame_masks': len(frame_masks),
                    'sample_mask_shapes': sample_mask_shapes,
                },
            }
        finally:
            if cleanup is not None:
                cleanup.cleanup()
