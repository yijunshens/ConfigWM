from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np

COLOR_RGB = {
    "red": np.array([255, 0, 0], dtype=np.uint8),
    "blue": np.array([0, 92, 255], dtype=np.uint8),
    "green": np.array([25, 210, 90], dtype=np.uint8),
    "orange": np.array([255, 135, 0], dtype=np.uint8),
    "purple": np.array([175, 70, 255], dtype=np.uint8),
    "cyan": np.array([22, 198, 216], dtype=np.uint8),
    "yellow": np.array([255, 220, 35], dtype=np.uint8),
    "pink": np.array([255, 85, 180], dtype=np.uint8),
    "brown": np.array([150, 88, 45], dtype=np.uint8),
}


def blend_gaussian_patch(
    frame: np.ndarray, center_xy: np.ndarray, radius: float, rgb: np.ndarray
) -> None:
    """Match the dataset builder's normalized Gaussian alpha compositor."""
    height, width = frame.shape[:2]
    cx = float(center_xy[0])
    cy = float(center_xy[1])
    std = max(1e-06, float(radius))
    pad = int(np.ceil(std * 4.0))
    x0 = max(0, int(np.floor(cx)) - pad)
    x1 = min(width, int(np.floor(cx)) + pad + 1)
    y0 = max(0, int(np.floor(cy)) - pad)
    y1 = min(height, int(np.floor(cy)) + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy = np.arange(y0, y1, dtype=np.float32)[:, None]
    xx = np.arange(x0, x1, dtype=np.float32)[None, :]
    dist2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
    alpha = np.exp(-dist2 / (2.0 * std * std)).astype(np.float32)
    max_alpha = float(alpha.max())
    if max_alpha > 0:
        alpha /= max_alpha
    patch = frame[y0:y1, x0:x1].astype(np.float32)
    color = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3)
    patch = patch * (1.0 - alpha[..., None]) + color * alpha[..., None]
    frame[y0:y1, x0:x1] = patch.astype(frame.dtype)


def make_instance_overlay_class(legacy_cls: type) -> type:
    """Create a drop-in replacement for the legacy RGB-difference overlay."""

    class TwoRoomInstanceMetadataOverlay(legacy_cls):
        """Render donor distractors from centers, colors, radii, and order."""

        def _build_index(self) -> None:
            import h5py

            self._scene_index = {}
            for path in self._h5_paths():
                metadata_paths = sorted(path.parent.glob("core_*_scene_metadata.jsonl"))
                rows: dict[int, dict[str, Any]] = {}
                with metadata_paths[0].open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        rows[int(row["local_scene_idx"])] = row
                with h5py.File(path, "r") as h5:
                    original_episode_ids = np.asarray(
                        h5["scene_original_episode_idx"], dtype=np.int64
                    )
                    source_offsets = np.asarray(
                        h5["scene_source_offset"], dtype=np.int64
                    )
                    episode_lengths = np.asarray(h5["scene_ep_len"], dtype=np.int64)
                    source_height, source_width = map(
                        int, h5["source_pixels"].shape[1:3]
                    )
                    center_object_count = int(h5["object_centers_xy"].shape[1])
                for scene_idx, original_episode_idx in enumerate(
                    original_episode_ids.tolist()
                ):
                    row = rows.get(scene_idx)
                    objects = list(row.get("objects", []))
                    self._scene_index[int(original_episode_idx)] = {
                        "path": path,
                        "source_offset": int(source_offsets[scene_idx]),
                        "ep_len": int(episode_lengths[scene_idx]),
                        "objects": objects,
                        "source_height": source_height,
                        "source_width": source_width,
                    }
            missing = [
                episode
                for episode in self.episodes_idx
                if episode not in self._scene_index
            ]
            too_short = []
            for episode_idx, start_step in zip(self.episodes_idx, self.start_steps):
                meta = self._scene_index[int(episode_idx)]
                required_step = int(start_step) + self.goal_offset_steps - 1
                if int(meta["ep_len"]) <= required_step:
                    too_short.append(
                        (int(episode_idx), int(meta["ep_len"]), required_step)
                    )

        def _preload_selected_episodes(self) -> None:
            self._episode_centers: dict[int, np.ndarray] = {}
            for episode_idx in dict.fromkeys(self.episodes_idx):
                meta = self._scene_index[int(episode_idx)]
                h5 = self._handle(Path(meta["path"]))
                offset = int(meta["source_offset"])
                length = int(meta["ep_len"])
                centers = np.asarray(
                    h5["object_centers_xy"][offset : offset + length], dtype=np.float32
                )
                self._episode_centers[int(episode_idx)] = centers

        def apply_frame_batch(
            self, images: np.ndarray, *, steps: np.ndarray | list[int]
        ) -> np.ndarray:
            if not self.enabled:
                return images
            out = np.asarray(images).copy()
            is_float_01 = (
                np.issubdtype(out.dtype, np.floating) and float(np.nanmax(out)) <= 1.5
            )
            for env_idx in range(min(out.shape[0], len(self.episodes_idx))):
                episode_idx = int(self.episodes_idx[env_idx])
                meta = self._scene_index[episode_idx]
                donor_step = int(self.start_steps[env_idx]) + (
                    int(steps[env_idx]) - int(self.base_start_steps[env_idx])
                )
                donor_step = int(
                    np.clip(donor_step, 0, max(0, int(meta["ep_len"]) - 1))
                )
                centers = self._episode_centers[episode_idx][donor_step]
                height, width = out[env_idx].shape[:2]
                scale_x = float(width) / float(meta["source_width"])
                scale_y = float(height) / float(meta["source_height"])
                radius_scale = float(np.sqrt(scale_x * scale_y))
                for obj_idx, obj in enumerate(meta["objects"][1:], start=1):
                    center = np.array(
                        [centers[obj_idx, 0] * scale_x, centers[obj_idx, 1] * scale_y],
                        dtype=np.float32,
                    )
                    color = COLOR_RGB[str(obj["color"])].astype(np.float32)
                    if is_float_01:
                        color /= 255.0
                    blend_gaussian_patch(
                        out[env_idx], center, float(obj["radius"]) * radius_scale, color
                    )
            return out

        def metadata(self) -> dict[str, Any]:
            return {
                "enabled": self.enabled,
                "render_mode": "tworoom_instance_metadata_gaussian_v1",
                "dataset_dir": str(self.dataset_dir),
                "variant": self.dataset_dir.name,
                "base_episodes_idx": [int(value) for value in self.base_episodes_idx],
                "base_start_steps": [int(value) for value in self.base_start_steps],
                "donor_episodes_idx": [int(value) for value in self.episodes_idx],
                "donor_start_steps": [int(value) for value in self.start_steps],
                "goal_offset_steps": int(self.goal_offset_steps),
                "eval_budget": int(self.eval_budget),
                "palette": {name: value.tolist() for name, value in COLOR_RGB.items()},
            }

    TwoRoomInstanceMetadataOverlay.__name__ = "TwoRoomInstanceMetadataOverlay"
    return TwoRoomInstanceMetadataOverlay
