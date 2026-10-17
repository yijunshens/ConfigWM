"""Continuous camera pose distraction for OGBench environments.

This variant keeps the task subject visible by orbiting the camera around the
actual task-relevant object or handle, rather than around a coarse scene center.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stable_worldmodel.envs.dmcontrol.camera_distract import (
    cart2sphere,
    clip_cam_position,
    get_lookat_xmat_no_roll,
    rotvec2mat,
    sphere2cart,
)


@dataclass
class OGBenchCameraDistractParams:
    horizontal_delta: float
    vertical_delta: float
    roll_delta: float
    zoom_in_percent: float
    zoom_out_percent: float


def params_from_scale(scale: float) -> OGBenchCameraDistractParams:
    scale = float(np.clip(scale, 0.0, 1.0))
    return OGBenchCameraDistractParams(
        horizontal_delta=np.pi * 0.14 * scale,
        vertical_delta=np.pi * 0.08 * scale,
        roll_delta=np.pi * 0.06 * scale,
        zoom_in_percent=0.08 * scale,
        zoom_out_percent=0.16 * scale,
    )


class OGBenchCameraDistractor:
    def __init__(
        self,
        env,
        camera_name: str,
        scale: float,
        mode: str = 'sample',
        seed: int | None = None,
    ) -> None:
        self._env = env
        self._camera_name = camera_name
        self._scale = float(np.clip(scale, 0.0, 1.0))
        self._mode = str(mode).lower()
        if self._mode not in {'sample', 'calibrated', 'extreme'}:
            raise ValueError(f'Unsupported OGB camera distract mode: {mode}')
        self._params = params_from_scale(self._scale)
        self._random = np.random.RandomState(seed)

        self._lookat = self._infer_lookat_point()
        self._start_pos = np.array(self._env.cameras[self._camera_name]['pos'], dtype=np.float64)
        rel = self._start_pos - self._lookat
        start_r, start_h, start_v = cart2sphere(rel)
        self._start_radius = float(start_r)
        self._start_h = float(start_h)
        self._start_v = float(start_v)

        self._min_h = self._start_h - self._params.horizontal_delta
        self._max_h = self._start_h + self._params.horizontal_delta
        self._min_v = max(0.10, self._start_v - self._params.vertical_delta)
        self._max_v = min(np.pi - 0.10, self._start_v + self._params.vertical_delta)
        self._min_r = max(0.05, self._start_radius * (1.0 - self._params.zoom_in_percent))
        self._max_r = self._start_radius * (1.0 + self._params.zoom_out_percent)
        self._min_roll = -self._params.roll_delta
        self._max_roll = self._params.roll_delta
        self._resolved_pose: tuple[float, float, float, float] | None = None

    @property
    def enabled(self) -> bool:
        return self._scale > 0.0

    def _subject_cube_pos(self, block_idx: int = 0) -> np.ndarray:
        return np.array(self._env._data.joint(f'object_joint_{block_idx}').qpos[:3], dtype=np.float64)

    def _infer_lookat_point(self) -> np.ndarray:
        if hasattr(self._env, '_target_task'):
            task = getattr(self._env, '_target_task')
            if task == 'cube':
                block_idx = int(getattr(self._env, '_target_block', 0))
                return self._subject_cube_pos(block_idx)
            if task == 'button':
                button_idx = int(getattr(self._env, '_target_button', 0))
                qpos = np.array(self._env._data.joint(f'buttonbox_joint_{button_idx}').qpos.copy(), dtype=np.float64)
                return qpos[:3]
            if task == 'drawer' and hasattr(self._env, '_drawer_site_id'):
                return np.array(self._env._data.site_xpos[self._env._drawer_site_id], dtype=np.float64)
            if task == 'window' and hasattr(self._env, '_window_site_id'):
                return np.array(self._env._data.site_xpos[self._env._window_site_id], dtype=np.float64)
        if hasattr(self._env, '_target_block'):
            return self._subject_cube_pos(int(getattr(self._env, '_target_block', 0)))
        if hasattr(self._env, '_object_sampling_bounds'):
            bounds = np.asarray(self._env._object_sampling_bounds, dtype=np.float64)
            xy = np.mean(bounds, axis=0)
            return np.array([xy[0], xy[1], 0.05], dtype=np.float64)
        return np.array([0.4, 0.0, 0.05], dtype=np.float64)

    def _sample_value(self, center: float, lower: float, upper: float) -> float:
        if self._mode == 'sample':
            return float(self._random.uniform(lower, upper))
        choose_upper = bool(self._random.randint(0, 2))
        target = upper if choose_upper else lower
        if self._mode == 'extreme':
            return float(target)
        return float(center + (target - center) * self._scale)

    def _resolve_pose(self) -> tuple[float, float, float, float]:
        if self._resolved_pose is None:
            h = self._sample_value(self._start_h, self._min_h, self._max_h)
            v = self._sample_value(self._start_v, self._min_v, self._max_v)
            r = self._sample_value(self._start_radius, self._min_r, self._max_r)
            roll = self._sample_value(0.0, self._min_roll, self._max_roll)
            self._resolved_pose = (h, v, r, roll)
        return self._resolved_pose

    def apply(self) -> None:
        if not self.enabled:
            return
        h, v, r, roll = self._resolve_pose()
        sphere = clip_cam_position(
            np.array([r, h, v], dtype=np.float64),
            self._min_r,
            self._max_r,
            self._min_h,
            self._max_h,
            self._min_v,
            self._max_v,
        )
        camera_pos = self._lookat + sphere2cart(sphere)
        lookat_mat = get_lookat_xmat_no_roll(self._lookat, camera_pos)
        roll_mat = rotvec2mat(roll, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        xmat = np.dot(np.reshape(lookat_mat, (3, 3)), roll_mat)

        cam_id = mujoco_camera_id(self._env, self._camera_name)
        self._env._data.cam_xpos[cam_id] = camera_pos
        self._env._data.cam_xmat[cam_id] = xmat.reshape(-1)


def _xmat_from_xyaxes(xyaxes) -> np.ndarray:
    xaxis = np.array(xyaxes[:3], dtype=np.float64)
    yaxis = np.array(xyaxes[3:], dtype=np.float64)
    zaxis = np.cross(xaxis, yaxis)
    zaxis = zaxis / (np.linalg.norm(zaxis) + 1e-8)
    return np.stack([xaxis, yaxis, zaxis], axis=0)


def mujoco_camera_id(env, camera_name: str) -> int:
    return env._model.camera(camera_name).id
