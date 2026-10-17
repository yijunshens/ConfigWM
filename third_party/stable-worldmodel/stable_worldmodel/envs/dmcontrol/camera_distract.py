"""Continuous camera pose distraction for DMControl environments.

This is inspired by the camera wrapper from `distracting_control`, but exposed
as a lightweight utility for stable-worldmodel DMControl wrappers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


CAMERA_MODES = ['fixed', 'track', 'trackcom', 'targetbody', 'targetbodycom']


def _mat_from_theta(cos_theta: float, sin_theta: float, axis: np.ndarray) -> np.ndarray:
    row1 = [
        cos_theta + axis[0] ** 2.0 * (1.0 - cos_theta),
        axis[0] * axis[1] * (1.0 - cos_theta) - axis[2] * sin_theta,
        axis[0] * axis[2] * (1.0 - cos_theta) + axis[1] * sin_theta,
    ]
    row2 = [
        axis[1] * axis[0] * (1.0 - cos_theta) + axis[2] * sin_theta,
        cos_theta + axis[1] ** 2.0 * (1.0 - cos_theta),
        axis[1] * axis[2] * (1.0 - cos_theta) - axis[0] * sin_theta,
    ]
    row3 = [
        axis[2] * axis[0] * (1.0 - cos_theta) - axis[1] * sin_theta,
        axis[2] * axis[1] * (1.0 - cos_theta) + axis[0] * sin_theta,
        cos_theta + axis[2] ** 2.0 * (1.0 - cos_theta),
    ]
    return np.stack([row1, row2, row3])


def rotvec2mat(theta: float, vec: np.ndarray) -> np.ndarray:
    axis = vec / (np.sqrt(np.sum(vec**2.0)) + 1e-8)
    return _mat_from_theta(np.cos(theta), np.sin(theta), axis)


def cart2sphere(cart: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.sum(cart**2.0))
    h_angle = np.arctan2(cart[1], cart[0])
    v_angle = np.arctan2(np.sqrt(np.sum(cart[:2] ** 2.0)), cart[2])
    return np.array([r, h_angle, v_angle], dtype=np.float64)


def sphere2cart(sphere: np.ndarray) -> np.ndarray:
    r, h_angle, v_angle = sphere
    x = r * np.sin(v_angle) * np.cos(h_angle)
    y = r * np.sin(v_angle) * np.sin(h_angle)
    z = r * np.cos(v_angle)
    return np.array([x, y, z], dtype=np.float64)


def clip_cam_position(
    position: np.ndarray,
    min_radius: float,
    max_radius: float,
    min_h_angle: float,
    max_h_angle: float,
    min_v_angle: float,
    max_v_angle: float,
) -> np.ndarray:
    clipped = np.array(position, dtype=np.float64, copy=True)
    clipped[0] = np.clip(clipped[0], min_radius, max_radius)
    clipped[1] = np.clip(clipped[1], min_h_angle, max_h_angle)
    clipped[2] = np.clip(clipped[2], min_v_angle, max_v_angle)
    return clipped


def get_lookat_xmat(agent_pos: np.ndarray, camera_pos: np.ndarray) -> np.ndarray:
    delta_vec = agent_pos - camera_pos
    delta_vec /= np.sqrt(np.sum(delta_vec**2.0)) + 1e-8
    y_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    axis = np.cross(y_vec, delta_vec)
    sin_theta = np.sqrt(np.sum(axis**2.0))
    cos_theta = float(np.dot(delta_vec, y_vec))
    axis /= np.sqrt(np.sum(axis**2.0)) + 1e-4
    return _mat_from_theta(cos_theta, sin_theta, axis)


def get_lookat_xmat_no_roll(
    agent_pos: np.ndarray, camera_pos: np.ndarray
) -> np.ndarray:
    delta_vec = agent_pos - camera_pos
    delta_vec /= np.sqrt(np.sum(delta_vec**2.0)) + 1e-8
    yaw = np.arctan2(delta_vec[0], delta_vec[1])
    pitch = np.arctan2(delta_vec[2], np.sqrt(np.sum(delta_vec[:2] ** 2.0)))
    pitch += np.pi / 2.0
    cos_pitch, _, cos_yaw = np.cos([pitch, 0.0, -yaw])
    sin_pitch, _, sin_yaw = np.sin([pitch, 0.0, -yaw])
    return np.array(
        [
            [
                cos_yaw,
                -sin_yaw * cos_pitch,
                sin_yaw * sin_pitch,
            ],
            [
                sin_yaw,
                cos_yaw * cos_pitch,
                -cos_yaw * sin_pitch,
            ],
            [0.0, sin_pitch, cos_pitch],
        ],
        dtype=np.float64,
    )


def get_lookat_point(physics, camera_id: int) -> np.ndarray:
    dist_to_agent = (
        physics.named.data.cam_xpos[camera_id] - physics.named.data.subtree_com[1]
    )
    dist_to_agent = np.sqrt(np.sum(dist_to_agent**2.0))
    initial_viewing_mat = copy.deepcopy(physics.named.data.cam_xmat[camera_id])
    initial_viewing_mat = np.reshape(initial_viewing_mat, (3, 3))
    z_vec = np.array([0.0, 0.0, -dist_to_agent], dtype=np.float64)
    rotated_vec = np.dot(initial_viewing_mat, z_vec)
    return rotated_vec + physics.named.data.cam_xpos[camera_id]


@dataclass
class CameraDistractParams:
    horizontal_delta: float
    vertical_delta: float
    roll_delta: float
    vel_std: float
    max_vel: float
    roll_std: float
    max_roll_vel: float
    max_zoom_in_percent: float
    max_zoom_out_percent: float
    limit_to_upper_quadrant: bool


def params_from_scale(
    domain_name: str, scale: float, dynamic: bool
) -> CameraDistractParams:
    scale = float(np.clip(scale, 0.0, 1.0))
    return CameraDistractParams(
        vertical_delta=np.pi / 2.0 * scale,
        horizontal_delta=np.pi / 2.0 * scale,
        roll_delta=np.pi / 2.0 * scale,
        vel_std=0.1 * scale if dynamic else 0.0,
        max_vel=0.4 * scale if dynamic else 0.0,
        roll_std=np.pi / 300.0 * scale if dynamic else 0.0,
        max_roll_vel=np.pi / 50.0 * scale if dynamic else 0.0,
        max_zoom_in_percent=0.5 * scale,
        max_zoom_out_percent=1.5 * scale,
        limit_to_upper_quadrant='reacher' not in domain_name,
    )


class DMControlCameraDistractor:
    def __init__(
        self,
        env,
        camera_id: int,
        domain_name: str,
        scale: float,
        dynamic: bool,
        mode: str = 'sample',
        seed: int | None = None,
    ) -> None:
        self._env = env
        self._camera_id = int(camera_id)
        self._domain_name = domain_name
        self._scale = float(np.clip(scale, 0.0, 1.0))
        self._dynamic = bool(dynamic)
        self._mode = str(mode).lower()
        if self._mode not in {'sample', 'extreme', 'calibrated'}:
            raise ValueError(f'Unsupported camera distract mode: {mode}')
        self._params = params_from_scale(domain_name, self._scale, self._dynamic)
        self._random_state = np.random.RandomState(seed=seed)

        self._camera_type: str | None = None
        self._cam_initial_lookat_point: np.ndarray | None = None
        self._camera_vel: np.ndarray | None = None
        self._max_h_angle: float | None = None
        self._max_v_angle: float | None = None
        self._min_h_angle: float | None = None
        self._min_v_angle: float | None = None
        self._radius: float | None = None
        self._roll_vel: float | None = None
        self._vel_scaling: float | None = None
        self._start_h_angle: float | None = None
        self._start_v_angle: float | None = None
        self._start_radius: float | None = None
        self._h_angle: float | None = None
        self._v_angle: float | None = None
        self._roll: float | None = None
        self._min_radius: float | None = None
        self._max_radius: float | None = None
        self._max_roll: float | None = None
        self._min_roll: float | None = None

    @property
    def enabled(self) -> bool:
        return self._scale > 0.0

    def _sample_initial_value(
        self,
        center: float,
        lower: float,
        upper: float,
    ) -> float:
        if self._mode == 'sample':
            return float(self._random_state.uniform(lower, upper))
        choose_upper = bool(self._random_state.randint(0, 2))
        target = upper if choose_upper else lower
        return float(center + (target - center) * self._scale)

    def setup_camera(self) -> None:
        if not self.enabled:
            return

        mode_idx = int(self._env.physics.model.cam_mode[self._camera_id])
        self._camera_type = CAMERA_MODES[mode_idx]
        self._cam_initial_lookat_point = get_lookat_point(
            self._env.physics, self._camera_id
        )

        start_pos = np.array(
            self._env.physics.named.data.cam_xpos[self._camera_id], copy=True
        )
        if self._camera_type != 'fixed':
            start_pos -= self._env.physics.named.data.subtree_com[1]

        start_r, start_h_angle, start_v_angle = cart2sphere(start_pos)
        self._start_radius = float(start_r)
        self._start_h_angle = float(start_h_angle)
        self._start_v_angle = float(start_v_angle)
        self._vel_scaling = start_r / 4.0
        self._max_h_angle = start_h_angle + self._params.horizontal_delta
        self._min_h_angle = start_h_angle - self._params.horizontal_delta
        self._max_v_angle = start_v_angle + self._params.vertical_delta
        self._min_v_angle = start_v_angle - self._params.vertical_delta

        if self._params.limit_to_upper_quadrant:
            self._max_v_angle = min(self._max_v_angle, np.pi / 2.0)
            self._min_v_angle = max(self._min_v_angle, 0.0)
            self._max_h_angle = min(self._max_h_angle, 0.0)
            self._min_h_angle = max(self._min_h_angle, -np.pi)

        self._max_roll = self._params.roll_delta
        self._min_roll = -self._params.roll_delta
        self._min_radius = max(
            start_r - start_r * self._params.max_zoom_in_percent, 0.0
        )
        self._max_radius = start_r + start_r * self._params.max_zoom_out_percent

        self._h_angle = self._sample_initial_value(
            self._start_h_angle,
            self._min_h_angle,
            self._max_h_angle,
        )
        self._v_angle = self._sample_initial_value(
            self._start_v_angle,
            self._min_v_angle,
            self._max_v_angle,
        )
        self._radius = self._sample_initial_value(
            self._start_radius,
            self._min_radius,
            self._max_radius,
        )
        self._roll = self._sample_initial_value(0.0, self._min_roll, self._max_roll)

        vel = self._random_state.randn(3)
        vel /= np.sqrt(np.sum(vel**2.0)) + 1e-8
        vel *= self._random_state.uniform(0.0, self._params.max_vel)
        self._camera_vel = vel
        self._roll_vel = self._random_state.uniform(
            -self._params.max_roll_vel, self._params.max_roll_vel
        )

    def apply(self) -> None:
        if not self.enabled:
            return
        if self._camera_type is None:
            self.setup_camera()

        assert self._camera_vel is not None
        assert self._h_angle is not None
        assert self._v_angle is not None
        assert self._radius is not None
        assert self._roll is not None
        assert self._roll_vel is not None
        assert self._vel_scaling is not None
        assert self._min_radius is not None
        assert self._max_radius is not None
        assert self._min_h_angle is not None
        assert self._max_h_angle is not None
        assert self._min_v_angle is not None
        assert self._max_v_angle is not None
        assert self._cam_initial_lookat_point is not None
        assert self._camera_type is not None

        if self._dynamic:
            vel_delta = self._random_state.randn(3)
            self._camera_vel += vel_delta * self._params.vel_std * self._vel_scaling
            self._roll_vel += self._random_state.randn() * self._params.roll_std

            vel_norm = np.sqrt(np.sum(self._camera_vel**2.0))
            if vel_norm > self._params.max_vel * self._vel_scaling:
                self._camera_vel *= (self._params.max_vel * self._vel_scaling) / (
                    vel_norm + 1e-8
                )
            self._roll_vel = np.clip(
                self._roll_vel,
                -self._params.max_roll_vel,
                self._params.max_roll_vel,
            )

        cart_cam_pos = sphere2cart(
            np.array([self._radius, self._h_angle, self._v_angle], dtype=np.float64)
        )
        sphere_cam_pos2 = cart2sphere(cart_cam_pos + self._camera_vel)
        sphere_cam_pos2 = clip_cam_position(
            sphere_cam_pos2,
            self._min_radius,
            self._max_radius,
            self._min_h_angle,
            self._max_h_angle,
            self._min_v_angle,
            self._max_v_angle,
        )

        self._camera_vel = sphere2cart(sphere_cam_pos2) - cart_cam_pos
        self._radius, self._h_angle, self._v_angle = sphere_cam_pos2

        roll2 = np.clip(self._roll + self._roll_vel, self._min_roll, self._max_roll)
        self._roll_vel = roll2 - self._roll
        self._roll = roll2

        cart_cam_pos = sphere2cart(sphere_cam_pos2)
        if self._params.limit_to_upper_quadrant:
            lookat_method = get_lookat_xmat_no_roll
        else:
            lookat_method = get_lookat_xmat

        if self._camera_type == 'fixed':
            lookat_mat = lookat_method(
                self._cam_initial_lookat_point,
                cart_cam_pos,
            )
        else:
            cart_cam_pos += self._env.physics.named.data.subtree_com[1]
            lookat_mat = lookat_method(
                get_lookat_point(self._env.physics, self._camera_id),
                cart_cam_pos,
            )

        roll_mat = rotvec2mat(self._roll, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        xmat = np.dot(np.reshape(lookat_mat, (3, 3)), roll_mat)
        self._env.physics.named.data.cam_xpos[self._camera_id] = cart_cam_pos
        self._env.physics.named.data.cam_xmat[self._camera_id] = xmat.flatten()
