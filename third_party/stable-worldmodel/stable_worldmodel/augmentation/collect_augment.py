"""Collect raw HDF5 datasets from stable-worldmodel envs, then augment them."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import stable_worldmodel as swm
from stable_worldmodel.augmentation.hdf5_static_copy import (
    ObjectCopyConfig,
    augment_hdf5_dataset,
)
from stable_worldmodel.augmentation.sam3_backend import Sam3ImageBackend
from stable_worldmodel.policy import RandomPolicy


DEFAULT_DMC_EXPERT_ROOT = Path('/data/model/stable-worldmodel-dmc-experts')

DMC_EXPERT_NAME = {
    'swm/CartpoleDMControl-v0': 'cartpole',
    'swm/WalkerDMControl-v0': 'walker',
    'swm/QuadrupedDMControl-v0': 'quadruped',
    'swm/BallInCupDMControl-v0': 'ball_in_cup',
    'swm/FingerDMControl-v0': 'finger',
    'swm/HopperDMControl-v0': 'hopper',
    'swm/CheetahDMControl-v0': 'cheetah',
    'swm/ReacherDMControl-v0': 'reacher',
    'swm/PendulumDMControl-v0': 'pendulum',
}


@dataclass
class CollectAugmentConfig:
    env_id: str
    object_name: str
    num: int
    episodes: int
    output_dir: Path
    seed: int = 0
    policy_name: str = 'auto'
    colors: list[str] | None = None
    drop_original: bool = False
    debug_episode_idx: int | None = None
    debug_only: bool = False
    confidence_threshold: float = 0.5
    checkpoint_path: Path | None = None
    sam3_repo_path: Path | None = None
    expert_ckpt_root: Path | None = None
    options: dict[str, Any] | None = None
    variation: list[str] | None = None
    variation_values: dict[str, Any] | None = None
    force_variation_all: bool = False
    camera_distract_scale: float = 0.0
    camera_distract_dynamic: bool = False
    camera_distract_mode: str = 'sample'


def normalize_env_id(env_id: str) -> str:
    if '/' not in env_id:
        return f'swm/{env_id}'
    return env_id


def build_output_paths(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        'output_dir': output_dir,
        'raw_h5': output_dir / 'raw.h5',
        'augmented_h5': output_dir / 'augmented.h5',
    }


def build_collect_options(
    options: dict[str, Any] | None,
    variation: list[str] | None = None,
    variation_values: dict[str, Any] | None = None,
    force_variation_all: bool = False,
) -> dict[str, Any]:
    merged = dict(options or {})
    if force_variation_all:
        merged['variation'] = ['all']
    elif variation is not None:
        merged['variation'] = variation
    if variation_values is not None:
        merged['variation_values'] = variation_values
    return merged


def get_world_kwargs(
    env_id: str,
    camera_distract_scale: float = 0.0,
    camera_distract_dynamic: bool = False,
    camera_distract_mode: str = 'sample',
) -> dict[str, Any]:
    env_id = normalize_env_id(env_id)
    base = {
        'num_envs': 1,
        'image_shape': (224, 224),
    }
    if env_id in {'swm/TwoRoom-v1', 'swm/PushT-v1'}:
        return {**base, 'render_mode': 'rgb_array'}
    if env_id == 'swm/OGBCube-v0':
        os.environ.setdefault('MUJOCO_GL', 'egl')
        return {
            **base,
            'env_type': 'single',
            'multiview': True,
            'width': 224,
            'height': 224,
            'visualize_info': False,
            'terminate_at_goal': False,
            'mode': 'data_collection',
            'camera_distract_scale': camera_distract_scale,
            'camera_distract_dynamic': camera_distract_dynamic,
            'camera_distract_mode': camera_distract_mode,
        }
    if env_id == 'swm/OGBScene-v0':
        os.environ.setdefault('MUJOCO_GL', 'egl')
        return {
            **base,
            'multiview': False,
            'width': 224,
            'height': 224,
            'visualize_info': False,
            'terminate_at_goal': False,
            'mode': 'data_collection',
            'camera_distract_scale': camera_distract_scale,
            'camera_distract_dynamic': camera_distract_dynamic,
            'camera_distract_mode': camera_distract_mode,
        }
    if 'DMControl' in env_id:
        os.environ.setdefault('MUJOCO_GL', 'egl')
        if env_id in {
            'swm/ReacherDMControl-v0',
            'swm/FingerDMControl-v0',
            'swm/BallInCupDMControl-v0',
        }:
            return {
                **base,
                'camera_distract_scale': camera_distract_scale,
                'camera_distract_dynamic': camera_distract_dynamic,
                'camera_distract_mode': camera_distract_mode,
            }
        return base
    return {**base, 'render_mode': 'rgb_array'}


def _build_dmc_expert(env_id: str, ckpt_root: Path | None):
    from stable_worldmodel.envs.dmcontrol import ExpertPolicy

    env_name = DMC_EXPERT_NAME.get(env_id)
    root = ckpt_root or DEFAULT_DMC_EXPERT_ROOT
    if env_name is None:
        raise KeyError(f'No DMC expert mapping for {env_id}')
    policy_dir = root / env_name / 'expert_policy'
    return ExpertPolicy(
        ckpt_path=policy_dir / 'expert_policy.zip',
        vec_normalize_path=policy_dir / 'vec_normalize.pkl',
    )


def create_policy(
    env_id: str,
    policy_name: str = 'auto',
    expert_ckpt_root: Path | None = None,
):
    env_id = normalize_env_id(env_id)
    if policy_name == 'random':
        return RandomPolicy()

    if env_id == 'swm/PushT-v1':
        from stable_worldmodel.envs.pusht import WeakPolicy

        if policy_name in {'auto', 'weak'}:
            return WeakPolicy(dist_constraint=100)
        raise ValueError(f'Unsupported policy {policy_name} for {env_id}')

    if env_id == 'swm/TwoRoom-v1':
        from stable_worldmodel.envs.two_room import ExpertPolicy

        if policy_name in {'auto', 'expert'}:
            return ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05)
        raise ValueError(f'Unsupported policy {policy_name} for {env_id}')

    if env_id == 'swm/OGBCube-v0' or env_id == 'swm/OGBScene-v0':
        from stable_worldmodel.envs.ogbench import ExpertPolicy

        if policy_name in {'auto', 'expert'}:
            return ExpertPolicy()
        raise ValueError(f'Unsupported policy {policy_name} for {env_id}')

    if env_id == 'swm/SimpleNavigation-v0':
        from stable_worldmodel.envs.simple_nav import ExpertPolicy

        if policy_name in {'auto', 'expert'}:
            return ExpertPolicy()
        raise ValueError(f'Unsupported policy {policy_name} for {env_id}')

    if 'DMControl' in env_id:
        if policy_name == 'random':
            return RandomPolicy()
        if policy_name in {'auto', 'expert'}:
            return _build_dmc_expert(env_id, expert_ckpt_root)
        raise ValueError(f'Unsupported policy {policy_name} for {env_id}')

    if policy_name == 'auto':
        return RandomPolicy()
    raise ValueError(f'Unsupported policy {policy_name} for {env_id}')


def default_world_factory(env_id: str, world_kwargs: dict[str, Any]):
    return swm.World(env_id, **world_kwargs)


def collect_and_augment_dataset(
    config: CollectAugmentConfig,
    world_factory: Callable[[str, dict[str, Any]], Any] = default_world_factory,
    policy_factory: Callable[[str, str, Path | None], Any] = create_policy,
    backend_factory: Callable[[], Any] | None = None,
    augmenter: Callable[[ObjectCopyConfig, Any], dict[str, Any]] = augment_hdf5_dataset,
) -> dict[str, Any]:
    env_id = normalize_env_id(config.env_id)
    paths = build_output_paths(config.output_dir)

    world_kwargs = get_world_kwargs(
        env_id,
        camera_distract_scale=config.camera_distract_scale,
        camera_distract_dynamic=config.camera_distract_dynamic,
        camera_distract_mode=config.camera_distract_mode,
    )
    world = world_factory(env_id, world_kwargs)
    policy = policy_factory(env_id, config.policy_name, config.expert_ckpt_root)
    world.set_policy(policy)

    collect_options = build_collect_options(
        config.options,
        variation=config.variation,
        variation_values=config.variation_values,
        force_variation_all=config.force_variation_all,
    )
    world.record_dataset(
        dataset_name='raw',
        episodes=config.episodes,
        seed=config.seed,
        cache_dir=paths['output_dir'],
        options=collect_options,
    )
    if hasattr(world, 'close'):
        world.close()

    augment_config = ObjectCopyConfig(
        input_h5=paths['raw_h5'],
        output_h5=paths['augmented_h5'],
        env_id=env_id,
        object_name=config.object_name,
        num=config.num,
        seed=config.seed,
        colors=config.colors,
        drop_original=config.drop_original,
        debug_episode_idx=config.debug_episode_idx,
        debug_only=config.debug_only,
        confidence_threshold=config.confidence_threshold,
    )

    if backend_factory is None:
        backend = Sam3ImageBackend(
            checkpoint_path=config.checkpoint_path,
            sam3_repo_path=config.sam3_repo_path,
            confidence_threshold=config.confidence_threshold,
        )
    else:
        backend = backend_factory()

    result = augmenter(augment_config, backend=backend)
    result.update(
        {
            'raw_h5': str(paths['raw_h5']),
            'augmented_h5': str(paths['augmented_h5']),
            'output_dir': str(paths['output_dir']),
            'policy_name': config.policy_name,
        }
    )
    return result
