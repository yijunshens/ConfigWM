"""Registry for environment-specific object augmentation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


COLOR_NAME_TO_RGB = {
    'red': np.array([255, 64, 64], dtype=np.uint8),
    'green': np.array([64, 200, 96], dtype=np.uint8),
    'blue': np.array([64, 128, 255], dtype=np.uint8),
    'yellow': np.array([255, 220, 64], dtype=np.uint8),
    'cyan': np.array([64, 220, 255], dtype=np.uint8),
    'magenta': np.array([220, 64, 255], dtype=np.uint8),
    'orange': np.array([255, 160, 64], dtype=np.uint8),
    'white': np.array([240, 240, 240], dtype=np.uint8),
    'gray': np.array([128, 128, 128], dtype=np.uint8),
    'purple': np.array([160, 96, 220], dtype=np.uint8),
    'pink': np.array([192, 96, 96], dtype=np.uint8),
    'brown': np.array([170, 110, 70], dtype=np.uint8),
    'black': np.array([16, 16, 16], dtype=np.uint8),
}


@dataclass(frozen=True)
class ObjectSpec:
    object_name: str
    prompt_templates: tuple[str, ...]
    shape_hint: str
    variation_color_key: str | None = None
    variation_color_index: int | None = None
    default_color_name: str | None = None
    focus_template: str = 'focus on the {descriptor} to finish the task'


def normalize_env_id(env_id: str) -> str:
    if '/' not in env_id:
        env_id = f'swm/{env_id}'
    return env_id


OBJECT_REGISTRY: dict[str, dict[str, ObjectSpec]] = {
    'swm/TwoRoom-v1': {
        'ball': ObjectSpec(
            object_name='ball',
            prompt_templates=('ball', '{color} ball'),
            shape_hint='ball',
            variation_color_key='variation.agent.color',
            default_color_name='red',
        ),
    },
    'swm/PushT-v1': {
        'block': ObjectSpec(
            object_name='block',
            prompt_templates=('T-shaped block', '{color} block', 'block'),
            shape_hint='block',
            variation_color_key='variation.block.color',
        ),
        'agent': ObjectSpec(
            object_name='agent',
            prompt_templates=('agent', '{color} agent', 'ball', '{color} ball'),
            shape_hint='ball',
            variation_color_key='variation.agent.color',
        ),
    },
    'swm/BallInCupDMControl-v0': {
        'ball': ObjectSpec(
            object_name='ball',
            prompt_templates=('ball', '{color} ball', 'circle', '{color} circle'),
            shape_hint='ball',
            variation_color_key='variation.ball.color',
        ),
        'u_shaped_holder': ObjectSpec(
            object_name='u_shaped_holder',
            prompt_templates=(
                'U-shaped holder',
                'small {color} U-shaped holder',
                '{color} U-shaped holder',
            ),
            shape_hint='U-shaped holder',
            variation_color_key='variation.agent.color',
        ),
    },
    'swm/FingerDMControl-v0': {
        'robotic_finger': ObjectSpec(
            object_name='robotic_finger',
            prompt_templates=('robotic finger', '{color} robotic finger'),
            shape_hint='robotic finger',
            variation_color_key='variation.agent.color',
        ),
        'capsule': ObjectSpec(
            object_name='capsule',
            prompt_templates=(
                'capsule',
                'capsule-shaped object',
                'spinner',
                '{color} capsule',
                '{color} capsule-shaped object',
                '{color} spinner',
            ),
            shape_hint='capsule-shaped object',
            variation_color_key='variation.spinner.color',
        ),
    },
    'swm/ReacherDMControl-v0': {
        'v_shaped_object': ObjectSpec(
            object_name='v_shaped_object',
            prompt_templates=('V-shaped object', 'v shape', '{color} V-shaped object', '{color} v shape'),
            shape_hint='V-shaped object',
            variation_color_key='variation.agent.color',
        ),
    },
    'swm/OGBCube-v0': {
        'cube': ObjectSpec(
            object_name='cube',
            prompt_templates=('cube', 'block', '{color} cube', '{color} block'),
            shape_hint='cube',
            variation_color_key='variation.cube.color',
        ),
    },
    'swm/OGBScene-v0': {
        'cube': ObjectSpec(
            object_name='cube',
            prompt_templates=(
                'small box',
                'small cube',
                '{color} small box',
                '{color} small cube',
            ),
            shape_hint='small box',
            variation_color_key='variation.cube.color',
        ),
        'button': ObjectSpec(
            object_name='button',
            prompt_templates=('button', '{color} button'),
            shape_hint='button',
            variation_color_key='variation.lock_color',
            variation_color_index=0,
        ),
        'drawer_handle': ObjectSpec(
            object_name='drawer_handle',
            prompt_templates=('drawer handle', '{color} drawer handle'),
            shape_hint='drawer handle',
            variation_color_key='variation.lock_color',
            variation_color_index=0,
        ),
    },
}


SMOKE_ENV_OBJECTS: dict[str, tuple[str, str]] = {
    'ballincup_ball': ('swm/BallInCupDMControl-v0', 'ball'),
    'finger_capsule': ('swm/FingerDMControl-v0', 'capsule'),
    'reacher_v_shape': ('swm/ReacherDMControl-v0', 'v_shaped_object'),
    'ogbcube_cube': ('swm/OGBCube-v0', 'cube'),
    'ogbscene_cube': ('swm/OGBScene-v0', 'cube'),
    'pusht_block': ('swm/PushT-v1', 'block'),
}


def get_object_spec(env_id: str, object_name: str) -> ObjectSpec:
    normalized_env = normalize_env_id(env_id)
    if normalized_env not in OBJECT_REGISTRY:
        raise KeyError(f'unsupported env for augmentation: {normalized_env}')
    if object_name not in OBJECT_REGISTRY[normalized_env]:
        raise KeyError(
            f"unsupported object '{object_name}' for env {normalized_env}"
        )
    return OBJECT_REGISTRY[normalized_env][object_name]


def get_smoke_object_spec(smoke_env_name: str) -> ObjectSpec:
    if smoke_env_name not in SMOKE_ENV_OBJECTS:
        raise KeyError(f'unsupported smoke env for prompt resolution: {smoke_env_name}')
    env_id, object_name = SMOKE_ENV_OBJECTS[smoke_env_name]
    return get_object_spec(env_id, object_name)


def _normalize_rgb(rgb: np.ndarray | list[float] | list[int] | None) -> np.ndarray | None:
    if rgb is None:
        return None
    arr = np.asarray(rgb, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    arr = arr[:3]
    if np.max(arr) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def describe_color_name(rgb: np.ndarray | list[float] | list[int] | None) -> str | None:
    arr = _normalize_rgb(rgb)
    if arr is None:
        return None
    best_name = None
    best_dist = None
    for name, ref in COLOR_NAME_TO_RGB.items():
        dist = float(np.linalg.norm(arr.astype(np.float32) - ref.astype(np.float32)))
        if best_dist is None or dist < best_dist:
            best_name = name
            best_dist = dist
    return best_name


def build_prompt_candidates(
    spec: ObjectSpec,
    color_name: str | None = None,
    shape_hint: str | None = None,
) -> list[str]:
    resolved_color_name = color_name or spec.default_color_name
    shape = shape_hint or spec.shape_hint
    prompts: list[str] = []
    for template in spec.prompt_templates:
        if '{color}' in template and not resolved_color_name:
            continue
        prompts.append(template.format(color=resolved_color_name or '', shape=shape).strip())
    if spec.shape_hint not in prompts:
        prompts.append(spec.shape_hint)
    deduped: list[str] = []
    seen = set()
    for prompt in prompts:
        if prompt and prompt not in seen:
            seen.add(prompt)
            deduped.append(prompt)
    return deduped


def build_smoke_prompt_candidates(
    smoke_env_name: str,
    color_name: str | None = None,
) -> list[str]:
    spec = get_smoke_object_spec(smoke_env_name)
    return build_prompt_candidates(
        spec,
        color_name=color_name,
        shape_hint=spec.shape_hint,
    )


def describe_core_object(
    spec: ObjectSpec,
    color_rgb: np.ndarray | list[float] | list[int] | None,
    shape_hint: str | None = None,
) -> dict[str, Any]:
    normalized_rgb = _normalize_rgb(color_rgb)
    color_name = describe_color_name(normalized_rgb)
    shape = shape_hint or spec.shape_hint
    descriptor = f'{color_name} {shape}'.strip() if color_name else shape
    return {
        'core_object_name': spec.object_name,
        'core_object_color': color_name,
        'core_object_color_rgb': normalized_rgb.tolist() if normalized_rgb is not None else None,
        'core_object_shape_hint': shape,
        'task_focus_short': descriptor,
        'task_focus_text': spec.focus_template.format(descriptor=descriptor),
    }
