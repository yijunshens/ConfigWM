"""Resolve portable release manifests and prepare lightweight Configurable Encoder/control views.

The generated view contains only manifests and links into the downloaded data.
It preserves the existing adapters' file names and ordering without copying HDF5.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath


FILE_ROLES = ('frames', 'dynamics', 'query_pairs', 'metadata', 'source')


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def resolve_file(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or '..' in value.parts or '\\' in relative:
        raise ValueError(f'Expected a dataset-root-relative path: {relative}')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Data reference leaves the dataset directory: {relative}')
    if not path.is_file():
        raise FileNotFoundError(f'Missing data file: {path}. Download the selected task and its dependencies.')
    return path


def dataset_root(collection: Path) -> Path:
    for parent in (collection, *collection.parents):
        if (parent / 'dataset.json').is_file():
            return parent
    raise FileNotFoundError(f'Cannot find dataset.json above {collection}')


# Dataset manifests and existing download commands retain their published keys.
TASK_KEYS = {"configurable_encoder": "stage_a", "configwm": "stage_b"}


def canonical_task_name(value: str) -> str:
    return {published: name for name, published in TASK_KEYS.items()}.get(value, value)


def selected_manifests(index: dict, env: str, purpose: str, variant: str = 'd0') -> list[str]:
    selected = index[env][TASK_KEYS.get(purpose, purpose)]
    if isinstance(selected, dict): selected = selected[variant]
    return [selected] if isinstance(selected, str) else list(selected)


def manifest_files(manifest: dict) -> set[str]:
    files = {manifest[k] for k in ('query_metadata', 'base') if k in manifest}
    for shard in manifest.get('shards', []):
        files.update(shard[k] for k in FILE_ROLES if k in shard)
    return files


def prepare_data_root(data_root, output_dir, env: str, purpose: str, variant: str = 'd0') -> Path:
    root = Path(data_root).expanduser().resolve()
    if not (root / 'dataset.json').is_file():
        return root
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    view = Path(tempfile.mkdtemp(prefix=f'{env}-{purpose}-{variant}-', dir=output))
    index = read_json(root / 'dataset.json')
    environments = ('pusht', 'tworoom', 'ogbcube') if env == 'unified' else (env,)
    seen = set()

    def link(relative: str, destination: Path):
        target = resolve_file(root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != target: raise ValueError(f'Conflicting data reference: {destination}')
            return
        destination.symlink_to(os.path.relpath(target, destination.parent))

    def save(destination: Path, value: dict):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

    def trajectories(environment, variant_name, manifest, destination):
        shards = []
        for item in manifest['shards']:
            name = PurePosixPath(item['frames']).parent.name
            basename = f'dataset_core_{variant_name}_color_query_full.h5' if environment == 'tworoom' else 'dataset.h5'
            link(item['frames'], destination / name / basename)
            dynamics = item.get('dynamics', item['frames'])
            with h5py.File(resolve_file(root, dynamics), 'r') as h:
                count = len(h['ep_len']); frames = int(h['action'].shape[0])
                offsets = np.asarray(h['ep_offset'][:], dtype=np.int64)
                if 'original_episode_idx' in h:
                    episodes = [int(x) for x in h['original_episode_idx'][:]]
                else:
                    key = 'episode_idx' if 'episode_idx' in h else 'ep_idx'
                    episodes = [int(x) for x in h[key][offsets]]
            shard = {'path': f'{name}/{basename}', 'num_episodes': count, 'total_frames': frames, 'episodes': episodes}
            if environment == 'tworoom':
                link(dynamics, destination / name / 'base_trajectories.h5')
                shard.pop('path')
                shard['paths'] = {'color_query_h5':f'{name}/{basename}', 'base_trajectories_h5':f'{name}/base_trajectories.h5'}
            if environment == 'ogbcube':
                shard.update(ep_start=episodes[0], ep_end=episodes[-1] + 1)
            if 'metadata' in item:
                metadata_name = f'core_{variant_name}_scene_metadata.jsonl' if environment == 'tworoom' else 'episode_metadata.jsonl'
                link(item['metadata'], destination / name / metadata_name)
            shards.append(shard)
        dataset = {'pusht':'pusht_v13_goldbrown_style_9color_far_nested_d0_d8',
                   'tworoom':f'tworoom_9color_native_gaussian_core_{variant_name}_color_query_full',
                   'ogbcube':'ogbcube_9color_core_fixedslots_d0_d8_alpha_full10000'}[environment]
        # Query order is part of the training contract, not a sort order.
        colors = {'pusht':['gray','red','orange','yellow','pink','purple','brown','cyan','navy'],
                  'tworoom':['red','blue','green','orange','purple','cyan','yellow','pink','brown'],
                  'ogbcube':['silver','green','yellow','cyan','red','blue','orange','purple','pink']}[environment]
        save(destination/'manifest.json', {'dataset':dataset, 'variant':variant_name, 'colors':colors, 'shards':shards})
        if 'query_metadata' in manifest:
            link(manifest['query_metadata'], destination/'query_metadata.jsonl')

    def load(relative):
        if relative in seen: return
        seen.add(relative)
        manifest = read_json(resolve_file(root, relative))
        for dependency in manifest.get('dependencies', []): load(dependency)
        parts = PurePosixPath(relative).parts
        environment, kind = parts[:2]
        if kind == 'trajectories':
            v = parts[2]
            trajectories(environment, v, manifest, view/environment/v)
            if environment == 'pusht' and v == 'd0':
                (view/environment/'clean').symlink_to('d0', target_is_directory=True)
        elif kind == 'perception':
            dest = view/environment/('train' if environment != 'tworoom' else '')
            legacy = {'colors':manifest['colors'], 'shards':[]}
            for item in manifest['shards']:
                name = PurePosixPath(item['query_pairs']).parent.name
                basename = 'dataset_source9_stride5.h5' if environment == 'tworoom' else 'dataset.h5'
                link(item['query_pairs'], dest/name/basename)
                source = PurePosixPath(item['source']); source_variant = source.parts[2]; source_name = source.parent.name
                with h5py.File(resolve_file(root, item['query_pairs']), 'r') as h:
                    frames = int(h['source_pixels'].shape[0])
                shard = {'path':f'{name}/{basename}', 'total_frames':frames}
                if environment == 'tworoom':
                    shard.update(output_h5_path=shard['path'], parent_h5_path=f'd8/{source_name}/dataset_core_d8_color_query_full.h5')
                elif environment == 'ogbcube':
                    shard.update(source_variant=source_variant, source_shard=source_name, num_groups=frames)
                legacy['shards'].append(shard)
            legacy['dataset'] = {'pusht':'pusht_9color_far_source9_query_h5_diffexact_v2', 'tworoom':'tworoom_9color_source9_stride5', 'ogbcube':'ogbcube_d8_same_source_query9_to_d0'}[environment]
            if environment == 'pusht':
                legacy.update(format='pusht_source9_query_hdf5', source_variant='d8', query_template='{color} T block', frame_stride=5)
            save(dest/'manifest.json', legacy)
        elif kind == 'control':
            v = parts[2]; dest = view/'eval'/environment/v
            if 'base' in manifest:
                basename = 'pusht_expert_train.h5' if environment == 'pusht' else 'tworoom.h5'
                link(manifest['base'], view/'eval'/environment/basename)
            if environment == 'pusht':
                trajectories(environment, v, manifest, dest)
            elif environment == 'tworoom':
                for item in manifest['shards']:
                    name = PurePosixPath(item['frames']).parent.name
                    link(item['frames'], dest/name/'dataset_core_d8_color_query_full.h5')
                    link(item['metadata'], dest/name/'core_d8_scene_metadata.jsonl')
        else:
            raise ValueError(f'Unknown collection: {relative}')

    for environment in environments:
        for relative in selected_manifests(index, environment, purpose, variant): load(relative)
    return view
