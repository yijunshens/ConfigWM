"""Stable World Model CLI commands."""

from importlib.metadata import version as pkg_version
import json
from pathlib import Path
from typing import Annotated

import typer
from rich import print
from rich.table import Table


app = typer.Typer()


def _version_callback(value: bool):
    if value:
        typer.echo(
            f'stable-worldmodel version: {pkg_version("stable-worldmodel")}'
        )
        raise typer.Exit()


def _detect_folder_format(folder) -> str:
    for sub in sorted(folder.iterdir()):
        if sub.is_dir():
            if any(sub.glob('*.mp4')):
                return 'Video'
            if any(sub.glob('*.jpeg')) or any(sub.glob('*.jpg')):
                return 'Image'
    return 'Folder'


def _format_size(n_bytes: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n_bytes < 1024:
            return f'{n_bytes:.1f} {unit}'
        n_bytes /= 1024
    return f'{n_bytes:.1f} PB'


def _inspect_hdf5_dataset(path) -> None:
    import h5py

    with h5py.File(path, 'r') as f:
        ep_len = f['ep_len'][:]
        columns = {
            k: (f[k].shape, str(f[k].dtype))
            for k in sorted(f.keys())
            if k not in ('ep_len', 'ep_offset')
        }

    size = _format_size(path.stat().st_size)
    print(f'[bold]Name:[/bold]     {path.stem}')
    print('[bold]Format:[/bold]   HDF5')
    print(f'[bold]Path:[/bold]     {path}')
    print(f'[bold]Size:[/bold]     {size}')
    print(f'[bold]Episodes:[/bold] {len(ep_len)}')
    print(f'[bold]Steps:[/bold]    {int(ep_len.sum())}')
    print(f'[bold]Ep length:[/bold] {int(ep_len.min())} – {int(ep_len.max())}')

    table = Table(title='Columns')
    table.add_column('Column', style='cyan', no_wrap=True)
    table.add_column('Shape', style='yellow')
    table.add_column('Dtype', style='magenta')
    for col, (shape, dtype) in columns.items():
        table.add_row(col, str(shape), dtype)
    print(table)


def _inspect_folder_dataset(path) -> None:
    import numpy as np

    ep_len = np.load(path / 'ep_len.npz')['arr_0']
    fmt = _detect_folder_format(path)
    npz_size = sum(p.stat().st_size for p in path.glob('*.npz'))

    print(f'[bold]Name:[/bold]     {path.name}')
    print(f'[bold]Format:[/bold]   {fmt}')
    print(f'[bold]Path:[/bold]     {path}')
    print(f'[bold]Size:[/bold]     {_format_size(npz_size)} (metadata only)')
    print(f'[bold]Episodes:[/bold] {len(ep_len)}')
    print(f'[bold]Steps:[/bold]    {int(ep_len.sum())}')
    print(f'[bold]Ep length:[/bold] {int(ep_len.min())} – {int(ep_len.max())}')

    table = Table(title='Columns')
    table.add_column('Column', style='cyan', no_wrap=True)
    table.add_column('Shape', style='yellow')
    table.add_column('Dtype', style='magenta')

    for p in sorted(path.iterdir()):
        if p.suffix == '.npz' and p.stem not in ('ep_len', 'ep_offset'):
            arr = np.load(p)['arr_0']
            table.add_row(p.stem, str(arr.shape), str(arr.dtype))

    for p in sorted(path.iterdir()):
        if p.is_dir():
            table.add_row(p.name, '(folder)', 'image/video')

    print(table)


def _format_space(space) -> tuple[str, str, str]:
    """Return (type_label, range_str, init_str) for a leaf space."""
    from stable_worldmodel import spaces as swm_spaces

    init = space.init_value if hasattr(space, 'init_value') else None
    init_str = str(init) if init is not None else '-'

    if isinstance(space, swm_spaces.RGBBox):
        return 'RGBBox', '[0,255]^3', init_str
    if isinstance(space, swm_spaces.Box):
        low = space.low.flat[0] if space.low.size == 1 else space.low.tolist()
        high = (
            space.high.flat[0] if space.high.size == 1 else space.high.tolist()
        )
        shape = '' if space.shape == () else f' shape={list(space.shape)}'
        return 'Box', f'[{low}, {high}]{shape}', init_str
    if isinstance(space, swm_spaces.Discrete):
        end = space.start + space.n - 1
        return 'Discrete', f'[{space.start}, {end}]', init_str
    return type(space).__name__, '-', init_str


def _get_space_at_path(variation_space, dotted_path: str):
    space = variation_space
    for part in dotted_path.split('.'):
        space = space.spaces[part]
    return space


@app.command()
def datasets():
    """List all datasets in the cache directory."""
    from stable_worldmodel.data.utils import get_cache_dir

    cache_dir = get_cache_dir()
    table = Table(title=f'Datasets in {cache_dir}')
    table.add_column('Name', justify='left', style='cyan', no_wrap=True)
    table.add_column('Format', justify='left', style='magenta')
    table.add_column('Size', justify='right', style='yellow')

    rows = []

    for h5_path in sorted(cache_dir.glob('*.h5')):
        size = _format_size(h5_path.stat().st_size)
        rows.append((h5_path.stem, 'HDF5', size))

    for folder in sorted(cache_dir.iterdir()):
        if not folder.is_dir() or not (folder / 'ep_len.npz').exists():
            continue
        npz_size = sum(p.stat().st_size for p in folder.glob('*.npz'))
        rows.append(
            (
                folder.name,
                _detect_folder_format(folder),
                _format_size(npz_size),
            )
        )

    if not rows:
        print(f'No datasets found in {cache_dir}')
    else:
        for row in rows:
            table.add_row(*row)
        print(table)


@app.command()
def inspect(
    name: Annotated[str, typer.Argument(help='Dataset name to inspect.')],
):
    """Show detailed info for a dataset."""
    from stable_worldmodel.data.utils import get_cache_dir

    cache_dir = get_cache_dir()
    h5_path = cache_dir / f'{name}.h5'
    folder_path = cache_dir / name

    if h5_path.exists():
        _inspect_hdf5_dataset(h5_path)
    elif folder_path.is_dir() and (folder_path / 'ep_len.npz').exists():
        _inspect_folder_dataset(folder_path)
    else:
        print(f'[red]Dataset not found: {name}[/red]')
        print('Run [cyan]swm datasets[/cyan] to see available datasets.')
        raise typer.Exit(1)


@app.command()
def envs():
    """List all registered environments."""
    table = Table(title='Registered SWM Environments')
    table.add_column(
        'Environment ID', justify='left', style='cyan', no_wrap=True
    )
    table.add_column('Type', justify='left', style='magenta', no_wrap=True)

    from stable_worldmodel.envs import WORLDS

    continuous = sorted(e for e in WORLDS if 'Discrete' not in e)
    discrete = sorted(e for e in WORLDS if 'Discrete' in e)

    for env_id in continuous:
        table.add_row(env_id, 'Continuous')
    if discrete:
        table.add_section()
        for env_id in discrete:
            table.add_row(env_id, 'Discrete')

    print(table)


@app.command()
def fovs(
    env: Annotated[
        str, typer.Argument(help='Environment ID (e.g. PushT-v1).')
    ],
):
    """List factors of variation for the given environment."""
    import gymnasium as gym

    from stable_worldmodel.envs import WORLDS

    if '/' not in env:
        env = f'swm/{env}'

    if env not in WORLDS:
        print(f'[red]Unknown environment: {env}[/red]')
        print('Run [cyan]swm envs[/cyan] to see available environments.')
        raise typer.Exit(1)

    try:
        environment = gym.make(env)
        unwrapped = environment.unwrapped
    except Exception as e:
        print(f'[red]Failed to instantiate {env}: {e}[/red]')
        raise typer.Exit(1)

    if not hasattr(unwrapped, 'variation_space'):
        print(f'[yellow]{env} has no variation_space.[/yellow]')
        raise typer.Exit()

    vs = unwrapped.variation_space
    names = vs.names()

    table = Table(title=f'Factors of Variation — {env}')
    table.add_column('Factor', style='cyan', no_wrap=True)
    table.add_column('Type', style='magenta')
    table.add_column('Range', style='yellow')
    table.add_column('Default', style='green')

    for name in names:
        space = _get_space_at_path(vs, name)
        type_label, range_str, init_str = _format_space(space)
        table.add_row(name, type_label, range_str, init_str)

    print(table)
    environment.close()


@app.command('augment-hdf5')
def augment_hdf5(
    input_h5: Annotated[str, typer.Option(help='Input HDF5 dataset path.')],
    output_h5: Annotated[
        str | None,
        typer.Option(help='Output HDF5 path. Optional in --debug-only mode.'),
    ] = None,
    env_id: Annotated[str, typer.Option(help='Environment ID, e.g. swm/TwoRoom-v1.')] = '',
    object_name: Annotated[str, typer.Option(help='Stable object name to duplicate.')] = '',
    num: Annotated[int, typer.Option(help='Number of distractor copies to insert.')] = 1,
    seed: Annotated[int, typer.Option(help='Random seed for position/color sampling.')] = 0,
    colors: Annotated[
        str | None,
        typer.Option(help='Comma-separated color names, e.g. yellow,blue.'),
    ] = None,
    drop_original: Annotated[
        bool,
        typer.Option(help='Erase the original object before inserting copies.'),
    ] = False,
    debug_episode_idx: Annotated[
        int | None,
        typer.Option(help='Episode index to export debug artifacts for.'),
    ] = None,
    debug_only: Annotated[
        bool,
        typer.Option(help='Only export debug artifacts; do not write output HDF5.'),
    ] = False,
    confidence_threshold: Annotated[
        float,
        typer.Option(help='Minimum segmentation confidence threshold.'),
    ] = 0.5,
    checkpoint_path: Annotated[
        str | None,
        typer.Option(help='SAM3 checkpoint path.'),
    ] = None,
    sam3_repo_path: Annotated[
        str | None,
        typer.Option(help='Path to local SAM3 repository if not importable.'),
    ] = None,
):
    """Augment an HDF5 dataset with static object-copy distractors."""
    from stable_worldmodel.augmentation.hdf5_static_copy import (
        ObjectCopyConfig,
        augment_hdf5_dataset,
    )
    from stable_worldmodel.augmentation.sam3_backend import Sam3ImageBackend

    if not env_id:
        print('[red]--env-id is required[/red]')
        raise typer.Exit(1)
    if not object_name:
        print('[red]--object-name is required[/red]')
        raise typer.Exit(1)
    if not output_h5 and not debug_only:
        print('[red]--output-h5 is required unless --debug-only is set[/red]')
        raise typer.Exit(1)

    parsed_colors = None
    if colors:
        parsed_colors = [
            item.strip().lower() for item in colors.split(',') if item.strip()
        ]

    config = ObjectCopyConfig(
        input_h5=Path(input_h5),
        output_h5=Path(output_h5) if output_h5 else None,
        env_id=env_id,
        object_name=object_name,
        num=num,
        seed=seed,
        colors=parsed_colors,
        drop_original=drop_original,
        debug_episode_idx=debug_episode_idx,
        debug_only=debug_only,
        confidence_threshold=confidence_threshold,
    )
    backend = Sam3ImageBackend(
        checkpoint_path=checkpoint_path,
        sam3_repo_path=sam3_repo_path,
        confidence_threshold=confidence_threshold,
    )
    result = augment_hdf5_dataset(config, backend=backend)
    print_json = typer.style(str(result), fg=typer.colors.GREEN)
    typer.echo(print_json)


@app.command('debug-tworoom-dynamic-toy')
def debug_tworoom_dynamic_toy(
    input_h5: Annotated[str, typer.Option(help='Input HDF5 dataset path.')] = '',
    output_dir: Annotated[str, typer.Option(help='Directory to store toy outputs.')] = '',
    env_id: Annotated[str, typer.Option(help='Environment ID, e.g. swm/TwoRoom-v1.')] = '',
    object_name: Annotated[str, typer.Option(help='Stable object name to track.')] = '',
    donor_episode_idx: Annotated[int, typer.Option(help='Donor episode index.')] = 0,
    target_episode_idx: Annotated[int, typer.Option(help='Target episode index.')] = 0,
    reference_frames_to_try: Annotated[
        int, typer.Option(help='How many early frames to try for initialization.')
    ] = 5,
    color: Annotated[str, typer.Option(help='Color name for the transformed donor.')] = 'blue',
    scale: Annotated[float, typer.Option(help='Global scale factor for the donor patch.')] = 1.0,
    translate_x: Annotated[int, typer.Option(help='Global x translation in pixels.')] = 0,
    translate_y: Annotated[int, typer.Option(help='Global y translation in pixels.')] = 0,
    confidence_threshold: Annotated[
        float, typer.Option(help='Minimum segmentation confidence threshold.')
    ] = 0.05,
    checkpoint_path: Annotated[
        str | None,
        typer.Option(help='SAM3 checkpoint path.'),
    ] = None,
    sam3_repo_path: Annotated[
        str | None,
        typer.Option(help='Path to local SAM3 repository if not importable.'),
    ] = None,
):
    """Run a toy TwoRoom dynamic donor tracking + composition debug case."""
    from stable_worldmodel.augmentation.sam3_backend import (
        Sam3ImageBackend,
        Sam3VideoBackend,
    )
    from stable_worldmodel.augmentation.tworoom_dynamic_toy import (
        TwoRoomDynamicToyConfig,
        run_tworoom_dynamic_donor_toy,
    )

    if not input_h5:
        print('[red]--input-h5 is required[/red]')
        raise typer.Exit(1)
    if not output_dir:
        print('[red]--output-dir is required[/red]')
        raise typer.Exit(1)
    if not env_id:
        print('[red]--env-id is required[/red]')
        raise typer.Exit(1)
    if not object_name:
        print('[red]--object-name is required[/red]')
        raise typer.Exit(1)

    config = TwoRoomDynamicToyConfig(
        input_h5=Path(input_h5),
        output_dir=Path(output_dir),
        env_id=env_id,
        object_name=object_name,
        donor_episode_idx=donor_episode_idx,
        target_episode_idx=target_episode_idx,
        reference_frames_to_try=reference_frames_to_try,
        color=color.lower(),
        scale=scale,
        translate_x=translate_x,
        translate_y=translate_y,
        confidence_threshold=confidence_threshold,
    )
    image_backend = Sam3ImageBackend(
        checkpoint_path=checkpoint_path,
        sam3_repo_path=sam3_repo_path,
        confidence_threshold=confidence_threshold,
    )
    video_backend = Sam3VideoBackend(
        checkpoint_path=checkpoint_path,
        sam3_repo_path=sam3_repo_path,
    )
    result = run_tworoom_dynamic_donor_toy(
        config=config,
        image_backend=image_backend,
        video_backend=video_backend,
    )
    print_json = typer.style(str(result), fg=typer.colors.GREEN)
    typer.echo(print_json)


@app.command('collect-augment-hdf5')
def collect_augment_hdf5(
    env_id: Annotated[str, typer.Option(help='Environment ID, e.g. swm/PushT-v1.')] = '',
    object_name: Annotated[str, typer.Option(help='Stable object name to duplicate.')] = '',
    episodes: Annotated[int, typer.Option(help='Number of raw episodes to collect.')] = 1,
    output_dir: Annotated[str, typer.Option(help='Directory to store raw/augmented outputs.')] = '',
    num: Annotated[int, typer.Option(help='Number of distractor copies to insert.')] = 1,
    seed: Annotated[int, typer.Option(help='Seed for collection and augmentation.')] = 0,
    policy: Annotated[
        str,
        typer.Option(help='Policy to use during collection: auto, expert, weak, random.'),
    ] = 'auto',
    colors: Annotated[
        str | None,
        typer.Option(help='Comma-separated color names, e.g. yellow,blue.'),
    ] = None,
    drop_original: Annotated[
        bool,
        typer.Option(help='Erase the original object before inserting copies.'),
    ] = False,
    debug_episode_idx: Annotated[
        int | None,
        typer.Option(help='Episode index to export debug artifacts for.'),
    ] = None,
    debug_only: Annotated[
        bool,
        typer.Option(help='Only export debug artifacts; do not write output HDF5.'),
    ] = False,
    confidence_threshold: Annotated[
        float,
        typer.Option(help='Minimum segmentation confidence threshold.'),
    ] = 0.5,
    checkpoint_path: Annotated[
        str | None,
        typer.Option(help='SAM3 checkpoint path.'),
    ] = None,
    sam3_repo_path: Annotated[
        str | None,
        typer.Option(help='Path to local SAM3 repository if not importable.'),
    ] = None,
    expert_ckpt_root: Annotated[
        str | None,
        typer.Option(help='Root directory for expert checkpoints, e.g. DMC experts.'),
    ] = None,
    variation: Annotated[
        str | None,
        typer.Option(
            help='Comma-separated variation keys to randomize during collection. '
            'If omitted, uses the environment default reset behavior.',
        ),
    ] = None,
    variation_values_json: Annotated[
        str | None,
        typer.Option(
            help='JSON object for explicit variation_values, following world.reset() semantics.',
        ),
    ] = None,
    variation_all: Annotated[
        bool,
        typer.Option(help='Force variation=[\"all\"] during collection.'),
    ] = False,
    camera_distract_scale: Annotated[
        float,
        typer.Option(help='Continuous camera distraction scale in [0, 1] for supported 3D envs.'),
    ] = 0.0,
    camera_distract_dynamic: Annotated[
        bool,
        typer.Option(
            help='Enable dynamic camera motion when supported. Currently mainly useful for DMControl.',
        ),
    ] = False,
    camera_distract_mode: Annotated[
        str,
        typer.Option(
            help='Camera distraction mode: sample, calibrated, or extreme.',
        ),
    ] = 'sample',
):
    """Collect a raw HDF5 dataset from an env and immediately augment it."""
    from stable_worldmodel.augmentation.collect_augment import (
        CollectAugmentConfig,
        collect_and_augment_dataset,
    )

    if not env_id:
        print('[red]--env-id is required[/red]')
        raise typer.Exit(1)
    if not object_name:
        print('[red]--object-name is required[/red]')
        raise typer.Exit(1)
    if not output_dir:
        print('[red]--output-dir is required[/red]')
        raise typer.Exit(1)

    parsed_colors = None
    if colors:
        parsed_colors = [
            item.strip().lower() for item in colors.split(',') if item.strip()
        ]
    parsed_variation = None
    if variation:
        parsed_variation = [item.strip() for item in variation.split(',') if item.strip()]
    parsed_variation_values = None
    if variation_values_json:
        parsed_variation_values = json.loads(variation_values_json)

    config = CollectAugmentConfig(
        env_id=env_id,
        object_name=object_name,
        num=num,
        episodes=episodes,
        output_dir=Path(output_dir),
        seed=seed,
        policy_name=policy,
        colors=parsed_colors,
        drop_original=drop_original,
        debug_episode_idx=debug_episode_idx,
        debug_only=debug_only,
        confidence_threshold=confidence_threshold,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        sam3_repo_path=Path(sam3_repo_path) if sam3_repo_path else None,
        expert_ckpt_root=Path(expert_ckpt_root) if expert_ckpt_root else None,
        variation=parsed_variation,
        variation_values=parsed_variation_values,
        force_variation_all=variation_all,
        camera_distract_scale=camera_distract_scale,
        camera_distract_dynamic=camera_distract_dynamic,
        camera_distract_mode=camera_distract_mode,
    )
    result = collect_and_augment_dataset(config)
    print_json = typer.style(str(result), fg=typer.colors.GREEN)
    typer.echo(print_json)


@app.command()
def checkpoints(
    filter: Annotated[
        str | None,
        typer.Argument(
            help='Optional substring to filter by run or checkpoint name.',
            show_default=False,
        ),
    ] = None,
):
    """List model checkpoints available in the cache directory."""
    from stable_worldmodel.data.utils import get_cache_dir

    cache_dir = get_cache_dir()
    table = Table(title=f'Checkpoints in {cache_dir}')
    table.add_column('Run', justify='left', style='cyan', no_wrap=True)
    table.add_column('Checkpoint', justify='left', style='magenta')

    def _ckpt_name(p):
        return p.stem.removesuffix('_object')

    def _by_mtime(p):
        return p.stat().st_mtime

    groups: list[tuple[str, list[str]]] = []

    import re

    pattern = re.compile(filter) if filter else None

    def _matches(run: str, ckpt: str) -> bool:
        if pattern is None:
            return True
        return bool(pattern.search(ckpt) or pattern.search(run))

    # Root-level checkpoints (directly in cache_dir)
    root_files = sorted(cache_dir.glob('*_object.ckpt'), key=_by_mtime)
    if root_files:
        names = [
            _ckpt_name(p) for p in root_files if _matches('', _ckpt_name(p))
        ]
        if names:
            groups.append(('', names))

    # Per-directory checkpoints
    for folder in sorted(cache_dir.iterdir()):
        if not folder.is_dir():
            continue
        ckpt_files = sorted(folder.glob('*_object.ckpt'), key=_by_mtime)
        if not ckpt_files:
            continue
        run_name = folder.name
        names = [
            _ckpt_name(p)
            for p in ckpt_files
            if _matches(run_name, _ckpt_name(p))
        ]
        if not names:
            continue
        groups.append((run_name, names))

    if not groups:
        msg = f'No checkpoints found in {cache_dir}'
        if filter:
            msg += f' matching pattern [bold]{filter}[/bold]'
        print(msg)
    else:
        first = True
        for run_name, ckpt_names in groups:
            if not first:
                table.add_section()
            first = False
            for i, ckpt in enumerate(ckpt_names):
                table.add_row(run_name if i == 0 else '', ckpt)
        print(table)


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            '--version',
            '-v',
            callback=_version_callback,
            is_eager=True,
            help='Show installed version.',
        ),
    ] = None,
):
    """Stable World Model - World Model Research Made Simple."""


if __name__ == '__main__':
    app()
