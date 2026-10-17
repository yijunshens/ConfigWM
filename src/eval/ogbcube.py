from __future__ import annotations
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import h5py
import hdf5plugin
import numpy as np
from .common import (
    TSWMProjectorCostModel,
    _apply_dataset_state_callables,
    _broadcast_goal_history,
    _build_action_context_from_history,
    _build_reset_options,
    _dataset_name_and_cache_dir,
    _extract_latest_value,
    _fit_action_scaler_from_dataset,
    _get_episode_ids_column,
    _get_world_model_action,
    _image_transform,
    _load_eval_sequences,
    _stack_batch_history_buffers,
    _update_action_history_buffers,
    _update_history_buffers,
)


@dataclass(frozen=True)
class _OGBShardInfo:
    path: Path
    ep_start: int
    ep_end: int


def _ogb_load_manifest(base: Path, variant: str) -> list[_OGBShardInfo]:
    manifest = json.loads(
        (base / variant / "manifest.json").read_text(encoding="utf-8")
    )
    shards: list[_OGBShardInfo] = []
    next_ep_start = 0
    for item in manifest["shards"]:
        raw_episodes = item.get("episodes", [])
        ep_start = int(item.get("ep_start", next_ep_start))
        if "ep_end" in item:
            ep_end = int(item["ep_end"])
        elif isinstance(raw_episodes, int):
            ep_end = ep_start + int(raw_episodes)
        elif raw_episodes:
            episodes = [int(ep) for ep in raw_episodes]
            ep_start = int(item.get("ep_start", min(episodes)))
            ep_end = max(episodes) + 1
        else:
            ep_end = ep_start
        shards.append(
            _OGBShardInfo(
                path=base / variant / item["path"], ep_start=ep_start, ep_end=ep_end
            )
        )
        next_ep_start = ep_end
    return shards


class _OGBVariantReader:
    def __init__(self, base: Path, variant: str) -> None:
        self.base = base.expanduser().resolve()
        self.variant = str(variant)
        self.shards = _ogb_load_manifest(self.base, self.variant)
        self._metadata: dict[int, dict[str, Any]] | None = None
        self._h5_cache: dict[Path, h5py.File] = {}

    def shard_for_episode(self, episode: int) -> _OGBShardInfo:
        for shard in self.shards:
            if shard.ep_start <= int(episode) < shard.ep_end:
                return shard
        raise KeyError(f"episode {episode} not found in {self.variant}")

    def metadata(self, episode: int) -> dict[str, Any]:
        if self._metadata is None:
            metadata: dict[int, dict[str, Any]] = {}
            for shard in self.shards:
                meta_path = shard.path.parent / "episode_metadata.jsonl"
                for line in meta_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    metadata[int(item["episode"])] = item
            self._metadata = metadata
        return self._metadata[int(episode)]

    def frame(self, episode: int, step: int) -> np.ndarray:
        shard = self.shard_for_episode(int(episode))
        h5 = self._h5_cache.get(shard.path)
        if h5 is None:
            h5 = h5py.File(shard.path, "r")
            self._h5_cache[shard.path] = h5
        local_episode = int(episode) - shard.ep_start
        ep_offset = int(h5["ep_offset"][local_episode])
        ep_len = int(h5["ep_len"][local_episode])
        clamped_step = max(0, min(int(step), ep_len - 1))
        return np.asarray(h5["pixels"][ep_offset + clamped_step])


def _ogb_meta_distractor_colors(meta: dict[str, Any]) -> set[str]:
    colors: set[str] = set()
    for key in ("distractor_colors", "donor_colors"):
        value = meta.get(key)
        if isinstance(value, (list, tuple, set)):
            colors.update(
                (str(item).strip().lower() for item in value if str(item).strip())
            )
        elif value is not None and str(value).strip():
            colors.add(str(value).strip().lower())
    for key in ("distractors", "donors"):
        value = meta.get(key)
        if not isinstance(value, (list, tuple)):
            continue
        for item in value:
            if isinstance(item, dict):
                for color_key in (
                    "color_name",
                    "color",
                    "core_color",
                    "block_color",
                    "cube_color",
                    "donor_color",
                    "distractor_color",
                    "color_token",
                ):
                    value = item.get(color_key)
                    if value is not None and str(value).strip():
                        colors.add(str(value).strip().lower())
            elif item is not None and str(item).strip():
                colors.add(str(item).strip().lower())
    return colors


def _sample_ogbcube_donor_layouts(
    *, base: Path, variant: str, num_eval: int, seed: int, core_color: str
) -> tuple[list[int], list[int], dict[str, Any]]:
    wanted = str(core_color).strip().lower()
    reader = _OGBVariantReader(base, variant)
    candidates: list[int] = []
    matching_core = 0
    excluded_same_color_distractor = 0
    color_counts: dict[str, int] = defaultdict(int)
    episode_ids = sorted(
        (
            ep
            for shard in reader.shards
            for ep in range(int(shard.ep_start), int(shard.ep_end))
        )
    )
    for episode in episode_ids:
        meta = reader.metadata(int(episode))
        color = str(meta.get("core_color", "")).strip().lower()
        if color:
            color_counts[color] += 1
        if color != wanted:
            continue
        matching_core += 1
        if wanted in _ogb_meta_distractor_colors(meta):
            excluded_same_color_distractor += 1
            continue
        candidates.append(int(episode))
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(
        len(candidates), size=int(num_eval), replace=len(candidates) < int(num_eval)
    )
    donor_episodes = [int(candidates[int(idx)]) for idx in picked.tolist()]
    donor_starts = [0 for _ in donor_episodes]
    metadata = {
        "requested": wanted,
        "sample_mode": "independent_donor_layout",
        "variant": str(variant),
        "base": str(Path(base).expanduser().resolve()),
        "core_color_counts": dict(sorted(color_counts.items())),
        "matching_core_episodes": int(matching_core),
        "excluded_same_color_distractor": int(excluded_same_color_distractor),
        "eligible_episodes": int(len(candidates)),
        "sampled": int(len(donor_episodes)),
        "seed": int(seed),
    }
    return (donor_episodes, donor_starts, metadata)


def _ogb_grid_spec_from_meta(
    meta: dict[str, Any], variant: str
) -> tuple[int, int, int]:
    grid_text = str(meta.get("mosaic_grid") or "").strip().lower()
    if "x" in grid_text:
        left, right = grid_text.split("x", 1)
        if left.isdigit() and right.isdigit() and (int(left) == int(right)):
            grid = int(left)
        else:
            raise ValueError(f"Unsupported OGB-Cube mosaic_grid={grid_text!r}")
    else:
        variant_text = str(variant).strip().lower()
        if variant_text.startswith("d") and variant_text[1:].isdigit():
            idx = int(variant_text[1:])
            if idx == 0:
                grid = 1
            elif idx <= 3:
                grid = 2
            else:
                grid = 3
        else:
            grid = 2
    if grid == 1:
        return (1, 224, 0)
    if grid == 2:
        return (2, 111, 2)
    if grid == 3:
        return (3, 72, 4)
    raise ValueError(f"Unsupported OGB-Cube grid={grid} from {grid_text!r}")


def _ogb_resize_cell(image: np.ndarray, size: int = 111) -> np.ndarray:
    return cv2.resize(
        np.asarray(image, dtype=np.uint8), (size, size), interpolation=cv2.INTER_AREA
    )


def _ogb_slot_origin(
    slot: int, *, grid: int = 2, cell_size: int = 111, border: int = 2
) -> tuple[int, int]:
    row = int(slot) // int(grid)
    col = int(slot) % int(grid)
    return (row * (cell_size + border), col * (cell_size + border))


def _ogb_paste_cell(
    canvas: np.ndarray,
    slot: int,
    patch: np.ndarray,
    *,
    grid: int = 2,
    cell_size: int = 111,
    border: int = 2,
) -> None:
    y0, x0 = _ogb_slot_origin(slot, grid=grid, cell_size=cell_size, border=border)
    canvas[y0 : y0 + cell_size, x0 : x0 + cell_size] = patch


def _ogb_empty_cell_from_frame(frame: np.ndarray, cell_size: int = 111) -> np.ndarray:
    corners = np.concatenate(
        [
            frame[:24, :24].reshape(-1, 3),
            frame[:24, -24:].reshape(-1, 3),
            frame[-24:, :24].reshape(-1, 3),
            frame[-24:, -24:].reshape(-1, 3),
        ],
        axis=0,
    )
    color = np.median(corners, axis=0).astype(np.uint8)
    return np.broadcast_to(color[None, None, :], (cell_size, cell_size, 3)).copy()


def _ogb_base_canvas(
    empty_cell: np.ndarray, *, grid: int = 2, cell_size: int = 111, border: int = 2
) -> np.ndarray:
    height = cell_size * grid + border * (grid - 1)
    canvas = np.zeros((height, height, 3), dtype=np.uint8)
    canvas[:] = (18, 24, 33)
    for slot in range(grid * grid):
        _ogb_paste_cell(
            canvas, slot, empty_cell, grid=grid, cell_size=cell_size, border=border
        )
    return canvas


class OGBCubeLive2x2Renderer:
    """Rebuild official OGB-Cube 2x2 disturbed pixels during live rollout."""

    def __init__(
        self,
        *,
        base: Path,
        variant: str,
        episodes_idx: list[int],
        donor_episodes_idx: list[int] | None = None,
        base_start_steps: list[int] | None = None,
        donor_start_steps: list[int] | None = None,
    ) -> None:
        self.base = base.expanduser().resolve()
        self.variant = str(variant)
        self.d0 = _OGBVariantReader(self.base, "d0")
        self.disturbed = _OGBVariantReader(self.base, self.variant)
        self.base_episodes_idx = [int(x) for x in episodes_idx]
        self.base_start_steps = (
            [int(x) for x in base_start_steps]
            if base_start_steps is not None
            else [0 for _ in self.base_episodes_idx]
        )
        self.donor_episodes_idx = (
            [int(x) for x in donor_episodes_idx]
            if donor_episodes_idx is not None
            else [int(x) for x in episodes_idx]
        )
        self.donor_start_steps = (
            [int(x) for x in donor_start_steps]
            if donor_start_steps is not None
            else list(self.base_start_steps)
        )
        self.metas = [
            self.disturbed.metadata(int(ep)) for ep in self.donor_episodes_idx
        ]
        self.queries = np.asarray(
            [f"{str(meta['core_color'])} arm" for meta in self.metas], dtype=object
        )
        self.grid_specs = [
            _ogb_grid_spec_from_meta(meta, self.variant) for meta in self.metas
        ]
        self._base_canvases: list[np.ndarray] = []
        for meta, (grid, cell_size, border) in zip(self.metas, self.grid_specs):
            core_episode = int(meta.get("source_episode", meta["episode"]))
            first_frame = self.d0.frame(core_episode, 0)
            empty_cell = _ogb_empty_cell_from_frame(first_frame, cell_size=cell_size)
            self._base_canvases.append(
                _ogb_base_canvas(
                    empty_cell, grid=grid, cell_size=cell_size, border=border
                )
            )

    def _donor_step(self, env_idx: int, base_step: int) -> int:
        return int(self.donor_start_steps[env_idx]) + (
            int(base_step) - int(self.base_start_steps[env_idx])
        )

    def render_batch(
        self, live_core_frames: np.ndarray, step_indices: np.ndarray
    ) -> np.ndarray:
        frames = np.asarray(live_core_frames)
        rendered = []
        flat_steps = np.asarray(step_indices).reshape(-1)
        for env_idx, meta in enumerate(self.metas):
            grid, cell_size, border = self.grid_specs[env_idx]
            step = self._donor_step(env_idx, int(flat_steps[env_idx]))
            canvas = self._base_canvases[env_idx].copy()
            _ogb_paste_cell(
                canvas,
                int(meta["core_cell"]),
                _ogb_resize_cell(frames[env_idx], size=cell_size),
                grid=grid,
                cell_size=cell_size,
                border=border,
            )
            for donor in meta.get("distractors", []):
                donor_episode = int(
                    donor.get("donor_source_episode", donor["donor_episode"])
                )
                donor_frame = self.d0.frame(donor_episode, step)
                _ogb_paste_cell(
                    canvas,
                    int(donor["cell"]),
                    _ogb_resize_cell(donor_frame, size=cell_size),
                    grid=grid,
                    cell_size=cell_size,
                    border=border,
                )
            rendered.append(canvas)
        return np.stack(rendered, axis=0).astype(np.uint8, copy=False)

    def render_history(
        self, core_history: np.ndarray, step_indices: np.ndarray
    ) -> np.ndarray:
        arr = np.asarray(core_history)
        if arr.ndim != 5:
            return self.render_batch(arr, step_indices)
        out = arr.copy()
        capacity = out.shape[1]
        base_steps = np.asarray(step_indices, dtype=np.int64)
        for hist_idx in range(capacity):
            hist_steps = np.maximum(0, base_steps - (capacity - 1 - hist_idx))
            out[:, hist_idx] = self.render_batch(out[:, hist_idx], hist_steps)
        return out

    def render_goal(
        self, core_goal: np.ndarray, goal_step_indices: np.ndarray
    ) -> np.ndarray:
        arr = np.asarray(core_goal)
        if arr.ndim != 5:
            return self.render_batch(arr, goal_step_indices)
        out = arr.copy()
        for hist_idx in range(out.shape[1]):
            out[:, hist_idx] = self.render_batch(out[:, hist_idx], goal_step_indices)
        return out

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "variant": self.variant,
            "base": str(self.base),
            "base_episodes_idx": [int(x) for x in self.base_episodes_idx],
            "base_start_steps": [int(x) for x in self.base_start_steps],
            "donor_episodes_idx": [int(x) for x in self.donor_episodes_idx],
            "donor_start_steps": [int(x) for x in self.donor_start_steps],
            "donor_core_colors": [str(meta.get("core_color")) for meta in self.metas],
            "donor_distractor_colors": [
                sorted(_ogb_meta_distractor_colors(meta)) for meta in self.metas
            ],
            "grid_specs": [
                {
                    "grid": int(grid),
                    "cell_size": int(cell_size),
                    "border": int(border),
                    "mosaic_grid": str(meta.get("mosaic_grid")),
                }
                for meta, (grid, cell_size, border) in zip(self.metas, self.grid_specs)
            ],
        }


def _sample_eval_starts(
    dataset,
    *,
    num_eval: int,
    goal_offset_steps: int,
    seed: int,
    eligible_episode_ids: set[int] | None = None,
):
    episode_col = _get_episode_ids_column(dataset)
    episode_idx = dataset.get_col_data(episode_col)
    step_idx = dataset.get_col_data("step_idx")
    unique_episode_ids = np.array(
        sorted({int(ep) for ep in episode_idx}), dtype=np.int64
    )
    lengths = []
    for ep_id in unique_episode_ids:
        lengths.append(int(np.max(step_idx[episode_idx == ep_id]) + 1))
    lengths = np.asarray(lengths, dtype=np.int64)
    max_start_idx = lengths - goal_offset_steps - 1
    max_start_idx_dict = {
        int(ep_id): int(max_start_idx[i]) for i, ep_id in enumerate(unique_episode_ids)
    }
    max_start_per_row = np.asarray(
        [max_start_idx_dict[int(ep_id)] for ep_id in episode_idx], dtype=np.int64
    )
    valid_mask = step_idx <= max_start_per_row
    if eligible_episode_ids is not None:
        valid_mask &= np.asarray(
            [int(ep) in eligible_episode_ids for ep in episode_idx], dtype=bool
        )
    valid_indices = np.nonzero(valid_mask)[0]
    rng = np.random.default_rng(seed)
    sampled_positions = rng.choice(len(valid_indices), size=num_eval, replace=False)
    sampled = np.sort(valid_indices[sampled_positions])
    rows = dataset.get_row_data(sampled)
    return {
        "episode_ids": rows[episode_col].tolist(),
        "start_steps": rows["step_idx"].tolist(),
        "row_ids": rows["id"].tolist() if "id" in rows else sampled.tolist(),
        "valid_start_points": int(valid_mask.sum()),
    }


def _build_policy_info(
    *,
    pixel_history_buffers: np.ndarray,
    action_history_buffers: np.ndarray | None,
    goal_images: np.ndarray,
    row_ids: np.ndarray,
    step_indices: np.ndarray,
    goal_step_indices: np.ndarray,
    terminated: np.ndarray,
    history_size: int,
    frame_skip: int,
    needs_flush: np.ndarray,
    queries: np.ndarray | list[str] | None = None,
) -> dict[str, np.ndarray]:
    stacked_pixels = _stack_batch_history_buffers(
        pixel_history_buffers, history_size=history_size, frame_skip=frame_skip
    )
    id_history = np.broadcast_to(
        row_ids[:, None], (row_ids.shape[0], history_size)
    ).copy()
    step_history = np.broadcast_to(
        step_indices[:, None], (step_indices.shape[0], history_size)
    ).copy()
    goal_step_history = np.broadcast_to(
        goal_step_indices[:, None], (goal_step_indices.shape[0], history_size)
    ).copy()
    info = {
        "pixels": stacked_pixels,
        "goal": goal_images,
        "id": id_history,
        "step_idx": step_history,
        "goal_step_idx": goal_step_history,
        "terminated": terminated.astype(bool, copy=True),
        "_needs_flush": needs_flush.astype(bool, copy=True),
    }
    if queries is not None:
        info["queries"] = np.asarray(queries, dtype=object).reshape(
            row_ids.shape[0], -1
        )
    if action_history_buffers is not None:
        info["action_context"] = _build_action_context_from_history(
            action_history_buffers, history_size=history_size, frame_skip=frame_skip
        )
    return info


def run_control_eval(
    args: argparse.Namespace, bridge: TSWMProjectorCostModel, tswm_config
) -> dict[str, Any]:
    import stable_worldmodel as swm

    eval_data_dir = (
        args.eval_data_dir.expanduser().resolve()
        if args.eval_data_dir is not None
        else Path(tswm_config.data.data_dir).expanduser().resolve()
    )
    model_history_size = int(tswm_config.data.num_hist)
    model_frame_skip = int(tswm_config.data.frameskip)
    world_history_size = (
        model_history_size
        if args.world_history_size is None
        else int(args.world_history_size)
    )
    world_frame_skip = (
        model_frame_skip
        if args.world_frame_skip is None
        else int(args.world_frame_skip)
    )
    dataset_name, cache_dir = _dataset_name_and_cache_dir(eval_data_dir)
    dataset = swm.data.HDF5Dataset(
        dataset_name, frameskip=1, num_steps=1, cache_dir=cache_dir
    )
    requested_cache_keys = list(args.keys_to_cache or [])
    keys_to_cache = [key for key in requested_cache_keys if key in dataset.column_names]
    if keys_to_cache:
        dataset = swm.data.HDF5Dataset(
            dataset_name,
            frameskip=1,
            num_steps=1,
            cache_dir=cache_dir,
            keys_to_cache=keys_to_cache,
        )
    normalize_actions = bool(getattr(tswm_config.data, "normalize_actions", False))
    action_scaler = None
    process = {}
    if normalize_actions:
        action_scaler = _fit_action_scaler_from_dataset(dataset)
        process["action"] = action_scaler
        bridge.set_action_normalizer(
            action_scaler.mean_.reshape(-1), action_scaler.scale_.reshape(-1)
        )
    eligible_episode_ids = None
    sampled_core_color_filter = None
    if args.ogbcube_core_color_filter is not None:
        wanted = str(args.ogbcube_core_color_filter).strip().lower()
        decoupled_donor = bool(args.decouple_donor_trajectory)
        if decoupled_donor or args.ogbcube_live_2x2_variant is None:
            filter_variant = "d0"
        else:
            filter_variant = str(args.ogbcube_live_2x2_variant)
        filter_reader = _OGBVariantReader(args.ogbcube_2x2_base, filter_variant)
        episode_col = _get_episode_ids_column(dataset)
        dataset_episode_ids = sorted(
            {int(ep) for ep in dataset.get_col_data(episode_col)}
        )
        matching_core_episode_ids: list[int] = []
        excluded_same_color_distractor = 0
        eligible_episode_ids = set()
        for ep in dataset_episode_ids:
            meta = filter_reader.metadata(int(ep))
            if str(meta.get("core_color", "")).strip().lower() != wanted:
                continue
            matching_core_episode_ids.append(int(ep))
            if (
                args.ogbcube_live_2x2_variant is not None
                and (not decoupled_donor)
                and (wanted in _ogb_meta_distractor_colors(meta))
            ):
                excluded_same_color_distractor += 1
                continue
            eligible_episode_ids.add(int(ep))
        if not eligible_episode_ids:
            suffix = ""
            if args.ogbcube_live_2x2_variant is not None and (not decoupled_donor):
                suffix = " after excluding same-color distractors"
            raise RuntimeError(
                f"No OGB-Cube episodes matched core_color={wanted!r} in {filter_variant}{suffix}."
            )
        sampled_core_color_filter = {
            "requested": wanted,
            "sample_mode": "official_clean_base_core_filter"
            if decoupled_donor or args.ogbcube_live_2x2_variant is None
            else "pre_sample_exact",
            "variant": str(filter_variant),
            "base": str(Path(args.ogbcube_2x2_base).expanduser().resolve()),
            "matching_core_episodes": int(len(matching_core_episode_ids)),
            "excluded_same_color_distractor": int(excluded_same_color_distractor),
            "eligible_episodes": int(len(eligible_episode_ids)),
            "exclude_same_color_distractor": bool(not decoupled_donor),
        }
    sampled_starts = _sample_eval_starts(
        dataset,
        num_eval=int(args.num_eval),
        goal_offset_steps=int(args.goal_offset_steps),
        seed=int(args.seed),
        eligible_episode_ids=eligible_episode_ids,
    )
    episodes_idx = [int(x) for x in sampled_starts["episode_ids"]]
    start_steps = [int(x) for x in sampled_starts["start_steps"]]
    row_ids = np.asarray(sampled_starts["row_ids"], dtype=np.int64)
    if sampled_core_color_filter is not None:
        sampled_core_color_filter["kept"] = int(len(episodes_idx))
    base_core_color_filter = (
        dict(sampled_core_color_filter)
        if sampled_core_color_filter is not None
        else None
    )
    num_eval = len(episodes_idx)
    donor_episodes_idx = None
    donor_start_steps = None
    if (
        bool(args.decouple_donor_trajectory)
        and args.ogbcube_live_2x2_variant is not None
    ):
        donor_seed = int(
            args.seed
            if args.donor_trajectory_seed is None
            else args.donor_trajectory_seed
        )
        (
            donor_episodes_idx,
            donor_start_steps,
            donor_core_color_filter,
        ) = _sample_ogbcube_donor_layouts(
            base=args.ogbcube_2x2_base,
            variant=str(args.ogbcube_live_2x2_variant),
            num_eval=num_eval,
            seed=donor_seed,
            core_color=str(args.ogbcube_core_color_filter),
        )
        donor_core_color_filter["kept"] = int(num_eval)
        if base_core_color_filter is not None:
            base_core_color_filter["kept"] = int(num_eval)
            sampled_core_color_filter = {
                "requested": str(args.ogbcube_core_color_filter).strip().lower(),
                "sample_mode": "official_clean_base_plus_independent_donor",
                "base_filter": base_core_color_filter,
                "donor_filter": donor_core_color_filter,
                "kept": int(num_eval),
            }
        else:
            sampled_core_color_filter = donor_core_color_filter
    sequences = _load_eval_sequences(
        dataset,
        episodes_idx=episodes_idx,
        start_steps=start_steps,
        goal_offset_steps=int(args.goal_offset_steps),
        history_size=model_history_size,
        frame_skip=model_frame_skip,
    )
    prefer_full_state = "pusht" in str(args.env_name).lower()
    transform = _image_transform(
        dino_model_name=tswm_config.model.dino_model_name, img_size=int(args.img_size)
    )
    plan_config = swm.policy.PlanConfig(
        horizon=int(args.horizon),
        receding_horizon=min(int(args.receding_horizon), int(args.horizon)),
        history_len=model_history_size,
        action_block=int(tswm_config.data.target_offset),
        warm_start=bool(args.warm_start),
    )
    solver = swm.solver.CEMSolver(
        model=bridge,
        batch_size=max(1, min(int(args.solver_batch_size), num_eval)),
        num_samples=int(args.num_samples),
        var_scale=float(args.var_scale),
        n_steps=max(1, int(args.cem_steps)),
        topk=min(int(args.topk), int(args.num_samples)),
        device=bridge.device,
        seed=int(args.seed),
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform={"pixels": transform, "goal": transform},
    )
    world = swm.World(
        env_name=args.env_name,
        num_envs=num_eval,
        image_shape=(int(args.img_size), int(args.img_size)),
        history_size=world_history_size,
        frame_skip=world_frame_skip,
        max_episode_steps=max(
            int(args.eval_budget) * 2, int(args.goal_offset_steps) + 1
        ),
        verbose=0,
    )
    world.set_policy(policy)
    seeds, options = _build_reset_options(
        sequences["current_step"], sequences["goal_step"], num_envs=num_eval
    )
    if seeds is None:
        seeds = [int(args.seed) + i for i in range(num_eval)]
    world.reset(seed=seeds, options=options)
    _apply_dataset_state_callables(
        world,
        sequences["current_step"],
        sequences["goal_step"],
        prefer_full_state=prefer_full_state,
    )
    pixel_history_buffers = np.asarray(sequences["history_buffers"]["pixels"]).copy()
    action_history_buffers = (
        np.asarray(sequences["history_buffers"]["action"], dtype=np.float32).copy()
        if "action" in sequences["history_buffers"]
        else None
    )
    goal_images = _broadcast_goal_history(
        np.asarray(sequences["goal_step"]["goal"]),
        history_size=int(tswm_config.data.num_hist),
    )
    step_indices = np.asarray(start_steps, dtype=np.int64)
    goal_step_indices = step_indices + int(args.goal_offset_steps) - 1
    live_2x2_renderer = None
    eval_queries = None
    if args.ogbcube_live_2x2_variant is not None:
        live_2x2_renderer = OGBCubeLive2x2Renderer(
            base=args.ogbcube_2x2_base,
            variant=args.ogbcube_live_2x2_variant,
            episodes_idx=episodes_idx,
            donor_episodes_idx=donor_episodes_idx,
            base_start_steps=start_steps,
            donor_start_steps=donor_start_steps,
        )
        pixel_history_buffers = live_2x2_renderer.render_history(
            pixel_history_buffers, step_indices
        )
        goal_images = live_2x2_renderer.render_goal(goal_images, goal_step_indices)
    episode_successes = np.zeros(num_eval, dtype=bool)
    needs_flush = np.zeros(num_eval, dtype=bool)
    policy_infos = _build_policy_info(
        pixel_history_buffers=pixel_history_buffers,
        action_history_buffers=action_history_buffers,
        goal_images=goal_images,
        row_ids=row_ids,
        step_indices=step_indices,
        goal_step_indices=goal_step_indices,
        terminated=np.zeros(num_eval, dtype=bool),
        history_size=model_history_size,
        frame_skip=model_frame_skip,
        needs_flush=needs_flush,
        queries=eval_queries,
    )
    bridge._goal_cache.clear()
    bridge._history_cache.clear()
    for step_idx in range(int(args.eval_budget)):
        actions = _get_world_model_action(policy, policy_infos)
        (
            world.states,
            world.rewards,
            world.terminateds,
            world.truncateds,
            world.infos,
        ) = world.envs.step(actions)
        episode_successes = np.logical_or(
            episode_successes, np.asarray(world.terminateds, dtype=bool)
        )
        if hasattr(world.envs.unwrapped, "_autoreset_envs"):
            world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))
        next_step_indices = step_indices + 1
        if live_2x2_renderer is not None:
            live_core_pixels = _extract_latest_value(
                world.infos["pixels"], world_history_size
            )
            world.infos["pixels"] = live_2x2_renderer.render_batch(
                live_core_pixels, next_step_indices
            )
        pixel_history_buffers = _update_history_buffers(
            pixel_history_buffers, world.infos, world_history_size=world_history_size
        )
        if action_history_buffers is not None:
            action_history_buffers = _update_action_history_buffers(
                action_history_buffers, actions
            )
        step_indices = next_step_indices
        needs_flush = np.logical_or(
            np.asarray(world.terminateds, dtype=bool),
            np.asarray(world.truncateds, dtype=bool),
        )
        policy_infos = _build_policy_info(
            pixel_history_buffers=pixel_history_buffers,
            action_history_buffers=action_history_buffers,
            goal_images=goal_images,
            row_ids=row_ids,
            step_indices=step_indices,
            goal_step_indices=goal_step_indices,
            terminated=np.asarray(world.terminateds, dtype=bool),
            history_size=model_history_size,
            frame_skip=model_frame_skip,
            needs_flush=needs_flush,
            queries=eval_queries,
        )
        if np.all(episode_successes):
            break
    return float(np.mean(episode_successes.astype(np.float32)) * 100.0)
