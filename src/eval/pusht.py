from __future__ import annotations
import argparse
import json
from collections import defaultdict
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
    _fit_action_scaler_from_dataset,
    _get_episode_ids_column,
    _get_world_model_action,
    _image_transform,
    _load_eval_sequences,
    _stack_batch_history_buffers,
    _update_action_history_buffers,
    _update_history_buffers,
)

OGBCUBE_PALETTE = {
    "red": (226, 50, 62),
    "blue": (36, 104, 226),
    "green": (24, 188, 72),
    "orange": (238, 140, 34),
    "purple": (160, 74, 212),
    "cyan": (0, 190, 210),
}
DINO_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINO_IMAGE_STD = (0.229, 0.224, 0.225)


def _pusht_matched_core_color_episode_ids(
    dataset_dir: Path, *, core_color: str
) -> tuple[set[int], dict[str, Any]]:
    wanted = str(core_color).strip().lower()
    root = Path(dataset_dir).expanduser().resolve()
    paths = sorted(root.glob("shard_*/episode_metadata.jsonl"))
    if not paths:
        paths = sorted(root.glob("**/episode_metadata.jsonl"))

    def metadata_episode_id(item: dict[str, Any]) -> int:
        for key in (
            "episode",
            "global_episode_idx",
            "episode_idx",
            "ep_idx",
            "local_episode_idx",
        ):
            value = item.get(key)
            if value is not None:
                return int(value)
        raise KeyError(
            "Push-T metadata item has no episode id field; expected one of episode/global_episode_idx/episode_idx/ep_idx/local_episode_idx"
        )

    eligible: set[int] = set()
    total = 0
    matching_core = 0
    same_color_donor = 0
    color_counts: dict[str, int] = defaultdict(int)
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            total += 1
            color = str(item.get("core_color", "")).strip().lower()
            if color:
                color_counts[color] += 1
            if color != wanted:
                continue
            matching_core += 1
            donor_colors: set[str] = set()
            for key in ("distractor_colors", "donor_colors"):
                value = item.get(key)
                if isinstance(value, list):
                    donor_colors.update(
                        (str(x).strip().lower() for x in value if str(x).strip())
                    )
            for key in ("distractors", "donors"):
                value = item.get(key)
                if isinstance(value, list):
                    for donor in value:
                        if isinstance(donor, dict) and donor.get("color") is not None:
                            donor_colors.add(str(donor["color"]).strip().lower())
            if wanted in donor_colors:
                same_color_donor += 1
                continue
            eligible.add(metadata_episode_id(item))
    return (
        eligible,
        {
            "requested": wanted,
            "sample_mode": "pre_sample_exact",
            "dataset_dir": str(root),
            "metadata_files": int(len(paths)),
            "total_episodes": int(total),
            "core_color_counts": dict(sorted(color_counts.items())),
            "matching_core_episodes": int(matching_core),
            "excluded_same_color_donor": int(same_color_donor),
            "eligible_episodes": int(len(eligible)),
        },
    )


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
        eligible = np.asarray(
            sorted((int(x) for x in eligible_episode_ids)), dtype=np.int64
        )
        valid_mask = np.logical_and(valid_mask, np.isin(episode_idx, eligible))
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


def _sample_donor_trajectory_starts_from_pusht_variant(
    dataset_dir: Path,
    *,
    num_eval: int,
    goal_offset_steps: int,
    eval_budget: int,
    seed: int,
    core_color: str,
) -> tuple[list[int], list[int], dict[str, Any]]:
    import h5py

    eligible_episode_ids, donor_filter = _pusht_matched_core_color_episode_ids(
        dataset_dir, core_color=core_color
    )
    root = Path(dataset_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    max_required = int(goal_offset_steps) + int(eval_budget) - 1
    candidates: list[tuple[int, int]] = []
    for shard in manifest.get("shards", []):
        h5_path = root / str(shard["path"])
        with h5py.File(h5_path, "r") as h5:
            episode_ids = np.asarray(h5["episode_idx"], dtype=np.int64)
            step_idx = np.asarray(h5["step_idx"], dtype=np.int64)
            for ep in sorted(set((int(x) for x in episode_ids.tolist()))):
                if ep not in eligible_episode_ids:
                    continue
                ep_steps = step_idx[episode_ids == ep]
                if ep_steps.size == 0:
                    continue
                max_step = int(np.max(ep_steps))
                max_start = max_step - max_required
                if max_start < 0:
                    continue
                candidates.append((int(ep), int(max_start)))
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(
        len(candidates), size=int(num_eval), replace=len(candidates) < int(num_eval)
    )
    donor_episodes: list[int] = []
    donor_starts: list[int] = []
    for idx in picked.tolist():
        ep, max_start = candidates[int(idx)]
        donor_episodes.append(int(ep))
        donor_starts.append(int(rng.integers(0, max_start + 1)))
    donor_filter = dict(donor_filter)
    donor_filter.update(
        {
            "sample_mode": "independent_donor_trajectory",
            "num_candidates": int(len(candidates)),
            "sampled": int(len(donor_episodes)),
            "seed": int(seed),
            "max_required_steps": int(max_required),
        }
    )
    return (donor_episodes, donor_starts, donor_filter)


class PushTMatchedDatasetOverlay:
    def __init__(
        self,
        *,
        disturbed_dir: Path,
        clean_dir: Path,
        episodes_idx: list[int],
        start_steps: list[int],
        donor_episodes_idx: list[int] | None = None,
        donor_start_steps: list[int] | None = None,
        goal_offset_steps: int,
        eval_budget: int,
        threshold: float,
    ) -> None:
        self.disturbed_dir = disturbed_dir.expanduser().resolve()
        self.clean_dir = clean_dir.expanduser().resolve()
        self.base_episodes_idx = [int(x) for x in episodes_idx]
        self.base_start_steps = [int(x) for x in start_steps]
        self.episodes_idx = (
            [int(x) for x in donor_episodes_idx]
            if donor_episodes_idx is not None
            else [int(x) for x in episodes_idx]
        )
        self.start_steps = (
            [int(x) for x in donor_start_steps]
            if donor_start_steps is not None
            else [int(x) for x in start_steps]
        )
        self.goal_offset_steps = int(goal_offset_steps)
        self.eval_budget = int(eval_budget)
        self.threshold = float(threshold)
        self.disturbed = self._open_variant(self.disturbed_dir)
        self.clean = self._open_variant(self.clean_dir)
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.variant = self.disturbed_dir.name

    @property
    def enabled(self) -> bool:
        return bool(self.episodes_idx)

    @staticmethod
    def _open_variant(root: Path) -> dict[str, Any]:
        import h5py

        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = []
        for shard in manifest.get("shards", []):
            h5_path = root / str(shard["path"])
            meta_path = h5_path.parent / "episode_metadata.jsonl"
            episodes = [int(x) for x in shard.get("episodes", [])]
            shards.append(
                {
                    "path": h5_path,
                    "meta_path": meta_path,
                    "episodes": episodes,
                    "episode_set": set(episodes),
                }
            )
        return {"root": root, "manifest": manifest, "shards": shards, "h5py": h5py}

    @staticmethod
    def _find_shard(opened: dict[str, Any], episode_idx: int) -> dict[str, Any]:
        for shard in opened["shards"]:
            if int(episode_idx) in shard["episode_set"]:
                return shard
        raise KeyError(f"episode {episode_idx} not found under {opened['root']}")

    @staticmethod
    def _read_frame(opened: dict[str, Any], episode_idx: int, step: int) -> np.ndarray:
        shard = PushTMatchedDatasetOverlay._find_shard(opened, episode_idx)
        h5py = opened["h5py"]
        with h5py.File(shard["path"], "r") as h5:
            episode_ids = np.asarray(h5["episode_idx"])
            steps = np.asarray(h5["step_idx"])
            mask = np.logical_and(episode_ids == int(episode_idx), steps == int(step))
            rows = np.flatnonzero(mask)
            if rows.size == 0:
                ep_rows = np.flatnonzero(episode_ids == int(episode_idx))
                row = int(ep_rows[min(max(int(step), 0), len(ep_rows) - 1)])
            else:
                row = int(rows[0])
            return np.asarray(h5["pixels"][row], dtype=np.uint8)

    def _donor_frame_and_mask(
        self, episode_idx: int, step: int
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (int(episode_idx), int(step))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        disturbed = self._read_frame(self.disturbed, int(episode_idx), int(step))
        clean = self._read_frame(self.clean, int(episode_idx), int(step))
        diff = np.max(
            np.abs(disturbed.astype(np.int16) - clean.astype(np.int16)), axis=-1
        )
        mask = diff > float(self.threshold)
        self._cache[key] = (disturbed, mask)
        return (disturbed, mask)

    @staticmethod
    def _resize_like(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        if image.shape[:2] == shape:
            return image
        return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

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
            donor_step = int(self.start_steps[env_idx]) + (
                int(steps[env_idx]) - int(self.base_start_steps[env_idx])
            )
            donor, mask = self._donor_frame_and_mask(
                self.episodes_idx[env_idx], donor_step
            )
            donor = self._resize_like(donor, out[env_idx].shape[:2])
            mask = self._resize_like(
                mask.astype(np.uint8), out[env_idx].shape[:2]
            ).astype(bool)
            if is_float_01:
                out[env_idx][mask] = (donor[mask].astype(np.float32) / 255.0).astype(
                    out.dtype
                )
            else:
                out[env_idx][mask] = donor[mask].astype(out.dtype)
        return out

    def apply_history(
        self, images: np.ndarray, *, step_indices: np.ndarray, frame_skip: int
    ) -> np.ndarray:
        arr = np.asarray(images)
        if arr.ndim != 5:
            return self.apply_frame_batch(arr, steps=step_indices)
        out = arr.copy()
        capacity = out.shape[1]
        for hist_idx in range(capacity):
            hist_steps = np.maximum(
                0, np.asarray(step_indices, dtype=np.int64) - (capacity - 1 - hist_idx)
            )
            out[:, hist_idx] = self.apply_frame_batch(
                out[:, hist_idx], steps=hist_steps
            )
        return out

    def apply_goal(self, images: np.ndarray, *, start_steps: list[int]) -> np.ndarray:
        goal_steps = (
            np.asarray(start_steps, dtype=np.int64) + int(self.goal_offset_steps) - 1
        )
        arr = np.asarray(images)
        if arr.ndim == 5:
            out = arr.copy()
            for hist_idx in range(out.shape[1]):
                out[:, hist_idx] = self.apply_frame_batch(
                    out[:, hist_idx], steps=goal_steps
                )
            return out
        return self.apply_frame_batch(arr, steps=goal_steps)

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "render_mode": "pusht_matched_dataset_donor_pixels",
            "disturbed_dir": str(self.disturbed_dir),
            "clean_dir": str(self.clean_dir),
            "variant": self.variant,
            "threshold": float(self.threshold),
            "base_episodes_idx": [int(x) for x in self.base_episodes_idx],
            "base_start_steps": [int(x) for x in self.base_start_steps],
            "donor_episodes_idx": [int(x) for x in self.episodes_idx],
            "donor_start_steps": [int(x) for x in self.start_steps],
            "goal_offset_steps": int(self.goal_offset_steps),
            "eval_budget": int(self.eval_budget),
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
    matched_core_color_filter = None
    if (
        not bool(args.decouple_donor_trajectory)
        and args.matched_dataset_overlay_dir is not None
        and (args.matched_core_color_filter is not None)
    ):
        (
            eligible_episode_ids,
            matched_core_color_filter,
        ) = _pusht_matched_core_color_episode_ids(
            args.matched_dataset_overlay_dir,
            core_color=str(args.matched_core_color_filter),
        )
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
    if matched_core_color_filter is not None:
        matched_core_color_filter["kept"] = int(len(episodes_idx))
    num_eval = len(episodes_idx)
    donor_episodes_idx = None
    donor_start_steps = None
    if (
        bool(args.decouple_donor_trajectory)
        and args.matched_dataset_overlay_dir is not None
    ):
        donor_seed = int(
            args.seed
            if args.donor_trajectory_seed is None
            else args.donor_trajectory_seed
        )
        (
            donor_episodes_idx,
            donor_start_steps,
            matched_core_color_filter,
        ) = _sample_donor_trajectory_starts_from_pusht_variant(
            args.matched_dataset_overlay_dir,
            num_eval=num_eval,
            goal_offset_steps=int(args.goal_offset_steps),
            eval_budget=int(args.eval_budget),
            seed=donor_seed,
            core_color=str(args.matched_core_color_filter),
        )
        matched_core_color_filter["kept"] = int(num_eval)
    sequences = _load_eval_sequences(
        dataset,
        episodes_idx=episodes_idx,
        start_steps=start_steps,
        goal_offset_steps=int(args.goal_offset_steps),
        history_size=model_history_size,
        frame_skip=model_frame_skip,
    )
    matched_overlay = None
    if args.matched_dataset_overlay_dir is not None:
        matched_overlay = PushTMatchedDatasetOverlay(
            disturbed_dir=args.matched_dataset_overlay_dir,
            clean_dir=args.matched_clean_dataset_dir,
            episodes_idx=episodes_idx,
            start_steps=start_steps,
            donor_episodes_idx=donor_episodes_idx,
            donor_start_steps=donor_start_steps,
            goal_offset_steps=int(args.goal_offset_steps),
            eval_budget=int(args.eval_budget),
            threshold=float(args.matched_overlay_threshold),
        )
    donor = None
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
    step_indices = np.asarray(start_steps, dtype=np.int64)
    goal_step_indices = step_indices + int(args.goal_offset_steps) - 1
    pixel_history_buffers = np.asarray(sequences["history_buffers"]["pixels"]).copy()
    if donor is not None and donor.enabled:
        pixel_history_buffers = donor.apply_history(
            pixel_history_buffers, step=0, frame_skip=model_frame_skip
        )
    action_history_buffers = (
        np.asarray(sequences["history_buffers"]["action"], dtype=np.float32).copy()
        if "action" in sequences["history_buffers"]
        else None
    )
    goal_base_images = np.asarray(sequences["goal_step"]["goal"])
    if matched_overlay is not None and matched_overlay.enabled:
        goal_base_images = matched_overlay.apply_goal(
            goal_base_images, start_steps=start_steps
        )
    if donor is not None and donor.enabled:
        goal_base_images = donor.apply_goal(goal_base_images)
    goal_images = _broadcast_goal_history(
        goal_base_images, history_size=int(tswm_config.data.num_hist)
    )
    episode_successes = np.zeros(num_eval, dtype=bool)
    needs_flush = np.zeros(num_eval, dtype=bool)

    def build_policy_info_for_current_buffers(
        terminated: np.ndarray,
    ) -> dict[str, np.ndarray]:
        visual_pixel_buffers = pixel_history_buffers
        if matched_overlay is not None and matched_overlay.enabled:
            visual_pixel_buffers = matched_overlay.apply_history(
                visual_pixel_buffers,
                step_indices=step_indices,
                frame_skip=model_frame_skip,
            )
        return _build_policy_info(
            pixel_history_buffers=visual_pixel_buffers,
            action_history_buffers=action_history_buffers,
            goal_images=goal_images,
            row_ids=row_ids,
            step_indices=step_indices,
            goal_step_indices=goal_step_indices,
            terminated=terminated,
            history_size=model_history_size,
            frame_skip=model_frame_skip,
            needs_flush=needs_flush,
        )

    policy_infos = build_policy_info_for_current_buffers(np.zeros(num_eval, dtype=bool))
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
        pixel_history_buffers = _update_history_buffers(
            pixel_history_buffers, world.infos, world_history_size=world_history_size
        )
        next_step_indices = step_indices + 1
        if donor is not None and donor.enabled:
            pixel_history_buffers = donor.apply_history(
                pixel_history_buffers, step=step_idx + 1, frame_skip=model_frame_skip
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
        policy_infos = build_policy_info_for_current_buffers(
            np.asarray(world.terminateds, dtype=bool)
        )
        if np.all(episode_successes):
            break
    return float(np.mean(episode_successes.astype(np.float32)) * 100.0)
