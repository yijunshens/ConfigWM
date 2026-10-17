from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Any
import hdf5plugin
import numpy as np
import torch
import torch.nn.functional as F


def _to_device_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device)
    raise TypeError(f"Unsupported info_dict value type: {type(value)!r}")


def _permute_last_channel_to_first(images: torch.Tensor) -> torch.Tensor:
    if images.shape[-1] == 3:
        dims = list(range(images.ndim))
        return images.permute(*dims[:-3], dims[-1], dims[-3], dims[-2]).contiguous()
    return images


class _ActionStandardScaler:
    """Small StandardScaler-compatible adapter for LeWM policy.process."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean_ = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        self.scale_ = np.asarray(std, dtype=np.float32).reshape(1, -1)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.mean_) / self.scale_).astype(np.float32, copy=False)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x * self.scale_ + self.mean_).astype(np.float32, copy=False)


def _fit_action_scaler_from_dataset(dataset: Any) -> _ActionStandardScaler:
    actions = np.asarray(dataset.get_col_data("action"), dtype=np.float64)
    action_dim = int(actions.shape[-1])
    flat = np.nan_to_num(actions.reshape(-1, action_dim), copy=False)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-06, 1.0, std)
    return _ActionStandardScaler(mean.astype(np.float32), std.astype(np.float32))


class TSWMProjectorCostModel(torch.nn.Module):
    """Expose TSWM post-projector dynamics as a LEWM-style get_cost model."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        history_len: int,
        action_block: int,
        eval_query: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.history_len = int(history_len)
        self.action_block = int(action_block)
        self.eval_query = None if eval_query is None else str(eval_query)
        self.interpolate_pos_encoding = True
        self._goal_cache: dict[str, torch.Tensor] = {}
        self._history_cache: dict[str, torch.Tensor] = {}
        self.register_buffer("_action_mean", torch.empty(0), persistent=False)
        self.register_buffer("_action_std", torch.empty(0), persistent=False)
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def state_dim(self) -> int:
        return self.model.state_dim

    def set_action_normalizer(self, mean: np.ndarray, std: np.ndarray) -> None:
        mean_tensor = torch.as_tensor(
            mean, dtype=torch.float32, device=self.device
        ).reshape(-1)
        std_tensor = torch.as_tensor(
            std, dtype=torch.float32, device=self.device
        ).reshape(-1)
        std_tensor = torch.where(
            std_tensor < 1e-06, torch.ones_like(std_tensor), std_tensor
        )
        self._action_mean = mean_tensor
        self._action_std = std_tensor

    def _maybe_normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self._action_mean.numel() == 0:
            return actions
        mean = self._action_mean.to(device=actions.device, dtype=actions.dtype)
        std = self._action_std.to(device=actions.device, dtype=actions.dtype)
        view_shape = [1] * actions.ndim
        view_shape[-1] = int(mean.numel())
        return (actions - mean.reshape(view_shape)) / std.reshape(view_shape)

    def _ensure_history_pixels(
        self, pixels: Any, *, batch_size: int, num_samples: int
    ) -> torch.Tensor:
        if torch.is_tensor(pixels):
            tensor = pixels
        elif isinstance(pixels, np.ndarray):
            tensor = torch.from_numpy(pixels)
        else:
            raise TypeError(f"Unsupported pixels type: {type(pixels)!r}")
        if tensor.ndim == 5:
            tensor = tensor.unsqueeze(1)
        if tensor.shape[1] > 1:
            tensor = tensor[:, :1]
        tensor = _permute_last_channel_to_first(tensor)
        tensor = tensor.to(self.device).float()
        if tensor.shape[1] == 1 and num_samples > 1:
            tensor = tensor.expand(-1, num_samples, -1, -1, -1, -1)
        return tensor

    def _ensure_goal_pixels(
        self, goal: Any, *, batch_size: int, num_samples: int
    ) -> torch.Tensor:
        if torch.is_tensor(goal):
            tensor = goal
        elif isinstance(goal, np.ndarray):
            tensor = torch.from_numpy(goal)
        else:
            raise TypeError(f"Unsupported goal type: {type(goal)!r}")
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(1)
        if tensor.shape[1] > 1:
            tensor = tensor[:, :1]
        if tensor.ndim == 6:
            tensor = tensor[:, :, -1]
        tensor = _permute_last_channel_to_first(tensor)
        tensor = tensor.to(self.device).float()
        if num_samples > 1:
            tensor = tensor.expand(-1, num_samples, -1, -1, -1)
        return tensor

    def _action_chunk_shape(self, flat_action_dim: int) -> int:
        return flat_action_dim // self.action_block

    def _encode_history_states(
        self, info_dict: dict[str, Any], pixels: Any, batch_size: int, num_samples: int
    ) -> torch.Tensor:
        cache_key = self._cache_key(info_dict, batch_size, step_key="step_idx")
        if cache_key is not None and cache_key in self._history_cache:
            cached = self._history_cache[cache_key]
            return cached[:, None, ...].expand(-1, num_samples, *cached.shape[1:])
        pixels = self._ensure_history_pixels(
            pixels, batch_size=batch_size, num_samples=num_samples
        )
        _, _, hist_len, channels, height, width = pixels.shape
        base_pixels = pixels[:, 0]
        queries = (
            [self.eval_query] * batch_size if self.eval_query is not None else None
        )
        features = self.model.observation_encoder(
            base_pixels.reshape(batch_size, hist_len, channels, height, width),
            queries=queries,
        )
        states = self.model.projector(features)
        if cache_key is not None:
            self._history_cache[cache_key] = states.detach()
        return states[:, None, ...].expand(-1, num_samples, *states.shape[1:])

    def _cache_key(
        self, info_dict: dict[str, Any], batch_size: int, *, step_key: str
    ) -> str | None:
        if "id" not in info_dict or step_key not in info_dict:
            return None
        row_id = self._metadata_column(info_dict["id"], batch_size)
        step_idx = self._metadata_column(info_dict[step_key], batch_size)
        return f"{row_id.detach().cpu().tolist()}::{step_idx.detach().cpu().tolist()}"

    def _metadata_column(self, value: Any, batch_size: int) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
        else:
            tensor = torch.as_tensor(np.asarray(value))
        return tensor.reshape(batch_size, -1)[:, 0]

    def _encode_goal_state(
        self, info_dict: dict[str, Any], batch_size: int, num_samples: int
    ) -> torch.Tensor:
        cache_key = self._cache_key(info_dict, batch_size, step_key="goal_step_idx")
        if cache_key is not None and cache_key in self._goal_cache:
            cached = self._goal_cache[cache_key]
            return cached[:, None, ...].expand(-1, num_samples, *cached.shape[1:])
        goal = self._ensure_goal_pixels(
            info_dict["goal"], batch_size=batch_size, num_samples=num_samples
        )
        base_goal = goal[:, 0]
        queries = (
            [self.eval_query] * batch_size if self.eval_query is not None else None
        )
        features = self.model.observation_encoder(
            base_goal[:, None, ...], queries=queries
        )
        state = self.model.projector(features).squeeze(1)
        if cache_key is not None:
            self._goal_cache[cache_key] = state.detach()
        return state[:, None, ...].expand(-1, num_samples, *state.shape[1:])

    def _ensure_action_context(
        self,
        info_dict: dict[str, Any],
        batch_size: int,
        num_samples: int,
        per_step_action_dim: int,
    ) -> torch.Tensor | None:
        if self.history_len <= 1 or "action_context" not in info_dict:
            return None
        tensor = _to_device_tensor(info_dict["action_context"], self.device).float()
        if tensor.ndim == 4:
            tensor = tensor[:, None, ...]
        expected_context = self.history_len - 1
        tensor = self._maybe_normalize_actions(tensor)
        if tensor.shape[1] == 1 and num_samples > 1:
            tensor = tensor.expand(-1, num_samples, -1, -1, -1)
        return tensor.reshape(
            batch_size * num_samples,
            expected_context,
            self.action_block,
            per_step_action_dim,
        )

    def _rollout_predicted_states(
        self,
        history_states: torch.Tensor,
        action_candidates: torch.Tensor,
        action_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_samples, horizon, flat_action_dim = action_candidates.shape
        per_step_action_dim = self._action_chunk_shape(flat_action_dim)
        if history_states.ndim == 4:
            history_flat = history_states.reshape(
                batch_size * num_samples, self.history_len, -1
            )
        elif history_states.ndim == 5:
            history_flat = history_states.reshape(
                batch_size * num_samples,
                self.history_len,
                history_states.shape[-2],
                history_states.shape[-1],
            )
        else:
            raise ValueError(
                f"Unsupported history state shape: {tuple(history_states.shape)}"
            )
        predicted: list[torch.Tensor] = []
        past_action_context = action_context
        for step in range(horizon):
            step_action = action_candidates[:, :, step, :]
            step_action = step_action.reshape(
                batch_size * num_samples, self.action_block, per_step_action_dim
            )
            full_action_context = None
            if past_action_context is not None:
                full_action_context = torch.cat(
                    [past_action_context, step_action[:, None, ...]], dim=1
                )
            pred_state = self.model.predict_next(
                history_flat, step_action, action_context=full_action_context
            )
            predicted.append(
                pred_state.reshape(batch_size, num_samples, *pred_state.shape[1:])
            )
            if self.history_len == 1:
                history_flat = pred_state[:, None, ...]
            else:
                history_flat = torch.cat(
                    [history_flat[:, 1:, ...], pred_state[:, None, ...]], dim=1
                )
            if past_action_context is not None and full_action_context is not None:
                past_action_context = full_action_context[:, 1:, ...].contiguous()
        return torch.stack(predicted, dim=2)

    def criterion(
        self, predicted_states: torch.Tensor, goal_state: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(
            predicted_states[:, :, -1, ...], goal_state, reduction="none"
        ).mean(dim=tuple(range(2, predicted_states.ndim - 1)))

    @torch.inference_mode()
    def get_cost(
        self, info_dict: dict[str, Any], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        action_candidates = _to_device_tensor(action_candidates, self.device).float()
        batch_size, num_samples = action_candidates.shape[:2]
        raw_pixels = info_dict["pixels"]
        goal_state = self._encode_goal_state(info_dict, batch_size, num_samples)
        history_states = self._encode_history_states(
            info_dict, raw_pixels, batch_size, num_samples
        )
        per_step_action_dim = self._action_chunk_shape(action_candidates.shape[-1])
        action_context = self._ensure_action_context(
            info_dict, batch_size, num_samples, per_step_action_dim
        )
        predicted_states = self._rollout_predicted_states(
            history_states, action_candidates, action_context=action_context
        )
        return self.criterion(predicted_states, goal_state)


def _image_transform(*, dino_model_name: str, img_size: int):
    from src.data.preprocessor.dinov2 import DINOv2Preprocessor
    from torchvision.transforms import v2 as transforms

    preprocessor = DINOv2Preprocessor(
        model_name=dino_model_name, resolution=img_size, use_fast=True
    )
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=preprocessor._processor.image_mean,
                std=preprocessor._processor.image_std,
            ),
            transforms.Resize(size=img_size),
        ]
    )


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _episode_column_to_numpy(value: Any, *, column_name: str) -> np.ndarray:
    array = _as_numpy(value)
    if (
        column_name.startswith("pixels")
        and array.ndim == 4
        and (array.shape[1] in (1, 3))
    ):
        array = np.transpose(array, (0, 2, 3, 1))
    return array


def _dataset_name_and_cache_dir(data_dir: str | Path) -> tuple[str, Path]:
    path = Path(data_dir).expanduser().resolve()
    if path.name == "manifest.json":
        path = path.parent
    if path.suffix in {".h5", ".hdf5"}:
        return (path.stem, path.parent)
    return (path.name, path.parent)


def _get_episode_ids_column(dataset) -> str:
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def _build_full_history_buffer(
    sequence: np.ndarray, *, history_size: int, frame_skip: int
) -> np.ndarray:
    array = np.asarray(sequence)
    capacity = history_size * frame_skip
    if array.shape[0] >= capacity:
        return array[-capacity:].copy()
    pad_count = capacity - array.shape[0]
    pad = np.repeat(array[:1], pad_count, axis=0)
    return np.concatenate([pad, array], axis=0)


def _stack_from_history_buffer(
    full_buffer: np.ndarray, *, history_size: int, frame_skip: int
) -> np.ndarray:
    stacked = full_buffer[frame_skip - 1 :: frame_skip]
    return stacked


def _stack_batch_history_buffers(
    full_buffers: np.ndarray, *, history_size: int, frame_skip: int
) -> np.ndarray:
    stacked = full_buffers[:, frame_skip - 1 :: frame_skip]
    return stacked


def _extract_latest_value(value: Any, history_size: int) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim >= 2 and array.shape[1] == history_size:
        return array[:, -1]
    return array


def _load_eval_sequences(
    dataset,
    *,
    episodes_idx: list[int],
    start_steps: list[int],
    goal_offset_steps: int,
    history_size: int,
    frame_skip: int,
) -> dict[str, Any]:
    history_span = (history_size - 1) * frame_skip
    ep_idx_arr = np.asarray(episodes_idx, dtype=np.int64)
    start_steps_arr = np.asarray(start_steps, dtype=np.int64)
    history_start_steps = np.maximum(0, start_steps_arr - history_span)
    end_steps = start_steps_arr + goal_offset_steps
    data = dataset.load_chunk(ep_idx_arr, history_start_steps, end_steps)
    columns = dataset.column_names
    buffer_per_key: dict[str, list[np.ndarray]] = defaultdict(list)
    init_step_per_key: dict[str, list[np.ndarray]] = defaultdict(list)
    goal_step_per_key: dict[str, list[np.ndarray]] = defaultdict(list)
    target_frames_per_env: list[np.ndarray] = []
    for i, episode in enumerate(data):
        anchor_idx = int(start_steps_arr[i] - history_start_steps[i])
        pixels_for_video: np.ndarray | None = None
        for col in columns:
            if col.startswith("goal"):
                continue
            column_array = _episode_column_to_numpy(episode[col], column_name=col)
            if column_array.dtype.kind in ("U", "S", "O"):
                continue
            if column_array.shape[0] == 0:
                continue
            history_buffer = _build_full_history_buffer(
                column_array[: anchor_idx + 1],
                history_size=history_size,
                frame_skip=frame_skip,
            )
            stacked = _stack_from_history_buffer(
                history_buffer, history_size=history_size, frame_skip=frame_skip
            )
            buffer_per_key[col].append(history_buffer)
            init_step_per_key[col].append(stacked)
            goal_key = "goal" if col == "pixels" else f"goal_{col}"
            goal_step_per_key[goal_key].append(column_array[-1])
            if col == "pixels":
                pixels_for_video = column_array[anchor_idx:]
        target_frames_per_env.append(pixels_for_video)
    buffers = {k: np.stack(v) for k, v in buffer_per_key.items()}
    init_step = {k: np.stack(v) for k, v in init_step_per_key.items()}
    goal_step = {k: np.stack(v) for k, v in goal_step_per_key.items()}
    target_frames = np.stack(target_frames_per_env)
    current_step = {
        k: v[:, -1] if v.ndim >= 2 and v.shape[1] == history_size else v
        for k, v in init_step.items()
    }
    return {
        "history_buffers": buffers,
        "init_step": init_step,
        "goal_step": goal_step,
        "current_step": current_step,
        "target_frames": target_frames,
    }


def _build_reset_options(
    current_step: dict[str, np.ndarray],
    goal_step: dict[str, np.ndarray],
    *,
    num_envs: int,
) -> tuple[list[int] | None, list[dict[str, Any]]]:
    seeds = None
    if "seed" in current_step:
        seeds = [int(x) for x in np.asarray(current_step["seed"]).reshape(-1).tolist()]
    options = [{} for _ in range(num_envs)]
    variation_prefix = "variation."
    variation_keys = [k for k in current_step.keys() if k.startswith(variation_prefix)]
    if variation_keys:
        for i in range(num_envs):
            options[i]["variation"] = [
                k.removeprefix(variation_prefix) for k in variation_keys
            ]
            options[i]["variation_values"] = {
                k.removeprefix(variation_prefix): np.asarray(current_step[k])[i]
                for k in variation_keys
            }
    return (seeds, options)


def _state_key(
    current_step: dict[str, np.ndarray], *, prefer_full_state: bool = False
) -> str | None:
    candidates = (
        ("state", "pos_agent", "proprio")
        if prefer_full_state
        else ("proprio", "pos_agent", "state")
    )
    for key in candidates:
        if key in current_step:
            return key
    return None


def _goal_state_key(
    goal_step: dict[str, np.ndarray], *, prefer_full_state: bool = False
) -> str | None:
    candidates = (
        ("goal_state", "goal_pos_agent", "goal_proprio")
        if prefer_full_state
        else ("goal_proprio", "goal_pos_agent", "goal_state")
    )
    for key in candidates:
        if key in goal_step:
            return key
    return None


def _apply_dataset_state_callables(
    world: Any,
    current_step: dict[str, np.ndarray],
    goal_step: dict[str, np.ndarray],
    *,
    env_indices: np.ndarray | list[int] | None = None,
    prefer_full_state: bool = False,
) -> None:
    state_key = _state_key(current_step, prefer_full_state=prefer_full_state)
    goal_key = _goal_state_key(goal_step, prefer_full_state=prefer_full_state)
    has_ogb_state = "qpos" in current_step and "qvel" in current_step
    has_ogb_goal = "goal_privileged_block_0_pos" in goal_step
    if (
        state_key is None
        and goal_key is None
        and (not has_ogb_state)
        and (not has_ogb_goal)
    ):
        return
    indices = range(world.num_envs) if env_indices is None else env_indices
    envs = world.envs.unwrapped.envs
    state_values = None if state_key is None else np.asarray(current_step[state_key])
    goal_values = None if goal_key is None else np.asarray(goal_step[goal_key])
    qpos_values = None if not has_ogb_state else np.asarray(current_step["qpos"])
    qvel_values = None if not has_ogb_state else np.asarray(current_step["qvel"])
    target_pos_values = (
        None
        if not has_ogb_goal
        else np.asarray(goal_step["goal_privileged_block_0_pos"])
    )
    target_quat_values = (
        np.asarray(goal_step["goal_privileged_block_0_quat"])
        if "goal_privileged_block_0_quat" in goal_step
        else None
    )
    for env_idx in indices:
        env_unwrapped = envs[int(env_idx)].unwrapped
        if qpos_values is not None and qvel_values is not None:
            if hasattr(env_unwrapped, "set_state"):
                env_unwrapped.set_state(
                    qpos=qpos_values[int(env_idx)], qvel=qvel_values[int(env_idx)]
                )
            elif hasattr(env_unwrapped, "_set_state"):
                env_unwrapped._set_state(
                    np.concatenate(
                        [qpos_values[int(env_idx)], qvel_values[int(env_idx)]], axis=0
                    )
                )
        if target_pos_values is not None and hasattr(env_unwrapped, "set_target_pos"):
            kwargs = {"cube_id": 0, "target_pos": target_pos_values[int(env_idx)]}
            if target_quat_values is not None:
                kwargs["target_quat"] = target_quat_values[int(env_idx)]
            env_unwrapped.set_target_pos(**kwargs)
        if state_values is not None and hasattr(env_unwrapped, "_set_state"):
            env_unwrapped._set_state(state_values[int(env_idx)])
        if goal_values is not None and hasattr(env_unwrapped, "_set_goal_state"):
            env_unwrapped._set_goal_state(goal_values[int(env_idx)])


def _broadcast_goal_history(goal_images: np.ndarray, history_size: int) -> np.ndarray:
    return np.broadcast_to(
        goal_images[:, None, ...],
        (goal_images.shape[0], history_size, *goal_images.shape[1:]),
    ).copy()


def _build_action_context_from_history(
    action_history_buffers: np.ndarray, *, history_size: int, frame_skip: int
) -> np.ndarray:
    if history_size <= 1:
        return np.empty(
            (
                action_history_buffers.shape[0],
                0,
                frame_skip,
                action_history_buffers.shape[-1],
            ),
            dtype=np.float32,
        )
    action_history_buffers = np.asarray(action_history_buffers, dtype=np.float32)
    blocks = []
    for history_idx in range(history_size - 1):
        start = frame_skip - 1 + history_idx * frame_skip
        end = start + frame_skip
        blocks.append(action_history_buffers[:, start:end, :])
    return np.stack(blocks, axis=1).astype(np.float32, copy=False)


def _update_history_buffers(
    pixel_history_buffers: np.ndarray,
    env_infos: dict[str, Any],
    *,
    world_history_size: int,
) -> np.ndarray:
    latest_pixels = _extract_latest_value(env_infos["pixels"], world_history_size)
    latest_pixels = np.asarray(latest_pixels)
    return np.concatenate(
        [pixel_history_buffers[:, 1:], latest_pixels[:, None, ...]], axis=1
    )


def _update_action_history_buffers(
    action_history_buffers: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    latest_actions = np.asarray(actions, dtype=np.float32)
    if latest_actions.ndim == 1:
        latest_actions = latest_actions[None, :]
    # The last slot belongs to the current observation's outgoing action.
    updated = action_history_buffers.copy()
    updated[:, -1, :] = latest_actions
    return np.concatenate(
        [updated[:, 1:], np.zeros_like(updated[:, :1])], axis=1
    )


def _get_world_model_action(policy, policy_infos: dict[str, np.ndarray]) -> np.ndarray:
    action_buffer = getattr(policy, "_action_buffer", None)
    if action_buffer is not None and len(action_buffer) > 0:
        action = action_buffer.popleft()
        action = action.reshape(*policy.env.action_space.shape)
        action = action.numpy()
        if "action" in policy.process:
            action = policy.process["action"].inverse_transform(action)
        return action
    return policy.get_action(policy_infos)
