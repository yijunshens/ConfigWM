"""Helpers for analyzing OGBScene data-collection rollouts."""

from __future__ import annotations

from collections.abc import Sequence


def drawer_open_amount(drawer_pos: float) -> float:
    """Drawer openness where closed=0 and more open is positive."""
    return max(0.0, -float(drawer_pos))


def window_open_amount(window_pos: float) -> float:
    """Window openness where closed=0 and more open is positive."""
    return max(0.0, float(window_pos))


def is_in_drawer_region(
    block_pos: Sequence[float], drawer_handle_pos: Sequence[float]
) -> bool:
    """Mirror SceneEnv._is_in_drawer using exported privileged state."""
    drawer_pos_y = float(drawer_handle_pos[1])
    drawer_low = [0.21, drawer_pos_y - 0.27, 0.0]
    drawer_high = [0.45, drawer_pos_y - 0.07, 0.15]
    return all(low <= value <= high for value, low, high in zip(block_pos, drawer_low, drawer_high))


def summarize_episode_trace(trace: list[dict]) -> dict:
    """Reduce a per-step trace to a compact summary for classification."""
    if not trace:
        raise ValueError("trace must be non-empty")

    start = trace[0]
    end = trace[-1]

    target_segments: list[str] = []
    prev_target = None
    button_flip_count = 0
    prev_buttons = tuple(start["button_states"])
    max_drawer_open = 0.0
    max_window_open = 0.0
    ever_in_drawer = False

    for step in trace:
        target = str(step["target_task"])
        if target != prev_target:
            target_segments.append(target)
            prev_target = target

        buttons = tuple(step["button_states"])
        button_flip_count += sum(
            int(old != new) for old, new in zip(prev_buttons, buttons)
        )
        prev_buttons = buttons

        max_drawer_open = max(
            max_drawer_open, drawer_open_amount(step["drawer_pos"])
        )
        max_window_open = max(
            max_window_open, window_open_amount(step["window_pos"])
        )
        ever_in_drawer = ever_in_drawer or is_in_drawer_region(
            step["block_pos"], step["drawer_handle_pos"]
        )

    return {
        "start": start,
        "end": end,
        "metrics": {
            "target_segments": target_segments,
            "button_flip_count": button_flip_count,
            "max_drawer_open": max_drawer_open,
            "max_window_open": max_window_open,
            "ever_in_drawer": ever_in_drawer,
        },
    }


def classify_episode(summary: dict) -> str | None:
    """Assign a human-readable high-level label to a data-collection rollout."""
    start = summary["start"]
    end = summary["end"]
    metrics = summary["metrics"]

    start_drawer_open = drawer_open_amount(start["drawer_pos"])
    end_drawer_open = drawer_open_amount(end["drawer_pos"])
    start_window_open = window_open_amount(start["window_pos"])
    end_window_open = window_open_amount(end["window_pos"])
    end_in_drawer = is_in_drawer_region(
        end["block_pos"], end["drawer_handle_pos"]
    )

    if (
        (metrics.get("ever_in_drawer", False) or end_in_drawer)
        and "cube" in metrics["target_segments"]
        and (end_drawer_open > 0.07 or metrics["max_drawer_open"] > 0.10)
    ):
        return "put_in_drawer_like"

    if (
        start_drawer_open < 0.04
        and start_window_open < 0.04
        and (end_drawer_open > 0.10 or end_window_open > 0.10)
    ):
        return "open_like"

    if (
        (metrics["button_flip_count"] > 0 or "button" in metrics["target_segments"])
        and metrics["max_drawer_open"] > 0.08
        and metrics["max_window_open"] > 0.08
        and end_drawer_open < 0.04
        and end_window_open < 0.04
    ):
        return "button_then_close_like"

    return None


def classify_open_put_close_sequence(summary: dict) -> bool:
    """Detect a composed drawer-open -> cube-in-drawer -> drawer-close episode."""
    start = summary["start"]
    end = summary["end"]
    metrics = summary["metrics"]

    start_drawer_open = drawer_open_amount(start["drawer_pos"])
    end_drawer_open = drawer_open_amount(end["drawer_pos"])
    end_in_drawer = is_in_drawer_region(
        end["block_pos"], end["drawer_handle_pos"]
    )
    target_segments = metrics.get("target_segments", [])

    return bool(
        start_drawer_open < 0.04
        and metrics.get("max_drawer_open", 0.0) > 0.10
        and end_drawer_open < 0.04
        and (metrics.get("ever_in_drawer", False) or end_in_drawer)
        and "cube" in target_segments
        and target_segments.count("drawer") >= 1
    )
