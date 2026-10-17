from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

_OKLAB_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float32,
)
_OKLAB_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float32,
)
_OKLAB_M1_INV = np.linalg.inv(_OKLAB_M1).astype(np.float32)
_OKLAB_M2_INV = np.linalg.inv(_OKLAB_M2).astype(np.float32)


def zero_rgb_where_alpha_is_zero(patch_rgba: np.ndarray) -> np.ndarray:
    out = np.asarray(patch_rgba, dtype=np.uint8).copy()
    alpha_zero = out[..., 3] == 0
    if np.any(alpha_zero):
        out[alpha_zero, :3] = 0
    return out


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb_arr = np.asarray(rgb, dtype=np.float32)
    return np.where(
        rgb_arr <= 0.04045,
        rgb_arr / 12.92,
        ((rgb_arr + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb_arr = np.asarray(rgb, dtype=np.float32)
    rgb_arr = np.clip(rgb_arr, 0.0, None)
    return np.where(
        rgb_arr <= 0.0031308,
        12.92 * rgb_arr,
        1.055 * (rgb_arr ** (1.0 / 2.4)) - 0.055,
    ).astype(np.float32)


def _linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = np.asarray(rgb, dtype=np.float32) @ _OKLAB_M1.T
    lms_cuberoot = np.cbrt(lms)
    return (lms_cuberoot @ _OKLAB_M2.T).astype(np.float32)


def _oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lms_cuberoot = np.asarray(lab, dtype=np.float32) @ _OKLAB_M2_INV.T
    lms = lms_cuberoot ** 3
    return (lms @ _OKLAB_M1_INV.T).astype(np.float32)


def _target_hue_rad(target_rgb: np.ndarray) -> float:
    rgb = np.asarray(target_rgb, dtype=np.float32).reshape(-1)[:3] / 255.0
    lab = _linear_to_oklab(_srgb_to_linear(rgb[None, :]))[0]
    return float(np.arctan2(float(lab[2]), float(lab[1])))


def _visible_values(channel: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float32)[np.asarray(visible_mask, dtype=bool)]
    if values.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return values.astype(np.float32)


def _match_visible_channel_stats(
    channel: np.ndarray,
    visible_mask: np.ndarray,
    *,
    target_mean: float,
    target_std: float | None = None,
    blend: float = 1.0,
    floor: float | None = None,
    ceil: float | None = None,
) -> np.ndarray:
    out = np.asarray(channel, dtype=np.float32).copy()
    visible = np.asarray(visible_mask, dtype=bool)
    if not np.any(visible):
        return out

    current = out[visible]
    cur_mean = float(current.mean())
    cur_std = float(current.std())
    if target_std is None or cur_std < 1e-6 or float(target_std) <= 1e-6:
        remapped = current - cur_mean + float(target_mean)
    else:
        remapped = (
            (current - cur_mean)
            * (float(target_std) / max(cur_std, 1e-6))
            + float(target_mean)
        )
    remapped = (1.0 - float(blend)) * current + float(blend) * remapped
    if floor is not None or ceil is not None:
        lo = -np.inf if floor is None else float(floor)
        hi = np.inf if ceil is None else float(ceil)
        remapped = np.clip(remapped, lo, hi)
    out[visible] = remapped.astype(np.float32)
    return out


def _rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    rgb_arr = np.asarray(rgb, dtype=np.float32)
    return (
        0.299 * rgb_arr[..., 0]
        + 0.587 * rgb_arr[..., 1]
        + 0.114 * rgb_arr[..., 2]
    ).astype(np.float32)


def mask_orientation_rad(
    mask_bool: np.ndarray | None,
    *,
    min_points: int = 8,
    min_axis_ratio: float = 1.15,
) -> float | None:
    if mask_bool is None:
        return None
    mask = np.asarray(mask_bool, dtype=bool)
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    if ys.size < min_points:
        return None
    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_idx = int(np.argmax(eigvals))
    minor_idx = 1 - major_idx
    major_val = float(eigvals[major_idx])
    minor_val = float(max(eigvals[minor_idx], 1e-6))
    if major_val <= 1e-6 or major_val / minor_val < float(min_axis_ratio):
        return None
    major = eigvecs[:, major_idx]
    return float(np.arctan2(float(major[1]), float(major[0])))


def wrap_angle_delta_rad(delta: float) -> float:
    return float((float(delta) + np.pi) % (2.0 * np.pi) - np.pi)


def fill_pose_track(
    centers_xy: np.ndarray,
    center_valid: np.ndarray,
    *,
    angles_rad: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[int]]:
    centers = np.asarray(centers_xy, dtype=np.float32).copy()
    valid = np.asarray(center_valid, dtype=bool).copy()
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers_xy must have shape [N, 2]")

    filled_frame_indices: list[int] = []
    valid_indices = np.flatnonzero(valid & np.isfinite(centers).all(axis=1))
    if valid_indices.size > 0:
        for frame_idx in range(len(centers)):
            if valid[frame_idx] and np.isfinite(centers[frame_idx]).all():
                continue
            prev_candidates = valid_indices[valid_indices < frame_idx]
            next_candidates = valid_indices[valid_indices > frame_idx]
            prev_idx = None if prev_candidates.size == 0 else int(prev_candidates[-1])
            next_idx = None if next_candidates.size == 0 else int(next_candidates[0])
            if prev_idx is None and next_idx is None:
                continue
            if prev_idx is not None and next_idx is not None:
                t = float(frame_idx - prev_idx) / float(next_idx - prev_idx)
                centers[frame_idx] = (
                    (1.0 - t) * centers[prev_idx] + t * centers[next_idx]
                ).astype(np.float32)
            elif prev_idx is not None:
                centers[frame_idx] = centers[prev_idx]
            else:
                centers[frame_idx] = centers[next_idx]
            valid[frame_idx] = True
            filled_frame_indices.append(int(frame_idx))

    filled_angles: np.ndarray | None
    if angles_rad is None:
        filled_angles = None
    else:
        filled_angles = np.asarray(angles_rad, dtype=np.float32).copy()
        angle_known = np.isfinite(filled_angles)
        known_indices = np.flatnonzero(angle_known)
        if known_indices.size > 0:
            for frame_idx in range(len(filled_angles)):
                if angle_known[frame_idx]:
                    continue
                prev_candidates = known_indices[known_indices < frame_idx]
                next_candidates = known_indices[known_indices > frame_idx]
                prev_idx = None if prev_candidates.size == 0 else int(prev_candidates[-1])
                next_idx = None if next_candidates.size == 0 else int(next_candidates[0])
                if prev_idx is None and next_idx is None:
                    continue
                if prev_idx is not None and next_idx is not None:
                    t = float(frame_idx - prev_idx) / float(next_idx - prev_idx)
                    delta = wrap_angle_delta_rad(
                        float(filled_angles[next_idx]) - float(filled_angles[prev_idx])
                    )
                    filled_angles[frame_idx] = float(filled_angles[prev_idx]) + float(t) * delta
                elif prev_idx is not None:
                    filled_angles[frame_idx] = filled_angles[prev_idx]
                else:
                    filled_angles[frame_idx] = filled_angles[next_idx]

    return centers, filled_angles, valid, filled_frame_indices


def normalize_visible_luminance(
    luminance: np.ndarray,
    visible_mask: np.ndarray,
    *,
    reference_mask: np.ndarray | None = None,
    shadow_floor: float = 0.4,
    shadow_gamma: float = 0.8,
    contrast: float = 0.85,
    reference_floor: float = 0.68,
    reference_contrast: float = 0.4,
) -> np.ndarray:
    luma = np.clip(np.asarray(luminance, dtype=np.float32), 0.0, 1.0).copy()
    visible = np.asarray(visible_mask, dtype=bool)
    if not np.any(visible):
        return luma

    if reference_mask is None:
        sample_mask = visible
    else:
        sample_mask = np.asarray(reference_mask, dtype=bool) & visible
        if not np.any(sample_mask):
            sample_mask = visible

    sample = luma[sample_mask]
    lo = float(np.percentile(sample, 5.0))
    hi = float(np.percentile(sample, 95.0))
    if hi - lo < 1e-6:
        luma[visible] = np.clip(np.maximum(luma[visible], shadow_floor), shadow_floor, 1.0)
        return luma

    normalized = np.clip((luma - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    softened = shadow_floor + (1.0 - shadow_floor) * np.power(normalized, shadow_gamma)
    if contrast != 1.0:
        midpoint = float(np.median(softened[visible]))
        softened = np.clip(midpoint + (softened - midpoint) * contrast, shadow_floor, 1.0)
    if reference_mask is not None:
        reference = np.asarray(reference_mask, dtype=bool) & visible
        if np.any(reference):
            ref_values = softened[reference]
            ref_midpoint = max(float(np.median(ref_values)), reference_floor)
            flattened = ref_midpoint + (ref_values - ref_midpoint) * reference_contrast
            softened = softened.copy()
            softened[reference] = np.clip(
                np.maximum(flattened, reference_floor),
                reference_floor,
                1.0,
            )
    luma[visible] = softened[visible]
    return luma


def compress_visible_luminance(
    luminance: np.ndarray,
    visible_mask: np.ndarray,
    *,
    core_mask: np.ndarray | None = None,
    strength: float = 0.85,
) -> np.ndarray:
    out = np.clip(np.asarray(luminance, dtype=np.float32), 0.0, 1.0).copy()
    visible = np.asarray(visible_mask, dtype=bool)
    if not np.any(visible):
        return out
    if core_mask is None:
        core = visible
    else:
        core = np.asarray(core_mask, dtype=bool) & visible
        if not np.any(core):
            core = visible

    midpoint = float(np.median(out[core]))
    keep = float(np.clip(1.0 - strength, 0.0, 1.0))
    out[visible] = np.clip(
        midpoint + (out[visible] - midpoint) * keep,
        0.0,
        1.0,
    )
    return out


def template_luminance_reference(template_patch_rgba: np.ndarray) -> dict[str, float]:
    patch = zero_rgb_where_alpha_is_zero(template_patch_rgba).astype(np.float32)
    alpha = patch[..., 3] > 0
    if not np.any(alpha):
        return {
            "shadow_floor": 0.4,
            "reference_floor": 0.68,
            "contrast": 0.85,
            "reference_contrast": 0.4,
            "core_p10": 0.68,
            "core_median": 0.74,
            "core_p90": 0.9,
            "blend": 0.65,
        }

    core = _patch_core_mask(patch[..., 3].astype(np.uint8)) & alpha
    if not np.any(core):
        core = alpha

    rgb = patch[..., :3]
    luminance = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
    normalized = normalize_visible_luminance(
        luminance,
        alpha,
        reference_mask=core,
    )
    core_values = normalized[core]
    core_p10 = float(np.percentile(core_values, 10.0))
    core_median = float(np.median(core_values))
    core_p90 = float(np.percentile(core_values, 90.0))
    shadow_floor = float(np.clip(core_p10, 0.4, 0.9))
    reference_floor = float(np.clip(max(core_median, shadow_floor), shadow_floor, 0.95))
    spread = max(core_p90 - core_p10, 1e-6)
    reference_contrast = float(np.clip(0.25 + 0.5 * spread, 0.22, 0.55))
    blend = float(np.clip(0.5 + 0.4 * (reference_floor - 0.5), 0.5, 0.8))
    return {
        "shadow_floor": shadow_floor,
        "reference_floor": reference_floor,
        "contrast": 0.85,
        "reference_contrast": reference_contrast,
        "core_p10": core_p10,
        "core_median": core_median,
        "core_p90": core_p90,
        "blend": blend,
    }


def align_visible_luminance_to_template(
    luminance: np.ndarray,
    visible_mask: np.ndarray,
    *,
    core_mask: np.ndarray | None = None,
    template_reference: dict[str, float] | None = None,
) -> np.ndarray:
    out = np.clip(np.asarray(luminance, dtype=np.float32), 0.0, 1.0).copy()
    visible = np.asarray(visible_mask, dtype=bool)
    if template_reference is None or not np.any(visible):
        return out

    if core_mask is None:
        core = visible
    else:
        core = np.asarray(core_mask, dtype=bool) & visible
        if not np.any(core):
            core = visible

    current_values = out[core]
    cur_lo = float(np.percentile(current_values, 10.0))
    cur_hi = float(np.percentile(current_values, 90.0))
    target_lo = float(template_reference.get("core_p10", template_reference.get("shadow_floor", 0.4)))
    target_mid = float(template_reference.get("core_median", template_reference.get("reference_floor", 0.68)))
    target_hi = float(template_reference.get("core_p90", max(target_mid, target_lo + 1e-3)))
    blend = float(np.clip(template_reference.get("blend", 0.65), 0.0, 1.0))
    floor = float(template_reference.get("shadow_floor", min(target_lo, target_mid)))

    if cur_hi - cur_lo < 1e-6:
        remapped = np.full_like(out, target_mid)
    else:
        normalized = np.clip((out - cur_lo) / max(cur_hi - cur_lo, 1e-6), 0.0, 1.0)
        remapped = target_lo + normalized * max(target_hi - target_lo, 1e-6)

    out_visible = np.clip(
        (1.0 - blend) * out[visible] + blend * remapped[visible],
        floor,
        1.0,
    )
    out[visible] = out_visible
    return out


def recolor_patch_rgba_preserve_material(
    patch_rgba: np.ndarray,
    target_rgb: np.ndarray,
    *,
    template_patch_rgba: np.ndarray | None = None,
    hard_edge: bool = False,
    return_clip_ratio: bool = False,
    low_chroma_thresh: float = 0.01,
) -> np.ndarray | tuple[np.ndarray, float]:
    patch_u8 = zero_rgb_where_alpha_is_zero(patch_rgba)
    if hard_edge:
        patch_u8 = harden_patch_alpha(patch_u8)
    patch_float = patch_u8.astype(np.float32) / 255.0
    rgb = patch_float[..., :3]
    alpha = patch_float[..., 3:4]
    alpha_mask = alpha[..., 0] > 0
    if not np.any(alpha_mask):
        if return_clip_ratio:
            return patch_u8.copy(), 0.0
        return patch_u8.copy()

    core_mask = alpha[..., 0] >= (224.0 / 255.0)
    if not np.any(core_mask):
        core_mask = alpha_mask
    template_reference = (
        template_luminance_reference(template_patch_rgba)
        if template_patch_rgba is not None
        else None
    )

    lin = _srgb_to_linear(rgb)
    lab = _linear_to_oklab(lin)
    source_l_chan = lab[..., 0:1].copy()
    l_chan = lab[..., 0:1]
    a_chan = lab[..., 1:2]
    b_chan = lab[..., 2:3]
    l_chan[..., 0] = normalize_visible_luminance(
        l_chan[..., 0],
        alpha_mask,
        reference_mask=core_mask,
        shadow_floor=(
            float(template_reference["shadow_floor"])
            if template_reference is not None
            else 0.4
        ),
        reference_floor=(
            float(template_reference["reference_floor"])
            if template_reference is not None
            else 0.68
        ),
        contrast=(
            float(template_reference["contrast"])
            if template_reference is not None
            else 0.85
        ),
        reference_contrast=(
            float(template_reference["reference_contrast"])
            if template_reference is not None
            else 0.4
        ),
    )
    if hard_edge:
        l_chan[..., 0] = compress_visible_luminance(
            l_chan[..., 0],
            alpha_mask,
            core_mask=core_mask,
            strength=0.9,
        )

    chroma = np.sqrt(a_chan * a_chan + b_chan * b_chan)
    hue_rad = _target_hue_rad(target_rgb)
    new_a = chroma * np.cos(hue_rad)
    new_b = chroma * np.sin(hue_rad)
    low_c_mask = chroma < float(low_chroma_thresh)
    new_a = np.where(low_c_mask, a_chan, new_a)
    new_b = np.where(low_c_mask, b_chan, new_b)

    new_lab = np.concatenate([l_chan, new_a, new_b], axis=-1)
    source_visible_l = _visible_values(source_l_chan[..., 0], alpha_mask)
    if source_visible_l.size > 0:
        source_mean = float(source_visible_l.mean())
        source_std = float(source_visible_l.std())
        new_lab[..., 0] = _match_visible_channel_stats(
            new_lab[..., 0],
            alpha_mask,
            target_mean=source_mean,
            target_std=max(source_std, 1e-3),
            blend=0.82 if not hard_edge else 0.7,
            floor=0.0,
            ceil=1.0,
        )
        if hard_edge:
            new_lab[..., 0] = compress_visible_luminance(
                new_lab[..., 0],
                alpha_mask,
                core_mask=core_mask,
                strength=0.82,
            )
    new_lin = _oklab_to_linear(new_lab)
    new_rgb = _linear_to_srgb(new_lin)

    out_of_gamut = ((new_rgb < 0.0) | (new_rgb > 1.0)).any(axis=-1)
    clip_ratio = float(out_of_gamut[alpha_mask].mean()) if np.any(alpha_mask) else 0.0
    new_rgb = np.clip(new_rgb, 0.0, 1.0)
    new_rgb[~alpha_mask] = 0.0

    recolored = np.concatenate([new_rgb, alpha], axis=-1)
    recolored_u8 = np.round(np.clip(recolored, 0.0, 1.0) * 255.0).astype(np.uint8)
    recolored_u8 = zero_rgb_where_alpha_is_zero(recolored_u8)
    if hard_edge:
        recolored_u8 = harden_patch_alpha(recolored_u8)
    if return_clip_ratio:
        return recolored_u8, clip_ratio
    return recolored_u8


def soften_round_object_patch_rgba(
    patch_rgba: np.ndarray,
    *,
    shadow_floor: float = 0.52,
    contrast: float = 0.78,
    compress_strength: float = 0.28,
) -> np.ndarray:
    patch_u8 = zero_rgb_where_alpha_is_zero(patch_rgba)
    alpha_mask = patch_u8[..., 3] > 0
    if not np.any(alpha_mask):
        return patch_u8.copy()

    patch_float = patch_u8.astype(np.float32) / 255.0
    rgb = patch_float[..., :3]
    luminance = _rgb_to_luminance(rgb * 255.0) / 255.0
    normalized = normalize_visible_luminance(
        luminance,
        alpha_mask,
        reference_mask=_patch_core_mask(patch_u8[..., 3]) & alpha_mask,
        shadow_floor=shadow_floor,
        contrast=contrast,
        reference_floor=max(shadow_floor, 0.64),
        reference_contrast=0.5,
    )
    softened = compress_visible_luminance(
        normalized,
        alpha_mask,
        core_mask=_patch_core_mask(patch_u8[..., 3]) & alpha_mask,
        strength=compress_strength,
    )

    safe_luminance = np.maximum(luminance, 1e-3)
    gain = np.clip(softened / safe_luminance, 0.7, 1.8)
    out_rgb = np.clip(rgb * gain[..., None], 0.0, 1.0)
    out_rgb[~alpha_mask] = 0.0
    out = np.concatenate([out_rgb, patch_float[..., 3:4]], axis=-1)
    out_u8 = np.round(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return zero_rgb_where_alpha_is_zero(out_u8)


def render_round_object_patch_rgba(
    patch_rgba: np.ndarray,
    target_rgb: np.ndarray,
    *,
    residual_strength: float = 0.03,
    ambient: float = 0.58,
    diffuse: float = 0.34,
    specular_strength: float = 0.08,
    specular_power: float = 18.0,
) -> np.ndarray:
    patch_u8 = zero_rgb_where_alpha_is_zero(patch_rgba)
    alpha_mask = patch_u8[..., 3] > 0
    if not np.any(alpha_mask):
        return patch_u8.copy()

    patch_float = patch_u8.astype(np.float32) / 255.0
    rgb = patch_float[..., :3]
    ys, xs = np.where(alpha_mask)
    cx = float(xs.mean())
    cy = float(ys.mean())
    radius = max(1.0, float(np.sqrt(float(np.count_nonzero(alpha_mask)) / np.pi)))

    height, width = patch_u8.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    d = np.sqrt(dx * dx + dy * dy)

    alpha = np.clip(radius + 0.5 - d, 0.0, 1.0).astype(np.float32)
    visible = alpha > 0.0
    if not np.any(visible):
        return patch_u8.copy()

    safe_d = np.maximum(d, 1e-6)
    scale = np.minimum(1.0, radius / safe_d)
    x_in = dx * scale
    y_in = dy * scale
    z_in = np.sqrt(np.maximum(radius * radius - x_in * x_in - y_in * y_in, 0.0))
    nx = x_in / radius
    ny = y_in / radius
    nz = z_in / radius

    light_dir = np.array([-0.3, -0.5, 0.8], dtype=np.float32)
    light_dir /= max(float(np.linalg.norm(light_dir)), 1e-6)
    ndotl = np.clip(
        nx * float(light_dir[0]) + ny * float(light_dir[1]) + nz * float(light_dir[2]),
        0.0,
        1.0,
    )
    shade = float(ambient) + float(diffuse) * ndotl
    spec = np.power(ndotl, float(specular_power)) * float(specular_strength)

    target_arr = np.asarray(target_rgb, dtype=np.float32).reshape(-1)[:3] / 255.0
    target_lin = _srgb_to_linear(target_arr[None, :])[0]
    rgb_lin = target_lin[None, None, :] * shade[..., None] + spec[..., None]

    luminance = _rgb_to_luminance(rgb * 255.0) / 255.0
    blur_sigma = max(1.0, 0.4 * radius)
    premult_u8 = np.round(np.clip(_premultiply_patch_rgba(patch_u8), 0.0, 1.0) * 255.0).astype(np.uint8)
    blurred_img = Image.fromarray(premult_u8, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    blurred_float = np.asarray(blurred_img, dtype=np.float32) / 255.0
    blurred_u8 = _unpremultiply_patch_rgba(blurred_float)
    blurred_luminance = _rgb_to_luminance(blurred_u8[..., :3].astype(np.float32)) / 255.0
    residual = (luminance - blurred_luminance).astype(np.float32)
    residual_visible = residual[alpha_mask]
    if residual_visible.size > 0:
        residual = residual / max(float(residual_visible.std()), 1e-3)
    rgb_lin *= np.clip(1.0 + float(residual_strength) * residual[..., None], 0.9, 1.1)

    rgb_srgb = np.clip(_linear_to_srgb(rgb_lin), 0.0, 1.0)
    rgb_srgb[~visible] = 0.0
    out = np.concatenate([rgb_srgb, alpha[..., None]], axis=-1)
    out_u8 = np.round(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return zero_rgb_where_alpha_is_zero(out_u8)


def adapt_round_object_patch_rgba_to_background(
    patch_rgba: np.ndarray,
    background_rgb: np.ndarray,
    *,
    gain_floor: float = 0.85,
    gain_ceil: float = 1.15,
    gain_strength: float = 0.15,
) -> np.ndarray:
    patch_u8 = zero_rgb_where_alpha_is_zero(patch_rgba)
    alpha_mask = patch_u8[..., 3] > 0
    if not np.any(alpha_mask):
        return patch_u8.copy()

    background = np.asarray(background_rgb, dtype=np.uint8)
    if background.ndim == 1:
        background = background.reshape(1, 1, 3)
    bg_l = _rgb_to_luminance(background.astype(np.float32))
    patch_l = _rgb_to_luminance(patch_u8[..., :3].astype(np.float32))
    current_mean = float(np.mean(patch_l[alpha_mask]))
    bg_mean = float(np.median(bg_l))
    ratio = bg_mean / max(current_mean, 1e-3)
    gain = float(np.clip(1.0 + float(gain_strength) * (ratio - 1.0), gain_floor, gain_ceil))

    out = patch_u8.copy()
    rgb = out[..., :3].astype(np.float32) * gain
    out[..., :3] = np.round(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)
    return zero_rgb_where_alpha_is_zero(out)


def adapt_patch_rgba_to_background(
    patch_rgba: np.ndarray,
    background_rgb: np.ndarray,
    *,
    hard_edge: bool = False,
    lighting_strength: float = 0.7,
    chroma_strength: float = 0.0,
) -> np.ndarray:
    patch_u8 = zero_rgb_where_alpha_is_zero(patch_rgba)
    if hard_edge:
        patch_u8 = harden_patch_alpha(patch_u8)
    alpha_mask = patch_u8[..., 3] > 0
    if not np.any(alpha_mask):
        return patch_u8.copy()

    background = np.asarray(background_rgb, dtype=np.uint8)
    if background.ndim == 1:
        background = background.reshape(1, 1, 3)
    bg_lin = _srgb_to_linear(background.astype(np.float32) / 255.0)
    bg_lab = _linear_to_oklab(bg_lin)
    bg_l = bg_lab[..., 0]
    bg_ab = bg_lab[..., 1:3]
    bg_mean = float(bg_l.mean())
    bg_std = float(bg_l.std())
    bg_ab_mean = bg_ab.reshape(-1, 2).mean(axis=0).astype(np.float32)

    patch_float = patch_u8.astype(np.float32) / 255.0
    patch_lin = _srgb_to_linear(patch_float[..., :3])
    patch_lab = _linear_to_oklab(patch_lin)
    patch_l = patch_lab[..., 0]
    patch_visible = _visible_values(patch_l, alpha_mask)
    if patch_visible.size == 0:
        return patch_u8.copy()

    target_std = max(
        0.4 * float(patch_visible.std()) + 0.6 * max(bg_std, 1e-3),
        1e-3,
    )
    patch_lab[..., 0] = _match_visible_channel_stats(
        patch_lab[..., 0],
        alpha_mask,
        target_mean=(
            (1.0 - float(lighting_strength)) * float(patch_visible.mean())
            + float(lighting_strength) * bg_mean
        ),
        target_std=target_std,
        blend=float(lighting_strength),
        floor=0.0,
        ceil=1.0,
    )
    if chroma_strength > 0.0:
        patch_lab[..., 1] = (
            (1.0 - float(chroma_strength)) * patch_lab[..., 1]
            + float(chroma_strength) * float(bg_ab_mean[0])
        )
        patch_lab[..., 2] = (
            (1.0 - float(chroma_strength)) * patch_lab[..., 2]
            + float(chroma_strength) * float(bg_ab_mean[1])
        )

    adapted_lin = _oklab_to_linear(patch_lab)
    adapted_rgb = np.clip(_linear_to_srgb(adapted_lin), 0.0, 1.0)
    adapted_rgb[~alpha_mask] = 0.0
    adapted = np.concatenate([adapted_rgb, patch_float[..., 3:4]], axis=-1)
    adapted_u8 = np.round(np.clip(adapted, 0.0, 1.0) * 255.0).astype(np.uint8)
    adapted_u8 = zero_rgb_where_alpha_is_zero(adapted_u8)
    if hard_edge:
        adapted_u8 = harden_patch_alpha(adapted_u8)
    return adapted_u8


def render_contact_shadow_rgba(
    alpha_mask: np.ndarray,
    *,
    background_rgb: np.ndarray,
    offset_xy: tuple[int, int] = (0, 1),
    darkness: float = 0.72,
    max_alpha: int = 132,
) -> np.ndarray:
    alpha_arr = np.asarray(alpha_mask)
    if alpha_arr.ndim == 3:
        alpha_arr = alpha_arr[..., 3]
    alpha_bool = np.asarray(alpha_arr > 0, dtype=bool)
    h, w = alpha_bool.shape[:2]
    shadow = np.zeros((h, w, 4), dtype=np.uint8)
    if not np.any(alpha_bool):
        return shadow

    ys, xs = np.where(alpha_bool)
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    width = max(1.0, float(x1 - x0 + 1))
    height = max(1.0, float(y1 - y0 + 1))
    cx = float((x0 + x1) * 0.5 + float(offset_xy[0]))
    cy = float(y1 + 0.2 * height + float(offset_xy[1]))
    sigma_x = max(1.0, 0.55 * width)
    sigma_y = max(1.0, 0.22 * height)

    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(
        -0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2)
    ).astype(np.float32)
    blob *= (yy >= (y0 - 1)).astype(np.float32)
    blob_alpha = np.clip(blob * float(max_alpha), 0.0, 255.0)
    if float(blob_alpha.max()) <= 0.0:
        return shadow

    bg_rgb = np.asarray(background_rgb, dtype=np.float32).reshape(-1)[:3]
    shadow_rgb = np.clip(bg_rgb * float(darkness), 0.0, 255.0).astype(np.uint8)
    shadow[..., :3] = shadow_rgb[None, None, :]
    shadow[..., 3] = np.round(blob_alpha).astype(np.uint8)
    shadow = zero_rgb_where_alpha_is_zero(shadow)
    return shadow


def _patch_core_mask(alpha: np.ndarray) -> np.ndarray:
    alpha_arr = np.asarray(alpha, dtype=np.uint8)
    if alpha_arr.size == 0:
        return np.zeros_like(alpha_arr, dtype=bool)
    core = alpha_arr >= 224
    if np.any(core):
        return core
    return alpha_arr > 0


def _patch_crop_edge_contact_fraction(alpha_mask: np.ndarray) -> float:
    mask = np.asarray(alpha_mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return 1.0
    border = np.concatenate(
        [
            mask[0, :],
            mask[-1, :],
            mask[:, 0],
            mask[:, -1],
        ]
    )
    return float(np.count_nonzero(border) / max(border.size, 1))


def _patch_directional_asymmetry(core_mask: np.ndarray, luminance: np.ndarray) -> float:
    mask = np.asarray(core_mask, dtype=bool)
    if not np.any(mask):
        return 1.0
    luma = np.asarray(luminance, dtype=np.float32)

    def _masked_mean(values: np.ndarray, region_mask: np.ndarray) -> float:
        active = region_mask & mask
        if not np.any(active):
            return float(luma[mask].mean())
        return float(values[active].mean())

    ys, xs = np.where(mask)
    y_mid = int(np.floor((int(ys.min()) + int(ys.max())) / 2))
    x_mid = int(np.floor((int(xs.min()) + int(xs.max())) / 2))
    left = np.zeros_like(mask, dtype=bool)
    left[:, : x_mid + 1] = True
    right = np.zeros_like(mask, dtype=bool)
    right[:, x_mid + 1 :] = True
    top = np.zeros_like(mask, dtype=bool)
    top[: y_mid + 1, :] = True
    bottom = np.zeros_like(mask, dtype=bool)
    bottom[y_mid + 1 :, :] = True
    denom = max(float(np.mean(luma[mask])), 1.0)
    lr = abs(_masked_mean(luma, left) - _masked_mean(luma, right)) / denom
    tb = abs(_masked_mean(luma, top) - _masked_mean(luma, bottom)) / denom
    return float(max(lr, tb))


def _patch_relative_dark_fraction(core_mask: np.ndarray, luminance: np.ndarray) -> float:
    mask = np.asarray(core_mask, dtype=bool)
    if not np.any(mask):
        return 1.0
    luma = np.asarray(luminance, dtype=np.float32)
    core_luma = luma[mask]
    threshold = max(8.0, 0.5 * float(np.median(core_luma)))
    return float(np.count_nonzero(core_luma < threshold) / max(core_luma.size, 1))


def patch_quality_metrics(patch_rgba: np.ndarray) -> dict[str, float | int]:
    patch = np.asarray(patch_rgba, dtype=np.uint8)
    alpha = patch[..., 3]
    visible = alpha > 0
    area = int(np.count_nonzero(visible))
    if area == 0:
        return {
            'score': float('-inf'),
            'area': 0,
            'core_area': 0,
            'dark_fraction': 1.0,
            'asymmetry': 1.0,
            'edge_contact': 1.0,
        }

    core = _patch_core_mask(alpha)
    core_area = int(np.count_nonzero(core))
    if core_area < max(9, int(round(0.2 * area))):
        return {
            'score': float('-inf'),
            'area': area,
            'core_area': core_area,
            'dark_fraction': 1.0,
            'asymmetry': 1.0,
            'edge_contact': 1.0,
        }

    rgb = patch[..., :3].astype(np.float32)
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    edge_contact = _patch_crop_edge_contact_fraction(visible)
    asymmetry = _patch_directional_asymmetry(core, luminance)
    dark_fraction = _patch_relative_dark_fraction(core, luminance)
    score = (
        np.log1p(float(core_area))
        - 1.35 * dark_fraction
        - 0.9 * asymmetry
        - 0.2 * edge_contact
    )
    return {
        'score': float(score),
        'area': area,
        'core_area': core_area,
        'dark_fraction': float(dark_fraction),
        'asymmetry': float(asymmetry),
        'edge_contact': float(edge_contact),
    }


def select_best_valid_patch_index(step_valid: np.ndarray, patches: list[np.ndarray]) -> int | None:
    valid_indices = np.flatnonzero(np.asarray(step_valid, dtype=bool))
    if valid_indices.size == 0:
        return None
    candidates: list[tuple[int, dict[str, float | int]]] = []
    legacy_best_idx = None
    legacy_best_area = -1
    for idx in valid_indices.tolist():
        patch = np.asarray(patches[int(idx)], dtype=np.uint8)
        metrics = patch_quality_metrics(patch)
        area = int(metrics['area'])
        candidates.append((int(idx), metrics))
        if area > legacy_best_area:
            legacy_best_area = area
            legacy_best_idx = int(idx)

    quality_best_idx, quality_best_metrics = max(
        candidates,
        key=lambda item: (float(item[1]['score']), int(item[1]['area'])),
    )
    if legacy_best_idx is None:
        return int(quality_best_idx)

    legacy_metrics = next(metrics for idx, metrics in candidates if idx == legacy_best_idx)
    quality_best_score = float(quality_best_metrics['score'])
    legacy_score = float(legacy_metrics['score'])
    legacy_contaminated = (
        float(legacy_metrics['dark_fraction']) > 0.02
        or float(legacy_metrics['asymmetry']) > 0.45
    )
    best_cleaner = (
        float(quality_best_metrics['dark_fraction']) + 1e-6
        < float(legacy_metrics['dark_fraction'])
    )
    if (
        int(quality_best_idx) != int(legacy_best_idx)
        and quality_best_score < legacy_score + 0.2
        and not (legacy_contaminated and best_cleaner)
    ):
        return int(legacy_best_idx)
    return int(quality_best_idx)


def harden_patch_alpha(
    patch_rgba: np.ndarray,
    *,
    alpha_threshold: int | None = None,
) -> np.ndarray:
    patch = zero_rgb_where_alpha_is_zero(patch_rgba)
    alpha = np.asarray(patch[..., 3], dtype=np.uint8)
    if alpha_threshold is None:
        if np.any(alpha == 255):
            keep = alpha == 255
        else:
            keep = alpha >= 128
    else:
        keep = alpha >= int(alpha_threshold)
    out = patch.copy()
    out[..., 3] = np.where(keep, 255, 0).astype(np.uint8)
    out[out[..., 3] == 0, :3] = 0
    return out


def _premultiply_patch_rgba(patch_rgba: np.ndarray) -> np.ndarray:
    patch = zero_rgb_where_alpha_is_zero(patch_rgba).astype(np.float32) / 255.0
    patch[..., :3] *= patch[..., 3:4]
    return patch


def _unpremultiply_patch_rgba(patch_rgba: np.ndarray, *, eps: float = 1.0 / 255.0) -> np.ndarray:
    premultiplied = np.asarray(patch_rgba, dtype=np.float32)
    alpha = np.clip(premultiplied[..., 3:4], 0.0, 1.0)
    safe_alpha = np.where(alpha >= eps, alpha, 1.0)
    rgb = np.where(alpha >= eps, premultiplied[..., :3] / safe_alpha, 0.0)
    out = np.concatenate([np.clip(rgb, 0.0, 1.0), alpha], axis=-1)
    out_u8 = np.round(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return zero_rgb_where_alpha_is_zero(out_u8)


def _resample_premultiplied_patch_rgba(
    patch_rgba: np.ndarray,
    transform,
) -> np.ndarray:
    premultiplied = _premultiply_patch_rgba(patch_rgba)
    premultiplied_u8 = np.round(np.clip(premultiplied, 0.0, 1.0) * 255.0).astype(np.uint8)
    transformed = transform(Image.fromarray(premultiplied_u8, mode="RGBA"))
    transformed_float = np.asarray(transformed, dtype=np.float32) / 255.0
    return _unpremultiply_patch_rgba(transformed_float)


def scale_patch_rgba(
    patch_rgba: np.ndarray,
    scale_factor: float,
    *,
    resample: int = Image.BILINEAR,
) -> np.ndarray:
    patch_u8 = zero_rgb_where_alpha_is_zero(patch_rgba)
    if abs(float(scale_factor) - 1.0) < 1e-6:
        return patch_u8
    height, width = patch_u8.shape[:2]
    out_w = max(1, int(round(width * float(scale_factor))))
    out_h = max(1, int(round(height * float(scale_factor))))
    return _resample_premultiplied_patch_rgba(
        patch_u8,
        lambda img: img.resize((out_w, out_h), resample=resample),
    )


def rotate_patch_rgba(
    patch_rgba: np.ndarray,
    angle_deg: float,
    *,
    resample: int = Image.BILINEAR,
    expand: bool = True,
) -> np.ndarray:
    patch_u8 = zero_rgb_where_alpha_is_zero(patch_rgba)
    if abs(float(angle_deg)) < 1e-6:
        return patch_u8
    return _resample_premultiplied_patch_rgba(
        patch_u8,
        lambda img: img.rotate(
            angle=float(angle_deg),
            resample=resample,
            expand=expand,
            fillcolor=(0, 0, 0, 0),
        ),
    )


def render_template_patch_rgba(
    template_patch_rgba: np.ndarray,
    *,
    global_scale_factor: float = 1.0,
    rotation_delta_rad: float = 0.0,
    hard_edge: bool = False,
) -> np.ndarray:
    resample = Image.NEAREST if hard_edge else Image.BILINEAR
    rendered = scale_patch_rgba(
        template_patch_rgba,
        global_scale_factor,
        resample=resample,
    )
    if abs(float(rotation_delta_rad)) < 1e-6:
        return rendered
    return rotate_patch_rgba(
        rendered,
        float(np.rad2deg(rotation_delta_rad)),
        resample=resample,
    )


def render_template_patch_from_masks(
    template_patch_rgba: np.ndarray,
    *,
    template_mask: np.ndarray | None,
    current_mask: np.ndarray | None,
    global_scale_factor: float = 1.0,
    hard_edge: bool = False,
    allow_rotation: bool = True,
) -> np.ndarray:
    if not allow_rotation:
        rotation_delta = 0.0
    else:
        template_angle = mask_orientation_rad(template_mask)
        current_angle = mask_orientation_rad(current_mask)
        if template_angle is None or current_angle is None:
            rotation_delta = 0.0
        else:
            rotation_delta = wrap_angle_delta_rad(float(current_angle) - float(template_angle))
    return render_template_patch_rgba(
        template_patch_rgba,
        global_scale_factor=float(global_scale_factor),
        rotation_delta_rad=float(rotation_delta),
        hard_edge=hard_edge,
    )


def estimate_patch_background_rgb(
    patch_rgba: np.ndarray,
    *,
    alpha_threshold: int = 64,
) -> np.ndarray:
    patch_u8 = zero_rgb_where_alpha_is_zero(patch_rgba)
    rgb = patch_u8[..., :3].astype(np.float32)
    alpha = patch_u8[..., 3]
    candidates = rgb[(alpha > 0) & (alpha <= alpha_threshold)]
    if candidates.size == 0:
        candidates = rgb[(alpha > 0) & (alpha < 255)]
    if candidates.size == 0:
        border_rgb = np.concatenate(
            [
                rgb[0, :, :],
                rgb[-1, :, :],
                rgb[:, 0, :],
                rgb[:, -1, :],
            ],
            axis=0,
        )
        border_alpha = np.concatenate(
            [
                alpha[0, :],
                alpha[-1, :],
                alpha[:, 0],
                alpha[:, -1],
            ],
            axis=0,
        )
        candidates = border_rgb[border_alpha > 0]
        if candidates.size == 0:
            candidates = border_rgb
    return np.median(candidates, axis=0).astype(np.uint8)


def decontaminate_patch_rgba(
    patch_rgba: np.ndarray,
    *,
    background_rgb: np.ndarray | None = None,
    alpha_floor: int = 16,
    eps: float = 1e-6,
) -> np.ndarray:
    out = zero_rgb_where_alpha_is_zero(patch_rgba).astype(np.float32)
    if background_rgb is None:
        background_rgb = estimate_patch_background_rgb(patch_rgba)
    background_rgb = np.asarray(background_rgb, dtype=np.float32)

    alpha_u8 = out[..., 3].copy()
    if alpha_floor > 0:
        alpha_u8 = alpha_u8.copy()
        alpha_u8[alpha_u8 <= float(alpha_floor)] = 0.0
    out[..., 3] = alpha_u8

    alpha = out[..., 3:4] / 255.0
    rgb = out[..., :3]
    nonzero = alpha[..., 0] > eps
    if np.any(nonzero):
        fg = (rgb[nonzero] - (1.0 - alpha[nonzero]) * background_rgb[None, :]) / np.clip(
            alpha[nonzero],
            eps,
            1.0,
        )
        rgb[nonzero] = np.clip(fg, 0.0, 255.0)
    out[..., :3] = rgb
    out_u8 = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return zero_rgb_where_alpha_is_zero(out_u8)
