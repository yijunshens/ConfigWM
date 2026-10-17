"""Resolve supported backbone names and cache official V-JEPA weights."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
import urllib.request


MODEL_IDS = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-giant": "facebook/dinov2-giant",
    "dinov3-7b": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "lingbot-giant": "robbyant/lingbot-vision-vit-giant",
}
VJEPA_FILES = {
    "vjepa2-vitl": "vitl.pt",
    "vjepa2-vith": "vith.pt",
    "vjepa2-vitg": "vitg.pt",
    "vjepa2-vitg-384": "vitg-384.pt",
    "vjepa2.1-vitb-384": "vjepa2_1_vitb_dist_vitG_384.pt",
    "vjepa2.1-vitl-384": "vjepa2_1_vitl_dist_vitG_384.pt",
    "vjepa2.1-vitg-384": "vjepa2_1_vitg_384.pt",
    "vjepa2.1-vitG-384": "vjepa2_1_vitG_384.pt",
}
VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2/"
VJEPA_GIGANTIC_URL = VJEPA_BASE_URL + VJEPA_FILES["vjepa2.1-vitG-384"]


def resolve_backbone(value: str) -> tuple[str, str | None, str | None]:
    """Return preset, optional local HF/LingBot model, and native checkpoint."""
    name = str(value).rstrip("/")
    if name in MODEL_IDS or name == "vjepa2.1-gigantic":
        return name, None, None
    for preset, model_id in MODEL_IDS.items():
        if name in {model_id, "https://huggingface.co/" + model_id}:
            return preset, None, None
    if name in {VJEPA_GIGANTIC_URL, "vjepa2.1-vitG-384"}:
        return "vjepa2.1-gigantic", None, None

    path = Path(name).expanduser()
    if path.is_dir():
        config_path = path / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            key = (config.get("model_type"), config.get("hidden_size"), config.get("patch_size"))
            preset = {
                ("dinov2", 384, 14): "dinov2-small",
                ("dinov2", 1536, 14): "dinov2-giant",
                ("dinov3_vit", 4096, 16): "dinov3-7b",
            }.get(key)
            if preset is not None:
                return preset, str(path.resolve()), None
        if (path / "model.pt").is_file() and path.name == "lingbot-vision-vit-giant":
            return "lingbot-giant", str(path.resolve()), None
        if (path / VJEPA_FILES["vjepa2.1-vitG-384"]).is_file():
            return "vjepa2.1-gigantic", None, str((path / VJEPA_FILES["vjepa2.1-vitG-384"]).resolve())
    if path.is_file() and path.name == VJEPA_FILES["vjepa2.1-vitG-384"]:
        return "vjepa2.1-gigantic", None, str(path.resolve())
    raise ValueError(
        f"Unsupported backbone {value!r}. Use a preset name, its official model ID/URL, "
        "or a recognized local model. For custom weights, select a preset and use "
        "--vision-model (HF/LingBot) or --backbone-checkpoint (V-JEPA)."
    )


def check_loading_info(info: dict) -> None:
    """Reject incomplete HF weights instead of training random missing layers."""
    problems = {key: info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if problems:
        summary = {key: list(value)[:5] for key, value in problems.items()}
        raise ValueError(f"Backbone weights do not match the model: {summary}")


def check_backbone(model, expected_output_dim=None, expected_patch_size=None, expected_model_type=None) -> None:
    config = getattr(model, "config", None)
    if expected_model_type is not None and getattr(config, "model_type", None) != expected_model_type:
        raise ValueError(f"Backbone model type does not match {expected_model_type!r}")
    dim = getattr(config, "hidden_size", getattr(model, "embed_dim", None))
    patch = getattr(config, "patch_size", getattr(model, "patch_size", None))
    if expected_output_dim is not None and dim != expected_output_dim:
        raise ValueError(f"Backbone feature dimension is {dim}, expected {expected_output_dim}")
    if isinstance(patch, (list, tuple)):
        patch = patch[0] if len(set(patch)) == 1 else patch
    if expected_patch_size is not None and patch != expected_patch_size:
        raise ValueError(f"Backbone patch size is {patch}, expected {expected_patch_size}")


def download_checkpoint(url: str, destination: Path, expected_size: int | None = None) -> Path:
    """Download once across processes, resume partial files, check response length."""
    from filelock import FileLock

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if destination.is_file():
            if expected_size is not None and destination.stat().st_size != expected_size:
                raise ValueError(f"Incomplete cached backbone: {destination}; use a different TORCH_HOME to download again")
            return destination
        if os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
            raise FileNotFoundError(f"Backbone is not cached in offline mode: {destination}")
        partial = destination.with_name(destination.name + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_size is not None and offset == expected_size:
            partial.rename(destination)
            return destination
        if expected_size is not None and offset > expected_size:
            raise ValueError(f"Oversized partial backbone file: {partial}")
        headers = {"User-Agent": "ConfigWM", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            append = status == 206 and offset > 0
            content_range = response.headers.get("Content-Range", "")
            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if not match or int(match[1]) != offset:
                    raise ValueError("Invalid backbone download range")
                total = int(match[3])
            elif status == 200:
                length = response.headers.get("Content-Length")
                if length is None:
                    raise ValueError("Backbone download did not provide a file length")
                total = int(length)
            else:
                raise ValueError(f"Unexpected backbone download response: {status}")
            if expected_size is not None and total != expected_size:
                raise ValueError(f"Backbone download size changed: {total}, expected {expected_size}")
            if offset and not append:
                # Preserve a partial download if the server cannot resume it.
                partial.rename(partial.with_name(partial.name + f".saved-{time.time_ns()}"))
            with partial.open("ab" if append else "wb") as handle:
                while block := response.read(8 * 1024 * 1024):
                    handle.write(block)
            if partial.stat().st_size != total:
                raise ValueError(f"Incomplete backbone download: {partial}; retry to resume")
        partial.rename(destination)
    return destination


def resolve_vjepa_checkpoint(model_name: str, checkpoint_path: str | None = None) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if path.is_dir():
            candidates = [path / VJEPA_FILES[model_name], path / "original/model.pth"]
            path = next((p for p in candidates if p.is_file()), candidates[0])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    import torch

    filename = VJEPA_FILES[model_name]
    destination = Path(torch.hub.get_dir()) / "checkpoints/configwm" / filename
    # Content-Length published by Meta for this native encoder+predictor file.
    size = 30238058912 if model_name == "vjepa2.1-vitG-384" else None
    return download_checkpoint(VJEPA_BASE_URL + filename, destination, size)
