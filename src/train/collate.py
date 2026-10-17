"""Data collation utilities for Configurable Encoder training."""

from __future__ import annotations

from typing import Any

import hashlib

import numpy as np
import torch

DEFAULT_MAX_STR_BYTES = 4096


def _stable_nonnegative_int_id(key: str) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)


def pack_strings(
    strings: list[str], max_bytes: int = DEFAULT_MAX_STR_BYTES
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack strings into byte tensors for efficient distributed communication.

    Args:
        strings: List of strings to pack
        max_bytes: Maximum bytes per string (truncates if longer)

    Returns:
        (bytes_tensor, lengths_tensor):
            - bytes_tensor: (B, max_bytes) uint8
            - lengths_tensor: (B,) int32
    """
    out = torch.zeros((len(strings), max_bytes), dtype=torch.uint8)
    lens = torch.zeros((len(strings),), dtype=torch.int32)
    truncated_count = 0
    for i, s in enumerate(strings):
        s = "" if s is None else str(s)
        b = s.encode("utf-8", errors="ignore")
        if len(b) > max_bytes:
            truncated_count += 1
            b = b[:max_bytes]
        if b:
            out[i, : len(b)] = torch.as_tensor(list(b), dtype=torch.uint8)
            lens[i] = len(b)
    # if truncated_count > 0:
    #     logging.warning(f"pack_strings: {truncated_count}/{len(strings)} strings truncated to {max_bytes} bytes")
    return out, lens


def unpack_strings(
    bytes_tensor: torch.Tensor, lengths: torch.Tensor
) -> list[str]:
    """Unpack byte tensors back to strings.

    Args:
        bytes_tensor: (B, max_bytes) uint8 tensor
        lengths: (B,) int32 tensor with actual lengths

    Returns:
        List of decoded strings
    """
    bytes_tensor = bytes_tensor.to("cpu")
    lengths = lengths.to("cpu")
    out: list[str] = []
    for i in range(bytes_tensor.size(0)):
        n = int(lengths[i])
        out.append(
            ""
            if n <= 0
            else bytes(bytes_tensor[i, :n].tolist()).decode(
                "utf-8", errors="ignore"
            )
        )
    return out


def extract_text_field(
    inputs: Dict[str, Any], field_name: str, required: bool = True
) -> Optional[List[Optional[str]]]:
    """Extract a text field (queries, targets, or text_input) from inputs dict.
    Handles both byte-packed format and direct list format.

    Args:
        inputs: The inputs dict (modified in place)
        field_name: 'queries', 'targets', or 'text_input'
        required: If True, raise KeyError if field is missing

    Returns:
        List of strings (or None per-sample for text_input with len=0), or None if not required and missing
    """
    bytes_key = f"{field_name}_bytes"
    lens_key = f"{field_name}_lens"

    if bytes_key in inputs:
        bytes_tensor = inputs.pop(bytes_key)
        lens_tensor = inputs.pop(lens_key)
        strings = unpack_strings(bytes_tensor, lens_tensor)

        # For text_input, len=0 means "no text for this sample" -> return None per-sample
        if field_name == "text_input":
            lens_cpu = lens_tensor.cpu()
            return [
                s if int(lens_cpu[i]) > 0 else None
                for i, s in enumerate(strings)
            ]
        return strings
    if field_name in inputs:
        value = inputs.pop(field_name)
        if value is None:
            return None
        if not isinstance(value, list) and hasattr(value, "tolist"):
            value = value.tolist()
        return value
    if required:
        raise KeyError(f"Missing required field: {field_name}")
    return None


class DataCollator:
    """Batch tensors; pack strings into fixed-width byte tensors for distributed training.

    Works with PreprocessedSample which has:
    - vision_input: tensor (T, C, H, W)
    - targets: List[str]
    - query: List[str]
    """

    def __init__(
        self,
        max_str_bytes: int = DEFAULT_MAX_STR_BYTES,
        pixel_shape: Optional[Tuple[int, ...]] = None,
    ):
        """Args:
        max_str_bytes: Max bytes per packed string
        pixel_shape: Expected (T, C, H, W) shape for vision_input. Required for text-only batches.
        """
        self.max_str_bytes = max_str_bytes
        self.pixel_shape = pixel_shape

    @staticmethod
    def _to_tensor(x: Any) -> torch.Tensor | None:
        """Convert numpy arrays or tensors to torch.Tensor."""
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x) if x.dtype != object else None
        return None

    def _stack_optional_vision(
        self,
        batch: list[dict[str, Any]],
        field_name: str,
        *,
        allow_zero_fill_from_pixel_shape: bool,
    ) -> torch.Tensor | None:
        has_vision_list = [b.get(field_name) is not None for b in batch]
        if not any(has_vision_list):
            if not allow_zero_fill_from_pixel_shape:
                return None
            if self.pixel_shape is None:
                raise ValueError(
                    f"Batch has no {field_name} and pixel_shape is not configured"
                )
            ref_shape, ref_dtype = self.pixel_shape, torch.float32
        else:
            if not allow_zero_fill_from_pixel_shape and not all(has_vision_list):
                raise ValueError(
                    f"Batch mixes missing and present {field_name} values"
                )
            ref_pix = next(
                self._to_tensor(b[field_name])
                for b in batch
                if b.get(field_name) is not None
            )
            ref_shape, ref_dtype = ref_pix.shape, ref_pix.dtype

        pix_list = []
        for b in batch:
            p = b.get(field_name)
            if p is not None:
                t = self._to_tensor(p)
                pix_list.append(t if t is not None else p)
            elif allow_zero_fill_from_pixel_shape:
                pix_list.append(torch.zeros(ref_shape, dtype=ref_dtype))
            else:
                pix_list.append(None)

        if not allow_zero_fill_from_pixel_shape:
            pix_list = [p for p in pix_list if p is not None]
            if not pix_list:
                return None
        return torch.stack(pix_list)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of PreprocessedSample dicts into batched tensors.

        Args:
            batch: List of dicts with keys: vision_input, targets, query, text_input (optional)

        Returns:
            Dict with:
            - vision_input: (B, T, C, H, W) tensor (zeros for text-only samples)
            - text_input_bytes: (B, MAX_STR_BYTES) uint8 tensor (for text-only samples)
            - text_input_lens: (B,) int32 tensor
            - queries_bytes: (B, MAX_STR_BYTES) uint8 tensor
            - queries_lens: (B,) int32 tensor
            - targets_bytes: (B, MAX_STR_BYTES) uint8 tensor
            - targets_lens: (B,) int32 tensor
        """
        # 1) source vision_input: stack into batch (zero-pad text-only samples)
        vision_input = self._stack_optional_vision(
            batch,
            "vision_input",
            allow_zero_fill_from_pixel_shape=True,
        )
        if vision_input is None:
            raise ValueError(
                "Batch has no vision_input and pixel_shape not configured in collator"
            )

        # 1b) optional target vision stream for paired-image mode
        target_vision_input = self._stack_optional_vision(
            batch,
            "target_vision_input",
            allow_zero_fill_from_pixel_shape=False,
        )
        target_mask = self._stack_optional_vision(
            batch,
            "target_mask",
            allow_zero_fill_from_pixel_shape=False,
        )
        paired_vision_mode = target_vision_input is not None
        if target_mask is not None and target_vision_input is None:
            raise ValueError("target_mask requires target_vision_input")


        # 2) text_input: alternative to vision (None packed as empty string with len=0)
        text_inputs = []
        for b in batch:
            ti = b.get("text_input")
            text_inputs.append(ti if ti is not None else "")
        ti_bytes, ti_lens = pack_strings(text_inputs, self.max_str_bytes)
        # Mark empty strings as "no text_input" by setting lens to 0
        for i, b in enumerate(batch):
            if b.get("text_input") is None:
                ti_lens[i] = 0

        # 3) Extract first query and first target from lists
        queries = []
        targets = []
        for b in batch:
            q = b.get("query", [])
            t = b.get("targets", [])
            if isinstance(q, list):
                queries.append(q[0] if q else "")
            else:
                queries.append(q or "")
            if isinstance(t, list):
                targets.append(t[0] if t else "")
            else:
                targets.append(t or "")

        # 4) Pack strings into byte tensors
        q_bytes, q_lens = pack_strings(queries, self.max_str_bytes)

        batch_out = {
            "vision_input": vision_input,
            "text_input_bytes": ti_bytes,
            "text_input_lens": ti_lens,
            "queries_bytes": q_bytes,
            "queries_lens": q_lens,
        }
        if target_vision_input is not None:
            batch_out["target_vision_input"] = target_vision_input
        if target_mask is not None:
            batch_out["target_mask"] = target_mask
        if not paired_vision_mode:
            t_bytes, t_lens = pack_strings(targets, self.max_str_bytes)
            batch_out["targets_bytes"] = t_bytes
            batch_out["targets_lens"] = t_lens

        # Optional local group ids for same-source multi-query contrastive loss.
        # For PushT aligned all-query samples, one group is one source frame plus
        # one source variant; the rows inside the group differ only by query.
        group_keys = []
        for b in batch:
            meta = b.get("meta") if isinstance(b.get("meta"), dict) else {}
            if meta.get("contrast_group_key") is not None:
                group_keys.append(str(meta["contrast_group_key"]))
            elif "v15_group_idx" in meta and "source_variant" in meta:
                group_keys.append(f"v15:{meta.get('v15_group_idx')}:{meta.get('source_variant')}")
            elif meta.get("task") in {
                "pusht_source5_query",
                "pusht_query_support",
            }:
                group_keys.append(
                    "source5:"
                    f"{meta.get('shard_idx')}:"
                    f"{meta.get('source_local_index')}"
                )
            else:
                group_keys.append(None)
        if any(key is not None for key in group_keys):
            group_ids = [
                -1 if key is None else _stable_nonnegative_int_id(key)
                for key in group_keys
            ]
            batch_out["contrast_group_ids"] = torch.tensor(group_ids, dtype=torch.long)

        positive_keys = []
        for b in batch:
            meta = b.get("meta") if isinstance(b.get("meta"), dict) else {}
            key = meta.get("contrast_positive_key")
            positive_keys.append(str(key) if key is not None else None)
        if any(key is not None for key in positive_keys):
            positive_ids = [
                -1 if key is None else _stable_nonnegative_int_id(key)
                for key in positive_keys
            ]
            batch_out["contrast_positive_ids"] = torch.tensor(
                positive_ids,
                dtype=torch.long,
            )

        # Optional mask for the ordinary Configurable Encoder prediction/SigReg loss.
        # Source5 query data uses all five query rows for same-source InfoNCE,
        # but only the row selecting the true core object should contribute to
        # the main prediction objective.
        main_loss_weights = []
        has_main_loss_weight = False
        for b in batch:
            meta = b.get("meta") if isinstance(b.get("meta"), dict) else {}
            if "main_loss_weight" in meta:
                weight = float(meta.get("main_loss_weight") or 0.0)
                has_main_loss_weight = True
            elif meta.get("task") == "pusht_source5_query":
                weight = 1.0 if bool(meta.get("pair_is_core_clean")) else 0.0
                has_main_loss_weight = True
            else:
                weight = 1.0
            main_loss_weights.append(weight)
        if has_main_loss_weight:
            batch_out["main_loss_mask"] = torch.tensor(
                main_loss_weights,
                dtype=torch.float32,
            )

        # Optional mask for losses that should only apply to aligned core->clean
        # rows. Source5 multi-query rows stay in the contrastive group, but they
        # should not become the source->clean predictor-alignment anchor.
        core_clean_weights = []
        has_core_clean_marker = False
        core_anchor_families = {"core_infonce", "core_state_supcon"}
        for b in batch:
            meta = b.get("meta") if isinstance(b.get("meta"), dict) else {}
            mix_family = meta.get("mix_family")
            if "pair_is_core_clean" in meta or mix_family in (
                "core_infonce",
                "core_state_supcon",
                "source5_multistate",
            ):
                has_core_clean_marker = True

            if mix_family is not None:
                is_core_clean_anchor = (
                    mix_family in core_anchor_families
                    and bool(meta.get("pair_is_core_clean"))
                )
            else:
                is_core_clean_anchor = bool(meta.get("pair_is_core_clean"))
            core_clean_weights.append(1.0 if is_core_clean_anchor else 0.0)
        if has_core_clean_marker:
            batch_out["core_clean_mask"] = torch.tensor(
                core_clean_weights,
                dtype=torch.float32,
            )

        # 5) Final guard: ensure all outputs are tensors
        for k, v in list(batch_out.items()):
            tv = self._to_tensor(v)
            if tv is not None:
                batch_out[k] = tv
            elif not isinstance(v, torch.Tensor):
                raise TypeError(
                    f"Collator must output tensors only; key '{k}' is {type(v)}"
                )

        return batch_out


__all__ = [
    "DEFAULT_MAX_STR_BYTES",
    "DataCollator",
    "extract_text_field",
    "pack_strings",
    "unpack_strings",
]


# Backward-compatible Python names; internal code uses the canonical names above.
VLDataCollator = DataCollator
__all__ += ['VLDataCollator']
