"""Sequence packing utilities for efficient training on short-text datasets.

P3 data has mean ~197 tokens per sample (out of 2048 max). Without packing,
~90% of compute is wasted on padding. Packing bins multiple short samples
into each fixed-length window with block-diagonal attention, giving 8-10×
better token utilization and MFU.

Design:
  - The model (Llama3Backbone) is unchanged — it still processes 2048 tokens.
  - Multiple samples are concatenated into a single window.
  - A block-diagonal attention mask prevents cross-sample attention.
  - Pooling is done per-sample using stored offsets/lengths.

Usage in Predictor.forward() when packed mode is active:
  - text_input contains pre-packed token embeddings (not raw strings)
  - A `pack_info` dict carries sample offsets/lengths for pooling
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PackInfo:
    """Metadata for a batch of packed sequences.

    Attributes:
        offsets: (B,) int64 — start position of each sample in its packed window
        lengths: (B,) int64 — number of real tokens for each sample
        window_indices: (B,) int64 — which packed window (row in the batch) each sample belongs to
        n_windows: int — number of packed windows (= collated batch size)
    """

    offsets: torch.Tensor  # (B,)
    lengths: torch.Tensor  # (B,)
    window_indices: torch.Tensor  # (B,)
    n_windows: int


def bin_pack_samples(
    token_counts: list[int],
    max_window_len: int,
    pad_to: int | None = None,
) -> list[list[int]]:
    """First-fit-decreasing bin packing.

    Args:
        token_counts: Token count per sample.
        max_window_len: Maximum tokens per packed window.
        pad_to: If given, each window is conceptually padded to this length.

    Returns:
        List of bins, each bin is a list of original sample indices.
    """
    # Sort descending by token count (better packing)
    order = sorted(range(len(token_counts)), key=lambda i: -token_counts[i])
    bins: list[list[int]] = []
    bin_used: list[int] = []

    for idx in order:
        count = token_counts[idx]
        if count > max_window_len:
            # Oversized: give it its own window (truncated by backbone)
            bins.append([idx])
            bin_used.append(min(count, max_window_len))
            continue
        # Find first bin that fits
        placed = False
        for b in range(len(bins)):
            if bin_used[b] + count <= max_window_len:
                bins[b].append(idx)
                bin_used[b] += count
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_used.append(count)

    return bins


def make_block_diagonal_mask(
    pack_info: PackInfo,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a [n_windows, 1, seq_len, seq_len] block-diagonal additive attention mask.

    Within each window, each sample can only attend to its own tokens
    (positions [offset, offset+length)). All other positions get -inf.

    Args:
        pack_info: Packing metadata for the batch.
        seq_len: Total sequence length (max_window_len, i.e. max_input_tokens).
        dtype: Output tensor dtype.
        device: Output tensor device.

    Returns:
        Additive attention mask [n_windows, 1, seq_len, seq_len],
        where 0.0 = attend, -inf = mask.
    """
    n = pack_info.n_windows
    # Start with everything masked (-inf)
    mask = torch.full(
        (n, seq_len, seq_len), float("-inf"), dtype=dtype, device=device
    )

    offsets = pack_info.offsets.to(device)
    lengths = pack_info.lengths.to(device)
    win_idx = pack_info.window_indices.to(device)

    # For each sample, allow attend within [offset, offset+length)
    for s in range(pack_info.offsets.shape[0]):
        w = int(win_idx[s].item())
        off = int(offsets[s].item())
        length = int(lengths[s].item())
        end = min(off + length, seq_len)
        mask[w, off:end, off:end] = 0.0

    # Fix padding positions (not in any sample): give them self-attention to avoid NaN
    # from all-inf softmax. Their outputs are discarded by unpack_pooled anyway.
    # Track which positions are "owned" by some sample
    occupied = torch.zeros(n, seq_len, dtype=torch.bool, device=device)
    for s in range(pack_info.offsets.shape[0]):
        w = int(win_idx[s].item())
        off = int(offsets[s].item())
        length = int(lengths[s].item())
        end = min(off + length, seq_len)
        occupied[w, off:end] = True
    # For unoccupied positions, allow self-attention (diagonal = 0)
    pad_idx = (~occupied).nonzero(as_tuple=False)  # (num_pad, 2)
    for w_idx, pos in pad_idx.tolist():
        mask[w_idx, pos, pos] = 0.0

    return mask.unsqueeze(1)  # [n_windows, 1, seq_len, seq_len]


def pack_text_batch(
    texts: list[str | None],
    tokenizer,
    max_window_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, PackInfo]:
    """Tokenize and bin-pack a list of text strings into packed windows.

    Args:
        texts: List of B text strings.
        tokenizer: HF tokenizer (from backbone).
        max_window_len: Max tokens per window (= max_input_tokens).
        device: Target device.
        dtype: Target dtype for embeddings (not used here, returned for convenience).

    Returns:
        input_ids: (n_windows, max_window_len) long tensor of packed token IDs
        attention_mask_2d: (n_windows, max_window_len) bool tensor (True = valid)
        pack_info: PackInfo with offsets/lengths/window_indices
    """
    B = len(texts)

    # Tokenize each text individually
    tokenized = tokenizer(
        [t if t is not None else "" for t in texts],
        padding=False,
        truncation=True,
        max_length=max_window_len,
        return_tensors=None,
        add_special_tokens=True,
    )
    token_ids_list = tokenized["input_ids"]  # list of list[int]
    lengths = [len(ids) for ids in token_ids_list]

    # Bin-pack
    bins = bin_pack_samples(lengths, max_window_len)

    n_windows = len(bins)
    pad_id = tokenizer.pad_token_id

    # Build packed input_ids and attention_mask_2d
    input_ids = torch.full(
        (n_windows, max_window_len), pad_id, dtype=torch.long, device=device
    )
    attention_mask_2d = torch.zeros(
        (n_windows, max_window_len), dtype=torch.bool, device=device
    )

    # PackInfo arrays
    offsets_list = [0] * B
    lengths_list = [0] * B
    window_indices_list = [0] * B

    for w, sample_indices in enumerate(bins):
        cursor = 0
        for s_idx in sample_indices:
            ids = token_ids_list[s_idx]
            length = len(ids)
            end = min(cursor + length, max_window_len)
            actual_len = end - cursor
            input_ids[w, cursor:end] = torch.tensor(
                ids[:actual_len], dtype=torch.long, device=device
            )
            attention_mask_2d[w, cursor:end] = True
            offsets_list[s_idx] = cursor
            lengths_list[s_idx] = actual_len
            window_indices_list[s_idx] = w
            cursor = end

    pack_info = PackInfo(
        offsets=torch.tensor(offsets_list, dtype=torch.long),
        lengths=torch.tensor(lengths_list, dtype=torch.long),
        window_indices=torch.tensor(window_indices_list, dtype=torch.long),
        n_windows=n_windows,
    )

    return input_ids, attention_mask_2d, pack_info


def unpack_pooled(
    hidden_states: torch.Tensor,
    pack_info: PackInfo,
) -> torch.Tensor:
    """Pool hidden states per original sample from packed windows.

    Args:
        hidden_states: (n_windows, seq_len, H) hidden states from backbone.
        pack_info: Packing metadata.

    Returns:
        pooled: (B, H) mean-pooled per original sample.
    """
    B = pack_info.offsets.shape[0]
    H = hidden_states.shape[-1]
    device = hidden_states.device

    offsets = pack_info.offsets.to(device)
    lengths = pack_info.lengths.to(device)
    win_idx = pack_info.window_indices.to(device)

    pooled = torch.zeros(B, H, device=device, dtype=hidden_states.dtype)

    for s in range(B):
        w = int(win_idx[s].item())
        off = int(offsets[s].item())
        length = int(lengths[s].item())
        if length > 0:
            pooled[s] = hidden_states[w, off : off + length].mean(dim=0)

    return pooled
