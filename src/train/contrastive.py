from __future__ import annotations

import torch
import torch.nn.functional as F


def _multi_positive_loss_and_top1(
    logits: torch.Tensor,
    positive_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average supervised-contrastive loss over one or more logit matrices."""
    if logits.dim() < 2 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError("Multi-positive logits must end with a square (B,B) matrix")
    batch_size = logits.shape[-1]
    if positive_ids.shape != (batch_size,):
        raise ValueError(
            "positive_ids must have shape (B,), got "
            f"{tuple(positive_ids.shape)} for batch size {batch_size}"
        )

    positive_mask = positive_ids[:, None].eq(positive_ids[None, :])
    positive_mask &= positive_ids[:, None].ge(0)
    valid_rows = positive_mask.any(dim=1)
    flat_logits = logits.reshape(-1, batch_size, batch_size)

    if not bool(valid_rows.any().item()):
        labels = torch.arange(batch_size, device=logits.device)
        expanded_labels = labels.expand(flat_logits.size(0), -1).reshape(-1)
        loss = F.cross_entropy(
            flat_logits.reshape(-1, batch_size),
            expanded_labels,
        )
        top1 = (
            flat_logits.argmax(dim=-1)
            .eq(labels[None, :])
            .float()
            .mean()
        )
        return loss, top1

    log_probs = flat_logits - torch.logsumexp(flat_logits, dim=-1, keepdim=True)
    positive_mask_f = positive_mask.to(log_probs.dtype)
    per_row = -(
        log_probs * positive_mask_f[None, :, :]
    ).sum(dim=-1) / positive_mask_f.sum(dim=-1).clamp_min(1.0)[None, :]
    loss = per_row[:, valid_rows].mean()

    predicted_indices = flat_logits.argmax(dim=-1)
    predicted_is_positive = positive_mask[
        torch.arange(batch_size, device=logits.device)[None, :],
        predicted_indices,
    ]
    top1 = predicted_is_positive[:, valid_rows].float().mean()
    return loss, top1


def patch_position_multi_positive_loss(
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    positive_ids: torch.Tensor,
    *,
    temperature: float,
    patch_chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-position InfoNCE with multi-positive image identities.

    Every patch competes only against the same spatial position from other
    images. Patch chunks are mathematically independent and are accumulated
    with equal weight, avoiding a full ``(N,B,B)`` logits allocation.
    """
    if source_tokens.dim() != 3 or target_tokens.dim() != 3:
        raise ValueError("Patch-position InfoNCE expects tensors with shape (B,N,D)")
    if source_tokens.shape != target_tokens.shape:
        raise ValueError(
            "Source and target patch tokens must have matching shapes, got "
            f"{tuple(source_tokens.shape)} and {tuple(target_tokens.shape)}"
        )
    if source_tokens.size(0) < 2:
        raise ValueError("Patch-position InfoNCE requires at least two images")
    if positive_ids.shape != (source_tokens.size(0),):
        raise ValueError(
            "positive_ids must have shape (B,), got "
            f"{tuple(positive_ids.shape)} for batch size {source_tokens.size(0)}"
        )
    patch_chunk_size = int(patch_chunk_size)
    if patch_chunk_size <= 0:
        raise ValueError("patch_chunk_size must be positive")
    temperature = max(float(temperature), 1.0e-6)

    patch_count = source_tokens.size(1)
    weighted_losses_s2t = []
    weighted_losses_t2s = []
    weighted_accs_s2t = []
    weighted_accs_t2s = []
    chunk_weights = []
    for start in range(0, patch_count, patch_chunk_size):
        end = min(start + patch_chunk_size, patch_count)
        similarities = torch.einsum(
            "bpd,cpd->pbc",
            source_tokens[:, start:end],
            target_tokens[:, start:end],
        )
        logits_s2t = similarities / temperature
        logits_t2s = logits_s2t.transpose(-1, -2)
        loss_s2t, acc_s2t = _multi_positive_loss_and_top1(
            logits_s2t, positive_ids
        )
        loss_t2s, acc_t2s = _multi_positive_loss_and_top1(
            logits_t2s, positive_ids
        )
        chunk_weight = end - start
        weighted_losses_s2t.append(loss_s2t * chunk_weight)
        weighted_losses_t2s.append(loss_t2s * chunk_weight)
        weighted_accs_s2t.append(acc_s2t * chunk_weight)
        weighted_accs_t2s.append(acc_t2s * chunk_weight)
        chunk_weights.append(chunk_weight)

    total_weight = float(sum(chunk_weights))
    return (
        torch.stack(weighted_losses_s2t).sum() / total_weight,
        torch.stack(weighted_losses_t2s).sum() / total_weight,
        torch.stack(weighted_accs_s2t).sum() / total_weight,
        torch.stack(weighted_accs_t2s).sum() / total_weight,
    )


def patch_position_logits(
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build bidirectional InfoNCE logits at matching spatial positions.

    Inputs must already be L2-normalized and have shape (B, N, D). Each source
    patch matches the target patch at the same position from the same image;
    patches at that position from other images are negatives.
    """
    if source_tokens.dim() != 3 or target_tokens.dim() != 3:
        raise ValueError("Patch-position InfoNCE expects tensors with shape (B,N,D)")
    if source_tokens.shape != target_tokens.shape:
        raise ValueError(
            "Source and target patch tokens must have matching shapes, got "
            f"{tuple(source_tokens.shape)} and {tuple(target_tokens.shape)}"
        )
    if source_tokens.size(0) < 2:
        raise ValueError("Patch-position InfoNCE requires at least two images")
    temperature = max(float(temperature), 1.0e-6)

    patch_similarities = torch.einsum(
        "bnd,cnd->nbc", source_tokens, target_tokens
    )
    patch_count, batch_size, _ = patch_similarities.shape
    source_to_target = patch_similarities.reshape(
        patch_count * batch_size, batch_size
    ) / temperature
    target_to_source = patch_similarities.transpose(-1, -2).reshape(
        patch_count * batch_size, batch_size
    ) / temperature
    labels = torch.arange(batch_size, device=source_tokens.device).repeat(patch_count)
    return source_to_target, target_to_source, labels


def patch_spatial_logits(
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build bidirectional InfoNCE logits across positions within each image."""
    if source_tokens.dim() != 3 or target_tokens.dim() != 3:
        raise ValueError("Patch-spatial InfoNCE expects tensors with shape (B,N,D)")
    if source_tokens.shape != target_tokens.shape:
        raise ValueError(
            "Source and target patch tokens must have matching shapes, got "
            f"{tuple(source_tokens.shape)} and {tuple(target_tokens.shape)}"
        )
    temperature = max(float(temperature), 1.0e-6)

    similarities = torch.einsum("bnd,bmd->bnm", source_tokens, target_tokens)
    batch_size, patch_count, _ = similarities.shape
    source_to_target = similarities.reshape(
        batch_size * patch_count, patch_count
    ) / temperature
    target_to_source = similarities.transpose(-1, -2).reshape(
        batch_size * patch_count, patch_count
    ) / temperature
    labels = torch.arange(patch_count, device=source_tokens.device).repeat(batch_size)
    return source_to_target, target_to_source, labels


__all__ = [
    "patch_position_logits",
    "patch_position_multi_positive_loss",
    "patch_spatial_logits",
]
