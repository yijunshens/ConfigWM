"""SigLIP: Sigmoid Loss for Language-Image Pre-Training.

Pairwise sigmoid contrastive loss that treats each (image, text) pair as an
independent binary classification. Unlike InfoNCE's softmax normalization,
no global view of the batch is needed — each pair contributes independently.

Reference: Zhai et al., "Sigmoid Loss for Language Image Pre-Training", 2023.
           https://arxiv.org/abs/2303.15343

Implementation aligned with open_clip's SigLipLoss (open_clip/src/open_clip/loss.py).
Simplified here because the Configurable Encoder trainer handles cross-GPU gathering externally.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIP(nn.Module):
    """Sigmoid pairwise contrastive loss.

    Args:
        init_logit_scale: Initial value for log-scale parameter (default: log(10) ≈ 2.30).
        init_logit_bias: Initial value for bias parameter (default: -10).
            The large negative bias prevents the massive negative:positive imbalance
            from causing destabilizing gradient spikes at the start of training.
        max_logit_scale: Upper clamp for log-scale (default: log(100) ≈ 4.60).
    """

    def __init__(
        self,
        init_logit_scale: float = math.log(10),
        init_logit_bias: float = -10.0,
        max_logit_scale: float = math.log(100),
    ):
        super().__init__()
        self.log_logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        self.logit_bias = nn.Parameter(torch.tensor(init_logit_bias))
        self.max_logit_scale = max_logit_scale

    def forward(self, x_pred, x_tgt):
        x_pred = F.normalize(x_pred, dim=-1)
        x_tgt = F.normalize(x_tgt, dim=-1)

        logit_scale = self.log_logit_scale.clamp(
            min=0, max=self.max_logit_scale
        ).exp()
        logits = logit_scale * x_pred @ x_tgt.T + self.logit_bias

        # Labels: +1 for matched pairs (diagonal), -1 for unmatched (off-diagonal)
        n = x_pred.shape[0]
        labels = 2 * torch.eye(n, device=x_pred.device, dtype=x_pred.dtype) - 1

        # Pairwise sigmoid loss: -log σ(label * logit) for each pair
        # Divide by n (not n²) — each image owns n pair evaluations
        loss = -F.logsigmoid(labels * logits).sum() / n

        return {
            "loss": loss,
            "logit_scale": logit_scale.detach(),
            "logit_bias": self.logit_bias.detach(),
        }
