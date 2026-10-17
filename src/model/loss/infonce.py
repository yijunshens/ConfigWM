"""InfoNCE contrastive loss."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(self, logit_scale=None, learnable_logit_scale=False):
        super().__init__()

        if logit_scale is None:
            logit_scale = 2.659  # log space: exp(2.659) ≈ 14.28

        self.learnable_logit_scale = learnable_logit_scale
        if learnable_logit_scale:
            # Store in log space as nn.Parameter (matches open_clip pattern)
            self.log_logit_scale = nn.Parameter(
                torch.tensor(float(logit_scale))
            )
        else:
            self.logit_scale = torch.tensor(float(logit_scale)).exp()

    def forward(self, x_pred, x_tgt, logit_scale=None):

        if logit_scale is None:
            if self.learnable_logit_scale:
                # Clamp in log space to [0, log(100)] → effective scale in [1, 100]
                logit_scale = self.log_logit_scale.clamp(
                    min=0, max=math.log(100)
                ).exp()
            else:
                logit_scale = self.logit_scale

        x_tgt = F.normalize(x_tgt, dim=-1)
        x_pred = F.normalize(x_pred, dim=-1)

        device = x_tgt.device
        logits_per_tgt = logit_scale * x_tgt @ x_pred.T
        logits_per_pred = logit_scale * x_pred @ x_tgt.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_tgt.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        total_loss = (
            F.cross_entropy(logits_per_tgt, labels)
            + F.cross_entropy(logits_per_pred, labels)
        ) / 2

        if self.learnable_logit_scale:
            return {"loss": total_loss, "logit_scale": logit_scale.detach()}
        return total_loss
