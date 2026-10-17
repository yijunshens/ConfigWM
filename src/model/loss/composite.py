"""Composite loss: (1-α)*Contrastive + α*SIGReg.

Allows smooth interpolation between contrastive (InfoNCE or SigLIP) and
non-contrastive (SIGReg) objectives. Requires gather_embeddings=true in trainer
config so that the contrastive component sees the full batch similarity matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .infonce import InfoNCE
from .siglip import SigLIP
from .sigreg import SIGReg


class CompositeLoss(nn.Module):
    """Weighted combination of a contrastive loss and SIGReg.

    loss = (1-α)*Contrastive(pred, tgt) + α*SIGReg(pred, tgt)

    where SIGReg = (1-λ)*MSE + λ*EP_test.

    Args:
        alpha: Weight for SIGReg. 0 = pure contrastive, 1 = pure SIGReg.
        contrastive_type: Which contrastive loss to use ("infonce" or "siglip").
        logit_scale: Temperature for InfoNCE (default: exp(2.659) ≈ 14.28).
            Ignored when contrastive_type="siglip".
        learnable_logit_scale: Whether InfoNCE temperature is learnable.
            Ignored when contrastive_type="siglip".
        siglip_init_logit_scale: Initial log-scale for SigLIP. Ignored when
            contrastive_type!="siglip".
        siglip_init_logit_bias: Initial bias for SigLIP. Ignored when
            contrastive_type!="siglip".
        siglip_max_logit_scale: Upper clamp for SigLIP log-scale. Ignored when
            contrastive_type!="siglip".
        regression: Regression type for SIGReg alignment term ("l2" or "l1").
        sigreg_weight: λ for SIGReg's internal (1-λ)*regression + λ*EP balance.
        num_proj: Number of random projection slices for EP test.
        knots: Quadrature points for Epps-Pulley test.
        regularize: Which embeddings to regularize ("both", "pred", "tgt").
        normalize_ep: Divide EP statistic by num_nodes for multi-node stability.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        contrastive_type: str = "infonce",
        logit_scale: float | None = None,
        learnable_logit_scale: bool = False,
        siglip_init_logit_scale: float | None = None,
        siglip_init_logit_bias: float | None = None,
        siglip_max_logit_scale: float | None = None,
        regression: str = "l2",
        sigreg_weight: float = 0.5,
        num_proj: int = 4096,
        knots: int = 17,
        regularize: str = "both",
        normalize_ep: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.contrastive_type = contrastive_type
        if contrastive_type == "infonce":
            self.contrastive = InfoNCE(
                logit_scale=logit_scale,
                learnable_logit_scale=learnable_logit_scale,
            )
        elif contrastive_type == "siglip":
            siglip_kwargs = {}
            if siglip_init_logit_scale is not None:
                siglip_kwargs["init_logit_scale"] = siglip_init_logit_scale
            if siglip_init_logit_bias is not None:
                siglip_kwargs["init_logit_bias"] = siglip_init_logit_bias
            if siglip_max_logit_scale is not None:
                siglip_kwargs["max_logit_scale"] = siglip_max_logit_scale
            self.contrastive = SigLIP(**siglip_kwargs)
        else:
            raise ValueError(
                f"Unsupported contrastive_type: {contrastive_type}"
            )
        self.sigreg = SIGReg(
            regression=regression,
            sigreg_weight=sigreg_weight,
            num_proj=num_proj,
            knots=knots,
            regularize=regularize,
            normalize_ep=normalize_ep,
            pre_gathered=True,
        )

    def forward(
        self, pred_embeds: torch.Tensor, tgt_embeds: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        sigreg_out = self.sigreg(pred_embeds, tgt_embeds)
        contrastive_out = self.contrastive(pred_embeds, tgt_embeds)

        # Contrastive returns scalar or dict
        if isinstance(contrastive_out, dict):
            contrastive_loss = contrastive_out["loss"]
        else:
            contrastive_loss = contrastive_out

        loss = (1 - self.alpha) * contrastive_loss + self.alpha * sigreg_out[
            "loss"
        ]

        out = {
            "loss": loss,
            "contrastive_loss": contrastive_loss.detach(),
            "sigreg_total": sigreg_out["loss"].detach(),
            "pred_loss": sigreg_out["pred_loss"],
            "sigreg_loss": sigreg_out["sigreg_loss"],
            "cos_matched": sigreg_out["cos_matched"],
            "pred_std": sigreg_out["pred_std"],
            "tgt_std": sigreg_out["tgt_std"],
            "pred_self_sim": sigreg_out["pred_self_sim"],
        }
        if isinstance(contrastive_out, dict):
            if "logit_scale" in contrastive_out:
                out["logit_scale"] = contrastive_out["logit_scale"]
            if "logit_bias" in contrastive_out:
                out["logit_bias"] = contrastive_out["logit_bias"]
        return out
