"""SIGReg loss: L2/L1 regression + Sketch Isotropic Gaussian Regularizer.

Vendored from the LeJEPA library (Balestriero & LeCun, 2025).
Core components: EppsPulley characteristic-function test + SlicingUnivariateTest
for multivariate extension via random projections.
"""

from __future__ import annotations

import torch
from torch import distributed as dist
from torch.distributed.nn import ReduceOp
from torch.distributed.nn import all_reduce as _all_reduce
import torch.nn as nn

# ---------------------------------------------------------------------------
# Distributed helper
# ---------------------------------------------------------------------------


def _dist_all_reduce(x: torch.Tensor, op: str = "AVG") -> torch.Tensor:
    """All-reduce that's a no-op when not in a distributed context."""
    if dist.is_available() and dist.is_initialized():
        return _all_reduce(x, ReduceOp.__dict__[op.upper()])
    return x


# ---------------------------------------------------------------------------
# Epps-Pulley univariate normality test (characteristic-function based)
# ---------------------------------------------------------------------------


class EppsPulley(nn.Module):
    """Fast Epps-Pulley test statistic via trapezoidal quadrature.

    Tests whether projected 1-D samples follow a standard normal distribution
    by comparing empirical and theoretical characteristic functions.
    """

    def __init__(self, t_max: float = 3, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = t.square().mul(0.5).neg().exp()  # exp(-t²/2)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    @property
    def world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the test statistic for each slice.

        Args:
            x: (*, N, K) — N samples, K slices (random projections).

        Returns:
            (*, K) — per-slice test statistics.
        """
        N = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_mean = torch.cos(x_t).mean(-3)  # (*, K, n_points)
        sin_mean = torch.sin(x_t).mean(-3)
        cos_mean = _dist_all_reduce(cos_mean)
        sin_mean = _dist_all_reduce(sin_mean)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N * self.world_size


# ---------------------------------------------------------------------------
# Slicing wrapper: multivariate → univariate via random projections
# ---------------------------------------------------------------------------


class SlicingUnivariateTest(nn.Module):
    """Extend a univariate test to multivariate data via random slicing."""

    def __init__(
        self,
        univariate_test: nn.Module,
        num_slices: int,
        reduction: str = "mean",
        clip_value: float | None = None,
    ):
        super().__init__()
        self.univariate_test = univariate_test
        self.num_slices = num_slices
        self.reduction = reduction
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None

    def _get_generator(
        self, device: torch.device, seed: int
    ) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sliced univariate test to multivariate samples.

        Args:
            x: (*, N, D) — N samples, D feature dimensions.

        Returns:
            Scalar if reduction='mean'/'sum', else (*, num_slices).
        """
        with torch.no_grad():
            seed = _dist_all_reduce(self.global_step.clone(), op="MAX").item()
            g = self._get_generator(x.device, seed)
            A = torch.randn(
                x.size(-1), self.num_slices, device=x.device, generator=g
            )
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            # Avoid in-place mutation here: autograd needs the unmodified statistic
            # tensor to backpropagate through the regularizer.
            stats = stats.masked_fill(stats < self.clip_value, 0)
        if self.reduction == "mean":
            return stats.mean()
        if self.reduction == "sum":
            return stats.sum()
        return stats


# ---------------------------------------------------------------------------
# SIGReg: regression + regularizer
# ---------------------------------------------------------------------------


class SIGReg(nn.Module):
    """LeJEPA-style loss: (1-λ)*prediction + λ*SIGReg.

    Args:
        regression: "l2" or "l1" for the prediction/alignment term.
        sigreg_weight: λ — trade-off between prediction and regularization.
        num_proj: Number of random projection slices for SIGReg.
        knots: Number of quadrature points for the Epps-Pulley test.
        regularize: "both", "pred", or "tgt" — which embeddings to regularize.
    """

    def __init__(
        self,
        regression: str = "l2",
        sigreg_weight: float = 0.05,
        num_proj: int = 1024,
        knots: int = 17,
        regularize: str = "both",
        normalize_ep: bool = False,
        pre_gathered: bool = False,
        pred_loss_scale: float = 1.0,
    ):
        super().__init__()
        self.regression_type = regression
        self.regularize = regularize
        self.normalize_ep = normalize_ep
        self.pre_gathered = pre_gathered
        self.pred_loss_scale = float(pred_loss_scale)
        if regression == "l2":
            self.regression_fn = nn.MSELoss()
        elif regression == "l1":
            self.regression_fn = nn.L1Loss()
        elif regression == "cosine":
            self.regression_fn = None  # handled in forward
        else:
            raise ValueError(f"Unsupported regression type: {regression}")

        self.regularizer = SlicingUnivariateTest(
            univariate_test=EppsPulley(n_points=knots),
            num_slices=num_proj,
            reduction="mean",
        )
        self.lam = sigreg_weight

    def forward(
        self, pred_embeds: torch.Tensor, tgt_embeds: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        import torch.nn.functional as F

        if pred_embeds.shape != tgt_embeds.shape:
            raise ValueError(
                "pred_embeds and tgt_embeds must have identical shapes before loss, "
                f"got {tuple(pred_embeds.shape)} vs {tuple(tgt_embeds.shape)}."
            )
        if pred_embeds.dim() > 2:
            pred_embeds = pred_embeds.reshape(-1, pred_embeds.size(-1))
            tgt_embeds = tgt_embeds.reshape(-1, tgt_embeds.size(-1))

        if self.regression_type == "cosine":
            pred_loss_unscaled = (
                1 - F.cosine_similarity(pred_embeds, tgt_embeds, dim=-1).mean()
            )
        else:
            pred_loss_unscaled = self.regression_fn(pred_embeds, tgt_embeds)
        pred_loss = pred_loss_unscaled * self.pred_loss_scale

        if self.regularize == "pred":
            sigreg_loss = self.regularizer(pred_embeds)
        elif self.regularize == "tgt":
            sigreg_loss = self.regularizer(tgt_embeds)
        else:
            sigreg_loss = (
                self.regularizer(pred_embeds) + self.regularizer(tgt_embeds)
            ) / 2

        # Epps-Pulley statistic = (err @ weights) * N * world_size.
        # world_size = gpus_per_node * num_nodes. At multi-node, the extra
        # num_nodes factor makes SIGReg loss grow linearly with node count,
        # overwhelming L2 alignment. Divide by num_nodes so 1-node behavior
        # (tuned λ) is preserved while multi-node gets same effective loss.
        if self.normalize_ep:
            world_size = self.regularizer.univariate_test.world_size
            # Infer num_nodes: world_size / local_world_size
            import os

            local_world_size = int(
                os.environ.get("LOCAL_WORLD_SIZE", world_size)
            )
            num_nodes = max(1, world_size // local_world_size)
            if num_nodes > 1:
                sigreg_loss = sigreg_loss / num_nodes

        # When embeddings are pre-gathered (e.g., for composite loss with InfoNCE),
        # N is already the global batch size and all_reduce was a no-op on identical
        # data. Divide out the spurious world_size factor from EP scaling.
        if self.pre_gathered:
            world_size = self.regularizer.univariate_test.world_size
            if world_size > 1:
                sigreg_loss = sigreg_loss / world_size

        loss = (1 - self.lam) * pred_loss + self.lam * sigreg_loss

        # Diagnostic stats
        with torch.no_grad():
            cos_matched = F.cosine_similarity(
                pred_embeds, tgt_embeds, dim=-1
            ).mean()
            pred_std = pred_embeds.std(dim=0).mean()
            tgt_std = tgt_embeds.std(dim=0).mean()
            pred_norm = F.normalize(pred_embeds, dim=-1)
            pred_self_sim = (pred_norm @ pred_norm.T).fill_diagonal_(0).mean()

        return {
            "loss": loss,
            "pred_loss": pred_loss.detach(),
            "pred_loss_unscaled": pred_loss_unscaled.detach(),
            "sigreg_loss": sigreg_loss.detach(),
            "sigreg_loss_for_grad": sigreg_loss,
            "cos_matched": cos_matched,
            "pred_std": pred_std,
            "tgt_std": tgt_std,
            "pred_self_sim": pred_self_sim,
        }
