"""Temporal-Straightening-style latent dynamics model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig
from .encoders import build_frozen_encoder


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero modulation used by the LeWM predictor blocks."""
    return x * (1 + scale) + shift


class _SingletonSafeBatchNorm1d(nn.BatchNorm1d):
    """Use running statistics when a training batch has one value per channel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                training=False, momentum=0.0, eps=self.eps,
            )
        return super().forward(x)


class ObservationProjector(nn.Module):
    def __init__(self, input_dim: int, adapter_dim: int, state_dim: int) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(adapter_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(-1, features.shape[-1])
        states = self.trunk(self.adapter(flat))
        return states.reshape(*features.shape[:-1], -1)


class LeWMMLPProjector(nn.Module):
    """LeWM-style MLP projector applied to the last feature dimension."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _SingletonSafeBatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(-1, features.shape[-1])
        projected = self.net(flat)
        return projected.reshape(*features.shape[:-1], -1)


class DinoGlobalProjector(nn.Module):
    """Paper-style DINO patch-token global projector from Temporal Straightening."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        state_dim: int,
        *,
        pool_hw: int = 1,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.pool_hw = int(pool_hw)
        self.mix = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, bias=False)
        self.gn0 = nn.GroupNorm(
            num_groups=max(1, min(int(gn_groups), int(hidden_dim))),
            num_channels=hidden_dim,
        )
        blocks: list[nn.Module] = []
        for _ in range(3):
            blocks.extend(
                [
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(
                        num_groups=max(1, min(int(gn_groups), int(hidden_dim))),
                        num_channels=hidden_dim,
                    ),
                    nn.GELU(),
                ]
            )
        self.down_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((self.pool_hw, self.pool_hw))
        self.head = nn.Conv2d(hidden_dim, state_dim, kernel_size=1, bias=False)
        self.ln = nn.LayerNorm(state_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-2] <= 1:
            raise ValueError(
                "DinoGlobalProjector expects DINO patch tokens with shape "
                f"(..., P, D), got {tuple(features.shape)}"
            )
        leading_shape = features.shape[:-2]
        num_patches = int(features.shape[-2])
        dim = int(features.shape[-1])
        side = int(num_patches**0.5)
        if side * side != num_patches:
            raise ValueError(f"Expected square patch token count, got {num_patches}")
        x = features.reshape(-1, num_patches, dim)
        x = x.view(x.shape[0], side, side, dim).permute(0, 3, 1, 2).contiguous()
        x = F.gelu(self.gn0(self.mix(x)))
        x = self.down_blocks(x)
        x = self.pool(x)
        x = self.head(x)
        if self.pool_hw == 1:
            x = x.flatten(1)
            x = self.ln(x)
            return x.reshape(*leading_shape, -1)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.ln(x)
        return x.reshape(*leading_shape, self.pool_hw * self.pool_hw, -1)


class DinoChannelAggProjector(nn.Module):
    """TS-style channel projector followed by a LeWM-style global MLP pooler.

    The projector keeps the DINO patch grid intact while reducing channels,
    then pools the low-dimensional spatial map into one global LeWM state.
    """

    def __init__(
        self,
        input_dim: int,
        spatial_hidden_dim: int,
        channel_dim: int,
        state_dim: int,
        *,
        spatial_hw: int,
        agg_hidden_dim: int = 2048,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.spatial_hw = int(spatial_hw)
        self.channel_dim = int(channel_dim)
        self.channel_projector = nn.Sequential(
            nn.Conv2d(input_dim, spatial_hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(
                num_groups=max(1, min(int(gn_groups), int(spatial_hidden_dim))),
                num_channels=spatial_hidden_dim,
            ),
            nn.GELU(),
            nn.Conv2d(spatial_hidden_dim, self.channel_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(
                num_groups=max(1, min(int(gn_groups), self.channel_dim)),
                num_channels=self.channel_dim,
            ),
            nn.GELU(),
        )
        self.aggregator = nn.Sequential(
            nn.Linear(self.spatial_hw * self.spatial_hw * self.channel_dim, agg_hidden_dim),
            _SingletonSafeBatchNorm1d(agg_hidden_dim),
            nn.GELU(),
            nn.Linear(agg_hidden_dim, state_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-2] <= 1:
            raise ValueError(
                "DinoChannelAggProjector expects DINO patch tokens with shape "
                f"(..., P, D), got {tuple(features.shape)}"
            )
        leading_shape = features.shape[:-2]
        num_patches = int(features.shape[-2])
        dim = int(features.shape[-1])
        side = int(num_patches**0.5)
        if side * side != num_patches:
            raise ValueError(f"Expected square patch token count, got {num_patches}")
        if side != self.spatial_hw:
            raise ValueError(
                f"Expected {self.spatial_hw}x{self.spatial_hw} patch grid, got {side}x{side}"
            )
        x = features.reshape(-1, num_patches, dim)
        x = x.view(x.shape[0], side, side, dim).permute(0, 3, 1, 2).contiguous()
        x = self.channel_projector(x)
        x = x.flatten(1)
        x = self.aggregator(x)
        return x.reshape(*leading_shape, -1)


class ActionEncoder(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        action_steps: int,
        action_emb_dim: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_steps = int(action_steps)
        self.per_step = nn.Sequential(
            nn.Linear(action_dim, action_emb_dim),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(action_steps * action_emb_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        if action_chunk.dim() != 3:
            raise ValueError(
                f"Expected action_chunk shape (B, T, A), got {tuple(action_chunk.shape)}"
            )
        if action_chunk.size(1) != self.action_steps:
            raise ValueError(
                f"Expected action_chunk length {self.action_steps}, got {action_chunk.size(1)}"
            )
        encoded = self.per_step(action_chunk)
        flat = encoded.reshape(encoded.shape[0], -1)
        return self.proj(flat)


class LeWMActionEncoder(nn.Module):
    """LeWM-style action-block embedder.

    Accepts either one action block with shape (B, S, A) or a context sequence
    with shape (B, T, S, A), where S is the action block length.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        action_steps: int,
        smoothed_dim: int,
        emb_dim: int,
        mlp_scale: int = 4,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_steps = int(action_steps)
        self.input_dim = self.action_dim * self.action_steps
        self.patch_embed = nn.Conv1d(self.input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, action_context: torch.Tensor) -> torch.Tensor:
        if action_context.dim() == 3:
            if action_context.size(1) != self.action_steps:
                raise ValueError(
                    f"Expected action block length {self.action_steps}, "
                    f"got {action_context.size(1)}"
                )
            x = action_context.reshape(action_context.shape[0], 1, self.input_dim)
        elif action_context.dim() == 4:
            if action_context.size(2) != self.action_steps:
                raise ValueError(
                    f"Expected action block length {self.action_steps}, "
                    f"got {action_context.size(2)}"
                )
            x = action_context.reshape(
                action_context.shape[0],
                action_context.shape[1],
                self.input_dim,
            )
        else:
            raise ValueError(
                "LeWMActionEncoder expects (B,S,A) or (B,T,S,A), "
                f"got {tuple(action_context.shape)}"
            )
        x = x.float().permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class LeWMFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeWMAttention(nn.Module):
    """Scaled dot-product attention with causal masking, matching LeWM."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.dropout = float(dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        x = self.norm(x)
        batch_size, seq_len, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            tensor.reshape(batch_size, seq_len, self.heads, self.dim_head)
            .transpose(1, 2)
            .contiguous()
            for tensor in qkv
        )
        drop = self.dropout if self.training else 0.0
        # Spatial DINO routes flatten B*patches into the attention batch. The
        # fused CUDA SDPA kernels can exceed their launch-grid limit there even
        # though seq_len is tiny, so use the math backend for large batches.
        if q.shape[0] * q.shape[1] > 65535 and q.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
                enable_cudnn=False,
            ):
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=drop,
                    is_causal=causal,
                )
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=drop,
                is_causal=causal,
            )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.heads * self.dim_head)
        return self.to_out(out)


class LeWMConditionalBlock(nn.Module):
    """LeWM transformer block with AdaLN-zero action conditioning."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = LeWMAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = LeWMFeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c.size(1) not in (1, x.size(1)):
            raise ValueError(
                f"Condition sequence length must be 1 or {x.size(1)}, got {c.size(1)}"
            )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class LeWMARPredictor(nn.Module):
    """Autoregressive LeWM predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                LeWMConditionalBlock(
                    hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim) if output_dim and output_dim != hidden_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len]
        x = self.dropout(x)
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer from LeWM."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = int(num_proj)
        t = torch.linspace(0, 3, int(knots), dtype=torch.float32)
        dt = 3 / (int(knots) - 1)
        weights = torch.full((int(knots),), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim != 3:
            raise ValueError(f"SIGReg expects (T,B,D), got {tuple(proj.shape)}")
        generator = torch.randn(
            proj.size(-1),
            self.num_proj,
            device=proj.device,
            dtype=proj.dtype,
        )
        generator = generator / generator.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        x_t = (proj @ generator).unsqueeze(-1) * self.t.to(dtype=proj.dtype)
        err = (
            (x_t.cos().mean(-3) - self.phi.to(dtype=proj.dtype)).square()
            + x_t.sin().mean(-3).square()
        )
        statistic = (err @ self.weights.to(dtype=proj.dtype)) * proj.size(-2)
        return statistic.mean()


class ViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        num_hist: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_hist + 1, state_dim))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=state_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(state_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim),
        )

    def forward(
        self,
        history_states: torch.Tensor,
        action_token: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.cat([history_states, action_token[:, None, :]], dim=1)
        tokens = self.dropout(tokens + self.pos_embed[:, : tokens.size(1), :])
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


@dataclass
class WorldModelOutputs:
    loss: torch.Tensor
    predicted_state: torch.Tensor
    target_state: torch.Tensor
    pred_loss: torch.Tensor
    curvature_loss: torch.Tensor | None = None
    sigreg_loss: torch.Tensor | None = None


class TSWorldModel(nn.Module):
    def __init__(
        self,
        *,
        observation_encoder: nn.Module,
        observation_dim: int,
        adapter_dim: int,
        state_dim: int,
        use_observation_projector: bool,
        projector_type: str,
        projector_spatial_hw: int | None,
        projector_channel_dim: int,
        projector_agg_hidden_dim: int,
        projector_pool_hw: int,
        curvature_weight: float,
        curvature_mode: str,
        action_emb_dim: int,
        action_steps: int,
        action_dim: int,
        num_hist: int,
        predictor_type: str,
        predictor_hidden_dim: int | None,
        predictor_depth: int,
        predictor_heads: int,
        predictor_mlp_dim: int,
        predictor_dim_head: int,
        predictor_emb_dropout: float,
        pred_projector_type: str,
        sigreg_weight: float,
        sigreg_knots: int,
        sigreg_num_proj: int,
        dropout: float,
        detach_target: bool = True,
    ) -> None:
        super().__init__()
        self.observation_encoder = observation_encoder
        self.use_observation_projector = bool(use_observation_projector)
        self.observation_dim = int(observation_dim)
        self.state_dim = int(state_dim)
        self.projector_type = str(projector_type)
        self.curvature_weight = float(curvature_weight)
        self.curvature_mode = str(curvature_mode)
        self.predictor_type = str(predictor_type)
        self.pred_projector_type = str(pred_projector_type)
        self.sigreg_weight = float(sigreg_weight)
        self.detach_target = bool(detach_target)
        if self.use_observation_projector:
            if self.projector_type == "mlp":
                self.projector = ObservationProjector(
                    input_dim=observation_dim,
                    adapter_dim=adapter_dim,
                    state_dim=state_dim,
                )
            elif self.projector_type == "lewm_mlp":
                self.projector = LeWMMLPProjector(
                    input_dim=observation_dim,
                    hidden_dim=adapter_dim,
                    output_dim=state_dim,
                )
            elif self.projector_type == "dino_global":
                self.projector = DinoGlobalProjector(
                    input_dim=observation_dim,
                    hidden_dim=adapter_dim,
                    state_dim=state_dim,
                    pool_hw=projector_pool_hw,
                )
            elif self.projector_type == "dino_channel_agg":
                self.projector = DinoChannelAggProjector(
                    input_dim=observation_dim,
                    spatial_hidden_dim=state_dim,
                    channel_dim=projector_channel_dim,
                    state_dim=state_dim,
                    spatial_hw=int(projector_spatial_hw or 16),
                    agg_hidden_dim=projector_agg_hidden_dim,
                )
            else:
                raise ValueError(f"Unsupported projector_type: {self.projector_type!r}")
        else:
            if int(observation_dim) != int(state_dim):
                raise ValueError(
                    "Disabling the observation projector requires state_dim to match "
                    f"observation_dim, got state_dim={state_dim}, observation_dim={observation_dim}"
                )
            self.projector = nn.Identity()
        if self.predictor_type == "vit":
            self.action_encoder = ActionEncoder(
                action_dim=action_dim,
                action_steps=action_steps,
                action_emb_dim=action_emb_dim,
                state_dim=state_dim,
            )
            self.predictor = ViTPredictor(
                state_dim=state_dim,
                num_hist=num_hist,
                depth=predictor_depth,
                heads=predictor_heads,
                mlp_dim=predictor_mlp_dim,
                dropout=dropout,
            )
        elif self.predictor_type == "lewm_ar":
            self.action_encoder = LeWMActionEncoder(
                action_dim=action_dim,
                action_steps=action_steps,
                smoothed_dim=action_emb_dim,
                emb_dim=state_dim,
            )
            hidden_dim = int(predictor_hidden_dim or state_dim)
            self.predictor = LeWMARPredictor(
                num_frames=num_hist,
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                depth=predictor_depth,
                heads=predictor_heads,
                mlp_dim=predictor_mlp_dim,
                dim_head=predictor_dim_head,
                dropout=dropout,
                emb_dropout=predictor_emb_dropout,
            )
        else:
            raise ValueError(f"Unsupported predictor_type: {self.predictor_type!r}")
        if self.pred_projector_type == "identity":
            self.pred_proj = nn.Identity()
        elif self.pred_projector_type == "lewm_mlp":
            pred_input_dim = int(predictor_hidden_dim or state_dim)
            self.pred_proj = LeWMMLPProjector(
                input_dim=pred_input_dim,
                hidden_dim=adapter_dim,
                output_dim=state_dim,
            )
        else:
            raise ValueError(f"Unsupported pred_projector_type: {self.pred_projector_type!r}")
        self.sigreg = (
            SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)
            if self.sigreg_weight > 0
            else None
        )

    def _curvature_loss(self, states: torch.Tensor) -> torch.Tensor | None:
        if self.curvature_weight <= 0:
            return None
        if self.curvature_mode != "cos":
            raise ValueError(f"Unsupported curvature_mode: {self.curvature_mode!r}")
        if states.ndim != 3 or states.shape[1] < 3:
            return states.new_zeros(())
        v1 = states[:, 1:-1, :] - states[:, :-2, :]
        v2 = states[:, 2:, :] - states[:, 1:-1, :]
        cosine = F.cosine_similarity(v1, v2, dim=-1, eps=1e-6)
        loss = 1.0 - cosine
        step1 = v1.norm(dim=-1)
        step2 = v2.norm(dim=-1)
        mask = (step1 > 1e-6) & (step2 > 1e-6)
        if torch.any(mask):
            return loss[mask].mean()
        return states.new_zeros(())

    def _lewm_predict_sequence(
        self,
        history_states: torch.Tensor,
        action_condition: torch.Tensor,
    ) -> torch.Tensor:
        if history_states.ndim == 3:
            return self.pred_proj(self.predictor(history_states, action_condition))
        if history_states.ndim != 4:
            raise ValueError(
                "LeWM AR predictor expects global states (B,T,D) or spatial states "
                f"(B,T,N,D), got {tuple(history_states.shape)}"
            )
        batch_size, num_frames, num_tokens, state_dim = history_states.shape
        token_states = (
            history_states.permute(0, 2, 1, 3)
            .reshape(batch_size * num_tokens, num_frames, state_dim)
            .contiguous()
        )
        token_condition = (
            action_condition[:, None, :, :]
            .expand(batch_size, num_tokens, num_frames, action_condition.shape[-1])
            .reshape(batch_size * num_tokens, num_frames, action_condition.shape[-1])
            .contiguous()
        )
        predicted = self.predictor(token_states, token_condition)
        predicted = self.pred_proj(predicted)
        return (
            predicted.reshape(batch_size, num_tokens, num_frames, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _sigreg_loss(self, states: torch.Tensor) -> torch.Tensor | None:
        if self.sigreg is None:
            return None
        if states.ndim == 3:
            proj = states.transpose(0, 1)
        elif states.ndim == 4:
            proj = states.permute(1, 0, 2, 3).reshape(
                states.shape[1],
                states.shape[0] * states.shape[2],
                states.shape[3],
            )
        else:
            raise ValueError(f"Unsupported SIGReg state shape: {tuple(states.shape)}")
        return self.sigreg(proj.float())

    def forward(
        self,
        action_chunk: torch.Tensor,
        action_context: torch.Tensor | None = None,
        history_pixel_values: torch.Tensor | None = None,
        future_pixel_values: torch.Tensor | None = None,
        history_states: torch.Tensor | None = None,
        future_state: torch.Tensor | None = None,
        queries: Sequence[str] | None = None,
    ) -> WorldModelOutputs:
        if history_states is None:
            if history_pixel_values is None or future_pixel_values is None:
                raise ValueError(
                    "Pixel mode requires history_pixel_values and future_pixel_values"
                )
            history_features = self.observation_encoder(history_pixel_values, queries=queries)
            history_states = self.projector(history_features)

            with torch.no_grad():
                future_features = self.observation_encoder(
                    future_pixel_values[:, None, ...],
                    queries=queries,
                )
            future_projected = self.projector(future_features).squeeze(1)
            target_state = future_projected.detach() if self.detach_target else future_projected
        else:
            if future_state is None:
                raise ValueError("Feature-cache mode requires future_state")
            if self.use_observation_projector:
                history_states = self.projector(history_states)
                future_projected = self.projector(future_state[:, None, :]).squeeze(1)
                target_state = future_projected.detach() if self.detach_target else future_projected
            else:
                target_state = future_state
                future_projected = target_state

        if self.predictor_type == "lewm_ar":
            condition_source = action_context if action_context is not None else action_chunk
            action_condition = self.action_encoder(condition_source)
            predicted_sequence = self._lewm_predict_sequence(history_states, action_condition)
            target_sequence = torch.cat(
                [history_states[:, 1:, ...], future_projected[:, None, ...]],
                dim=1,
            )
            if self.detach_target:
                target_sequence = target_sequence.detach()
            predicted_state = predicted_sequence[:, -1, ...]
            target_state = target_sequence[:, -1, ...]
            pred_loss = F.mse_loss(predicted_sequence, target_sequence)
        else:
            action_token = self.action_encoder(action_chunk)
            predicted_state = self.predictor(history_states, action_token)
            pred_loss = F.mse_loss(predicted_state, target_state)
        full_states = torch.cat([history_states, future_projected[:, None, ...]], dim=1)
        curvature_states = full_states if full_states.ndim == 3 else full_states.new_zeros(())
        curvature_loss = self._curvature_loss(curvature_states)
        sigreg_loss = self._sigreg_loss(full_states)
        loss = pred_loss
        if curvature_loss is not None:
            loss = loss + self.curvature_weight * curvature_loss
        if sigreg_loss is not None:
            loss = loss + self.sigreg_weight * sigreg_loss
        return WorldModelOutputs(
            loss=loss,
            predicted_state=predicted_state,
            target_state=target_state,
            pred_loss=pred_loss,
            curvature_loss=curvature_loss,
            sigreg_loss=sigreg_loss,
        )

    def forward_pixels(
        self,
        history_pixel_values: torch.Tensor,
        future_pixel_values: torch.Tensor,
        action_chunk: torch.Tensor,
        action_context: torch.Tensor | None = None,
        queries: Sequence[str] | None = None,
    ) -> WorldModelOutputs:
        return self(
            history_pixel_values=history_pixel_values,
            future_pixel_values=future_pixel_values,
            action_chunk=action_chunk,
            action_context=action_context,
            queries=queries,
        )

    def predict_next(
        self,
        history_states: torch.Tensor,
        action_chunk: torch.Tensor,
        action_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.predictor_type == "lewm_ar":
            condition_source = action_context if action_context is not None else action_chunk
            action_condition = self.action_encoder(condition_source)
            predicted_sequence = self._lewm_predict_sequence(history_states, action_condition)
            return predicted_sequence[:, -1, ...]
        action_token = self.action_encoder(action_chunk)
        return self.predictor(history_states, action_token)


def build_world_model(config: ExperimentConfig) -> TSWorldModel:
    encoder_bundle = build_frozen_encoder(
        encoder_type=config.model.encoder_type,
        dino_model_name=config.model.dino_model_name,
        dino_input_size=config.model.dino_input_size,
        encoder_checkpoint=config.model.encoder_checkpoint,
        encoder_backbone_dtype=config.model.encoder_backbone_dtype,
        query=config.model.query,
        readout=config.model.readout,
    )
    torch.manual_seed(int(config.seed))
    state_dim = (
        int(config.model.state_dim)
        if config.model.use_observation_projector
        else int(encoder_bundle.output_dim)
    )
    return TSWorldModel(
        observation_encoder=encoder_bundle.encoder,
        observation_dim=encoder_bundle.output_dim,
        adapter_dim=config.model.adapter_dim,
        state_dim=state_dim,
        use_observation_projector=config.model.use_observation_projector,
        projector_type=config.model.projector_type,
        projector_spatial_hw=config.model.projector_spatial_hw,
        projector_channel_dim=config.model.projector_channel_dim,
        projector_agg_hidden_dim=config.model.projector_agg_hidden_dim,
        projector_pool_hw=config.model.projector_pool_hw,
        curvature_weight=config.model.curvature_weight,
        curvature_mode=config.model.curvature_mode,
        action_emb_dim=config.model.action_emb_dim,
        action_steps=config.data.target_offset,
        action_dim=getattr(config.data, "action_dim", 2),
        num_hist=config.data.num_hist,
        predictor_type=config.model.predictor_type,
        predictor_hidden_dim=config.model.predictor_hidden_dim,
        predictor_depth=config.model.predictor_depth,
        predictor_heads=config.model.predictor_heads,
        predictor_mlp_dim=config.model.predictor_mlp_dim,
        predictor_dim_head=config.model.predictor_dim_head,
        predictor_emb_dropout=config.model.predictor_emb_dropout,
        pred_projector_type=config.model.pred_projector_type,
        sigreg_weight=config.model.sigreg_weight,
        sigreg_knots=config.model.sigreg_knots,
        sigreg_num_proj=config.model.sigreg_num_proj,
        dropout=config.model.dropout,
        detach_target=config.model.detach_target,
    )


def build_cached_world_model(config: ExperimentConfig, *, feature_dim: int) -> TSWorldModel:
    torch.manual_seed(int(config.seed))
    state_dim = (
        int(config.model.state_dim)
        if config.model.use_observation_projector
        else int(feature_dim)
    )
    return TSWorldModel(
        observation_encoder=nn.Identity(),
        observation_dim=int(feature_dim),
        adapter_dim=config.model.adapter_dim,
        state_dim=state_dim,
        use_observation_projector=config.model.use_observation_projector,
        projector_type=config.model.projector_type,
        projector_spatial_hw=config.model.projector_spatial_hw,
        projector_channel_dim=config.model.projector_channel_dim,
        projector_agg_hidden_dim=config.model.projector_agg_hidden_dim,
        projector_pool_hw=config.model.projector_pool_hw,
        curvature_weight=config.model.curvature_weight,
        curvature_mode=config.model.curvature_mode,
        action_emb_dim=config.model.action_emb_dim,
        action_steps=config.data.target_offset,
        action_dim=getattr(config.data, "action_dim", 2),
        num_hist=config.data.num_hist,
        predictor_type=config.model.predictor_type,
        predictor_hidden_dim=config.model.predictor_hidden_dim,
        predictor_depth=config.model.predictor_depth,
        predictor_heads=config.model.predictor_heads,
        predictor_mlp_dim=config.model.predictor_mlp_dim,
        predictor_dim_head=config.model.predictor_dim_head,
        predictor_emb_dropout=config.model.predictor_emb_dropout,
        pred_projector_type=config.model.pred_projector_type,
        sigreg_weight=config.model.sigreg_weight,
        sigreg_knots=config.model.sigreg_knots,
        sigreg_num_proj=config.model.sigreg_num_proj,
        dropout=config.model.dropout,
        detach_target=config.model.detach_target,
    )


__all__ = [
    "TSWorldModel",
    "LeWMARPredictor",
    "LeWMActionEncoder",
    "LeWMMLPProjector",
    "SIGReg",
    "ViTPredictor",
    "WorldModelOutputs",
    "build_cached_world_model",
    "build_world_model",
]
