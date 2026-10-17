"""Base predictor class for vision-language models."""

from abc import ABC, abstractmethod
import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PredictorBackbone(nn.Module, ABC):
    """Abstract base class for LLM backbones.

    Subclasses must implement:
        - hidden_size: int property
        - device: torch.device property
        - dtype: torch.dtype property
        - embed_text_tokens(texts, max_length) -> (embeddings, lengths)
        - forward(hidden_states, attention_mask) -> hidden_states (normalized)
    """

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Hidden dimension of the model."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device of the model parameters."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Dtype of the model parameters."""

    @abstractmethod
    def embed_text(
        self,
        texts: list[str],
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and embed text strings.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length (pad/truncate to this)

        Returns:
            embeddings: [B, max_length, H] token embeddings
            lengths: [B] actual token counts (before padding), on CPU
        """

    @abstractmethod
    def forward(
        self,
        input_embeds: torch.Tensor,
        input_lengths: torch.Tensor,
        query_embeds: torch.Tensor,
        query_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through transformer layers with normalization.

        Args:
            input_embeds: [B, max_input_tokens, H] input embeddings
            input_lengths: [B] valid input token counts
            query_embeds: [B, max_query_tokens, H] query embeddings
            query_lengths: [B] valid query token counts

        Returns:
            hidden_states: [B, max_input_tokens + max_query_tokens, H] normalized output
                aligned as [input_tokens, query_tokens], even if the backbone uses a
                different internal execution order.
        """


class Predictor(nn.Module):
    """Base predictor that processes vision OR text input with optional queries.

    Handles:
        - Input preparation (vision projection, padding)
        - Attention mask creation
        - Masked mean pooling
        - Output projection

    Delegates to backbone:
        - Text tokenization and embedding
        - Transformer forward pass with normalization

    Logical sequence structure: [input_tokens (max_input_tokens)] + [query_tokens (max_query_tokens)].
    Backbones may use a different internal execution order if needed, but must return
    hidden states in this layout.
    """

    @staticmethod
    def _create_backbone(config: dict) -> PredictorBackbone:
        """Create a backbone from config dict."""
        backbone_type = config["type"]

        if backbone_type in {"llama3", "smollm2"}:
            from .llama3 import Llama3Backbone

            return Llama3Backbone(
                model_name=config["model_name"],
                pretrained_weights=config.get("pretrained_weights", True),
                start_layer=config.get("start_layer", 0),
                num_layers=config.get("num_layers"),
                dtype=config.get("dtype", "float32"),
            )
        if backbone_type in {"qwen3_5", "qwen3.5"}:
            from .qwen3_5 import Qwen3_5Backbone

            return Qwen3_5Backbone(
                model_name=config["model_name"],
                pretrained_weights=config.get("pretrained_weights", True),
                start_layer=config.get("start_layer", 0),
                num_layers=config.get("num_layers"),
            )
        if backbone_type in {"qwen3_embedding", "qwen3-embedding"}:
            from .qwen3_embedding import Qwen3EmbeddingBackbone

            return Qwen3EmbeddingBackbone(
                model_name=config["model_name"],
                pretrained_weights=config.get("pretrained_weights", True),
                start_layer=config.get("start_layer", 0),
                num_layers=config.get("num_layers"),
            )
        raise ValueError(f"Unsupported backbone type: {backbone_type}")

    @staticmethod
    def _init_projection(proj: nn.Linear) -> nn.Linear:
        """Initialize projection layer with small weights."""
        nn.init.normal_(proj.weight, mean=0.0, std=1e-5)
        if proj.bias is not None:
            nn.init.zeros_(proj.bias)
        return proj

    def __init__(
        self,
        backbone: PredictorBackbone | dict,
        vision_dim: int,
        max_input_tokens: int = 2048,
        max_query_tokens: int = 256,
        pooling: str = "mean",
        query_pool_temperature: float | None = None,
        query_pool_exclude_first_token: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = self._create_backbone(backbone)
        self.max_input_tokens = int(max_input_tokens)
        self.max_query_tokens = int(max_query_tokens)
        self.hidden_size = self.backbone.hidden_size
        self.pooling = str(pooling)
        if self.pooling not in {"mean", "query_attention", "patch_tokens"}:
            raise ValueError(
                "predictor.pooling must be one of ['mean', 'query_attention', 'patch_tokens'], "
                f"got {self.pooling!r}."
            )
        if query_pool_temperature is None:
            query_pool_temperature = math.sqrt(float(self.hidden_size))
        self.query_pool_temperature = float(query_pool_temperature)
        if self.query_pool_temperature <= 0.0:
            raise ValueError("query_pool_temperature must be positive.")
        self.query_pool_exclude_first_token = bool(query_pool_exclude_first_token)
        logger.info(
            "ConfigurableEncoder predictor pooling=%s query_pool_temperature=%.4f exclude_first_token=%s",
            self.pooling,
            self.query_pool_temperature,
            self.query_pool_exclude_first_token,
        )

        # vision -> backbone projection
        self.input_projection = (
            self._init_projection(nn.Linear(vision_dim, self.hidden_size))
            if vision_dim > 0
            else None
        )

    def _prepare_input_embeddings(
        self,
        vision_embedding: torch.Tensor | None,
        text_input: list[str | None] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare input embeddings from vision or text, padded/truncated to max_input_tokens.

        Args:
            vision_embedding: [B, Nv, H] projected vision embedding (None = text-only mode)
            text_input: List of strings or None per sample (None = use vision)

        Returns:
            input_embeds: [B, max_input_tokens, H]
            input_lengths: [B] actual sequence lengths (CPU tensor)
        """
        # Infer batch size
        if vision_embedding is not None:
            B = vision_embedding.size(0)
        elif text_input is not None:
            B = len(text_input)
        else:
            raise ValueError(
                "At least one of vision_embedding or text_input must be provided."
            )

        H = self.hidden_size
        device = self.backbone.device
        dtype = self.backbone.dtype

        # Initialize input embeddings and lengths
        input_embeds = torch.zeros(
            B, self.max_input_tokens, H, device=device, dtype=dtype
        )
        input_lengths = torch.zeros(B, dtype=torch.long)
        has_text = [t is not None for t in (text_input or [None] * B)]

        # Fill vision samples (only for those without text)
        if vision_embedding is not None:
            Nv = vision_embedding.size(1)
            if Nv > self.max_input_tokens:
                if self.pooling == "patch_tokens":
                    raise ValueError(
                        f"Vision tokens ({Nv}) exceed max_input_tokens ({self.max_input_tokens}); "
                        "increase the configured capacity to preserve all spatial patches."
                    )
                logger.warning(
                    f"Vision tokens ({Nv}) exceed max_input_tokens ({self.max_input_tokens}). "
                    f"Truncating to first {self.max_input_tokens} tokens."
                )
            vision_len = min(Nv, self.max_input_tokens)
            for i in range(B):
                if not has_text[i]:
                    input_embeds[i, :vision_len] = vision_embedding[
                        i, :vision_len
                    ]
                    input_lengths[i] = vision_len

        # Fill text samples
        text_indices = [i for i, has in enumerate(has_text) if has]
        if text_indices:
            texts = [text_input[i] for i in text_indices]
            text_embeds, actual_lengths = self.backbone.embed_text(
                texts, self.max_input_tokens
            )
            for j, i in enumerate(text_indices):
                input_embeds[i] = text_embeds[j]
                input_lengths[i] = actual_lengths[j]

        return input_embeds, input_lengths

    def _masked_mean_pool(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Masked mean pooling over variable-length sequences.

        Args:
            hidden_states: [B, S, H] hidden states
            lengths: [B] actual sequence lengths per sample (CPU tensor)

        Returns:
            pooled: [B, H] mean-pooled over non-padding positions
        """
        _B, S, _H = hidden_states.shape
        device = hidden_states.device
        lengths = lengths.to(device)

        # Create mask: [B, S] where True = valid token
        positions = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
        mask = positions < lengths.unsqueeze(1)  # [B, S]
        mask_expanded = mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, S, 1]

        # Masked sum and mean
        masked_sum = (hidden_states * mask_expanded).sum(dim=1)  # [B, H]
        pooled = masked_sum / lengths.unsqueeze(1).clamp(min=1).to(
            hidden_states.dtype
        )  # [B, H]

        return pooled

    def _query_attention_pool(
        self,
        input_hidden: torch.Tensor,
        input_lengths: torch.Tensor,
        query_hidden: torch.Tensor,
        query_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Pool input tokens with attention scores induced by the query tokens."""
        if query_hidden.size(1) == 0:
            return self._masked_mean_pool(input_hidden, input_lengths)

        device = input_hidden.device
        input_lengths = input_lengths.to(device)
        query_lengths = query_lengths.to(device)

        q_positions = torch.arange(query_hidden.size(1), device=device).unsqueeze(0)
        q_mask = q_positions < query_lengths.unsqueeze(1)
        q_weights = q_mask.unsqueeze(-1).to(query_hidden.dtype)
        q_denom = query_lengths.clamp(min=1).to(query_hidden.dtype).unsqueeze(1)
        query_summary = (query_hidden * q_weights).sum(dim=1) / q_denom

        scores = torch.einsum(
            "bsh,bh->bs", input_hidden, query_summary
        ) / self.query_pool_temperature
        positions = torch.arange(input_hidden.size(1), device=device).unsqueeze(0)
        valid = positions < input_lengths.unsqueeze(1)
        if self.query_pool_exclude_first_token:
            without_first = valid & (positions > 0)
            valid = torch.where(without_first.any(dim=1, keepdim=True), without_first, valid)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=1).to(input_hidden.dtype)
        return (input_hidden * weights.unsqueeze(-1)).sum(dim=1)

    def forward(
        self,
        vision_embedding: torch.Tensor | None = None,
        text_input: list[str | None] | None = None,
        queries: list[str] | None = None,
    ) -> torch.Tensor:
        """Forward pass through predictor.

        Args:
            vision_embedding: [B, Nv, D] vision token embedding (None for text-only mode)
            text_input: List of text strings or None per sample (complements vision)
            queries: Optional query strings (empty string if None)

        Returns:
            Pooled output [B, output_dim]
        """
        # Determine batch size
        vision_token_count = None
        if vision_embedding is not None:
            B = vision_embedding.size(0)
            vision_token_count = min(int(vision_embedding.size(1)), self.max_input_tokens)
            # Project vision embedding to hidden dimension
            proj_weight = self.input_projection.weight
            vision_embedding = vision_embedding.to(
                device=proj_weight.device,
                dtype=proj_weight.dtype,
            )
            vision_embedding = self.input_projection(vision_embedding).to(
                dtype=self.backbone.dtype
            )
        elif text_input is not None:
            B = len(text_input)
        else:
            raise ValueError(
                "At least one of vision_embedding or text_input must be provided."
            )

        # Prepare input embeddings (vision or text, padded to max_input_tokens)
        input_embeds, input_lengths = self._prepare_input_embeddings(
            vision_embedding, text_input
        )

        # Prepare query embeddings
        if self.max_query_tokens > 0:
            query_embeds, query_lengths = self.backbone.embed_text(
                queries or [""] * B, self.max_query_tokens
            )
        else:
            H = self.backbone.hidden_size
            query_embeds = torch.empty(
                B, 0, H, device=input_embeds.device, dtype=input_embeds.dtype
            )
            query_lengths = torch.zeros(B, dtype=torch.long)

        # Forward through backbone (handles concat, masking, normalization)
        hidden_states = self.backbone(
            input_embeds, input_lengths, query_embeds, query_lengths
        )

        # Pool over input window only (first max_input_tokens)
        input_hidden = hidden_states[
            :, : self.max_input_tokens, :
        ]  # [B, max_input_tokens, H]
        if self.pooling == "patch_tokens":
            if vision_token_count is None or vision_token_count <= 1:
                raise ValueError("predictor.pooling='patch_tokens' requires CLS+patch vision tokens.")
            return input_hidden[:, 1:vision_token_count, :]  # [B, Nv-1, H]
        if self.pooling == "query_attention":
            query_hidden = hidden_states[
                :, self.max_input_tokens : self.max_input_tokens + self.max_query_tokens, :
            ]
            pooled = self._query_attention_pool(
                input_hidden, input_lengths, query_hidden, query_lengths
            )
        else:
            pooled = self._masked_mean_pool(input_hidden, input_lengths)  # [B, H]

        return pooled  # [B, hidden_size]

    def forward_packed(
        self,
        text_input: list[str | None],
    ) -> torch.Tensor:
        """Forward pass using sequence packing for high token utilization.

        Bins multiple short samples into each window using block-diagonal attention.
        Gives 8-10× better MFU for short-text datasets vs standard padded forward.

        Args:
            text_input: B text strings to pack and process.

        Returns:
            pooled: [B, hidden_size] per-sample mean-pooled output (same as forward()).
        """
        from src.train.packing import pack_text_batch, unpack_pooled

        device = self.backbone.device
        dtype = self.backbone.dtype

        # Tokenize and pack into windows
        input_ids, _attn_mask_2d, pack_info = pack_text_batch(
            text_input,
            tokenizer=self.backbone.tokenizer,
            max_window_len=self.max_input_tokens,
            device=device,
            dtype=dtype,
        )

        # Forward through packed backbone
        hidden_states = self.backbone.forward_packed(input_ids, pack_info)
        # hidden_states: (n_windows, max_input_tokens, H)

        # Unpack and pool per original sample
        pooled = unpack_pooled(hidden_states, pack_info)  # (B, H)

        return pooled  # [B, hidden_size]
