"""Llama 3 backbone for the predictor."""

import logging
import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .base import PredictorBackbone

logger = logging.getLogger(__name__)


class Llama3Backbone(PredictorBackbone):
    """Llama 3 backbone providing tokenization, embedding, and transformer forward.

    Handles:
        - Model loading and layer selection
        - Tokenization via HuggingFace tokenizer
        - Token embedding lookup
        - Transformer forward with RoPE
        - Output normalization (applied before returning)
    """

    def __init__(
        self,
        model_name: str,
        pretrained_weights: bool = True,
        start_layer: int = 0,
        num_layers: int | None = None,
        dtype: str = "float32",
    ) -> None:
        super().__init__()

        self.model_name = model_name
        is_local = os.path.isdir(model_name)

        # Parameter storage is independent of the caller's autocast policy.
        dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        if dtype not in dtypes:
            raise ValueError(f"Unsupported Llama dtype: {dtype!r}; use float32 or bfloat16")
        parameter_dtype = dtypes[dtype]
        # Use local_files_only when loading an existing model directory.
        if pretrained_weights:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, local_files_only=is_local, dtype=parameter_dtype
            )
        else:
            cfg = AutoConfig.from_pretrained(
                model_name, local_files_only=is_local
            )
            model = AutoModelForCausalLM.from_config(cfg, dtype=parameter_dtype)

        # Disable causal masking for bidirectional attention
        for layer in model.model.layers:
            layer.self_attn.is_causal = False

        self.config = model.config
        self._hidden_size = int(model.config.hidden_size)

        # Select layers
        all_layers = model.model.layers
        if num_layers is None or num_layers == -1:
            self.layers = all_layers
        else:
            self.layers = nn.ModuleList(
                all_layers[start_layer : start_layer + num_layers]
            )

        # Keep necessary components
        self.norm = model.model.norm
        self.rotary_emb = model.model.rotary_emb
        self.embed_tokens = model.model.embed_tokens

        # Tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, local_files_only=is_local
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = int(self.tokenizer.pad_token_id)

        self.gradient_checkpointing = False

    @property
    def hidden_size(self) -> int:
        """Hidden dimension of the model."""
        return self._hidden_size

    @property
    def device(self) -> torch.device:
        """Device of the model parameters."""
        return self.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the model parameters."""
        return next(self.layers[0].parameters()).dtype

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
        device = self.device
        dtype = self.embed_tokens.weight.dtype

        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = tokenized.input_ids.to(device)
        embeddings = self.embed_tokens(input_ids).to(dtype=dtype)
        lengths = tokenized.attention_mask.sum(dim=1)  # stays on CPU

        return embeddings, lengths

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
            hidden_states: [B, S, H] normalized output hidden states
        """
        # Concatenate input and query
        hidden_states = torch.cat(
            [input_embeds, query_embeds], dim=1
        )  # [B, S, H]
        B, S, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        max_input = input_embeds.size(1)
        input_lengths = input_lengths.to(device)
        query_lengths = query_lengths.to(device)

        # Create attention mask: valid in input region OR valid in query region
        pos = torch.arange(S, device=device)
        valid = (pos < input_lengths.unsqueeze(1)) | (
            (pos >= max_input) & (pos - max_input < query_lengths.unsqueeze(1))
        )
        attention_mask = torch.where(valid, 0.0, float("-inf")).to(dtype)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # [B, 1, 1, S]

        # Position embeddings (RoPE)
        position_ids = torch.arange(S, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Forward through transformer layers (with optional gradient checkpointing)
        def run_layer(layer, h, attn_mask, pos_ids, pos_emb):
            out = layer(
                h,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                position_embeddings=pos_emb,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            return out[0] if isinstance(out, tuple) else out

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                import torch.utils.checkpoint as cp_utils

                hidden_states = cp_utils.checkpoint(
                    run_layer,
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    position_embeddings,
                    use_reentrant=False,
                )
            else:
                hidden_states = run_layer(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    position_embeddings,
                )

        return self.norm(hidden_states)

    def forward_packed(
        self,
        input_ids: torch.Tensor,
        pack_info,
    ) -> torch.Tensor:
        """Forward pass for sequence-packed input (multiple samples per window).

        Uses a block-diagonal attention mask so each sample only attends to
        its own tokens within the packed window. This eliminates padding waste
        for short-text datasets.

        Args:
            input_ids: (n_windows, max_input_tokens) long tensor of packed token IDs
            pack_info: PackInfo with offsets/lengths/window_indices for unpacking

        Returns:
            hidden_states: (n_windows, max_input_tokens, H) normalized output
                Use unpack_pooled(hidden_states, pack_info) to get per-sample pooled output.
        """
        from src.train.packing import make_block_diagonal_mask

        device = self.device
        dtype = self.dtype
        n_windows, S = input_ids.shape

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids.to(device)).to(dtype=dtype)

        # Build block-diagonal additive attention mask [n_windows, 1, S, S]
        attention_mask = make_block_diagonal_mask(pack_info, S, dtype, device)

        # Document-local position IDs: each sample sees positions 0..length-1 (not global offsets).
        # This matches what the normal (non-packed) forward does, so RoPE embeddings are identical.
        position_ids = torch.zeros(
            n_windows, S, dtype=torch.long, device=device
        )
        for s_idx in range(pack_info.offsets.shape[0]):
            w = int(pack_info.window_indices[s_idx].item())
            off = int(pack_info.offsets[s_idx].item())
            length = int(pack_info.lengths[s_idx].item())
            end = min(off + length, S)
            actual_len = end - off
            if actual_len > 0:
                position_ids[w, off:end] = torch.arange(
                    actual_len, dtype=torch.long, device=device
                )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        def run_layer(layer, h, attn_mask, pos_ids, pos_emb):
            out = layer(
                h,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                position_embeddings=pos_emb,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            return out[0] if isinstance(out, tuple) else out

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                import torch.utils.checkpoint as cp_utils

                hidden_states = cp_utils.checkpoint(
                    run_layer,
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    position_embeddings,
                    use_reentrant=False,
                )
            else:
                hidden_states = run_layer(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    position_embeddings,
                )

        return self.norm(hidden_states)
