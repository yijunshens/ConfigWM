from __future__ import annotations

import math
import logging
from contextlib import nullcontext
from pathlib import Path
import shutil
import time

from src.train.collate import extract_text_field
from src.train.contrastive import (
    patch_position_logits,
    patch_position_multi_positive_loss,
    patch_spatial_logits,
)
from src.train.dist import all_gather_no_grad, all_gather_with_grad
from src.train.optim import LRMultiplierMixin
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import Trainer, TrainerCallback

# H200 peak BF16 TFLOP/s (SXM, 989 TFLOP/s theoretical)
_H200_PEAK_TFLOPS = 989.0


def _estimate_flops_per_sample(model, seq_len: int) -> float:
    """Estimate FLOPs per training sample using the 6×N×S approximation.

    Uses non-embedding parameters of the predictor backbone as N, since that is
    the dominant compute in a text-only Configurable Encoder step.  The factor 6 accounts for
    one forward pass (2×) plus one backward pass (4×) per sample token.

    Args:
        model: The (possibly DDP/FSDP-wrapped) ConfigurableEncoder model.
        seq_len: Sequence length (max_input_tokens).

    Returns:
        Estimated FLOPs per sample (float).  Returns 0.0 if structure is
        unrecognised so callers can skip MFU logging gracefully.
    """
    unwrapped = model.module if hasattr(model, "module") else model
    backbone = getattr(getattr(unwrapped, "predictor", None), "backbone", None)
    if backbone is None:
        return 0.0
    non_embed_params = sum(
        p.numel()
        for name, p in backbone.named_parameters()
        if "embed_tokens" not in name
    )
    return 6.0 * non_embed_params * seq_len


def _to_float_cpu(x: torch.Tensor) -> torch.Tensor:
    """Detach -> float32 -> CPU (no grad)."""
    with torch.no_grad():
        x = x.detach()
        x = x.float()
        if x.is_cuda:
            x = x.cpu()
    return x


def _reshape_for_norms(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape (N, D) for vector L2 norms along the last dim.

    If x is 1D, treat it as a single vector of size D.
    """
    if x.dim() == 1:
        return x.unsqueeze(0)  # (1, D)
    if x.dim() >= 2:
        D = x.shape[-1]
        return x.reshape(-1, D)
    return x.view(1, 1)


def _embedding_stats(x: torch.Tensor | None, prefix: str) -> dict[str, float]:
    """Return two metrics to monitor embedding health.

    Metrics:
      - f"{prefix}/norm": mean L2 norm across vectors.
      - f"{prefix}/std" : cross-sample (batch-wise) variability, computed as:
            1) reshape to (N, D) so each row is one embedding vector,
            2) compute per-dimension std across the N samples,
            3) take the mean over dimensions.
        This goes ~0 when all samples collapse to (approximately) the same vector.

    Notes:
      If x is 1D (a single vector) or batch size N < 2, we can't measure cross-sample variation,
        so std is defined as 0.0.
      • We keep the previous semantics for "norm": mean L2 norm of the vectors.
      • Uses unbiased=False to avoid NaNs at small batch sizes.
    """
    if x is None:
        return {}
    with torch.no_grad():
        x = _to_float_cpu(x)
        if x.numel() == 0:
            return {}

        # Ensure shape (N, D): every row is one embedding vector.
        mat = _reshape_for_norms(x)  # (N, D)

        # Mean L2 norm across vectors
        norms = mat.norm(dim=-1)
        norm_mean = float(norms.mean().item())

        # Batch-wise per-dimension std averaged over dims (collapse -> ~0)
        if mat.size(0) >= 2:
            centered = mat - mat.mean(dim=0, keepdim=True)
            per_dim_std = centered.std(dim=0, unbiased=False)  # (D,)
            std_batch = float(per_dim_std.mean().item())
        else:
            std_batch = 0.0

        return {f"{prefix}/norm": norm_mean, f"{prefix}/std": std_batch}


@torch.no_grad()
def _compute_effective_rank(z: torch.Tensor, eps: float = 1.0e-12) -> float:
    """Entropy effective rank of a batch of embeddings."""
    z = _reshape_for_norms(_to_float_cpu(z))
    if z.size(0) < 2 or z.numel() == 0:
        return 0.0

    z = z - z.mean(dim=0, keepdim=True)
    cov = z.T @ z / max(1, z.size(0) - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
    total = eigvals.sum()
    if (not bool(torch.isfinite(total).item())) or float(total.item()) <= eps:
        return 0.0

    p = eigvals / total
    p = p[p > eps]
    if p.numel() == 0:
        return 0.0
    return float(torch.exp(-(p * p.log()).sum()).item())


@torch.no_grad()
def _embedding_std(x: torch.Tensor) -> float:
    """Mean per-dimension std across embedding samples."""
    mat = _reshape_for_norms(_to_float_cpu(x))
    if mat.size(0) < 2 or mat.numel() == 0:
        return 0.0
    centered = mat - mat.mean(dim=0, keepdim=True)
    return float(centered.std(dim=0, unbiased=False).mean().item())


def _dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_sum_(x: torch.Tensor) -> torch.Tensor:
    if _dist_is_ready():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


class _IterableWrapper(IterableDataset):
    """Wrap a map-style dataset as a rank/worker-sharded IterableDataset."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
        else:
            rank = 0
            world_size = 1
        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers
        for index in range(shard_id, len(self._dataset), num_shards):
            yield self._dataset[index]

    def __len__(self):
        return len(self._dataset)


class _EpochEmbeddingDiagnosticsCallback(TrainerCallback):
    """Bridge HF callback flow into ConfigurableEncoderTrainer.on_epoch_end()."""

    def __init__(self, trainer: "ConfigurableEncoderTrainer") -> None:
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer.on_epoch_end()
        return control


class ConfigurableEncoderTrainer(LRMultiplierMixin, Trainer):
    def __init__(self, *args, **kwargs):
        # HF Trainer.__init__ calls accelerator.unwrap_model(model), which fails if the
        # model already contains a torch-compiled OptimizedModule (KeyError: '_orig_mod').
        # Solution: run super().__init__ first (with plain model), then compile the backbone.
        #
        # Additional constraint for compile_text_encoder:
        #   output_capturing.py (transformers 5.x) wraps HuggingFace PreTrainedModel.forward
        #   during super().__init__(). Compiling the WRAPPER fails with NameError: 'torch'.
        #   Fix: save the ORIGINAL auto_model.forward BEFORE super().__init__(), compile
        #   that original afterwards (bypassing the output_capturing wrapper entirely).
        import logging as _log

        _saved_te_forward = None
        if kwargs.get("args") is not None and getattr(
            kwargs["args"], "compile_text_encoder", False
        ):
            model_pre = kwargs.get("model")
            if model_pre is not None:
                te_pre = getattr(model_pre, "text_encoder", None)
                sbert_pre = getattr(te_pre, "model", None)
                tm_pre = sbert_pre[0] if sbert_pre is not None else None
                hf_pre = (
                    getattr(tm_pre, "auto_model", None)
                    if tm_pre is not None
                    else None
                )
                if hf_pre is not None:
                    _saved_te_forward = (
                        hf_pre.forward
                    )  # save BEFORE output_capturing wraps it
                    _log.info(
                        "compile_text_encoder: saved original auto_model.forward pre-init."
                    )

        super().__init__(*args, **kwargs)

        self.lr_multipliers = dict(
            getattr(self.args, "lr_multipliers", None) or {}
        )
        self.gather_embeddings = getattr(self.args, "gather_embeddings", True)

        # Compile backbone's forward method AFTER HF Trainer init.
        #
        # Why not torch.compile(backbone_module)?
        #   → Creates an OptimizedModule; accelerate 1.12.0's extract_model_from_parallel
        #     then calls model.__dict__["_orig_mod"] on the parent ConfigurableEncoder (not an OptimizedModule),
        #     causing KeyError: '_orig_mod' in trainer.train().
        #
        # Fix: compile backbone.forward (the function) instead of the whole module.
        #   → backbone stays a plain nn.Module; has_compiled_regions() returns False;
        #     accelerate never tries to unwrap it. Full forward-pass speedup still achieved.
        import logging as _log

        unwrapped = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        if getattr(self.args, "compile_backbone", False):
            backbone = getattr(
                getattr(unwrapped, "predictor", None), "backbone", None
            )
            if backbone is not None and not isinstance(
                backbone.forward, torch._dynamo.eval_frame.OptimizedModule
            ):
                _log.info(
                    "Compiling backbone.forward with torch.compile(inductor)..."
                )
                backbone.forward = torch.compile(
                    backbone.forward,
                    backend="inductor",
                    fullgraph=False,
                    dynamic=False,
                )
                _log.info(
                    "Backbone.forward compiled (kernel build deferred to first batch)."
                )
            else:
                _log.info(
                    "compile_backbone: backbone.forward already compiled or backbone not found."
                )

        if getattr(self.args, "compile_text_encoder", False):
            # Compile individual transformer layers inside the SentenceTransformer's HF model.
            #
            # Why not compile auto_model.forward or inner.forward?
            #   → output_capturing.py (transformers 5.x) wraps PreTrainedModel-adjacent code.
            #     When Dynamo traces through auto_model.forward, it reaches HuggingFace
            #     DecoderLayer classes that have been modified by output_capturing.
            #     During symbolic tracing, output_capturing's dynamic @wraps(func) code
            #     executes in a context where 'torch' is inaccessible → NameError.
            #     This happens regardless of which version of auto_model.forward we compile:
            #     the traced graph inevitably passes through output_capturing-modified layers.
            #
            # Fix: compile individual transformer decoder layers (e.g. Gemma3DecoderLayer).
            #   → Decoder layers are plain nn.Modules, NOT PreTrainedModel subclasses.
            #   → output_capturing never wraps them; Dynamo traces cleanly.
            #   → Same pattern as backbone compile: kernel fusion per-layer.
            #   → Requires pad_to_max_length=true in text encoder config for fixed shapes.
            text_encoder = getattr(unwrapped, "text_encoder", None)
            sbert = (
                getattr(text_encoder, "model", None)
                if text_encoder is not None
                else None
            )
            transformer_module = sbert[0] if sbert is not None else None
            hf_model = (
                getattr(transformer_module, "auto_model", None)
                if transformer_module is not None
                else None
            )
            # Navigate to the decoder layers. Two layouts exist:
            #   (a) hf_model.layers        — when auto_model is the text body (e.g. Gemma3TextModel)
            #   (b) hf_model.model.layers  — when auto_model is the full CausalLM wrapper
            layers = (
                getattr(hf_model, "layers", None)
                if hf_model is not None
                else None
            )
            if layers is None:
                inner_body = (
                    getattr(hf_model, "model", None)
                    if hf_model is not None
                    else None
                )
                layers = (
                    getattr(inner_body, "layers", None)
                    if inner_body is not None
                    else None
                )
            if layers is not None and len(layers) > 0:
                n = 0
                for layer in layers:
                    if not isinstance(
                        layer.forward, torch._dynamo.eval_frame.OptimizedModule
                    ):
                        layer.forward = torch.compile(
                            layer.forward,
                            backend="inductor",
                            fullgraph=False,
                            dynamic=False,
                        )
                        n += 1
                _log.info(
                    "compile_text_encoder: compiled %d decoder layers (type=%s).",
                    n,
                    type(layers[0]).__name__,
                )
            else:
                _log.info(
                    "compile_text_encoder: could not find .model.layers — skipping."
                )

        if getattr(self.args, "compile_vision_encoder", False):
            # Compile vision_encoder.forward (fixed patch-grid shape → dynamic=False).
            # Only active when vision encoder is enabled (image pretraining).
            vision_encoder = getattr(unwrapped, "vision_encoder", None)
            if vision_encoder is not None and not isinstance(
                vision_encoder.forward, torch._dynamo.eval_frame.OptimizedModule
            ):
                _log.info(
                    "Compiling vision_encoder.forward with torch.compile(inductor, dynamic=False)..."
                )
                vision_encoder.forward = torch.compile(
                    vision_encoder.forward,
                    backend="inductor",
                    fullgraph=False,
                    dynamic=False,
                )
                _log.info(
                    "vision_encoder.forward compiled (kernel build deferred to first batch)."
                )
            else:
                _log.info(
                    "compile_vision_encoder: vision encoder disabled or already compiled."
                )

        # Micro-step accumulators -> averaged at next HF log() call
        self._pending_sums: dict[str, float] = {}
        self._pending_counts: dict[str, int] = {}

        # MFU / throughput tracking
        self._mfu_last_log_time: float | None = None
        self._mfu_last_log_step: int = 0
        self._flops_per_sample: float = 0.0  # computed lazily after first step
        self._embedding_diag_last_epoch: float | None = None

        if getattr(self.args, "embedding_diagnostics_on_epoch_end", True):
            self.add_callback(_EpochEmbeddingDiagnosticsCallback(self))

    def _accumulate_pending(self, metrics: dict[str, float]) -> None:
        """Accumulate micro-step metrics to be averaged at the next logging event."""
        for k, v in metrics.items():
            if v is None or not math.isfinite(v):
                continue
            self._pending_sums[k] = self._pending_sums.get(k, 0.0) + float(v)
            self._pending_counts[k] = self._pending_counts.get(k, 0) + 1

    def _drain_pending_means(self) -> dict[str, float]:
        """Return averaged metrics and clear accumulators."""
        if not self._pending_sums:
            return {}
        out = {
            k: self._pending_sums[k] / max(1, self._pending_counts.get(k, 1))
            for k in self._pending_sums
        }
        self._pending_sums.clear()
        self._pending_counts.clear()
        return out

    def on_epoch_end(self) -> None:
        """Log epoch-end embedding health metrics without requiring do_eval."""
        if not getattr(self.args, "embedding_diagnostics_on_epoch_end", True):
            return

        epoch = self.state.epoch
        if epoch is None:
            return
        epoch_key = round(float(epoch), 6)
        if self._embedding_diag_last_epoch == epoch_key:
            return
        self._embedding_diag_last_epoch = epoch_key

        try:
            metrics = self._compute_epoch_embedding_diagnostics()
        except Exception:
            logging.warning(
                "Epoch-end embedding diagnostics failed; continuing training.",
                exc_info=True,
            )
            return
        if metrics:
            self.log(metrics)

    def _diagnostic_dataloader(self) -> DataLoader:
        eval_dataset = self.eval_dataset
        if isinstance(eval_dataset, dict):
            eval_dataset = next(iter(eval_dataset.values()), None)
        if eval_dataset is not None:
            return self.get_eval_dataloader(eval_dataset)
        return self.get_train_dataloader()

    def _compute_epoch_embedding_diagnostics(self) -> dict[str, float]:
        max_samples = int(
            getattr(self.args, "embedding_diagnostics_max_samples", 2048)
        )
        if max_samples <= 0:
            return {}

        model = self.model
        device = next(model.parameters()).device
        dataloader = self._diagnostic_dataloader()
        was_training = model.training
        model.eval()

        pred_chunks: list[torch.Tensor] = []
        tgt_chunks: list[torch.Tensor] = []
        global_samples = 0
        loss_sums: dict[str, float] = {}
        loss_counts: dict[str, int] = {}
        use_autocast = device.type == "cuda" and (
            bool(getattr(self.args, "bf16", False))
            or bool(getattr(self.args, "fp16", False))
        )
        autocast_dtype = (
            torch.bfloat16
            if bool(getattr(self.args, "bf16", False))
            else torch.float16
        )

        try:
            for batch in dataloader:
                inputs = dict(batch)
                vision_input = inputs.pop("vision_input", None)
                if vision_input is not None:
                    vision_input = vision_input.to(device)
                target_vision_input = inputs.pop("target_vision_input", None)
                if target_vision_input is not None:
                    target_vision_input = target_vision_input.to(device)
                target_mask = inputs.pop("target_mask", None)
                if target_mask is not None:
                    target_mask = target_mask.to(device)

                text_input = extract_text_field(
                    inputs, "text_input", required=False
                )
                queries = extract_text_field(
                    inputs, "queries", required=False
                )
                targets = extract_text_field(
                    inputs, "targets", required=False
                )

                with torch.no_grad(), torch.amp.autocast(
                    "cuda", dtype=autocast_dtype, enabled=use_autocast
                ):
                    outputs = model(
                        vision_input=vision_input,
                        target_vision_input=target_vision_input,
                        target_mask=target_mask,
                        text_input=text_input,
                        queries=queries,
                        targets=targets,
                        compute_loss=False,
                    )

                pred_embeds = outputs.get("predicted_embeddings")
                tgt_embeds = outputs.get("target_embeddings")
                if pred_embeds is None or tgt_embeds is None:
                    continue

                unwrapped_model = (
                    model.module if hasattr(model, "module") else model
                )
                pred_for_loss = pred_embeds
                tgt_for_loss = tgt_embeds
                if self.gather_embeddings:
                    pred_for_loss = all_gather_no_grad(pred_for_loss)
                    tgt_for_loss = all_gather_no_grad(tgt_for_loss)
                with torch.no_grad(), torch.amp.autocast(
                    "cuda", enabled=False
                ):
                    loss_out = (
                        unwrapped_model.loss_fn(
                            pred_for_loss.float(), tgt_for_loss.float()
                        )
                        if unwrapped_model.loss_fn is not None else None
                    )
                if isinstance(loss_out, dict):
                    for key in ("pred_loss", "sigreg_loss"):
                        value = loss_out.get(key)
                        if isinstance(value, torch.Tensor):
                            loss_sums[key] = loss_sums.get(key, 0.0) + float(
                                value.detach().float().item()
                            )
                            loss_counts[key] = loss_counts.get(key, 0) + 1

                pred_for_rank = all_gather_no_grad(pred_embeds.detach())
                tgt_for_rank = all_gather_no_grad(tgt_embeds.detach())
                remaining = max_samples - global_samples
                if remaining <= 0:
                    break
                pred_chunks.append(_to_float_cpu(pred_for_rank[:remaining]))
                tgt_chunks.append(_to_float_cpu(tgt_for_rank[:remaining]))
                global_samples += min(remaining, int(pred_for_rank.size(0)))
                if global_samples >= max_samples:
                    break
        finally:
            if was_training:
                model.train()

        if not pred_chunks or not tgt_chunks:
            return {}

        pred_all = torch.cat(pred_chunks, dim=0)
        tgt_all = torch.cat(tgt_chunks, dim=0)
        metrics = {
            "epoch/effective_rank/pred": _compute_effective_rank(pred_all),
            "epoch/effective_rank/tgt": _compute_effective_rank(tgt_all),
            "epoch/pred_embeds/std": _embedding_std(pred_all),
            "epoch/tgt_embeds/std": _embedding_std(tgt_all),
        }

        loss_keys = ("pred_loss", "sigreg_loss")
        packed = torch.tensor(
            [
                item
                for key in loss_keys
                for item in (loss_sums.get(key, 0.0), loss_counts.get(key, 0))
            ],
            dtype=torch.float32,
            device=device,
        )
        _dist_sum_(packed)
        values = packed.detach().cpu().tolist()
        for idx, key in enumerate(loss_keys):
            total = values[2 * idx]
            count = values[2 * idx + 1]
            if count > 0:
                metrics[f"epoch/loss/{key}"] = float(total / count)

        return metrics

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        device = next(model.parameters()).device

        # Handle vision_input (may be None for text-only batches)
        vision_input = inputs.pop("vision_input", None)
        if vision_input is not None:
            vision_input = vision_input.to(device)
        target_vision_input = inputs.pop("target_vision_input", None)
        if target_vision_input is not None:
            target_vision_input = target_vision_input.to(device)
        target_mask = inputs.pop("target_mask", None)
        if target_mask is not None:
            target_mask = target_mask.to(device)
        contrast_group_ids = inputs.pop("contrast_group_ids", None)
        if contrast_group_ids is not None:
            contrast_group_ids = contrast_group_ids.to(device)
        contrast_positive_ids = inputs.pop("contrast_positive_ids", None)
        if contrast_positive_ids is not None:
            contrast_positive_ids = contrast_positive_ids.to(device)
        main_loss_mask = inputs.pop("main_loss_mask", None)
        if main_loss_mask is not None:
            main_loss_mask = main_loss_mask.to(device)
        core_clean_mask = inputs.pop("core_clean_mask", None)
        if core_clean_mask is not None:
            core_clean_mask = core_clean_mask.to(device)

        # inputs contain tensors of byte-encoded strings, read out to str here
        text_input = extract_text_field(inputs, "text_input", required=False)
        queries = extract_text_field(inputs, "queries", required=False)
        targets = extract_text_field(inputs, "targets", required=False)

        outputs = model(
            vision_input=vision_input,
            target_vision_input=target_vision_input,
            target_mask=target_mask,
            text_input=text_input,
            queries=queries,
            targets=targets,
            compute_loss=False,
        )
        pred_embeds = outputs["predicted_embeddings"]
        tgt_embeds = outputs["target_embeddings"]
        qtarget_embeds = outputs.get("query_target_predicted_embeddings")
        aux_object_pred = outputs.get("aux_object_pred_embeddings")
        aux_object_target = outputs.get("aux_object_target_embeddings")
        aux_object_coverage = outputs.get("aux_object_mask_coverage")

        if self.gather_embeddings:
            pred_embeds = all_gather_with_grad(pred_embeds)
            if tgt_embeds.requires_grad:
                tgt_embeds = all_gather_with_grad(tgt_embeds)
            else:
                tgt_embeds = all_gather_no_grad(tgt_embeds)
            if main_loss_mask is not None:
                main_loss_mask = all_gather_no_grad(main_loss_mask)
            if core_clean_mask is not None:
                core_clean_mask = all_gather_no_grad(core_clean_mask)
            if contrast_group_ids is not None:
                contrast_group_ids = all_gather_no_grad(contrast_group_ids)
            if contrast_positive_ids is not None:
                contrast_positive_ids = all_gather_no_grad(contrast_positive_ids)
            if qtarget_embeds is not None:
                if qtarget_embeds.requires_grad:
                    qtarget_embeds = all_gather_with_grad(qtarget_embeds)
                else:
                    qtarget_embeds = all_gather_no_grad(qtarget_embeds)
            if aux_object_pred is not None:
                aux_object_pred = all_gather_with_grad(aux_object_pred)
            if aux_object_target is not None:
                if aux_object_target.requires_grad:
                    aux_object_target = all_gather_with_grad(aux_object_target)
                else:
                    aux_object_target = all_gather_no_grad(aux_object_target)
            if aux_object_coverage is not None:
                aux_object_coverage = all_gather_no_grad(aux_object_coverage)

        # Unwrap DDP/DeepSpeed wrapper to access loss_fn
        unwrapped_model = model.module if hasattr(model, "module") else model
        main_loss_weight = float(
            getattr(
                self.args,
                "main_loss_weight",
                getattr(unwrapped_model, "main_loss_weight", 1.0),
            )
        )
        main_loss_include_clean_pred = bool(
            getattr(self.args, "main_loss_include_clean_pred", False)
        )
        main_sigreg_only_weight = float(
            getattr(self.args, "main_sigreg_only_weight", 0.0)
        )
        if unwrapped_model.loss_fn is None and (
            main_loss_weight != 0.0
            or main_sigreg_only_weight != 0.0
            or main_loss_include_clean_pred
        ):
            raise ValueError(
                "Primary prediction/SIGReg loss requires a model loss configuration. "
                "For pure InfoNCE, set main_loss_weight=0 and leave primary "
                "SIGReg/clean-pred losses disabled."
            )
        skip_main_loss = (
            main_loss_weight == 0.0
            and main_sigreg_only_weight == 0.0
            and not main_loss_include_clean_pred
        )

        # upcast + disable AMP for numerics
        with torch.amp.autocast("cuda", enabled=False):
            pred_embeds = pred_embeds.float()
            tgt_embeds = tgt_embeds.float()
            pred_for_main_loss = pred_embeds
            tgt_for_main_loss = tgt_embeds
            main_loss_valid_fraction = None
            if bool(getattr(self.args, "main_loss_force_all_rows", False)):
                main_loss_mask = None
            if main_loss_mask is not None:
                main_loss_mask = main_loss_mask.float().reshape(-1)
                main_loss_keep = main_loss_mask > 0
                main_loss_valid_fraction = main_loss_keep.float().mean()
                if bool(main_loss_keep.any().item()):
                    pred_for_main_loss = pred_embeds[main_loss_keep]
                    tgt_for_main_loss = tgt_embeds[main_loss_keep]
            if skip_main_loss:
                loss_out = {"loss": pred_embeds.sum() * 0.0}
            else:
                loss_out = unwrapped_model.loss_fn(pred_for_main_loss, tgt_for_main_loss)
        loss = loss_out["loss"] if isinstance(loss_out, dict) else loss_out
        if main_loss_weight != 1.0:
            loss = loss * main_loss_weight
        main_combined_pred_loss = None
        main_combined_pred_loss_unscaled = None
        main_combined_sigreg_loss = None
        if main_loss_include_clean_pred:
            loss = loss * 0.0
        main_sigreg_only_loss = None
        if main_sigreg_only_weight != 0.0 and not main_loss_include_clean_pred:
            if not isinstance(loss_out, dict) or not isinstance(
                loss_out.get("sigreg_loss_for_grad"), torch.Tensor
            ):
                raise ValueError(
                    "main_sigreg_only_weight requires loss_out['sigreg_loss_for_grad']; "
                    "use the patched SIGReg loss implementation."
                )
            sigreg_lambda = float(
                getattr(
                    self.args,
                    "main_sigreg_only_lambda",
                    getattr(unwrapped_model.loss_fn, "lam", 1.0),
                )
            )
            if sigreg_lambda < 0.0:
                sigreg_lambda = float(getattr(unwrapped_model.loss_fn, "lam", 1.0))
            main_sigreg_only_loss = (
                loss_out["sigreg_loss_for_grad"]
                * sigreg_lambda
                * main_sigreg_only_weight
            )
            loss = loss + main_sigreg_only_loss
        qtarget_loss = None
        qtarget_loss_unscaled = None
        qtarget_cos = None
        qtarget_anchor_loss = None
        qtarget_anchor_loss_unscaled = None
        qtarget_anchor_cos = None
        qtarget_contrastive_loss = None
        qtarget_contrastive_loss_unscaled = None
        qtarget_contrastive_acc = None
        qtarget_contrastive_acc_t2s = None
        same_source_contrastive_loss = None
        same_source_contrastive_loss_unscaled = None
        same_source_contrastive_acc = None
        same_source_contrastive_acc_t2s = None
        same_source_contrastive_groups = None
        same_source_clean_contrastive_loss = None
        same_source_clean_contrastive_loss_unscaled = None
        same_source_clean_contrastive_acc = None
        same_source_clean_contrastive_groups = None
        same_source_clean_pred_loss = None
        same_source_clean_pred_loss_unscaled = None
        same_source_pred_clean_loss = None
        same_source_pred_clean_loss_unscaled = None
        same_source_pred_clean_cos = None
        same_source_pred_clean_valid_fraction = None
        same_source_margin_loss = None
        same_source_margin_loss_unscaled = None
        same_source_margin_acc = None
        same_source_margin_groups = None
        same_source_margin_active_pairs = None
        same_source_contrastive_target_space = str(
            getattr(self.args, "same_source_contrastive_target_space", "target_encoder")
        ).strip().lower()
        same_source_contrastive_pooling = str(
            getattr(self.args, "same_source_contrastive_pooling", "mean")
        ).strip().lower()
        if same_source_contrastive_pooling not in {
            "mean",
            "patch_position",
            "patch_all",
        }:
            raise ValueError(
                "same_source_contrastive_pooling must be 'mean', "
                f"'patch_position', or 'patch_all', got "
                f"{same_source_contrastive_pooling!r}"
            )
        same_source_contrastive_target_query_mode = str(
            getattr(self.args, "same_source_contrastive_target_query_mode", "same")
        ).strip()
        aux_object_loss = None
        aux_object_loss_unscaled = None
        aux_object_cos = None
        aux_object_valid_fraction = None
        aux_object_coverage_mean = None
        qtarget_weight = float(
            getattr(unwrapped_model, "qtarget_consistency_weight", 0.0)
        )
        if qtarget_embeds is not None and qtarget_weight > 0.0:
            with torch.amp.autocast("cuda", enabled=False):
                qtarget_embeds = qtarget_embeds.float()
                qtarget_loss_unscaled = F.mse_loss(pred_embeds, qtarget_embeds)
                qtarget_loss = (
                    qtarget_loss_unscaled
                    * qtarget_weight
                    * float(getattr(unwrapped_model, "qtarget_consistency_scale", 384.0))
                )
                qtarget_cos = F.cosine_similarity(
                    pred_embeds, qtarget_embeds, dim=-1
                ).mean()
            loss = loss + qtarget_loss
        qtarget_anchor_weight = float(
            getattr(unwrapped_model, "qtarget_anchor_weight", 0.0)
        )
        if qtarget_embeds is not None and qtarget_anchor_weight > 0.0:
            with torch.amp.autocast("cuda", enabled=False):
                qtarget_for_anchor = qtarget_embeds.float()
                tgt_for_anchor = tgt_embeds.detach().float()
                qtarget_anchor_loss_unscaled = F.mse_loss(
                    qtarget_for_anchor,
                    tgt_for_anchor,
                )
                qtarget_anchor_loss = (
                    qtarget_anchor_loss_unscaled
                    * qtarget_anchor_weight
                    * float(getattr(unwrapped_model, "qtarget_anchor_scale", 384.0))
                )
                qtarget_anchor_cos = F.cosine_similarity(
                    qtarget_for_anchor, tgt_for_anchor, dim=-1
                ).mean()
            loss = loss + qtarget_anchor_loss
        qtarget_contrastive_weight = float(
            getattr(unwrapped_model, "qtarget_contrastive_weight", 0.0)
        )
        if qtarget_embeds is not None and qtarget_contrastive_weight > 0.0:
            with torch.amp.autocast("cuda", enabled=False):
                qtarget_for_contrast = qtarget_embeds.float()
                if bool(
                    getattr(
                        unwrapped_model,
                        "qtarget_contrastive_detach_target",
                        True,
                    )
                ):
                    qtarget_for_contrast = qtarget_for_contrast.detach()
                pred_for_contrast = pred_embeds.float()
                if pred_for_contrast.dim() == 3:
                    pred_for_contrast = pred_for_contrast.mean(dim=1)
                if qtarget_for_contrast.dim() == 3:
                    qtarget_for_contrast = qtarget_for_contrast.mean(dim=1)
                pred_for_contrast = F.normalize(pred_for_contrast, dim=-1)
                qtarget_for_contrast = F.normalize(qtarget_for_contrast, dim=-1)
                temperature = float(
                    getattr(unwrapped_model, "qtarget_contrastive_temperature", 0.07)
                )
                temperature = max(temperature, 1.0e-6)
                logits_s2t = pred_for_contrast @ qtarget_for_contrast.T / temperature
                logits_t2s = logits_s2t.T
                labels = torch.arange(
                    logits_s2t.size(0), device=logits_s2t.device, dtype=torch.long
                )
                loss_s2t = F.cross_entropy(logits_s2t, labels)
                loss_t2s = F.cross_entropy(logits_t2s, labels)
                qtarget_contrastive_loss_unscaled = 0.5 * (loss_s2t + loss_t2s)
                qtarget_contrastive_loss = (
                    qtarget_contrastive_loss_unscaled * qtarget_contrastive_weight
                )
                qtarget_contrastive_acc = (
                    logits_s2t.argmax(dim=-1).eq(labels).float().mean()
                )
                qtarget_contrastive_acc_t2s = (
                    logits_t2s.argmax(dim=-1).eq(labels).float().mean()
                )
            loss = loss + qtarget_contrastive_loss
        same_source_contrastive_weight = float(
            getattr(unwrapped_model, "same_source_contrastive_weight", 0.0)
        )
        same_source_clean_contrastive_weight = float(
            getattr(unwrapped_model, "same_source_clean_contrastive_weight", 0.0)
        )
        same_source_clean_pred_weight = float(
            getattr(unwrapped_model, "same_source_clean_pred_weight", 0.0)
        )
        same_source_pred_clean_weight = float(
            getattr(unwrapped_model, "same_source_pred_clean_weight", 0.0)
        )
        same_source_margin_weight = float(
            getattr(
                self.args,
                "same_source_margin_weight",
                getattr(unwrapped_model, "same_source_margin_weight", 0.0),
            )
        )
        same_source_margin_value = float(
            getattr(
                self.args,
                "same_source_margin_value",
                getattr(unwrapped_model, "same_source_margin_value", 0.02),
            )
        )
        same_source_margin_mode = str(
            getattr(
                self.args,
                "same_source_margin_mode",
                getattr(unwrapped_model, "same_source_margin_mode", "mean"),
            )
        ).strip().lower()
        if same_source_contrastive_pooling in {"patch_position", "patch_all"} and (
            same_source_margin_weight > 0.0
            or same_source_clean_contrastive_weight > 0.0
        ):
            raise ValueError(
                "patch_position pooling currently supports only the primary "
                "same-source InfoNCE term"
            )
        same_source_target_embeds = tgt_embeds
        if (
            contrast_group_ids is not None
            and (same_source_contrastive_weight > 0.0 or same_source_margin_weight > 0.0)
            and same_source_contrastive_target_space == "predictor"
            and target_vision_input is not None
        ):
            detach_same_source_target = bool(
                getattr(
                    unwrapped_model,
                    "same_source_contrastive_detach_target",
                    True,
                )
            )
            target_predict_context = (
                torch.no_grad() if detach_same_source_target else nullcontext()
            )
            target_queries = queries
            target_query_mode_lower = same_source_contrastive_target_query_mode.lower()
            if target_query_mode_lower in {"none", "null", "no_query", "no-query"}:
                target_queries = [None] * int(target_vision_input.size(0))
            elif target_query_mode_lower not in {"same", "source"}:
                target_queries = [
                    same_source_contrastive_target_query_mode
                ] * int(target_vision_input.size(0))
            with target_predict_context:
                target_pred_outputs = model(
                    vision_input=target_vision_input,
                    text_input=[None] * int(target_vision_input.size(0)),
                    queries=target_queries,
                    compute_loss=False,
                )
                same_source_target_embeds = target_pred_outputs["predicted_embeddings"]
            if self.gather_embeddings:
                if same_source_target_embeds.requires_grad:
                    same_source_target_embeds = all_gather_with_grad(
                        same_source_target_embeds
                    )
                else:
                    same_source_target_embeds = all_gather_no_grad(
                        same_source_target_embeds
                    )
        same_source_clean_pred_embeds = None
        needs_clean_pred_embeds = (
            main_loss_include_clean_pred
            or (
                contrast_group_ids is not None
                and (
                    same_source_clean_contrastive_weight > 0.0
                    or same_source_clean_pred_weight > 0.0
                    or same_source_pred_clean_weight > 0.0
                )
            )
        )
        if needs_clean_pred_embeds and target_vision_input is not None:
            clean_pred_outputs = model(
                vision_input=target_vision_input,
                text_input=[None] * int(target_vision_input.size(0)),
                queries=queries,
                compute_loss=False,
            )
            same_source_clean_pred_embeds = clean_pred_outputs["predicted_embeddings"]
            if self.gather_embeddings:
                same_source_clean_pred_embeds = all_gather_with_grad(
                    same_source_clean_pred_embeds
                )
        if (
            contrast_group_ids is not None
            and (
                same_source_contrastive_weight > 0.0
                or same_source_margin_weight > 0.0
                or same_source_clean_contrastive_weight > 0.0
            )
            and pred_embeds.size(0) == tgt_embeds.size(0)
        ):
            with torch.amp.autocast("cuda", enabled=False):
                pred_for_group = pred_embeds.float()
                tgt_for_group = same_source_target_embeds.float()
                if bool(
                    getattr(
                        unwrapped_model,
                        "same_source_contrastive_detach_target",
                        True,
                    )
                ):
                    tgt_for_group = tgt_for_group.detach()
                if same_source_contrastive_pooling == "mean":
                    if pred_for_group.dim() == 3:
                        pred_for_group = pred_for_group.mean(dim=1)
                    if tgt_for_group.dim() == 3:
                        tgt_for_group = tgt_for_group.mean(dim=1)
                elif pred_for_group.dim() != 3 or tgt_for_group.dim() != 3:
                    raise ValueError(
                        "patch_position InfoNCE requires source and target "
                        "embeddings with shape (B,N,D)"
                    )
                pred_for_group = F.normalize(pred_for_group, dim=-1)
                tgt_for_group = F.normalize(tgt_for_group, dim=-1)
                clean_pred_for_group = None
                if same_source_clean_pred_embeds is not None:
                    clean_pred_for_group = same_source_clean_pred_embeds.float()
                    if clean_pred_for_group.dim() == 3:
                        clean_pred_for_group = clean_pred_for_group.mean(dim=1)
                    clean_pred_for_group = F.normalize(clean_pred_for_group, dim=-1)
                temperature = float(
                    getattr(unwrapped_model, "same_source_contrastive_temperature", 0.07)
                )
                temperature = max(temperature, 1.0e-6)
                clean_temperature = float(
                    getattr(
                        unwrapped_model,
                        "same_source_clean_contrastive_temperature",
                        temperature,
                    )
                )
                clean_temperature = max(clean_temperature, 1.0e-6)
                group_losses = []
                group_acc_s2t = []
                group_acc_t2s = []
                group_clean_losses = []
                group_clean_acc = []
                group_margin_losses = []
                group_margin_acc = []
                group_margin_active = []
                valid_group_count = 0

                def multi_positive_cross_entropy(
                    logits: torch.Tensor,
                    positive_ids: torch.Tensor,
                ) -> torch.Tensor:
                    pos_mask = positive_ids[:, None].eq(positive_ids[None, :])
                    pos_mask = pos_mask & positive_ids[:, None].ge(0)
                    valid = pos_mask.any(dim=1)
                    if not bool(valid.any().item()):
                        labels = torch.arange(
                            logits.size(0),
                            device=logits.device,
                            dtype=torch.long,
                        )
                        return F.cross_entropy(logits, labels)
                    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
                    pos_mask_f = pos_mask.float()
                    per_row = -(
                        log_probs * pos_mask_f
                    ).sum(dim=1) / pos_mask_f.sum(dim=1).clamp_min(1.0)
                    return per_row[valid].mean()

                def multi_positive_top1(
                    logits: torch.Tensor,
                    positive_ids: torch.Tensor,
                ) -> torch.Tensor:
                    pos_mask = positive_ids[:, None].eq(positive_ids[None, :])
                    pos_mask = pos_mask & positive_ids[:, None].ge(0)
                    valid = pos_mask.any(dim=1)
                    if not bool(valid.any().item()):
                        labels = torch.arange(
                            logits.size(0),
                            device=logits.device,
                            dtype=torch.long,
                        )
                        return logits.argmax(dim=-1).eq(labels).float().mean()
                    pred_idx = logits.argmax(dim=-1)
                    return pos_mask[
                        torch.arange(logits.size(0), device=logits.device),
                        pred_idx,
                    ][valid].float().mean()

                for group_id in torch.unique(contrast_group_ids):
                    if int(group_id.item()) < 0:
                        continue
                    mask = contrast_group_ids.eq(group_id)
                    if int(mask.sum().item()) < 2:
                        continue
                    pred_g = pred_for_group[mask]
                    tgt_g = tgt_for_group[mask]
                    positive_ids_g = None
                    if contrast_positive_ids is not None:
                        positive_ids_g = contrast_positive_ids[mask]
                        if int(positive_ids_g.ge(0).sum().item()) == 0:
                            positive_ids_g = None
                    if same_source_contrastive_pooling == "patch_position":
                        if positive_ids_g is None:
                            positive_ids_g = torch.arange(
                                pred_g.size(0),
                                device=pred_g.device,
                                dtype=torch.long,
                            )
                        (
                            loss_s2t,
                            loss_t2s,
                            acc_s2t,
                            acc_t2s,
                        ) = patch_position_multi_positive_loss(
                            pred_g,
                            tgt_g,
                            positive_ids_g,
                            temperature=temperature,
                            patch_chunk_size=int(
                                getattr(
                                    self.args,
                                    "same_source_contrastive_patch_chunk_size",
                                    16,
                                )
                            ),
                        )
                        if same_source_contrastive_weight > 0.0:
                            group_losses.append(0.5 * (loss_s2t + loss_t2s))
                            group_acc_s2t.append(acc_s2t)
                            group_acc_t2s.append(acc_t2s)
                        valid_group_count += 1
                        continue
                    if same_source_contrastive_pooling == "patch_all":
                        if positive_ids_g is not None and int(
                            torch.unique(positive_ids_g).numel()
                        ) != int(positive_ids_g.numel()):
                            raise ValueError(
                                "patch_all InfoNCE requires unique positive IDs "
                                "within each contrastive group"
                            )
                        logits_s2t, logits_t2s, labels = patch_position_logits(
                            pred_g,
                            tgt_g,
                            temperature=temperature,
                        )
                        spatial_logits_s2t = None
                        spatial_logits_t2s = None
                        spatial_labels = None
                        (
                            spatial_logits_s2t,
                            spatial_logits_t2s,
                            spatial_labels,
                        ) = patch_spatial_logits(
                            pred_g,
                            tgt_g,
                            temperature=temperature,
                        )
                        sim_s2t = None
                    else:
                        sim_s2t = pred_g @ tgt_g.T
                        logits_s2t = sim_s2t / temperature
                        logits_t2s = logits_s2t.T
                        labels = torch.arange(
                            logits_s2t.size(0),
                            device=logits_s2t.device,
                            dtype=torch.long,
                        )
                    if same_source_contrastive_weight > 0.0:
                        if (
                            positive_ids_g is None
                            or same_source_contrastive_pooling
                            in {"patch_position", "patch_all"}
                        ):
                            loss_s2t = F.cross_entropy(logits_s2t, labels)
                            loss_t2s = F.cross_entropy(logits_t2s, labels)
                            acc_t2s = logits_t2s.argmax(dim=-1).eq(labels).float().mean()
                            acc_s2t = logits_s2t.argmax(dim=-1).eq(labels).float().mean()
                            if same_source_contrastive_pooling == "patch_all":
                                if (
                                    spatial_logits_s2t is None
                                    or spatial_logits_t2s is None
                                    or spatial_labels is None
                                ):
                                    raise RuntimeError(
                                        "patch_all InfoNCE spatial logits were not built"
                                    )
                                loss_s2t = 0.5 * (
                                    loss_s2t
                                    + F.cross_entropy(
                                        spatial_logits_s2t, spatial_labels
                                    )
                                )
                                loss_t2s = 0.5 * (
                                    loss_t2s
                                    + F.cross_entropy(
                                        spatial_logits_t2s, spatial_labels
                                    )
                                )
                                acc_s2t = 0.5 * (
                                    acc_s2t
                                    + spatial_logits_s2t.argmax(dim=-1)
                                    .eq(spatial_labels)
                                    .float()
                                    .mean()
                                )
                                acc_t2s = 0.5 * (
                                    acc_t2s
                                    + spatial_logits_t2s.argmax(dim=-1)
                                    .eq(spatial_labels)
                                    .float()
                                    .mean()
                                )
                        else:
                            loss_s2t = multi_positive_cross_entropy(
                                logits_s2t,
                                positive_ids_g,
                            )
                            loss_t2s = multi_positive_cross_entropy(
                                logits_t2s,
                                positive_ids_g,
                            )
                            acc_t2s = multi_positive_top1(
                                logits_t2s,
                                positive_ids_g,
                            )
                        group_losses.append(
                            0.5 * (loss_s2t + loss_t2s)
                        )
                        group_acc_t2s.append(acc_t2s)
                    if same_source_margin_weight > 0.0:
                        pos = sim_s2t.diag()
                        offdiag = ~torch.eye(
                            sim_s2t.size(0),
                            dtype=torch.bool,
                            device=sim_s2t.device,
                        )
                        if same_source_margin_mode in {"hard", "hardmax", "max"}:
                            neg_mask = offdiag
                            if positive_ids_g is not None:
                                positive_pair_mask = positive_ids_g[:, None].eq(
                                    positive_ids_g[None, :]
                                ) & positive_ids_g[:, None].ge(0)
                                neg_mask = ~positive_pair_mask
                                neg_mask = neg_mask & offdiag
                            valid_rows = neg_mask.any(dim=1)
                            if bool(valid_rows.any().item()):
                                hard_wrong = sim_s2t.masked_fill(
                                    ~neg_mask,
                                    torch.finfo(sim_s2t.dtype).min,
                                ).max(dim=1).values
                                hinge = F.relu(
                                    same_source_margin_value
                                    + hard_wrong[valid_rows]
                                    - pos[valid_rows]
                                )
                            else:
                                hinge = sim_s2t.new_zeros((0,))
                        elif same_source_margin_mode in {"mean", "all", "pairwise"}:
                            neg_mask = offdiag
                            if positive_ids_g is not None:
                                positive_pair_mask = positive_ids_g[:, None].eq(
                                    positive_ids_g[None, :]
                                ) & positive_ids_g[:, None].ge(0)
                                neg_mask = (~positive_pair_mask) & offdiag
                            violations = (
                                same_source_margin_value
                                + sim_s2t
                                - pos[:, None]
                            )[neg_mask]
                            hinge = F.relu(violations)
                        else:
                            raise ValueError(
                                "Unsupported same_source_margin_mode: "
                                f"{same_source_margin_mode!r}"
                            )
                        if hinge.numel() > 0:
                            group_margin_losses.append(hinge.mean())
                            group_margin_active.append((hinge > 0).float().mean())
                        else:
                            group_margin_losses.append(sim_s2t.new_tensor(0.0))
                            group_margin_active.append(sim_s2t.new_tensor(0.0))
                        group_margin_acc.append(
                            sim_s2t.argmax(dim=-1).eq(labels).float().mean()
                        )
                    if (
                        same_source_clean_contrastive_weight > 0.0
                        and clean_pred_for_group is not None
                    ):
                        clean_pred_g = clean_pred_for_group[mask]
                        clean_logits = clean_pred_g @ tgt_g.T / clean_temperature
                        if positive_ids_g is None:
                            clean_loss = F.cross_entropy(clean_logits, labels)
                            clean_acc = (
                                clean_logits.argmax(dim=-1).eq(labels).float().mean()
                            )
                        else:
                            clean_loss = multi_positive_cross_entropy(
                                clean_logits,
                                positive_ids_g,
                            )
                            clean_acc = multi_positive_top1(
                                clean_logits,
                                positive_ids_g,
                            )
                        group_clean_losses.append(
                            clean_loss
                        )
                        group_clean_acc.append(clean_acc)
                    if (
                        positive_ids_g is None
                        or same_source_contrastive_pooling
                        in {"patch_position", "patch_all"}
                    ):
                        group_acc_s2t.append(
                            acc_s2t
                        )
                    else:
                        group_acc_s2t.append(
                            multi_positive_top1(logits_s2t, positive_ids_g)
                        )
                    valid_group_count += 1
                if group_losses:
                    same_source_contrastive_loss_unscaled = torch.stack(group_losses).mean()
                    same_source_contrastive_loss = (
                        same_source_contrastive_loss_unscaled
                        * same_source_contrastive_weight
                    )
                    same_source_contrastive_acc = torch.stack(group_acc_s2t).mean()
                    same_source_contrastive_acc_t2s = torch.stack(group_acc_t2s).mean()
                    same_source_contrastive_groups = valid_group_count
                    loss = loss + same_source_contrastive_loss
                if group_margin_losses:
                    same_source_margin_loss_unscaled = torch.stack(
                        group_margin_losses
                    ).mean()
                    same_source_margin_loss = (
                        same_source_margin_loss_unscaled * same_source_margin_weight
                    )
                    same_source_margin_acc = torch.stack(group_margin_acc).mean()
                    same_source_margin_active_pairs = torch.stack(
                        group_margin_active
                    ).mean()
                    same_source_margin_groups = valid_group_count
                    loss = loss + same_source_margin_loss
                if group_clean_losses:
                    same_source_clean_contrastive_loss_unscaled = torch.stack(
                        group_clean_losses
                    ).mean()
                    same_source_clean_contrastive_loss = (
                        same_source_clean_contrastive_loss_unscaled
                        * same_source_clean_contrastive_weight
                    )
                    same_source_clean_contrastive_acc = torch.stack(
                        group_clean_acc
                    ).mean()
                    same_source_clean_contrastive_groups = valid_group_count
                    loss = loss + same_source_clean_contrastive_loss
        if (
            same_source_pred_clean_weight > 0.0
            and same_source_clean_pred_embeds is not None
        ):
            with torch.amp.autocast("cuda", enabled=False):
                pred_for_pred_clean = pred_embeds.float()
                clean_for_pred_clean = same_source_clean_pred_embeds.float()
                if bool(
                    getattr(
                        unwrapped_model,
                        "same_source_pred_clean_detach_target",
                        True,
                    )
                ):
                    clean_for_pred_clean = clean_for_pred_clean.detach()
                pred_clean_keep = None
                if (
                    core_clean_mask is not None
                    and int(core_clean_mask.numel()) == int(pred_for_pred_clean.size(0))
                ):
                    pred_clean_keep = core_clean_mask.float().reshape(-1) > 0
                    same_source_pred_clean_valid_fraction = (
                        pred_clean_keep.float().mean()
                    )
                if pred_clean_keep is not None:
                    if bool(pred_clean_keep.any().item()):
                        pred_for_pred_clean = pred_for_pred_clean[pred_clean_keep]
                        clean_for_pred_clean = clean_for_pred_clean[pred_clean_keep]
                    else:
                        pred_for_pred_clean = pred_for_pred_clean[:0]
                        clean_for_pred_clean = clean_for_pred_clean[:0]
                if pred_for_pred_clean.numel() == 0:
                    same_source_pred_clean_loss_unscaled = pred_embeds.new_tensor(0.0)
                    same_source_pred_clean_loss = pred_embeds.new_tensor(0.0)
                    same_source_pred_clean_cos = pred_embeds.new_tensor(0.0)
                else:
                    pred_clean_flat = pred_for_pred_clean.reshape(
                        -1, pred_for_pred_clean.size(-1)
                    )
                    clean_pred_flat = clean_for_pred_clean.reshape(
                        -1, clean_for_pred_clean.size(-1)
                    )
                    same_source_pred_clean_cos = F.cosine_similarity(
                        pred_clean_flat,
                        clean_pred_flat,
                        dim=-1,
                    ).mean()
                    pred_clean_loss_type = str(
                        getattr(
                            unwrapped_model,
                            "same_source_pred_clean_loss",
                            "mse",
                        )
                    ).strip().lower()
                    if pred_clean_loss_type == "cosine":
                        same_source_pred_clean_loss_unscaled = (
                            1.0 - same_source_pred_clean_cos
                        )
                    elif pred_clean_loss_type in {"cosine_mse", "mse_cosine"}:
                        same_source_pred_clean_loss_unscaled = (
                            F.mse_loss(pred_for_pred_clean, clean_for_pred_clean)
                            + 1.0
                            - same_source_pred_clean_cos
                        )
                    elif pred_clean_loss_type == "mse":
                        same_source_pred_clean_loss_unscaled = F.mse_loss(
                            pred_for_pred_clean,
                            clean_for_pred_clean,
                        )
                    else:
                        raise ValueError(
                            "Unsupported same_source_pred_clean_loss: "
                            f"{pred_clean_loss_type!r}"
                        )
                    same_source_pred_clean_loss = (
                        same_source_pred_clean_loss_unscaled
                        * same_source_pred_clean_weight
                        * float(
                            getattr(
                                unwrapped_model,
                                "same_source_pred_clean_scale",
                                1.0,
                            )
                        )
                    )
            loss = loss + same_source_pred_clean_loss
        if main_loss_include_clean_pred:
            if same_source_clean_pred_embeds is None:
                raise ValueError(
                    "main_loss_include_clean_pred requires clean prediction embeddings; "
                    "ensure target_vision_input is available for the batch."
                )
            with torch.amp.autocast("cuda", enabled=False):
                clean_pred_for_main_loss = same_source_clean_pred_embeds.float()
                tgt_for_clean_main_loss = tgt_embeds.detach().float()
                if main_loss_mask is not None:
                    clean_main_keep = main_loss_mask.float().reshape(-1) > 0
                    if bool(clean_main_keep.any().item()):
                        clean_pred_for_main_loss = clean_pred_for_main_loss[clean_main_keep]
                        tgt_for_clean_main_loss = tgt_for_clean_main_loss[clean_main_keep]
                    else:
                        clean_pred_for_main_loss = clean_pred_for_main_loss[:0]
                        tgt_for_clean_main_loss = tgt_for_clean_main_loss[:0]
                combined_pred_for_loss = torch.cat(
                    [pred_for_main_loss, clean_pred_for_main_loss], dim=0
                )
                combined_tgt_for_loss = torch.cat(
                    [tgt_for_main_loss, tgt_for_clean_main_loss], dim=0
                )
                combined_loss_out = unwrapped_model.loss_fn(
                    combined_pred_for_loss,
                    combined_tgt_for_loss,
                )
                main_combined_pred_loss_unscaled = (
                    combined_loss_out["loss"]
                    if isinstance(combined_loss_out, dict)
                    else combined_loss_out
                )
                main_combined_pred_loss = (
                    main_combined_pred_loss_unscaled * main_loss_weight
                )
                loss = loss + main_combined_pred_loss
                if isinstance(combined_loss_out, dict):
                    main_combined_sigreg_loss = combined_loss_out.get(
                        "sigreg_loss_for_grad"
                    )
                if main_sigreg_only_weight != 0.0:
                    if not isinstance(main_combined_sigreg_loss, torch.Tensor):
                        raise ValueError(
                            "main_sigreg_only_weight with main_loss_include_clean_pred "
                            "requires combined loss_out['sigreg_loss_for_grad']."
                        )
                    sigreg_lambda = float(
                        getattr(
                            self.args,
                            "main_sigreg_only_lambda",
                            getattr(unwrapped_model.loss_fn, "lam", 1.0),
                        )
                    )
                    if sigreg_lambda < 0.0:
                        sigreg_lambda = float(
                            getattr(unwrapped_model.loss_fn, "lam", 1.0)
                        )
                    main_sigreg_only_loss = (
                        main_combined_sigreg_loss
                        * sigreg_lambda
                        * main_sigreg_only_weight
                    )
                    loss = loss + main_sigreg_only_loss
        if (
            same_source_clean_pred_weight > 0.0
            and same_source_clean_pred_embeds is not None
            and not main_loss_include_clean_pred
        ):
            with torch.amp.autocast("cuda", enabled=False):
                clean_pred_for_loss = same_source_clean_pred_embeds.float()
                tgt_for_clean_pred_loss = tgt_embeds.detach().float()
                if (
                    core_clean_mask is not None
                    and int(core_clean_mask.numel()) == int(clean_pred_for_loss.size(0))
                ):
                    clean_pred_keep = core_clean_mask.float().reshape(-1) > 0
                    if bool(clean_pred_keep.any().item()):
                        clean_pred_for_loss = clean_pred_for_loss[clean_pred_keep]
                        tgt_for_clean_pred_loss = tgt_for_clean_pred_loss[clean_pred_keep]
                if clean_pred_for_loss.numel() == 0:
                    same_source_clean_pred_loss_unscaled = pred_embeds.new_tensor(0.0)
                    same_source_clean_pred_loss = pred_embeds.new_tensor(0.0)
                else:
                    same_source_clean_pred_loss_unscaled = F.mse_loss(
                        clean_pred_for_loss,
                        tgt_for_clean_pred_loss,
                    )
                    same_source_clean_pred_loss = (
                        same_source_clean_pred_loss_unscaled
                        * same_source_clean_pred_weight
                    )
            loss = loss + same_source_clean_pred_loss
        aux_weight = float(
            getattr(unwrapped_model, "aux_object_grounding_weight", 0.0)
        )
        if (
            aux_object_pred is not None
            and aux_object_target is not None
            and aux_weight > 0.0
        ):
            with torch.amp.autocast("cuda", enabled=False):
                aux_object_pred = aux_object_pred.float()
                aux_object_target = aux_object_target.float()
                if aux_object_coverage is not None:
                    aux_object_coverage = aux_object_coverage.float().reshape(-1)
                    aux_object_coverage_mean = aux_object_coverage.mean()
                    aux_valid = aux_object_coverage > 0
                    aux_object_valid_fraction = aux_valid.float().mean()
                    if aux_valid.any():
                        aux_object_pred = aux_object_pred[aux_valid]
                        aux_object_target = aux_object_target[aux_valid]
                    else:
                        aux_object_pred = aux_object_pred[:0]
                        aux_object_target = aux_object_target[:0]
                if aux_object_pred.numel() == 0:
                    aux_object_loss_unscaled = pred_embeds.new_tensor(0.0)
                    aux_object_loss = pred_embeds.new_tensor(0.0)
                    aux_object_cos = pred_embeds.new_tensor(0.0)
                else:
                    aux_pred_for_loss = aux_object_pred
                    aux_target_for_loss = aux_object_target
                    if bool(
                        getattr(
                            unwrapped_model,
                            "aux_object_grounding_normalize_pred",
                            True,
                        )
                    ):
                        aux_pred_for_loss = F.normalize(aux_pred_for_loss, dim=-1)
                    if bool(
                        getattr(
                            unwrapped_model,
                            "aux_object_grounding_normalize_target",
                            True,
                        )
                    ):
                        aux_target_for_loss = F.normalize(
                            aux_target_for_loss, dim=-1
                        )
                    aux_object_loss_unscaled = F.mse_loss(
                        aux_pred_for_loss, aux_target_for_loss
                    )
                    aux_object_loss = (
                        aux_object_loss_unscaled
                        * aux_weight
                        * float(
                            getattr(
                                unwrapped_model,
                                "aux_object_grounding_scale",
                                1.0,
                            )
                        )
                    )
                    aux_object_cos = F.cosine_similarity(
                        aux_pred_for_loss, aux_target_for_loss, dim=-1
                    ).mean()
            loss = loss + aux_object_loss

        # per-microstep stats
        extra: dict[str, float] = {}
        if isinstance(loss_out, dict):
            for k, v in loss_out.items():
                if k != "loss" and isinstance(v, torch.Tensor):
                    extra[f"loss/{k}"] = v.item()
        if main_loss_mask is not None and main_loss_valid_fraction is not None:
            extra["loss/main_loss_valid_fraction"] = float(
                main_loss_valid_fraction.item()
            )
        if main_loss_weight != 1.0:
            extra["loss/main_loss_weight"] = main_loss_weight
        if main_loss_include_clean_pred:
            extra["loss/main_loss_include_clean_pred"] = 1.0
            if main_combined_pred_loss is not None:
                extra["loss/main_combined_pred_loss"] = float(
                    main_combined_pred_loss.detach().item()
                )
            if main_combined_pred_loss_unscaled is not None:
                extra["loss/main_combined_pred_loss_unscaled"] = float(
                    main_combined_pred_loss_unscaled.detach().item()
                )
            if isinstance(main_combined_sigreg_loss, torch.Tensor):
                extra["loss/main_combined_sigreg_loss"] = float(
                    main_combined_sigreg_loss.detach().item()
                )
        if main_sigreg_only_loss is not None:
            extra["loss/main_sigreg_only_loss"] = float(
                main_sigreg_only_loss.detach().item()
            )
            extra["loss/main_sigreg_only_weight"] = main_sigreg_only_weight
            extra["loss/main_sigreg_only_lambda"] = sigreg_lambda
        if bool(getattr(self.args, "main_loss_force_all_rows", False)):
            extra["loss/main_loss_force_all_rows"] = 1.0
        if qtarget_loss is not None and qtarget_loss_unscaled is not None:
            extra["loss/qtarget_consistency_loss"] = float(qtarget_loss.item())
            extra["loss/qtarget_consistency_loss_unscaled"] = float(
                qtarget_loss_unscaled.item()
            )
            if qtarget_cos is not None:
                extra["loss/qtarget_consistency_cos"] = float(qtarget_cos.item())
        if (
            qtarget_anchor_loss is not None
            and qtarget_anchor_loss_unscaled is not None
        ):
            extra["loss/qtarget_anchor_loss"] = float(qtarget_anchor_loss.item())
            extra["loss/qtarget_anchor_loss_unscaled"] = float(
                qtarget_anchor_loss_unscaled.item()
            )
            if qtarget_anchor_cos is not None:
                extra["loss/qtarget_anchor_cos"] = float(qtarget_anchor_cos.item())
        if (
            qtarget_contrastive_loss is not None
            and qtarget_contrastive_loss_unscaled is not None
        ):
            extra["loss/qtarget_contrastive_loss"] = float(
                qtarget_contrastive_loss.item()
            )
            extra["loss/qtarget_contrastive_loss_unscaled"] = float(
                qtarget_contrastive_loss_unscaled.item()
            )
            extra["loss/qtarget_contrastive_weight"] = float(
                qtarget_contrastive_weight
            )
            extra["loss/qtarget_contrastive_temperature"] = float(
                getattr(unwrapped_model, "qtarget_contrastive_temperature", 0.07)
            )
            if qtarget_contrastive_acc is not None:
                extra["loss/qtarget_contrastive_acc_s2t"] = float(
                    qtarget_contrastive_acc.item()
                )
            if qtarget_contrastive_acc_t2s is not None:
                extra["loss/qtarget_contrastive_acc_t2s"] = float(
                    qtarget_contrastive_acc_t2s.item()
                )
        if (
            same_source_contrastive_loss is not None
            and same_source_contrastive_loss_unscaled is not None
        ):
            extra["loss/same_source_contrastive_loss"] = float(
                same_source_contrastive_loss.item()
            )
            extra["loss/same_source_contrastive_loss_unscaled"] = float(
                same_source_contrastive_loss_unscaled.item()
            )
            extra["loss/same_source_contrastive_weight"] = float(
                same_source_contrastive_weight
            )
            extra["loss/same_source_contrastive_temperature"] = float(
                getattr(unwrapped_model, "same_source_contrastive_temperature", 0.07)
            )
            extra["loss/same_source_contrastive_target_space_predictor"] = float(
                same_source_contrastive_target_space == "predictor"
            )
            extra["loss/same_source_contrastive_patch_position"] = float(
                same_source_contrastive_pooling
                in {"patch_position", "patch_all"}
            )
            extra["loss/same_source_contrastive_patch_all"] = float(
                same_source_contrastive_pooling == "patch_all"
            )
            extra["loss/same_source_contrastive_target_query_same"] = float(
                same_source_contrastive_target_query_mode.lower() in {"same", "source"}
            )
            if same_source_contrastive_acc is not None:
                extra["loss/same_source_contrastive_acc_s2t"] = float(
                    same_source_contrastive_acc.item()
                )
            if same_source_contrastive_acc_t2s is not None:
                extra["loss/same_source_contrastive_acc_t2s"] = float(
                    same_source_contrastive_acc_t2s.item()
                )
            if same_source_contrastive_groups is not None:
                extra["loss/same_source_contrastive_groups"] = float(
                    same_source_contrastive_groups
                )
        if (
            same_source_clean_contrastive_loss is not None
            and same_source_clean_contrastive_loss_unscaled is not None
        ):
            extra["loss/same_source_clean_contrastive_loss"] = float(
                same_source_clean_contrastive_loss.item()
            )
            extra["loss/same_source_clean_contrastive_loss_unscaled"] = float(
                same_source_clean_contrastive_loss_unscaled.item()
            )
            extra["loss/same_source_clean_contrastive_weight"] = float(
                same_source_clean_contrastive_weight
            )
            extra["loss/same_source_clean_contrastive_temperature"] = float(
                getattr(
                    unwrapped_model,
                    "same_source_clean_contrastive_temperature",
                    getattr(unwrapped_model, "same_source_contrastive_temperature", 0.07),
                )
            )
            if same_source_clean_contrastive_acc is not None:
                extra["loss/same_source_clean_contrastive_acc_s2t"] = float(
                    same_source_clean_contrastive_acc.item()
                )
            if same_source_clean_contrastive_groups is not None:
                extra["loss/same_source_clean_contrastive_groups"] = float(
                    same_source_clean_contrastive_groups
                )
        if (
            same_source_clean_pred_loss is not None
            and same_source_clean_pred_loss_unscaled is not None
        ):
            extra["loss/same_source_clean_pred_loss"] = float(
                same_source_clean_pred_loss.item()
            )
            extra["loss/same_source_clean_pred_loss_unscaled"] = float(
                same_source_clean_pred_loss_unscaled.item()
            )
            extra["loss/same_source_clean_pred_weight"] = float(
                same_source_clean_pred_weight
            )
        if (
            same_source_pred_clean_loss is not None
            and same_source_pred_clean_loss_unscaled is not None
        ):
            extra["loss/same_source_pred_clean_loss"] = float(
                same_source_pred_clean_loss.item()
            )
            extra["loss/same_source_pred_clean_loss_unscaled"] = float(
                same_source_pred_clean_loss_unscaled.item()
            )
            extra["loss/same_source_pred_clean_weight"] = float(
                same_source_pred_clean_weight
            )
            extra["loss/same_source_pred_clean_scale"] = float(
                getattr(unwrapped_model, "same_source_pred_clean_scale", 1.0)
            )
            if same_source_pred_clean_cos is not None:
                extra["loss/same_source_pred_clean_cos"] = float(
                    same_source_pred_clean_cos.item()
                )
            if same_source_pred_clean_valid_fraction is not None:
                extra["loss/same_source_pred_clean_valid_fraction"] = float(
                    same_source_pred_clean_valid_fraction.item()
                )
        if (
            same_source_margin_loss is not None
            and same_source_margin_loss_unscaled is not None
        ):
            extra["loss/same_source_margin_loss"] = float(
                same_source_margin_loss.item()
            )
            extra["loss/same_source_margin_loss_unscaled"] = float(
                same_source_margin_loss_unscaled.item()
            )
            extra["loss/same_source_margin_weight"] = float(
                same_source_margin_weight
            )
            extra["loss/same_source_margin_value"] = float(
                same_source_margin_value
            )
            extra["loss/same_source_margin_target_space_predictor"] = float(
                same_source_contrastive_target_space == "predictor"
            )
            extra["loss/same_source_margin_target_query_same"] = float(
                same_source_contrastive_target_query_mode.lower()
                in {"same", "source"}
            )
            extra["loss/same_source_margin_mode_hard"] = float(
                same_source_margin_mode in {"hard", "hardmax", "max"}
            )
            if same_source_margin_acc is not None:
                extra["loss/same_source_margin_acc_s2t"] = float(
                    same_source_margin_acc.item()
                )
            if same_source_margin_active_pairs is not None:
                extra["loss/same_source_margin_active_pair_fraction"] = float(
                    same_source_margin_active_pairs.item()
                )
            if same_source_margin_groups is not None:
                extra["loss/same_source_margin_groups"] = float(
                    same_source_margin_groups
                )
        if (
            aux_object_loss is not None
            and aux_object_loss_unscaled is not None
        ):
            extra["loss/aux_object_grounding_loss"] = float(
                aux_object_loss.item()
            )
            extra["loss/aux_object_grounding_loss_unscaled"] = float(
                aux_object_loss_unscaled.item()
            )
            if aux_object_cos is not None:
                extra["loss/aux_object_grounding_cos"] = float(
                    aux_object_cos.item()
                )
            if aux_object_valid_fraction is not None:
                extra["loss/aux_object_valid_fraction"] = float(
                    aux_object_valid_fraction.item()
                )
            if aux_object_coverage_mean is not None:
                extra["loss/aux_object_mask_coverage"] = float(
                    aux_object_coverage_mean.item()
                )
            extra["loss/aux_object_grounding_weight"] = aux_weight
            extra["loss/aux_object_grounding_scale"] = float(
                getattr(unwrapped_model, "aux_object_grounding_scale", 1.0)
            )
            if isinstance(loss_out, dict) and isinstance(
                loss_out.get("pred_loss"), torch.Tensor
            ):
                pred_loss_value = float(loss_out["pred_loss"].detach().item())
                if pred_loss_value > 0:
                    extra["loss/aux_object_to_pred_loss_ratio"] = float(
                        aux_object_loss.detach().item() / pred_loss_value
                    )
        extra.update(_embedding_stats(pred_embeds, "pred_embeds"))
        extra.update(_embedding_stats(tgt_embeds, "tgt_embeds"))
        if qtarget_embeds is not None:
            extra.update(_embedding_stats(qtarget_embeds, "qtarget_embeds"))
        if aux_object_pred is not None:
            extra.update(_embedding_stats(aux_object_pred, "aux_object_pred"))
        if aux_object_target is not None:
            extra.update(_embedding_stats(aux_object_target, "aux_object_tgt"))
        self._accumulate_pending(extra)

        return (loss, outputs) if return_outputs else loss

    def log(
        self, logs: dict[str, float], start_time: float | None = None
    ) -> None:
        """Called by HF at its logging cadence.

        We merge averaged micro-step stats and throughput/MFU metrics here,
        then delegate to HF.
        """
        merged = dict(logs)

        # Merge pending micro-step stats
        pending_means = self._drain_pending_means()
        if pending_means:
            merged.update(pending_means)

        # Throughput and MFU
        now = time.monotonic()
        current_step = self.state.global_step
        if (
            self._mfu_last_log_time is not None
            and current_step > self._mfu_last_log_step
        ):
            elapsed = now - self._mfu_last_log_time
            steps_done = current_step - self._mfu_last_log_step
            world_size = getattr(self.args, "world_size", 1)
            grad_accum = getattr(self.args, "gradient_accumulation_steps", 1)
            global_batch = (
                self.args.per_device_train_batch_size * world_size * grad_accum
            )
            samples_per_sec = global_batch * steps_done / elapsed
            merged["train/samples_per_sec"] = round(samples_per_sec, 1)

            # Lazy FLOPs init (model may be wrapped after __init__)
            if self._flops_per_sample == 0.0:
                unwrapped = (
                    self.model.module
                    if hasattr(self.model, "module")
                    else self.model
                )
                predictor = getattr(unwrapped, "predictor", None)
                seq_len_actual = (
                    getattr(predictor, "max_input_tokens", 2048)
                    if predictor is not None
                    else 2048
                )
                self._flops_per_sample = _estimate_flops_per_sample(
                    self.model, seq_len_actual
                )

            if self._flops_per_sample > 0.0:
                flops_per_sec_per_gpu = (
                    self._flops_per_sample * samples_per_sec / world_size / 1e12
                )
                mfu = flops_per_sec_per_gpu / _H200_PEAK_TFLOPS
                merged["train/mfu"] = round(mfu * 100, 3)  # as percentage

        self._mfu_last_log_time = now
        self._mfu_last_log_step = current_step

        if pending_means or "train/samples_per_sec" in merged:
            super().log(merged)
        else:
            super().log(logs, start_time)

    def _save_checkpoint(self, model, trial):
        """Save checkpoint and copy configs directory into the checkpoint folder."""
        # Call parent's _save_checkpoint to handle normal checkpoint saving
        super()._save_checkpoint(model, trial)

        # Copy configs/ directory to the checkpoint folder (only on main process)
        if self.args.should_save:
            output_dir = Path(self.args.output_dir)
            configs_src = output_dir / "configs"

            # Get the checkpoint folder that was just created
            checkpoint_folder = (
                output_dir / f"checkpoint-{self.state.global_step}"
            )

            if configs_src.exists() and checkpoint_folder.exists():
                configs_dst = checkpoint_folder / "configs"
                shutil.copytree(configs_src, configs_dst, dirs_exist_ok=True)

    def prediction_step(
        self, model, inputs, prediction_loss_only: bool, ignore_keys=None
    ):
        unwrapped_model = model.module if hasattr(model, "module") else model
        has_targets = any(
            inputs.get(key) is not None
            for key in ("target_vision_input", "targets", "targets_bytes")
        )
        if unwrapped_model.loss_fn is None and has_targets:
            with torch.no_grad():
                loss, outputs = self.compute_loss(model, dict(inputs), return_outputs=True)
            loss = loss.mean().detach()
            if prediction_loss_only:
                return loss, None, None
            return (
                loss,
                outputs["predicted_embeddings"].detach(),
                outputs["target_embeddings"].detach(),
            )
        device = next(model.parameters()).device

        # Handle vision_input (may be None for text-only batches)
        vision_input = inputs.pop("vision_input", None)
        if vision_input is not None:
            vision_input = vision_input.to(device)
        target_vision_input = inputs.pop("target_vision_input", None)
        if target_vision_input is not None:
            target_vision_input = target_vision_input.to(device)
        target_mask = inputs.pop("target_mask", None)
        if target_mask is not None:
            target_mask = target_mask.to(device)

        text_input = extract_text_field(inputs, "text_input", required=False)
        queries = extract_text_field(inputs, "queries", required=False)
        targets = extract_text_field(inputs, "targets", required=False)

        with torch.no_grad():
            outputs = model(
                vision_input=vision_input,
                target_vision_input=target_vision_input,
                target_mask=target_mask,
                text_input=text_input,
                queries=queries,
                targets=targets,
                compute_loss=False,
            )

        loss = None
        pred_embeds = outputs["predicted_embeddings"]
        tgt_embeds = outputs.get("target_embeddings")

        if tgt_embeds is not None:
            if self.gather_embeddings:
                pred_embeds = all_gather_no_grad(pred_embeds)
                tgt_embeds = all_gather_no_grad(tgt_embeds)

            with torch.amp.autocast("cuda", enabled=False):
                pred_embeds = pred_embeds.float()
                tgt_embeds = tgt_embeds.float()
                loss_out = unwrapped_model.loss_fn(pred_embeds, tgt_embeds)

            loss = (
                (loss_out["loss"] if isinstance(loss_out, dict) else loss_out)
                .mean()
                .detach()
            )

        if prediction_loss_only:
            return (loss, None, None)

        pred_embeds = None if pred_embeds is None else pred_embeds.detach()
        tgt_embeds = None if tgt_embeds is None else tgt_embeds.detach()
        return (loss, pred_embeds, tgt_embeds)

    def _accel_info(self):
        acc = getattr(self, "accelerator", None)
        if acc is not None:
            return acc.num_processes, acc.process_index
        return getattr(self.args, "world_size", 1), getattr(
            self.args, "process_index", 0
        )

    def num_examples(self, dataloader: DataLoader) -> int:
        """Report global example counts for iterable distributed training."""
        num_examples = super().num_examples(dataloader)
        dataset = getattr(dataloader, "dataset", None)
        dataset = getattr(dataset, "dataset", dataset)
        if isinstance(dataset, IterableDataset):
            total_length = getattr(dataset, "_length", None)
            if isinstance(total_length, int):
                return total_length
            world_size, _ = self._accel_info()
            return num_examples * world_size
        return num_examples

    def get_train_dataloader(self) -> DataLoader:
        """Create DataLoader for streaming IterableDataset (no sampler needed)."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Wrap in IterableDataset if needed (for datasets with __iter__ but not IterableDataset)
        dataset = self.train_dataset
        if not isinstance(dataset, IterableDataset):
            dataset = _IterableWrapper(dataset)

        num_workers = self.args.dataloader_num_workers
        dataloader_kwargs = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "drop_last": self.args.dataloader_drop_last,
        }

        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = bool(
                getattr(self.args, "dataloader_persistent_workers", False)
            )
            prefetch_factor = getattr(
                self.args, "dataloader_prefetch_factor", None
            )
            if prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = prefetch_factor

        return DataLoader(
            dataset,
            **dataloader_kwargs,
        )
