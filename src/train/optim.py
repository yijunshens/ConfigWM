"""LR multiplier and optimizer utilities for ConfigurableEncoderTrainer."""

from __future__ import annotations

from collections import defaultdict
import logging


class LRMultiplierMixin:
    """Mixin class that provides LR multiplier functionality for Trainer subclasses.

    Expects:
        - self.args with learning_rate, weight_decay, lr_multipliers
        - self.optimizer
        - self.get_decay_parameter_names()
        - self.get_optimizer_cls_and_kwargs()
        - self.optimizer_cls_and_kwargs
    """

    def _lr_scale_for(self, name: str) -> float:
        """Compute LR scale for a parameter by multiplying all matching patterns."""
        s = 1.0
        for sub, mult in self.lr_multipliers.items():
            if sub in name:
                s *= float(mult)
        return s

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None:
            # Reproduce the historical first update before the warmup schedule.
            for group in self.optimizer.param_groups:
                group['lr'] = self.args.learning_rate * group.get('lr_scale', 1.0)

    def create_optimizer(self):
        """Setup the optimizer with per-parameter LR scaling diagnostics.

        Notes:
        • This keeps semantics identical to your current implementation.
        • It logs:
            - how many params matched each lr_multipliers pattern,
            - per-parameter final LR scale and whether multiple patterns stacked,
            - group summaries by (weight_decay, lr_scale),
            - warnings if optimizer kwargs will override your param_groups,
            - warnings for scale==0.0 params that still require_grad (not frozen).
        """
        if self.optimizer is not None:
            logging.info(
                "create_optimizer: existing optimizer found; returning it unchanged."
            )
            return self.optimizer

        # Works with plain, DDP, Deepspeed, SageMaker MP without imports
        opt_model = getattr(self, "model_wrapped", None) or self.model

        # Decay partition computed via HF helper (respects norm/bias exclusions)
        decay_params = set(self.get_decay_parameter_names(opt_model))

        # Resolve optimizer class + kwargs (handles bitsandbytes, LOMO, etc.)
        if self.optimizer_cls_and_kwargs is not None:
            opt_cls, opt_kwargs = self.optimizer_cls_and_kwargs
            logging.info(
                "create_optimizer: using provided optimizer_cls_and_kwargs -> %s",
                opt_cls.__name__,
            )
        else:
            opt_cls, opt_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
            logging.info(
                "create_optimizer: using get_optimizer_cls_and_kwargs -> %s",
                opt_cls.__name__,
            )

        base_lr = float(self.args.learning_rate)
        wd = float(self.args.weight_decay)

        # ---------- diagnostics collectors ----------
        pattern_hits = defaultdict(int)  # pattern -> #params matched
        pattern_multipliers = dict(getattr(self, "lr_multipliers", {}) or {})
        per_param_notes = []  # small sample of param->scale notes
        stacked_matches = 0  # params hit by >=2 patterns
        zero_scale_requires_grad = (
            0  # params with scale==0 and requires_grad==True
        )
        total_trainable = 0
        total_params = sum(p.numel() for p in opt_model.parameters())
        matched_any_pattern = 0

        # Keep a small sample of params for log readability
        SAMPLE_CAP = 1000

        # ---------- build param groups keyed by (weight_decay, lr_scale) ----------
        groups = {}  # key: (wd_here, scale) -> dict
        for n, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            total_trainable += p.numel()

            # Compute scale by multiplicative application of all matching substrings
            scale = 1.0
            hits = 0
            for sub, mult in pattern_multipliers.items():
                if sub and sub in n:
                    try:
                        scale *= float(mult)
                    except Exception:
                        logging.warning(
                            "create_optimizer: non-numeric multiplier for pattern '%s': %s",
                            sub,
                            mult,
                        )
                        continue
                    pattern_hits[sub] += 1
                    hits += 1

            if hits > 0:
                matched_any_pattern += 1
                if hits >= 2:
                    stacked_matches += 1
                    if len(per_param_notes) < SAMPLE_CAP:
                        per_param_notes.append(
                            f"[STACKED] {n} -> scale={scale:.6g} via {hits} patterns"
                        )

            # Detect scale==0.0 while still trainable
            if scale == 0.0 and p.requires_grad:
                zero_scale_requires_grad += 1
                if len(per_param_notes) < SAMPLE_CAP:
                    per_param_notes.append(
                        f"[ZERO-LR but grad ON] {n} -> scale=0.0; will still backprop, be clipped, etc."
                    )

            wd_here = wd if (n in decay_params) else 0.0
            key = (wd_here, scale)
            if key not in groups:
                groups[key] = {
                    "params": [],
                    "weight_decay": wd_here,
                    "lr": base_lr * scale,
                    "lr_scale": scale,
                    "_count": 0,
                    "_elems": 0,
                }
            groups[key]["params"].append(p)
            groups[key]["_count"] += 1
            groups[key]["_elems"] += p.numel()

            if len(per_param_notes) < SAMPLE_CAP and hits == 1:
                # Record a representative single-hit mapping
                per_param_notes.append(
                    f"{n} -> scale={scale:.6g}, wd={wd_here}"
                )

        param_groups = list(groups.values())

        # ---------- honor special optimizer kwargs that override groups ----------
        # NOTE: This *replaces* param_groups, which will bypass your LR scaling.
        overridden_key = None
        for k in ("params", "model", "optimizer_dict"):
            if k in opt_kwargs:
                overridden_key = k
                logging.warning(
                    "create_optimizer: optimizer kwarg '%s' overrides computed param_groups. "
                    "Your lr multipliers may be ignored by %s.",
                    k,
                    opt_cls.__name__,
                )
                param_groups = opt_kwargs.pop(k)

        # ---------- instantiate optimizer ----------
        self.optimizer = opt_cls(param_groups, **opt_kwargs)

        # ---------- final, compact report ----------
        self._log_optimizer_summary(
            total_params=total_params,
            total_trainable=total_trainable,
            pattern_multipliers=pattern_multipliers,
            pattern_hits=pattern_hits,
            matched_any_pattern=matched_any_pattern,
            stacked_matches=stacked_matches,
            zero_scale_requires_grad=zero_scale_requires_grad,
            param_groups=param_groups,
            per_param_notes=per_param_notes,
            overridden_key=overridden_key,
            sample_cap=SAMPLE_CAP,
        )

        return self.optimizer

    def _log_optimizer_summary(
        self,
        total_params: int,
        total_trainable: int,
        pattern_multipliers: dict[str, float],
        pattern_hits: dict[str, int],
        matched_any_pattern: int,
        stacked_matches: int,
        zero_scale_requires_grad: int,
        param_groups: list,
        per_param_notes: list,
        overridden_key: str | None,
        sample_cap: int,
    ) -> None:
        """Log a summary of optimizer configuration."""
        lines = []
        lines.append("create_optimizer: summary")
        lines.append(f"- total parameters: {total_params:,}")
        lines.append(f"- trainable parameters: {total_trainable:,}")
        lines.append(f"- lr_multipliers patterns: {len(pattern_multipliers)}")
        if pattern_multipliers:
            for sub, mult in pattern_multipliers.items():
                lines.append(
                    f"    • '{sub}': x{float(mult)} -> matched {pattern_hits.get(sub, 0)} tensors"
                )
        lines.append(f"- params matched any pattern: {matched_any_pattern}")
        lines.append(
            f"- params with stacked pattern matches: {stacked_matches}"
        )
        if zero_scale_requires_grad > 0:
            lines.append(
                f"- WARNING: {zero_scale_requires_grad} params have scale==0.0 but require_grad=True "
                "(not frozen; contribute grads, clipping, state)."
            )

        # Per-group stats
        lines.append(f"- optimizer param groups: {len(param_groups)}")
        for g in param_groups:
            # Some branch may have been overridden; be defensive
            lr_g = g.get("lr", None)
            sc_g = g.get("lr_scale", None)
            wd_g = g.get("weight_decay", None)
            cnt = g.get("_count", None)
            elems = g.get("_elems", None)
            if (
                cnt is not None
                and elems is not None
                and lr_g is not None
                and sc_g is not None
                and wd_g is not None
            ):
                lines.append(
                    f"    • group lr={lr_g:.6g} (scale={sc_g}), wd={wd_g}, tensors={cnt}, elements={elems:,}"
                )
            else:
                # If groups were overridden we may not have diagnostics
                lines.append(f"    • group keys: {sorted(g.keys())}")

        if overridden_key is not None:
            lines.append(
                f"- NOTICE: param groups were overridden by optimizer kwarg '{overridden_key}'. "
                "Verify that your custom optimizer applies LR scaling itself."
            )

        # Sample of per-parameter notes
        if per_param_notes:
            lines.append(
                f"- sample mappings ({min(len(per_param_notes), sample_cap)} shown):"
            )
            for s in per_param_notes[:sample_cap]:
                lines.append(f"    • {s}")

        logging.info("\n".join(lines))

