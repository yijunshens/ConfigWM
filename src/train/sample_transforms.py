"""Per-dataset sample transforms applied in the training mixer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import random
from typing import Any

from src.data.adapters.utils.text_augmentation import sample_sentence_chunk
from src.data.schema import Sample


@dataclass(frozen=True)
class DatasetSampleTransform:
    """Dataset-local sample transform configuration for training."""

    p_sample_sentence_chunk: float = 0.0

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> DatasetSampleTransform:
        raw_probability = spec.get("p_sample_sentence_chunk", 0.0)
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid p_sample_sentence_chunk={raw_probability!r}; expected a float in [0, 1]"
            ) from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("p_sample_sentence_chunk must be in [0, 1]")
        return cls(p_sample_sentence_chunk=probability)

    @property
    def enabled(self) -> bool:
        return self.p_sample_sentence_chunk > 0.0

    def apply(self, sample: Sample, *, rng: random.Random) -> Sample:
        """Apply configured text transforms to one raw sample."""
        if not self.enabled or not sample.targets:
            return sample

        if rng.random() >= self.p_sample_sentence_chunk:
            return sample

        updated_targets: list[str] = []
        changed = False
        for target in sample.targets:
            if not isinstance(target, str):
                return sample
            updated_target = sample_sentence_chunk(target, rng=rng) or target
            updated_targets.append(updated_target)
            changed = changed or updated_target != target

        if not changed:
            return sample

        return Sample(
            frames=sample.frames,
            label_map=sample.label_map,
            targets=updated_targets,
            meta=sample.meta,
            target_frames=sample.target_frames,
            query=sample.query,
            text_input=sample.text_input,
        )


__all__ = ["DatasetSampleTransform"]
