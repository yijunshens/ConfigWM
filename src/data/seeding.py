"""Bind Configurable Encoder data construction to the run's single seed."""

from collections.abc import Mapping
import copy
from typing import Any

from .adapters.registry import ADAPTER_ALIASES


_ANCHOR_FIELDS = {
    "pusht": "query_support_anchor_seed",
    "tworoom": "source9_anchor_seed",
    "ogbcube": "query_anchor_seed",
}


def validate_encoder_seed(seed: int) -> int:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Configurable Encoder seed must be an integer in [0, 2**32).")
    return seed


def seed_encoder_data(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Return a copy with all active adapter seed arguments bound to seed.

    Adapter argument names are internal plumbing, not independently tunable
    settings. Reject conflicting legacy values rather than silently changing
    the meaning of a saved multi-seed experiment.
    """
    seed = validate_encoder_seed(seed)
    result = copy.deepcopy(dict(config))
    if result.get("seed", seed) != seed:
        raise ValueError("Configurable Encoder data seed must equal the run seed.")
    result["seed"] = seed
    for section in ("train_datasets", "eval_datasets"):
        for spec in result.get(section, []):
            for key, value in spec.items():
                if (key == "seed" or key.endswith("_seed")) and value != seed:
                    raise ValueError(f"Remove independent Configurable Encoder {key}; use the run seed.")
            adapter_id = spec.get("id")
            adapter_id = ADAPTER_ALIASES.get(adapter_id, adapter_id)
            anchor = _ANCHOR_FIELDS.get(adapter_id)
            if anchor is not None:
                spec["core_balance_seed"] = seed
                spec[anchor] = seed
    return result
