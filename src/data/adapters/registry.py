"""Dataset registry for Configurable Encoder training."""

from __future__ import annotations

from importlib import import_module

from torch.utils.data import Dataset


ADAPTER_REGISTRY: dict[str, str] = {
    "realworld": "src.data.adapters.image.realworld:RealWorldDataset",
    "pusht": (
        "src.data.adapters.image.pusht:"
        "PushTCoreTemporalAnchoredQuerySupportExactCleanBankInfoNCEMixHDF5Dataset"
    ),
    "tworoom": (
        "src.data.adapters.image."
        "tworoom:"
        "TwoRoomNativeGaussianAnchoredSource9TemporalCleanBankInfoNCEMixHDF5Dataset"
    ),
    "ogbcube": (
        "src.data.adapters.image.ogbcube:"
        "OGBCubePushTStyleAnchoredQuery9CleanBankInfoNCEMixHDF5Dataset"
    ),
}


# Accept dataset IDs stored in older training configs and checkpoints.
ADAPTER_ALIASES = {
    "pusht_core_temporal_anchored_query_support_exact_cleanbank_infonce_mix_hdf5": "pusht",
    "tworoom_pusht_style_anchored_source9_cleanbank_hdf5": "tworoom",
    "ogbcube_pusht_style_anchored_query9_cleanbank_hdf5": "ogbcube",
}

ADAPTER_FAMILIES = {adapter_id: "image" for adapter_id in ADAPTER_REGISTRY}
_CLASS_CACHE: dict[str, type[Dataset]] = {}


def get_adapter(adapter_id: str) -> type[Dataset]:
    adapter_id = ADAPTER_ALIASES.get(adapter_id, adapter_id)
    if adapter_id in _CLASS_CACHE:
        return _CLASS_CACHE[adapter_id]
    if adapter_id not in ADAPTER_REGISTRY:
        available = ", ".join(sorted(ADAPTER_REGISTRY))
        raise KeyError(f"Unknown adapter {adapter_id!r}. Available: {available}")
    module_name, _, class_name = ADAPTER_REGISTRY[adapter_id].partition(":")
    adapter_cls = getattr(import_module(module_name), class_name)
    _CLASS_CACHE[adapter_id] = adapter_cls
    return adapter_cls


def get_adapter_family(adapter_id: str) -> str:
    adapter_id = ADAPTER_ALIASES.get(adapter_id, adapter_id)
    if adapter_id not in ADAPTER_FAMILIES:
        raise KeyError(f"Unknown adapter {adapter_id!r}")
    return ADAPTER_FAMILIES[adapter_id]


def import_all_adapters() -> dict[str, type[Dataset]]:
    return {adapter_id: get_adapter(adapter_id) for adapter_id in sorted(ADAPTER_REGISTRY)}


def list_adapters() -> list[str]:
    return sorted(ADAPTER_REGISTRY)


__all__ = [
    "ADAPTER_ALIASES",
    "ADAPTER_FAMILIES",
    "ADAPTER_REGISTRY",
    "get_adapter",
    "get_adapter_family",
    "import_all_adapters",
    "list_adapters",
]
