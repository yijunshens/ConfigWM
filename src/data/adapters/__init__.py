"""Public adapter exports."""

from __future__ import annotations

from ..schema import (
    ImageBoxesSample,
    ImageMasksSample,
    ImageTextSample,
    VideoMasksSample,
    Sample,
)
from ..specs import ResolvedDatasetSpec, dataset_label, resolve_dataset_spec
from .factory import create_adapter
from .registry import (
    ADAPTER_FAMILIES,
    ADAPTER_REGISTRY,
    get_adapter,
    get_adapter_family,
    import_all_adapters,
    list_adapters,
)
from .utils.unification import (
    from_boxes,
    from_masks,
    from_text,
)

__all__ = [
    "ADAPTER_FAMILIES",
    "ADAPTER_REGISTRY",
    "ImageBoxesSample",
    "ImageMasksSample",
    "ImageTextSample",
    "ResolvedDatasetSpec",
    "Sample",
    "VideoMasksSample",
    "create_adapter",
    "dataset_label",
    "from_boxes",
    "from_masks",
    "from_text",
    "get_adapter",
    "get_adapter_family",
    "import_all_adapters",
    "list_adapters",
    "resolve_dataset_spec",
]
