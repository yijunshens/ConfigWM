"""Evaluation-capable dataset protocols.

These protocols define optional interfaces that datasets can implement
to support different evaluation task types (MCQA, Classification, Retrieval).

The base Sample provides training data; protocols add eval-specific methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.data.schema import Sample


@runtime_checkable
class MCQACapable(Protocol):
    """Protocol for datasets that support MCQA (multiple-choice QA) evaluation.

    Required methods:
        - get_options(idx): Return list of answer options for sample
        - get_answer_idx(idx): Return 0-indexed correct answer index

    Example implementation:
        class RACEDataset(MCQACapable):
            def get_options(self, idx: int) -> List[str]:
                return self._dataset[idx]["options"]

            def get_answer_idx(self, idx: int) -> int:
                return ord(self._dataset[idx]["answer"]) - ord("A")
    """

    def __len__(self) -> int:
        """Number of samples in dataset."""
        ...

    def __getitem__(self, idx: int) -> Sample:
        """Get sample for training (returns Sample)."""
        ...

    def get_options(self, idx: int) -> list[str]:
        """Get answer options for sample at idx."""
        ...

    def get_answer_idx(self, idx: int) -> int:
        """Get 0-indexed correct answer for sample at idx."""
        ...


@runtime_checkable
class ClassificationCapable(Protocol):
    """Protocol for datasets that support classification evaluation.

    Required methods:
        - get_label_space(): Return list of all possible labels (dataset-level)
        - get_label(idx): Return ground-truth label for sample

    Example implementation:
        class SSv2Dataset(ClassificationCapable):
            @cached_property
            def label_space(self) -> List[str]:
                return sorted(set(item["label"] for item in self._dataset))

            def get_label_space(self) -> List[str]:
                return self.label_space

            def get_label(self, idx: int) -> str:
                return self._dataset[idx]["label"]
    """

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Sample: ...

    def get_label_space(self) -> list[str]:
        """Return list of all unique labels in dataset."""
        ...

    def get_label(self, idx: int) -> str:
        """Get ground-truth label for sample at idx."""
        ...


@runtime_checkable
class RetrievalCapable(Protocol):
    """Protocol for datasets that support retrieval evaluation.

    Two modes:
        1. Query → Gallery: queries point to relevant gallery items
        2. Paired: Each sample has (query, gallery_item) with relevance

    Required methods:
        - get_gallery(): Return list of gallery items (text or embeddings)
        - get_relevant_ids(idx): Return list of relevant gallery indices for query idx

    Example implementation:
        class COCORetrievalDataset(RetrievalCapable):
            def get_gallery(self) -> List[str]:
                return self._all_captions  # or image paths

            def get_relevant_ids(self, idx: int) -> List[int]:
                return self._query_to_gallery[idx]
    """

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Sample: ...

    def get_gallery(self) -> list:
        """Return gallery items (captions, image paths, or embeddings)."""
        ...

    def get_relevant_ids(self, idx: int) -> list[int]:
        """Return indices of gallery items relevant to query at idx."""
        ...


# Type alias for any eval-capable dataset
EvalCapableDataset = MCQACapable | ClassificationCapable | RetrievalCapable


def detect_eval_type(dataset) -> str | None:
    """Detect what type of evaluation a dataset supports.

    Returns:
        "mcqa", "classification", "retrieval", or None
    """
    if isinstance(dataset, MCQACapable):
        return "mcqa"
    if isinstance(dataset, ClassificationCapable):
        return "classification"
    if isinstance(dataset, RetrievalCapable):
        return "retrieval"
    return None
