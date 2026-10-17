from __future__ import annotations

import random


def sample_sentence_chunk(
    text: str,
    splitter: str = ".",
    rng: random.Random | None = None,
) -> str:
    """Sample a contiguous sentence chunk from text.

    The text is split on `splitter`, empty fragments are discarded, and one
    contiguous chunk is selected uniformly by chunk length and start position.
    A single sentence and the full text are both possible outcomes.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text)!r}")
    if not isinstance(splitter, str) or not splitter:
        raise ValueError("splitter must be a non-empty string")

    text = text.strip()
    if not text:
        return text

    raw_parts = text.split(splitter)
    has_trailing_splitter = text.endswith(splitter)

    sentences: list[str] = []
    for i, raw_part in enumerate(raw_parts):
        sentence = raw_part.strip()
        if not sentence:
            continue
        is_last_part = i == len(raw_parts) - 1
        if not is_last_part or has_trailing_splitter:
            sentence = f"{sentence}{splitter}"
        sentences.append(sentence)

    if not sentences:
        return text

    generator = rng if rng is not None else random
    chunk_length = generator.randint(1, len(sentences))
    start = generator.randint(0, len(sentences) - chunk_length)
    return " ".join(sentences[start : start + chunk_length]).strip()


__all__ = ["sample_sentence_chunk"]
