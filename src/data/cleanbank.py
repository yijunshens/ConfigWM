"""Clean-bank size helpers shared by the Configurable Encoder dataset adapters."""

from __future__ import annotations


def resolve_cleanbank_rows(
    bank_size: int,
    query_ratio: float,
    query_group_size: int,
) -> tuple[int, int, int]:
    """Return ``(bank_size, core_rows, query_rows)`` for one contrast bank.

    Query rows are rounded to complete query groups.  Thus ``512`` with a
    ``0.10`` ratio and a 9-row query group becomes ``458 core + 54 query``.
    """
    bank_size = int(bank_size)
    query_group_size = int(query_group_size)
    query_ratio = float(query_ratio)
    if bank_size <= 1:
        raise ValueError("bank_size must be greater than 1")
    if query_group_size <= 0:
        raise ValueError("query_group_size must be positive")
    if not 0.0 <= query_ratio <= 1.0:
        raise ValueError("query_ratio must be between 0 and 1")

    query_groups = int(round(bank_size * query_ratio / query_group_size))
    if query_ratio > 0.0:
        query_groups = max(1, query_groups)
    query_rows = query_groups * query_group_size
    if query_rows >= bank_size:
        query_rows = ((bank_size - 1) // query_group_size) * query_group_size
    core_rows = bank_size - query_rows
    if core_rows <= 0:
        raise ValueError("query_ratio leaves no core rows")
    return bank_size, core_rows, query_rows


__all__ = ["resolve_cleanbank_rows"]
