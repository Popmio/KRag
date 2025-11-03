from __future__ import annotations

from typing import Iterable, List, Sequence

from common.exceptions import VectorDimensionMismatchError, DataValidationError


def validate_vector_dimension(vector: Sequence[float], expected_dim: int) -> None:
    if vector is None:
        return
    if len(vector) != expected_dim:
        raise VectorDimensionMismatchError(
            f"Embedding dimension mismatch: got {len(vector)}, expected {expected_dim}"
        )


def normalize_vector(vector: Sequence[float]) -> List[float]:

    if vector is None:
        return []
    s = sum(v * v for v in vector)
    if s == 0:
        raise DataValidationError("Cannot normalize zero vector")
    norm = s ** 0.5
    return [float(v) / norm for v in vector]


