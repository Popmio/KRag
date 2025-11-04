import pytest

from common.utils.embedding_utils import validate_vector_dimension, normalize_vector
from common.exceptions import VectorDimensionMismatchError, DataValidationError
from common.models.KGnode import DocumentNode
from modelscope import AutoModel, AutoTokenizer

def test_validate_vector_dimension_ok():
    validate_vector_dimension([0.0] * 4, 4)


def test_validate_vector_dimension_mismatch():
    with pytest.raises(VectorDimensionMismatchError):
        validate_vector_dimension([0.0] * 3, 4)


def test_normalize_vector_ok():
    out = normalize_vector([3.0, 4.0])
    assert pytest.approx(out[0] ** 2 + out[1] ** 2, rel=1e-6) == 1.0


def test_normalize_vector_zero():
    with pytest.raises(DataValidationError):
        normalize_vector([0.0, 0.0])


def test_document_node_extra_forbid():
    with pytest.raises(Exception):
        DocumentNode(id="d1", file_name="f", unknown="x")


def test_document_node_embedding_dim():
    # Correct dim
    DocumentNode(id="d1", file_name_embedding=[0.0] * 768)
    # Wrong dim
    with pytest.raises(Exception):
        DocumentNode(id="d2", file_name_embedding=[0.0] * 10)
