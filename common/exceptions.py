class KRagError(Exception):
    """Base class for KRag exceptions."""


class SchemaValidationError(KRagError):
    """Raised when graph_schema.yaml is invalid."""


class NotFoundInSchemaError(KRagError):
    """Raised when a referenced label/property/relationship is not defined in schema."""


class DataValidationError(KRagError):
    """Raised when incoming data fails validation checks."""


class VectorDimensionMismatchError(DataValidationError):
    """Raised when an embedding vector length mismatches expected dimension."""


