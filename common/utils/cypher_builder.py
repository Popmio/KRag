import re
from typing import Dict, Any, Type
from common.exceptions import DataValidationError, KRagError
from pydantic import BaseModel, ValidationError

def q(identifier: str) -> str:
    return f"`{identifier}`"

def validate_identifier(name: str) -> None:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Invalid identifier: {name}")

def validate_node_properties(label: str, properties: Dict[str, Any], node_models:Dict[str, Type[BaseModel]]) -> Dict[str, Any]:
    model = node_models.get(label)
    if model is None:
        raise ValueError(f"Node label '{label}' is not allowed. Supported labels: {list(node_models.keys())}")
        # 宽松检查可替换
        # logger.warning(f"Creating node with unvalidated label: {label}")
        # return properties
    try:
        return model(**properties).model_dump()
    except ValidationError as e:
        raise DataValidationError(str(e))

def validate_rel_properties(rel_type: str, properties: Dict[str, Any], rel_models:Dict[str, Type[BaseModel]]) -> Dict[str, Any]:
    model = rel_models.get(rel_type)
    if model is None:
        raise ValueError(
            f"Relationship type '{rel_type}' is not allowed. Supported: {list(rel_models.keys())}"
        )
    try:
        return model(**(properties or {})).model_dump()
    except ValidationError as e:
        raise DataValidationError(str(e))