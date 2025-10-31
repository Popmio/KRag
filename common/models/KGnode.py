from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict

from common.utils.embedding_utils import validate_vector_dimension


EMBED_DIM = 768


class PathNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category_path_text: Optional[str] = None
    misc_info: Optional[str] = None
    category_path_embedding: Optional[List[float]] = None

    @field_validator("category_path_embedding")
    @classmethod
    def _check_cat_embedding(cls, v: Optional[List[float]]):
        if v is not None:
            validate_vector_dimension(v, EMBED_DIM)
        return v

class DocumentNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    file_name: Optional[str] = None
    doc_type: Optional[str] = None
    link: Optional[str] = None
    doc_code: Optional[str] = None
    file_name_embedding: Optional[List[float]] = None

    @field_validator("file_name_embedding")
    @classmethod
    def _check_file_embedding(cls, v: Optional[List[float]]):
        if v is not None:
            validate_vector_dimension(v, EMBED_DIM)
        return v


class TitleNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    document_id: str
    title_text: Optional[str] = None
    title_embedding: Optional[List[float]] = None

    @field_validator("title_embedding")
    @classmethod
    def _check_title_embedding(cls, v: Optional[List[float]]):
        if v is not None:
            validate_vector_dimension(v, EMBED_DIM)
        return v


class ClauseNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    heading: Optional[str] = None
    summary: Optional[str] = None
    heading_embedding: Optional[List[float]] = None

    @field_validator("heading_embedding")
    @classmethod
    def _check_head_embedding(cls, v: Optional[List[float]]):
        if v is not None:
            validate_vector_dimension(v, EMBED_DIM)
        return v


class ContentNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_id: str
    text: Optional[str] = None
    image_base64: Optional[str] = None
    multimodal_other: Optional[str] = None
    text_embedding: Optional[List[float]] = None

    @field_validator("text_embedding")
    @classmethod
    def _check_text_embedding(cls, v: Optional[List[float]]):
        if v is not None:
            validate_vector_dimension(v, EMBED_DIM)
        return v


class KeywordNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str

class OrganizationNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: Optional[str] = None


class ContainsRel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edge_type: Optional[str] = None


class PublishedByRel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    published_at: Optional[str] = None


class CitesRel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_level: Optional[str] = None


class HasKeywordRel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # no properties per schema


