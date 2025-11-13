from typing import Dict, Type
from pydantic import BaseModel


from common.models.KGnode import (
    PathNode,
    DocumentNode,
    TitleNode,
    ClauseNode,
    ContentNode,
    KeywordNode,
    OrganizationNode,
    ContainsRel,
    PublishedByRel,
    CitesRel,
    HasKeywordRel,
)

NODE_MODELS: Dict[str, Type[BaseModel]] = {
    "Path": PathNode,
    "Document": DocumentNode,
    "Title": TitleNode,
    "Clause": ClauseNode,
    "Content": ContentNode,
    "Keyword": KeywordNode,
    "Organization": OrganizationNode,
}
REL_MODELS: Dict[str, Type[BaseModel]] = {
    "CONTAINS": ContainsRel,
    "PUBLISHED_BY": PublishedByRel,
    "CITES": CitesRel,
    "HAS_KEYWORD": HasKeywordRel,
}