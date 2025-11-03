from fastapi import APIRouter, Depends

from interfaces.api.deps import get_neo4j_client
from core.graph.node_manager import NodeManager

router = APIRouter(prefix="/v1", tags=["graph"])


@router.get("/ping")
def ping() -> dict:
    return {"message": "pong"}


@router.get("/nodes/{label}/{key}/{value}")
def get_node(label: str, key: str, value: str, client = Depends(get_neo4j_client)) -> dict:
    nm = NodeManager(client)
    rec = nm.get_node(label, key, value)
    return rec or {}


