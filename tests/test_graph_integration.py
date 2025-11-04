import os
import uuid

import pytest

from core.graph.schema_manager import SchemaManager
from core.graph.node_manager import NodeManager
from core.graph.neo4j_client import Neo4jClient, Neo4jConfig


def get_client_or_skip():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "K6c3N5p8")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    try:
        client = Neo4jClient(Neo4jConfig(uri=uri, username=username, password=password, database=database))
        if not client.ping():
            pytest.skip("Neo4j not reachable (ping failed)")
        return client
    except Exception as e:
        pytest.skip(f"Neo4j not reachable: {e}")


def test_apply_schema(neo4j_client):
    client = neo4j_client
    sm = SchemaManager(client)
    cfg = sm.load_yaml("config/graph_schema.yaml")
    sm.apply(cfg)


def test_crud_and_relations(neo4j_client):
    client = neo4j_client
    nm = NodeManager(client)

    did = f"doc_{uuid.uuid4().hex[:8]}"
    eid = f"ent_{uuid.uuid4().hex[:8]}"
    kw = f"kw_{uuid.uuid4().hex[:6]}"

    # upsert nodes
    nm.merge_node("Document", key="id", properties={"id": did, "file_name": "t.pdf", "title_text": "Hello"})
    nm.merge_node("Entity", key="id", properties={"id": eid, "name": "Neo4j"})
    nm.merge_node("Keyword", key="text", properties={"text": kw})

    # read back
    doc = nm.get_node("Document", "id", did)
    assert doc and doc.get("node", {}).get("id") == did

    # relation
    nm.merge_relationship("Document", "id", did, "HAS_KEYWORD", "Keyword", "text", kw, {"weight": 0.9})

    # cleanup
    nm.delete_node("Document", "id", did)
    nm.delete_node("Entity", "id", eid)
    nm.delete_node("Keyword", "text", kw)


def test_batch_merge(neo4j_client):
    client = neo4j_client
    nm = NodeManager(client)
    items = [
        {"id": f"doc_{i}_{uuid.uuid4().hex[:6]}", "file_name": f"f{i}.pdf"}
        for i in range(3)
    ]
    cnt = nm.batch_merge_nodes("Document", key="id", items=items)
    assert cnt == len(items)
    # cleanup
    for it in items:
        nm.delete_node("Document", "id", it["id"])


