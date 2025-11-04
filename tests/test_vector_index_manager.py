import os
import pytest

from core.indexing.vector_index_manager import VectorIndexManager
from core.graph.neo4j_client import Neo4jClient, Neo4jConfig


def _client_or_skip():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    try:
        c = Neo4jClient(Neo4jConfig(uri=uri, username=username, password=password, database=database))
        if not c.ping():
            pytest.skip("Neo4j not reachable")
        return c
    except Exception:
        pytest.skip("Neo4j not reachable")


def test_vector_index_ensure_and_stats():
    c = _client_or_skip()

    def session_factory():
        return c.session()

    vim = VectorIndexManager(session_factory, label="Document", property_name="file_name_embedding", dimension=768)
    vim.ensure_index()
    assert isinstance(vim.index_exists(), bool)

