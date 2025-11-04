import os
import pytest

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


def test_ping_and_execute():
    c = _client_or_skip()
    assert c.ping()
    data = c.execute("RETURN 1 AS ok", readonly=True)
    assert isinstance(data, list) and data[0]["ok"] == 1

