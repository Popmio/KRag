import os
import sys

# Ensure project root is on sys.path so imports like `from core...` work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
from core.graph.neo4j_client import Neo4jClient, Neo4jConfig


@pytest.fixture(scope="session")
def neo4j_client():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "K6c3N5p8")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    client = None
    try:
        client = Neo4jClient(Neo4jConfig(uri=uri, username=username, password=password, database=database))
        if not client.ping():
            pytest.skip("Neo4j not reachable (ping failed)")
        yield client
    finally:
        if client is not None:
            client.close()


