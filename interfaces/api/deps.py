from __future__ import annotations

from functools import lru_cache
from typing import Generator

from pydantic_settings import BaseSettings, SettingsConfigDict

from core.graph.neo4j_client import Neo4jClient, Neo4jConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_neo4j_client() -> Neo4jClient:
    s = get_settings()
    cfg = Neo4jConfig(
        uri=s.NEO4J_URI,
        username=s.NEO4J_USERNAME,
        password=s.NEO4J_PASSWORD,
        database=s.NEO4J_DATABASE,
    )
    return Neo4jClient(cfg)


