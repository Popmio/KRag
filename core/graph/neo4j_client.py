from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, Optional

from neo4j import GraphDatabase, Driver, Session, basic_auth
from neo4j.exceptions import ServiceUnavailable, AuthError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:

    uri: str
    username: str
    password: str
    database: str = "neo4j"
    max_connection_pool_size: int = 100
    connection_acquisition_timeout: float = 30.0
    fetch_size: int = 1000


class Neo4jClient:

    def __init__(self, config: Neo4jConfig) -> None:
        self._config = config
        self._driver: Driver = GraphDatabase.driver(
            config.uri,
            auth=basic_auth(config.username, config.password),
            max_connection_pool_size=config.max_connection_pool_size,
            connection_acquisition_timeout=config.connection_acquisition_timeout,
        )

        self.verify_connectivity()

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            logger.exception("Error closing Neo4j driver")

    def verify_connectivity(self) -> None:
        try:
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s (db=%s)", self._config.uri, self._config.database)
        except AuthError as exc:
            logger.error("Neo4j authentication failed: %s", exc)
            raise
        except ServiceUnavailable as exc:
            logger.error("Neo4j service unavailable: %s", exc)
            raise

    def ping(self) -> bool:
        try:
            with self.session(readonly=True) as session:
                result = session.run("RETURN 1 AS ok")
                record = result.single()
                return bool(record and record[0] == 1)
        except Exception:
            return False

    @contextmanager
    def session(self, *, readonly: bool = False) -> Generator[Session, None, None]:

        default_access_mode = "READ" if readonly else "WRITE"
        session = self._driver.session(database=self._config.database,
                                       default_access_mode=default_access_mode,
                                       fetch_size=self._config.fetch_size)
        try:
            yield session
        finally:
            session.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type(ServiceUnavailable),
    )
    def execute(self, cypher: str, params: Optional[Dict[str, Any]] = None, *, readonly: bool = False) -> Iterable[Dict[str, Any]]:
        with self.session(readonly=readonly) as session:
            result = session.run(cypher, parameters=params or {})
            return result.data()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type(ServiceUnavailable),
    )
    def execute_single(self, cypher: str, params: Optional[Dict[str, Any]] = None, *, readonly: bool = False) -> Optional[Dict[str, Any]]:
        with self.session(readonly=readonly) as session:
            result = session.run(cypher, parameters=params or {})
            rec = result.single()
            return rec.data() if rec else None

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

class SessionConfig:
    READ = "READ"
    WRITE = "WRITE"


__all__ = [
    "Neo4jClient",
    "Neo4jConfig",
]


