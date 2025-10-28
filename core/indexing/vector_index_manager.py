from __future__ import annotations

from typing import Optional


class VectorIndexManager:
    """Manage Neo4j vector indexes and related checks.

    This class intentionally avoids depending on a specific Neo4j client wrapper.
    Callers should provide a session-like object exposing `run(cypher, **params)`.
    """

    def __init__(
        self,
        session_factory,
        *,
        label: str,
        property_name: str,
        dimension: int,
        similarity: str = "cosine",
        index_name: Optional[str] = None,
    ) -> None:
        self._session_factory = session_factory
        self.label = label
        self.property_name = property_name
        self.dimension = dimension
        self.similarity = similarity
        self.index_name = index_name or f"{label.lower()}_{property_name}_vec_idx"

    def ensure_index(self) -> None:
        """Create index if it doesn't exist, idempotently."""
        cypher = (
            """
            CREATE VECTOR INDEX %s IF NOT EXISTS
            FOR (n:%s) ON (n.%s)
            OPTIONS {indexConfig: {
              `vector.dimensions`: $dimension,
              `vector.similarity_function`: $similarity
            }}
            """
            % (self.index_name, self.label, self.property_name)
        )
        with self._session_factory() as session:
            session.run(
                cypher,
                dimension=self.dimension,
                similarity=self.similarity,
            )

    def drop_index(self) -> None:
        cypher = f"DROP INDEX {self.index_name} IF EXISTS"
        with self._session_factory() as session:
            session.run(cypher)

    def index_exists(self) -> bool:
        cypher = """
        SHOW INDEXES YIELD name, type
        WHERE name = $name AND type = 'VECTOR'
        RETURN count(*) AS cnt
        """
        with self._session_factory() as session:
            res = session.run(cypher, name=self.index_name)
            record = res.single()
            return bool(record and record[0] > 0)

    def stats(self) -> dict:
        """Return simple statistics for the vector property.

        Note: Neo4j doesn't expose per-index vector stats yet; this returns
        approximate node/property coverage.
        """
        cypher = (
            """
            MATCH (n:%s)
            WITH count(n) AS total,
                 count { (n.%s) IS NOT NULL } AS with_vec
            RETURN total, with_vec
            """
            % (self.label, self.property_name)
        )
        with self._session_factory() as session:
            rec = session.run(cypher).single()
            if not rec:
                return {"total": 0, "with_vec": 0}
            return {"total": int(rec[0] or 0), "with_vec": int(rec[1] or 0)}


