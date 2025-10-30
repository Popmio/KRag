from __future__ import annotations

import re
from typing import Any, Dict, Optional, List, Type

from .neo4j_client import Neo4jClient
from pydantic import BaseModel, ValidationError
from common.exceptions import DataValidationError, KRagError
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

# Label/relationship -> Pydantic model mapping for validation
_NODE_MODELS: Dict[str, Type[BaseModel]] = {
    "Path": PathNode,
    "Document": DocumentNode,
    "Title": TitleNode,
    "Clause": ClauseNode,
    "Content": ContentNode,
    "Keyword": KeywordNode,
    "Organization": OrganizationNode,
}

_REL_MODELS: Dict[str, Type[BaseModel]] = {
    "CONTAINS": ContainsRel,
    "PUBLISHED_BY": PublishedByRel,
    "CITES": CitesRel,
    "HAS_KEYWORD": HasKeywordRel,
}


class NodeManager:

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client
        self._unique_key_cache: Dict[str, Dict[str, bool]] = {}

    @staticmethod
    def _q(identifier: str) -> str:
        # Quote identifiers to avoid reserved word clashes; input is validated
        return f"`{identifier}`"

    @staticmethod
    def _validate_identifier(name: str) -> None:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            raise ValueError(f"Invalid identifier: {name}")

    @staticmethod
    def _validate_node_properties(label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        model = _NODE_MODELS.get(label)
        if model is None:
            raise ValueError(f"Node label '{label}' is not allowed. Supported labels: {list(_NODE_MODELS.keys())}")
            # 宽松检查可替换
            # logger.warning(f"Creating node with unvalidated label: {label}")
            # return properties
        try:
            return model(**properties).model_dump()
        except ValidationError as e:
            raise DataValidationError(str(e))

    @staticmethod
    def _validate_rel_properties(rel_type: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        model = _REL_MODELS.get(rel_type)
        if model is None:
            raise ValueError(
                f"Relationship type '{rel_type}' is not allowed. Supported: {list(_REL_MODELS.keys())}"
            )
        try:
            return model(**(properties or {})).model_dump()
        except ValidationError as e:
            raise DataValidationError(str(e))

    # -------------------
    # Uniqueness helpers
    # -------------------
    def _is_unique_key(self, label: str, key: str) -> bool:
        cache_for_label = self._unique_key_cache.setdefault(label, {})
        if key in cache_for_label:
            return cache_for_label[key]
        rows = self._client.execute(
            (
                "SHOW CONSTRAINTS YIELD type, entityType, labelsOrTypes, properties "
                "WHERE entityType = 'NODE' "
                "RETURN type, labelsOrTypes, properties"
            ),
            readonly=True,
        )
        is_unique = False
        for r in rows:
            labels = r.get("labelsOrTypes") or []
            props = r.get("properties") or []
            ctype = str(r.get("type") or "").upper()
            if label in labels and key in props and ("UNIQUE" in ctype or "UNIQUENESS" in ctype or "NODE_KEY" in ctype):
                is_unique = True
                break
        cache_for_label[key] = is_unique
        return is_unique

    def _ensure_unique_key(self, label: str, key: str) -> None:
        if not self._is_unique_key(label, key):
            raise KRagError(f"Property '{label}.{key}' is not unique; use multi-item variants (e.g., get_nodes)")

    # -------------------
    # Node CRUD
    # -------------------
    def merge_node(self, label: str, *, key: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_identifier(label)
        self._validate_identifier(key)
        self._ensure_unique_key(label, key)
        if key not in properties:
            raise DataValidationError(f"Primary key '{key}' missing from node properties")
        props = self._validate_node_properties(label, properties)
        props_no_key = {k: v for k, v in props.items() if k != key}
        cypher = (
            f"MERGE (n:{self._q(label)} {{{self._q(key)}: $keyVal}}) "
            "ON CREATE SET n += $propsNoKey "
            "WITH n, $propsNoKey AS props "
            "WITH n, props, [k IN keys(props) WHERE n[k] IS NULL] AS toSet "
            "FOREACH (k IN toSet | SET n[k] = props[k]) "
            "RETURN properties(n) AS node, toSet AS applied"
        )
        rec = self._client.execute_single(
            cypher,
            {"keyVal": props[key], "propsNoKey": props_no_key},
            readonly=False,
        )
        if not rec or not isinstance(rec.get("node"), dict):
            raise KRagError("Failed to merge node; no record returned from database")
        return rec

    def get_node(self, label: str, key: str, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            raise ValueError(f"Cannot query node by {key}=None")
        self._validate_identifier(label)
        self._validate_identifier(key)
        self._ensure_unique_key(label, key)
        cypher = (
            f"MATCH (n:{self._q(label)} {{{self._q(key)}: $val}}) "
            f"RETURN properties(n) AS node LIMIT 1"
        )
        return self._client.execute_single(cypher, {"val": value}, readonly=True)

    def get_nodes(self, label: str, key: str, values: List[Any]) -> List[Dict[str, Any]]:
        self._validate_identifier(label)
        self._validate_identifier(key)
        if not isinstance(values, list) or not values:
            raise ValueError("Invalid values: expected a non-empty list")
        values = [v for v in values if v is not None]
        if not values:
            raise ValueError(f"No valid (non-null) values provided for {label}.{key}")
        cypher = (
            f"UNWIND $vals AS v "
            f"MATCH (n:{self._q(label)} {{{self._q(key)}: v}}) "
            f"RETURN DISTINCT properties(n) AS node"
        )
        rows = self._client.execute(cypher, {"vals": values}, readonly=True)
        return list(rows)

    def get_node_with_neighbors(
        self,
        label: str,
        key: str,
        value: Any,
        *,
        direction: str = "both",  # 'out' | 'in' | 'both'
        rel_types: Optional[List[str]] = None,
        neighbor_labels: Optional[List[str]] = None,
        limit: Optional[int] = None,
        include_rel_props: bool = True,
    ) -> Optional[Dict[str, Any]]:
        self._validate_identifier(label)
        self._validate_identifier(key)
        # enforce unique node key for precise neighborhood retrieval
        self._ensure_unique_key(label, key)
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")
        if rel_types is not None:
            for rt in rel_types:
                self._validate_identifier(rt)
        if neighbor_labels is not None:
            for lb in neighbor_labels:
                self._validate_identifier(lb)

        # Build label filter using explicit label predicates to enable label index usage
        if neighbor_labels:
            label_predicate = " AND (" + " OR ".join([f"m:{self._q(lb)}" for lb in neighbor_labels]) + ")"
        else:
            label_predicate = ""

        cypher = (
            f"""
            MATCH (n:{self._q(label)} {{{self._q(key)}: $val}})
            WITH n
            WITH n,
                 [ (n)-[r]->(m)
                   WHERE ($rel_types IS NULL OR type(r) IN $rel_types){label_predicate}
                   | {{
                        direction: 'out',
                        rel: type(r),
                        rprops: CASE WHEN $include_rel_props THEN properties(r) ELSE {{}} END,
                        node: {{ labels: labels(m), props: properties(m) }}
                   }}
                 ] AS outs,
                 [ (m)-[r]->(n)
                   WHERE ($rel_types IS NULL OR type(r) IN $rel_types){label_predicate}
                   | {{
                        direction: 'in',
                        rel: type(r),
                        rprops: CASE WHEN $include_rel_props THEN properties(r) ELSE {{}} END,
                        node: {{ labels: labels(m), props: properties(m) }}
                   }}
                 ] AS ins,
                 properties(n) AS node
            WITH node,
                 (CASE WHEN $direction IN ['out','both'] THEN outs ELSE [] END) +
                 (CASE WHEN $direction IN ['in','both']  THEN ins  ELSE [] END) AS edges
            RETURN node, CASE WHEN $limit IS NULL THEN edges ELSE edges[0..$limit] END AS edges
            """
        )
        rec = self._client.execute_single(
            cypher,
            {
                "val": value,
                "direction": direction,
                "rel_types": rel_types,
                "neighbor_labels": neighbor_labels,
                "limit": limit,
                "include_rel_props": bool(include_rel_props),
            },
            readonly=True,
        )
        return rec

    def update_node(self, label: str, key: str, value: Any, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._validate_identifier(label)
        self._validate_identifier(key)
        self._ensure_unique_key(label, key)
        if value is None:
            raise ValueError(f"Cannot update node by {label}.{key}=None")
        if not isinstance(updates, dict) or not updates:
            raise ValueError("No updates provided")
        # Do not allow primary key update
        if key in updates:
            updates = {k: v for k, v in updates.items() if k != key}
        if not updates:
            raise ValueError("No updatable fields provided (primary key cannot be updated)")
        current = self.get_node(label, key, value)
        if not current or not isinstance(current.get("node"), dict):
            raise KRagError(f"Node not found for {label}.{key}={value}")
        # Validate by merging with current snapshot; use sanitized values for keys being updated only
        merged = {**current["node"], **updates}
        sanitized = self._validate_node_properties(label, merged)
        sanitized_updates = {k: sanitized.get(k) for k in updates.keys() if k != key}
        expected_snapshot = {k: current["node"].get(k) for k in sanitized_updates.keys()}
        sentinel = "__KRAG__NULL__SENTINEL__"
        cypher = (
            f"MATCH (n:{self._q(label)} {{{self._q(key)}: $val}}) "
            f"WITH n, $updates AS updates, $expected AS expected, $sentinel AS sentinel "
            f"WHERE ALL(k IN keys(updates) WHERE coalesce(properties(n)[k], sentinel) = coalesce(expected[k], sentinel)) "
            f"SET n += updates "
            f"RETURN properties(n) AS node"
        )
        params = {"val": value, "updates": sanitized_updates, "expected": expected_snapshot, "sentinel": sentinel}
        rec = self._client.execute_single(cypher, params, readonly=False)
        if not rec or not isinstance(rec.get("node"), dict):
            raise KRagError(
                f"Failed to update node {label}.{key}={value}; node may not exist or fields changed concurrently"
            )
        return rec

    def merge_nodes(self, label: str, *, key: str, items: List[Dict[str, Any]]) -> int:
        self._validate_identifier(label)
        self._validate_identifier(key)
        self._ensure_unique_key(label, key)
        rows: List[Dict[str, Any]] = []
        for it in items or []:
            if key not in it:
                raise DataValidationError(f"Primary key '{key}' missing from item")
            props = self._validate_node_properties(label, it)
            props_no_key = {k: v for k, v in props.items() if k != key}
            rows.append({"key": props[key], "propsNoKey": props_no_key})
        if not rows:
            return 0
        cypher = (
            f"UNWIND $rows AS row "
            f"MERGE (n:{self._q(label)} {{{self._q(key)}: row.key}}) "
            f"ON CREATE SET n += row.propsNoKey "
            f"WITH n, row.propsNoKey AS props "
            f"WITH n, props, [k IN keys(props) WHERE n[k] IS NULL] AS toSet "
            f"FOREACH (k IN toSet | SET n[k] = props[k]) "
            f"RETURN count(*) AS upserted"
        )
        rec = self._client.execute_single(cypher, {"rows": rows}, readonly=False)
        if not rec or not isinstance(rec.get("upserted"), int):
            raise KRagError("Batch merge failed; database returned no count")
        return int(rec["upserted"])

    def update_nodes(self, label: str, *, key: str, items: List[Dict[str, Any]]) -> int:
        self._validate_identifier(label)
        self._validate_identifier(key)
        if not items:
            return 0
        rows: List[Dict[str, Any]] = []
        # Validate per item by merging with current props, but only write provided fields
        for it in items:
            if key not in it:
                raise DataValidationError(f"Primary key '{key}' missing from item")
            k = it[key]
            self._ensure_unique_key(label, key)
            current = self.get_node(label, key, k)
            if not current or not isinstance(current.get("node"), dict):
                raise KRagError(f"Node not found for {label}.{key}={k}")
            updates_only = {kk: vv for kk, vv in it.items() if kk != key}
            if not updates_only:
                raise ValueError(f"No updatable fields provided for {label}.{key}={k}")
            merged = {**current["node"], **updates_only}
            sanitized = self._validate_node_properties(label, merged)
            sanitized_updates = {kk: sanitized.get(kk) for kk in updates_only.keys()}
            expected_snapshot = {kk: current["node"].get(kk) for kk in sanitized_updates.keys()}
            rows.append({"key": k, "updates": sanitized_updates, "expected": expected_snapshot})
        cypher = (
            f"UNWIND $rows AS row "
            f"MATCH (n:{self._q(label)} {{{self._q(key)}: row.key}}) "
            f"WITH n, row, $sentinel AS sentinel "
            f"WHERE ALL(k IN keys(row.updates) WHERE coalesce(properties(n)[k], sentinel) = coalesce(row.expected[k], sentinel)) "
            f"SET n += row.updates "
            f"RETURN 1 AS updated"
        )
        res = list(self._client.execute(cypher, {"rows": rows, "sentinel": "__KRAG__NULL__SENTINEL__"}, readonly=False))
        if len(res) != len(rows):
            raise KRagError(
                f"Updated {len(res)}/{len(rows)} nodes; some may not exist or were deleted concurrently"
            )
        return len(res)

    def update_nodes_by_property(self, label: str, key: str, value: Any, updates: Dict[str, Any]) -> int:
        self._validate_identifier(label)
        self._validate_identifier(key)
        if value is None:
            raise ValueError(f"Cannot update nodes by {label}.{key}=None")
        if not isinstance(updates, dict) or not updates:
            raise ValueError("No updates provided")
        # Do not allow updating the match key
        if key in updates:
            updates = {k: v for k, v in updates.items() if k != key}
        if not updates:
            raise ValueError("No updatable fields provided (match key cannot be updated)")
        # Validate update keys against model fields
        model = _NODE_MODELS.get(label)
        if model is None:
            raise ValueError(f"Node label '{label}' is not allowed. Supported labels: {list(_NODE_MODELS.keys())}")
        allowed_keys = set(getattr(model, "model_fields", {}).keys())
        unknown = [k for k in updates.keys() if k not in allowed_keys]
        if unknown:
            raise DataValidationError(f"Unknown properties for label {label}: {unknown}")
        cypher = (
            f"MATCH (n:{self._q(label)} {{{self._q(key)}: $val}}) "
            f"SET n += $updates "
            f"RETURN count(n) AS updated"
        )
        rec = self._client.execute_single(cypher, {"val": value, "updates": updates}, readonly=False)
        if not rec or not isinstance(rec.get("updated"), int):
            raise KRagError("Update by property failed; database returned no count")
        return int(rec.get("updated") or 0)
        
    def delete_node(self, label: str, key: str, value: Any) -> int:
        self._validate_identifier(label)
        self._validate_identifier(key)
        self._ensure_unique_key(label, key)
        if value is None:
            raise ValueError(f"Cannot delete node by {label}.{key}=None")
        cypher = (
            f"MATCH (n:{self._q(label)} {{{self._q(key)}: $val}}) "
            f"WITH collect(n) AS nodes "
            f"WITH nodes, size(nodes) AS cnt "
            f"CALL {{ WITH nodes, cnt "
            f"  WITH nodes WHERE cnt = 1 "
            f"  FOREACH (x IN nodes | DETACH DELETE x) "
            f"  RETURN 1 AS ok "
            f"}} "
            f"RETURN cnt AS matched"
        )
        rec = self._client.execute_single(cypher, {"val": value}, readonly=False)
        if not rec or not isinstance(rec.get("matched"), int):
            raise KRagError("Delete failed; database returned no match count")
        matched = int(rec.get("matched") or 0)
        if matched > 1:
            raise KRagError(
                f"Refusing to delete: found {matched} nodes for {label}.{key}={value}; fix duplicates first"
            )
        return 1 if matched == 1 else 0

    def delete_nodes(self, label: str, key: str, values: List[Any]) -> int:
        self._validate_identifier(label)
        self._validate_identifier(key)
        # Unique-only path; enforce unique key; refuse duplicates like delete_node
        self._ensure_unique_key(label, key)
        if not isinstance(values, list) or not values:
            return 0
        # Filter out None and deduplicate while preserving order
        seen: set = set()
        clean_values: List[Any] = []
        for v in values:
            if v is None:
                continue
            if v in seen:
                continue
            seen.add(v)
            clean_values.append(v)
        if not clean_values:
            return 0
        cypher = (
            f"UNWIND $vals AS v "
            f"OPTIONAL MATCH (n:{self._q(label)} {{{self._q(key)}: v}}) "
            f"WITH v, collect(n) AS nodes, size([x IN collect(n) WHERE x IS NOT NULL]) AS cnt "
            f"WITH collect({{v:v, nodes:nodes, cnt:cnt}}) AS rows "
            f"WITH rows, reduce(x=0, r IN rows | x + CASE WHEN r.cnt > 1 THEN 1 ELSE 0 END) AS dup_count "
            f"CALL {{ WITH rows, dup_count "
            f"  WITH rows WHERE dup_count = 0 "
            f"  UNWIND rows AS r "
            f"  WITH r WHERE r.cnt = 1 "
            f"  UNWIND r.nodes AS n "
            f"  DETACH DELETE n "
            f"  RETURN count(*) AS deleted_nodes "
            f"}} "
            f"RETURN dup_count AS dup_count, coalesce(deleted_nodes, 0) AS deleted_nodes"
        )
        rec = self._client.execute_single(cypher, {"vals": clean_values}, readonly=False)
        if not rec:
            raise KRagError("Batch delete failed; no result returned")
        dup_count = int(rec.get("dup_count") or 0)
        if dup_count > 0:
            raise KRagError(
                f"Refusing to delete: found duplicate nodes for some values of {label}.{key}; fix duplicates first"
            )
        return int(rec.get("deleted_nodes") or 0)

    def delete_nodes_by_property(self, label: str, key: str, values: List[Any]) -> int:
        self._validate_identifier(label)
        self._validate_identifier(key)
        if not isinstance(values, list) or not values:
            return 0
        # Filter out None and deduplicate while preserving order
        seen: set = set()
        clean_values: List[Any] = []
        for v in values:
            if v is None:
                continue
            if v in seen:
                continue
            seen.add(v)
            clean_values.append(v)
        if not clean_values:
            return 0
        cypher = (
            f"UNWIND $vals AS v "
            f"MATCH (n:{self._q(label)} {{{self._q(key)}: v}}) "
            f"WITH collect(n) AS nodes "
            f"UNWIND nodes AS n "
            f"DETACH DELETE n "
            f"RETURN count(*) AS deleted"
        )
        rec = self._client.execute_single(cypher, {"vals": clean_values}, readonly=False)
        if not rec or not isinstance(rec.get("deleted"), int):
            raise KRagError("Batch delete by property failed; database returned no count")
        return int(rec.get("deleted") or 0)

    # -------------------
    # Relationship CRUD
    # -------------------
    def create_relationship(
        self,
        from_label: str,
        from_key: str,
        from_value: Any,
        rel_type: str,
        to_label: str,
        to_key: str,
        to_value: Any,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        self._validate_identifier(rel_type)
        # endpoints must be uniquely addressed and exist
        self._ensure_unique_key(from_label, from_key)
        self._ensure_unique_key(to_label, to_key)
        rprops = self._validate_rel_properties(rel_type, properties or {})
        cypher = (
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: $fromVal}}) "
            f"MATCH (b:{self._q(to_label)} {{{self._q(to_key)}: $toVal}}) "
            f"CREATE (a)-[r:{self._q(rel_type)}]->(b) "
            f"SET r += $rprops "
            f"RETURN type(r) AS rel, properties(r) AS rprops"
        )
        rec = self._client.execute_single(
            cypher,
            {"fromVal": from_value, "toVal": to_value, "rprops": rprops},
            readonly=False,
        )
        if not rec:
            raise KRagError("Failed to create relationship; ensure both nodes exist")
        return rec

    def create_relationships(
        self,
        from_label: str,
        from_key: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        pairs: List[Dict[str, Any]],
    ) -> int:
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        self._validate_identifier(rel_type)
        # Enforce unique keys on both endpoints
        self._ensure_unique_key(from_label, from_key)
        self._ensure_unique_key(to_label, to_key)
        if not pairs:
            return 0
        rows: List[Dict[str, Any]] = []
        for p in pairs:
            fv = p.get("from_value")
            tv = p.get("to_value")
            rprops = self._validate_rel_properties(rel_type, p.get("properties") or {})
            rows.append({"fv": fv, "tv": tv, "props": rprops})

        # Atomic: detect missing endpoints and only create when all pairs are valid
        cypher = (
            f"UNWIND $rows AS row "
            f"OPTIONAL MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: row.fv}}) "
            f"OPTIONAL MATCH (b:{self._q(to_label)} {{{self._q(to_key)}: row.tv}}) "
            f"WITH row, a, b "
            f"WITH collect({{a:a, b:b, props: row.props, ok: a IS NOT NULL AND b IS NOT NULL}}) AS rs "
            f"WITH rs, reduce(m=0, r IN rs | m + CASE WHEN r.ok THEN 0 ELSE 1 END) AS missing "
            f"CALL {{ WITH rs, missing "
            f"  WITH rs WHERE missing = 0 "
            f"  UNWIND rs AS r "
            f"  WITH r WHERE r.ok "
            f"  CREATE (r.a)-[rel:{self._q(rel_type)}]->(r.b) "
            f"  SET rel += r.props "
            f"  RETURN count(*) AS created_inner "
            f"}} "
            f"RETURN missing AS missing, coalesce(created_inner, 0) AS created"
        )
        rec = self._client.execute_single(cypher, {"rows": rows}, readonly=False)
        if not rec:
            raise KRagError("Batch create relationships failed; no result returned")
        missing = int(rec.get("missing") or 0)
        if missing > 0:
            raise KRagError("Some pairs reference non-existent nodes; no relationships were created")
        created = int(rec.get("created") or 0)
        return created

    def create_relationships_by_property(
        self,
        from_label: str,
        from_key: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        pairs: List[Dict[str, Any]],
    ) -> int:
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        self._validate_identifier(rel_type)
        if not pairs:
            return 0
        rows: List[Dict[str, Any]] = []
        for p in pairs:
            fv = p.get("from_value")
            tv = p.get("to_value")
            rprops = self._validate_rel_properties(rel_type, p.get("properties") or {})
            rows.append({"fv": fv, "tv": tv, "props": rprops})
        # Partial: create for combinations whose endpoints exist; skip missing without failing whole batch
        cypher = (
            f"UNWIND $rows AS row "
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: row.fv}}) "
            f"MATCH (b:{self._q(to_label)} {{{self._q(to_key)}: row.tv}}) "
            f"CREATE (a)-[rel:{self._q(rel_type)}]->(b) "
            f"SET rel += row.props "
            f"RETURN count(rel) AS created"
        )
        rec = self._client.execute_single(cypher, {"rows": rows}, readonly=False)
        if not rec or not isinstance(rec.get("created"), int):
            raise KRagError("Batch create relationships by property failed; no count")
        return int(rec.get("created") or 0)

    def merge_relationship(
        self,
        from_label: str,
        from_key: str,
        from_value: Any,
        rel_type: str,
        to_label: str,
        to_key: str,
        to_value: Any,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        self._validate_identifier(rel_type)
        # Keys must be unique; do not create endpoints. If either side missing, skip silently.
        self._ensure_unique_key(from_label, from_key)
        self._ensure_unique_key(to_label, to_key)
        rprops = self._validate_rel_properties(rel_type, properties or {})
        cypher = (
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: $fromVal}}) "
            f"MATCH (b:{self._q(to_label)} {{{self._q(to_key)}: $toVal}}) "
            f"MERGE (a)-[r:{self._q(rel_type)}]->(b) "
            f"SET r += $rprops "
            f"RETURN type(r) AS rel, properties(r) AS rprops"
        )
        rec = self._client.execute_single(
            cypher,
            {"fromVal": from_value, "toVal": to_value, "rprops": rprops},
            readonly=False,
        )
        # If either endpoint missing, subquery returns no row → rec is None. Treat as skipped.
        return rec

    def merge_relationships(
        self,
        from_label: str,
        from_key: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        pairs: List[Dict[str, Any]],
    ) -> int:
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(rel_type)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        # Enforce unique keys on both endpoints
        self._ensure_unique_key(from_label, from_key)
        self._ensure_unique_key(to_label, to_key)
        if not pairs:
            return 0
        rows: List[Dict[str, Any]] = []
        for p in pairs:
            fv = p.get("from_value")
            tv = p.get("to_value")
            rprops = self._validate_rel_properties(rel_type, p.get("properties") or {})
            rows.append({"fv": fv, "tv": tv, "props": rprops})
        cypher = (
            f"UNWIND $rows AS row "
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: row.fv}}) "
            f"MATCH (b:{self._q(to_label)} {{{self._q(to_key)}: row.tv}}) "
            f"MERGE (a)-[r:{self._q(rel_type)}]->(b) "
            f"SET r += row.props "
            f"RETURN count(*) AS upserted"
        )
        rec = self._client.execute_single(cypher, {"rows": rows}, readonly=False)
        if not rec or not isinstance(rec.get("upserted"), int):
            raise KRagError("Batch merge relationships failed; no count")
        return int(rec.get("upserted") or 0)

    def merge_relationships_by_property(
        self,
        from_label: str,
        from_key: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        pairs: List[Dict[str, Any]],
    ) -> int:
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(rel_type)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        if not pairs:
            return 0
        rows: List[Dict[str, Any]] = []
        for p in pairs:
            fv = p.get("from_value")
            tv = p.get("to_value")
            rprops = self._validate_rel_properties(rel_type, p.get("properties") or {})
            rows.append({"fv": fv, "tv": tv, "props": rprops})
        cypher = (
            f"UNWIND $rows AS row "
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: row.fv}}) "
            f"MATCH (b:{self._q(to_label)} {{{self._q(to_key)}: row.tv}}) "
            f"MERGE (a)-[r:{self._q(rel_type)}]->(b) "
            f"SET r += row.props "
            f"RETURN count(*) AS upserted"
        )
        rec = self._client.execute_single(cypher, {"rows": rows}, readonly=False)
        if not rec or not isinstance(rec.get("upserted"), int):
            raise KRagError("Batch merge relationships by property failed; no count")
        return int(rec.get("upserted") or 0)

    def delete_relationship(
        self,
        from_label: str,
        from_key: str,
        from_value: Any,
        rel_type: str,
        to_label: str,
        to_key: str,
        to_value: Any,
        rel_props: Optional[Dict[str, Any]] = None,
    ) -> int:
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        self._validate_identifier(rel_type)
        # Enforce unique endpoints for precise single delete
        self._ensure_unique_key(from_label, from_key)
        self._ensure_unique_key(to_label, to_key)
        filters: List[str] = []
        params = {"fromVal": from_value, "toVal": to_value}
        if rel_props:
            for idx, (k, v) in enumerate(rel_props.items()):
                if not isinstance(k, str):
                    raise DataValidationError("Relationship property keys must be strings")
                pname = f"rp{idx}"
                filters.append(f"r.{self._q(k)} = ${pname}")
                params[pname] = v
        where_clause = (" WHERE " + " AND ".join(filters)) if filters else ""
        cypher = (
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: $fromVal}})"
            f"-[r:{self._q(rel_type)}]->"
            f"(b:{self._q(to_label)} {{{self._q(to_key)}: $toVal}})"
            f"{where_clause} "
            f"DELETE r "
            f"RETURN count(r) AS deleted"
        )
        rec = self._client.execute_single(
            cypher,
            params,
            readonly=False,
        )
        if not rec or not isinstance(rec.get("deleted"), int):
            raise KRagError("Delete relationship failed; no count")
        return int(rec.get("deleted") or 0)
        
    def delete_relationships(
        self,
        from_label: str,
        from_key: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        pairs: List[Dict[str, Any]],
        *,
        rel_types: Optional[List[str]] = None,
        rel_props: Optional[Dict[str, Any]] = None,
    ) -> int:
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(rel_type)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        # Enforce unique keys on both endpoints
        self._ensure_unique_key(from_label, from_key)
        self._ensure_unique_key(to_label, to_key)
        if not pairs:
            return 0
        # normalize and validate types list
        if rel_types is not None:
            if not rel_types:
                raise DataValidationError("rel_types cannot be empty")
            for rt in rel_types:
                self._validate_identifier(rt)
            types_param = rel_types
        else:
            types_param = [rel_type]
        # build and validate rows
        rows: List[Dict[str, Any]] = []
        for p in pairs:
            fv = p.get("from_value")
            tv = p.get("to_value")
            if fv is None or tv is None:
                raise DataValidationError("from_value/to_value must be provided and non-null")
            rows.append({"fv": fv, "tv": tv})
        filters = []
        params = {"rows": rows, "types": types_param}
        if rel_props:
            for idx, (k, v) in enumerate(rel_props.items()):
                if not isinstance(k, str):
                    raise DataValidationError("Relationship property keys must be strings")
                pname = f"rp{idx}"
                filters.append(f"r.{self._q(k)} = ${pname}")
                params[pname] = v
        where_props = (" AND " + " AND ".join(filters)) if filters else ""
        cypher = (
            f"UNWIND $rows AS row "
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: row.fv}}) "
            f"MATCH (b:{self._q(to_label)} {{{self._q(to_key)}: row.tv}}) "
            f"MATCH (a)-[r]->(b) "
            f"WHERE type(r) IN $types{where_props} "
            f"DELETE r "
            f"RETURN count(r) AS deleted"
        )
        rec = self._client.execute_single(cypher, params, readonly=False)
        if not rec or not isinstance(rec.get("deleted"), int):
            raise KRagError("Batch delete relationships failed; no count")
        return int(rec.get("deleted") or 0)

    def delete_relationships_by_property(
        self,
        from_label: str,
        from_key: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        pairs: List[Dict[str, Any]],
        *,
        rel_types: Optional[List[str]] = None,
        rel_props: Optional[Dict[str, Any]] = None,
    ) -> int:
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(rel_type)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        if not pairs:
            return 0
        # normalize and validate types list
        if rel_types is not None:
            if not rel_types:
                raise DataValidationError("rel_types cannot be empty")
            for rt in rel_types:
                self._validate_identifier(rt)
            types_param = rel_types
        else:
            types_param = [rel_type]
        # build and validate rows
        rows: List[Dict[str, Any]] = []
        for p in pairs:
            fv = p.get("from_value")
            tv = p.get("to_value")
            if fv is None or tv is None:
                raise DataValidationError("from_value/to_value must be provided and non-null")
            rows.append({"fv": fv, "tv": tv})
        filters = []
        params = {"rows": rows, "types": types_param}
        if rel_props:
            for idx, (k, v) in enumerate(rel_props.items()):
                if not isinstance(k, str):
                    raise DataValidationError("Relationship property keys must be strings")
                pname = f"rp{idx}"
                filters.append(f"r.{self._q(k)} = ${pname}")
                params[pname] = v
        where_props = (" AND " + " AND ".join(filters)) if filters else ""
        cypher = (
            f"UNWIND $rows AS row "
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: row.fv}}) "
            f"MATCH (b:{self._q(to_label)} {{{self._q(to_key)}: row.tv}}) "
            f"MATCH (a)-[r]->(b) "
            f"WHERE type(r) IN $types{where_props} "
            f"DELETE r "
            f"RETURN count(r) AS deleted"
        )
        rec = self._client.execute_single(cypher, params, readonly=False)
        if not rec or not isinstance(rec.get("deleted"), int):
            raise KRagError("Batch delete relationships by property failed; no count")
        return int(rec.get("deleted") or 0)

    def delete_all_relationships(
        self,
        from_label: str,
        from_key: str,
        from_value: Any,
        to_label: str,
        to_key: str,
        to_value: Any,
        *,
        both_directions: bool = False,
    ) -> int:
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        # unique endpoints
        self._ensure_unique_key(from_label, from_key)
        self._ensure_unique_key(to_label, to_key)
        cypher = (
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: $fromVal}}), "
            f"      (b:{self._q(to_label)} {{{self._q(to_key)}: $toVal}}) "
            f"OPTIONAL MATCH (a)-[r_out]->(b) "
            f"WITH a,b, collect(r_out) AS out_rels "
            f"FOREACH (x IN out_rels | DELETE x) "
            f"WITH a,b, size(out_rels) AS c_out "
            f"OPTIONAL MATCH (b)-[r_in]->(a) "
            f"WHERE $both "
            f"WITH c_out, collect(r_in) AS in_rels "
            f"FOREACH (x IN in_rels | DELETE x) "
            f"RETURN c_out + size(in_rels) AS deleted"
        )
        rec = self._client.execute_single(
            cypher,
            {"fromVal": from_value, "toVal": to_value, "both": bool(both_directions)},
            readonly=False,
        )
        if not rec or not isinstance(rec.get("deleted"), int):
            raise KRagError("Delete all relationships failed; no count")
        return int(rec.get("deleted") or 0)

    def delete_all_relationships_by_property(
        self,
        from_label: str,
        from_key: str,
        from_value: Any,
        to_label: str,
        to_key: str,
        to_value: Any,
        *,
        both_directions: bool = False,
    ) -> int:
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        cypher = (
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: $fromVal}}), "
            f"      (b:{self._q(to_label)} {{{self._q(to_key)}: $toVal}}) "
            f"OPTIONAL MATCH (a)-[r_out]->(b) "
            f"WITH a,b, collect(r_out) AS out_rels "
            f"FOREACH (x IN out_rels | DELETE x) "
            f"WITH a,b, size(out_rels) AS c_out "
            f"OPTIONAL MATCH (b)-[r_in]->(a) "
            f"WHERE $both "
            f"WITH c_out, collect(r_in) AS in_rels "
            f"FOREACH (x IN in_rels | DELETE x) "
            f"RETURN c_out + size(in_rels) AS deleted"
        )
        rec = self._client.execute_single(
            cypher,
            {"fromVal": from_value, "toVal": to_value, "both": bool(both_directions)},
            readonly=False,
        )
        if not rec or not isinstance(rec.get("deleted"), int):
            raise KRagError("Delete all relationships by property failed; no count")
        return int(rec.get("deleted") or 0)


    def get_relationships(
        self,
        from_label: str,
        from_key: str,
        from_value: Any,
        to_label: str,
        to_key: str,
        to_value: Any,
        *,
        rel_types: Optional[List[str]] = None,
        include_rel_props: bool = True,
    ) -> List[Dict[str, Any]]:
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        # enforce endpoints are uniquely addressable
        self._ensure_unique_key(from_label, from_key)
        self._ensure_unique_key(to_label, to_key)
        # validate types list if provided
        if rel_types is not None:
            if not rel_types:
                raise DataValidationError("rel_types cannot be empty")
            for rt in rel_types:
                self._validate_identifier(rt)
        cypher = (
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: $fromVal}}), "
            f"      (b:{self._q(to_label)} {{{self._q(to_key)}: $toVal}}) "
            f"MATCH (a)-[r]->(b) "
            f"WHERE ($types IS NULL OR type(r) IN $types) "
            f"RETURN type(r) AS rel, "
            f"       CASE WHEN $include THEN properties(r) ELSE {{}} END AS rprops, "
            f"       properties(a) AS from_node, properties(b) AS to_node"
        )
        params = {
            "fromVal": from_value,
            "toVal": to_value,
            "types": rel_types,
            "include": bool(include_rel_props),
        }
        rows = self._client.execute(cypher, params, readonly=True)
        return list(rows)

    def get_all_relationships(
        self,
        from_label: str,
        from_key: str,
        from_value: Any,
        to_label: str,
        to_key: str,
        to_value: Any,
        *,
        include_rel_props: bool = True,
    ) -> List[Dict[str, Any]]:
        # convenience wrapper to fetch all relationship types between two uniquely-addressed nodes
        return self.get_relationships(
            from_label,
            from_key,
            from_value,
            to_label,
            to_key,
            to_value,
            rel_types=None,
            include_rel_props=include_rel_props,
        )

