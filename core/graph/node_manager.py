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

    """
    节点与关系的 CRUD 统一管理器（基于 Neo4j Cypher）。

    设计要点（重要）：
    
    - 单项操作（get_node/merge_node/update_node/delete_node 等）要求 `(label.key)` 为唯一约束，避免返回多行导致的语义不确定。
    - 批量/按属性变体（…_by_property）不要求唯一性，可能命中大量实体，需谨慎使用并控制范围/批大小。
    - 并发控制：update_node / update_nodes 使用乐观并发控制（OCC），通过快照比对避免“丢失更新”。
    - merge_node / merge_nodes 采用“仅填充空字段”的软 Upsert 语义，避免盲目覆盖已有值。
    - 性能提示：
      - 高度连接节点在 get_node_with_neighbors 时可能返回大量边，建议使用 rel_types、neighbor_labels、limit 进行裁剪；
      - include_rel_props=True 会返回关系全部属性，可能放大结果体积；
      - by_property 变体（如 update_nodes_by_property、create_relationships_by_property）可能触发大范围扫描/更新，请仅在可控范围内使用。
    """

    def __init__(self, client: Neo4jClient) -> None:
        """
        初始化节点与关系管理器。

        Args:
            client: 已连接的 Neo4jClient。
        """
        self._client = client
        self._unique_key_cache: Dict[str, Dict[str, bool]] = {}

    @staticmethod
    def _q(identifier: str) -> str:
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


    async def _is_unique_key(self, label: str, key: str) -> bool:
        cache_for_label = self._unique_key_cache.setdefault(label, {})
        if key in cache_for_label:
            return cache_for_label[key]
        rows = await self._client.execute(
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

    async def _ensure_unique_key(self, label: str, key: str) -> None:
        if not await self._is_unique_key(label, key):
            raise KRagError(f"Property '{label}.{key}' is not unique; use multi-item variants (e.g., get_nodes)")


    async def merge_node(self, label: str, *, key: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        软 Upsert：仅填充空字段，避免覆盖已有值。

        Args:
            label: 节点标签（安全标识符）。
            key: 唯一键属性名（要求存在唯一约束）。
            properties: 节点属性字典，必须包含主键 `key`。

        Returns:
            包含 `node`(dict) 与 `applied`(List[str]) 的字典。

        Raises:
            ValueError: 标识符不合法或缺少主键。
            DataValidationError: 节点属性经 Pydantic 校验失败。
            KRagError: 数据库未返回记录等异常。

        Notes:
            - 并发：基于当前快照的“仅填空”判断，在极端并发下仍可能存在覆盖竞争；
              建议结合幂等/重试策略。
            - 请勿在 `properties` 中修改主键值；主键仅用于定位。
        """
        self._validate_identifier(label)
        self._validate_identifier(key)
        await self._ensure_unique_key(label, key)
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
        rec = await self._client.execute_single(
            cypher,
            {"keyVal": props[key], "propsNoKey": props_no_key},
            readonly=False,
        )
        if not rec or not isinstance(rec.get("node"), dict):
            raise KRagError("Failed to merge node; no record returned from database")
        return rec

    async def get_node(self, label: str, key: str, value: Any) -> Optional[Dict[str, Any]]:
        """
        按唯一键获取单个节点。

        Args:
            label: 节点标签（安全标识符）。
            key: 唯一键属性名（需唯一约束）。
            value: 唯一键取值（不可为 None）。

        Returns:
            {"node": dict} 或 None（未找到）。

        Raises:
            ValueError: 当 value 为 None 或标识符不合法。
            KRagError: 当键不具备唯一性要求时。
        """
        if value is None:
            raise ValueError(f"Cannot query node by {key}=None")
        self._validate_identifier(label)
        self._validate_identifier(key)
        await self._ensure_unique_key(label, key)
        cypher = (
            f"MATCH (n:{self._q(label)} {{{self._q(key)}: $val}}) "
            f"RETURN properties(n) AS node LIMIT 1"
        )
        return await self._client.execute_single(cypher, {"val": value}, readonly=True)

    async def get_nodes(self, label: str, key: str, values: List[Any]) -> List[Dict[str, Any]]:
        """
        批量按键获取节点（键可非唯一）。

        Args:
            label: 节点标签。
            key: 匹配属性名。
            values: 属性取值列表（会过滤 None）。

        Returns:
            列表：[ {"node": dict}, ... ]，已去重。

        Raises:
            ValueError: values 为空或全为 None，或标识符不合法。
        Notes:
            - 请控制 `values` 数量，避免过大 UNWIND 造成内存压力。
        """
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
        rows = await self._client.execute(cypher, {"vals": values}, readonly=True)
        return list(rows)

    async def get_node_with_neighbors(
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
        """
        获取节点与邻居（支持方向与类型筛选）。

        Args:
            label: 节点标签（需唯一键）。
            key: 唯一键属性名。
            value: 唯一键取值。
            direction: 邻接方向，'out' | 'in' | 'both'。
            rel_types: 仅返回这些关系类型；None 表示不限。
            neighbor_labels: 仅返回这些邻居标签；None 表示不限。
            limit: 返回的边数量上限（在聚合后切片）。
            include_rel_props: 是否包含关系属性。

        Returns:
            {"node": dict, "edges": [ {direction, rel, rprops, node{labels, props}}, ... ]} 或 None。

        Raises:
            ValueError: 参数不合法（方向/标识符）。
            KRagError: 唯一性要求不满足。

        Notes:
            - 高度连接节点可能产生大量结果；请结合 `rel_types`、`neighbor_labels` 与 `limit` 限制范围；
            - `limit` 在服务端聚合后切片，超大结果仍可能有内存压力；
            - 将 `include_rel_props=False` 可减小返回体积。
        """
        self._validate_identifier(label)
        self._validate_identifier(key)
        await self._ensure_unique_key(label, key)
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")
        if rel_types is not None:
            for rt in rel_types:
                self._validate_identifier(rt)
        if neighbor_labels is not None:
            for lb in neighbor_labels:
                self._validate_identifier(lb)

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
        rec = await self._client.execute_single(
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

    async def update_node(self, label: str, key: str, value: Any, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        条件更新（OCC，乐观并发控制）。

        Args:
            label: 节点标签（唯一键）。
            key: 唯一键属性名。
            value: 唯一键取值。
            updates: 待更新字段（禁止包含主键）。

        Returns:
            {"node": dict}，更新后的节点属性。

        Raises:
            ValueError: 参数不合法或无可更新字段。
            DataValidationError: 属性校验失败。
            KRagError: 未找到节点或并发冲突（快照不匹配）。
        """
        self._validate_identifier(label)
        self._validate_identifier(key)
        await self._ensure_unique_key(label, key)
        if value is None:
            raise ValueError(f"Cannot update node by {label}.{key}=None")
        if not isinstance(updates, dict) or not updates:
            raise ValueError("No updates provided")
        if key in updates:
            updates = {k: v for k, v in updates.items() if k != key}
        if not updates:
            raise ValueError("No updatable fields provided (primary key cannot be updated)")
        current = await self.get_node(label, key, value)
        if not current or not isinstance(current.get("node"), dict):
            raise KRagError(f"Node not found for {label}.{key}={value}")
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
        rec = await self._client.execute_single(cypher, params, readonly=False)
        if not rec or not isinstance(rec.get("node"), dict):
            raise KRagError(
                f"Failed to update node {label}.{key}={value}; node may not exist or fields changed concurrently"
            )
        return rec

    async def merge_nodes(self, label: str, *, key: str, items: List[Dict[str, Any]]) -> int:
        """
        批量软 Upsert：语义与 merge_node 一致（仅填充空字段）。

        Args:
            label: 节点标签（唯一键）。
            key: 唯一键属性名。
            items: 每项包含主键与属性的字典。

        Returns:
            成功 upsert 的条数。

        Raises:
            DataValidationError: 缺少主键或属性校验失败。
            KRagError: 数据库未返回计数。
        """
        self._validate_identifier(label)
        self._validate_identifier(key)
        await self._ensure_unique_key(label, key)
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
        rec = await self._client.execute_single(cypher, {"rows": rows}, readonly=False)
        if not rec or not isinstance(rec.get("upserted"), int):
            raise KRagError("Batch merge failed; database returned no count")
        return int(rec["upserted"])

    async def update_nodes(self, label: str, *, key: str, items: List[Dict[str, Any]]) -> int:
        """
        批量条件更新（OCC）。

        Args:
            label: 节点标签（唯一键）。
            key: 唯一键属性名。
            items: 每项包含主键与更新字段的字典。

        Returns:
            成功更新的条数（全部成功），否则抛错。

        Raises:
            ValueError: 缺少可更新字段。
            DataValidationError: 属性校验失败。
            KRagError: 任意条未找到或并发冲突（快照不匹配）。
        Notes:
            - 建议控制批大小，避免长事务导致锁竞争。
        """
        self._validate_identifier(label)
        self._validate_identifier(key)
        if not items:
            return 0
        rows: List[Dict[str, Any]] = []
        for it in items:
            if key not in it:
                raise DataValidationError(f"Primary key '{key}' missing from item")
            k = it[key]
            await self._ensure_unique_key(label, key)
            current = await self.get_node(label, key, k)
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
        rows_result = await self._client.execute(cypher, {"rows": rows, "sentinel": "__KRAG__NULL__SENTINEL__"}, readonly=False)
        res = list(rows_result)
        if len(res) != len(rows):
            raise KRagError(
                f"Updated {len(res)}/{len(rows)} nodes; some may not exist or were deleted concurrently"
            )
        return len(res)

    async def update_nodes_by_property(self, label: str, key: str, value: Any, updates: Dict[str, Any]) -> int:
        """
        按属性批量更新（非唯一键场景）。

        Args:
            label: 节点标签。
            key: 匹配属性名（可非唯一）。
            value: 匹配值（不可为 None）。
            updates: 更新字段（不可包含匹配键）。

        Returns:
            实际更新的节点数量。

        Raises:
            ValueError: 参数不合法或无可更新字段。
            DataValidationError: 存在未知字段（不在模型定义中）。
            KRagError: 数据库未返回计数。

        Notes:
            - 可能命中大量节点，属于“放大写入”；
            - 无 OCC 保护，适合作业/离线小范围修正；
            - 建议在上层限制选择性或分批执行。
        """
        self._validate_identifier(label)
        self._validate_identifier(key)
        if value is None:
            raise ValueError(f"Cannot update nodes by {label}.{key}=None")
        if not isinstance(updates, dict) or not updates:
            raise ValueError("No updates provided")
        if key in updates:
            updates = {k: v for k, v in updates.items() if k != key}
        if not updates:
            raise ValueError("No updatable fields provided (match key cannot be updated)")
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
        rec = await self._client.execute_single(cypher, {"val": value, "updates": updates}, readonly=False)
        if not rec or not isinstance(rec.get("updated"), int):
            raise KRagError("Update by property failed; database returned no count")
        return int(rec.get("updated") or 0)
        
    async def delete_node(self, label: str, key: str, value: Any) -> int:
        """
        删除单个节点（唯一键）。

        Args:
            label: 节点标签（唯一键）。
            key: 唯一键属性名。
            value: 唯一键取值。

        Returns:
            实际删除的数量（0 或 1）。

        Raises:
            ValueError: value 为空或标识符不合法。
            KRagError: 发现重复节点时拒绝删除。
        """
        self._validate_identifier(label)
        self._validate_identifier(key)
        await self._ensure_unique_key(label, key)
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
        rec = await self._client.execute_single(cypher, {"val": value}, readonly=False)
        if not rec or not isinstance(rec.get("matched"), int):
            raise KRagError("Delete failed; database returned no match count")
        matched = int(rec.get("matched") or 0)
        if matched > 1:
            raise KRagError(
                f"Refusing to delete: found {matched} nodes for {label}.{key}={value}; fix duplicates first"
            )
        return 1 if matched == 1 else 0

    async def delete_nodes(self, label: str, key: str, values: List[Any]) -> int:
        """
        批量删除（唯一键）。

        Args:
            label: 节点标签（唯一键）。
            key: 唯一键属性名。
            values: 待删除的唯一键取值列表。

        Returns:
            实际删除的数量。

        Raises:
            KRagError: 任意键存在重复节点时拒绝删除。
        Notes:
            - 会对输入进行去重与过滤 None。
        """
        self._validate_identifier(label)
        self._validate_identifier(key)
        await self._ensure_unique_key(label, key)
        if not isinstance(values, list) or not values:
            return 0
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
        rec = await self._client.execute_single(cypher, {"vals": clean_values}, readonly=False)
        if not rec:
            raise KRagError("Batch delete failed; no result returned")
        dup_count = int(rec.get("dup_count") or 0)
        if dup_count > 0:
            raise KRagError(
                f"Refusing to delete: found duplicate nodes for some values of {label}.{key}; fix duplicates first"
            )
        return int(rec.get("deleted_nodes") or 0)

    async def delete_nodes_by_property(self, label: str, key: str, values: List[Any]) -> int:
        """
        按属性批量删除（非唯一键）。

        Args:
            label: 节点标签。
            key: 匹配属性名（可非唯一）。
            values: 匹配值列表（会去重与过滤 None）。

        Returns:
            实际删除的节点数量。

        Raises:
            KRagError: 数据库未返回计数。

        Notes:
            - DETACH DELETE 会删除节点及其所有关系，属于破坏性操作；
            - 可能删除大量节点，务必范围可控并做好备份。
        """
        self._validate_identifier(label)
        self._validate_identifier(key)
        if not isinstance(values, list) or not values:
            return 0
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
        rec = await self._client.execute_single(cypher, {"vals": clean_values}, readonly=False)
        if not rec or not isinstance(rec.get("deleted"), int):
            raise KRagError("Batch delete by property failed; database returned no count")
        return int(rec.get("deleted") or 0)

    async def create_relationship(
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
        """
        创建单条关系（端点存在且唯一）。

        Args:
            from_label: 源端标签（唯一键）。
            from_key: 源端唯一键名。
            from_value: 源端唯一键值。
            rel_type: 关系类型。
            to_label: 目标端标签（唯一键）。
            to_key: 目标端唯一键名。
            to_value: 目标端唯一键值。
            properties: 关系属性字典（按模型校验）。

        Returns:
            {"rel": 类型, "rprops": 关系属性}。

        Raises:
            ValueError/DataValidationError: 标识符或属性校验失败。
            KRagError: 端点不存在或数据库未返回记录。
        """
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        self._validate_identifier(rel_type)
        await self._ensure_unique_key(from_label, from_key)
        await self._ensure_unique_key(to_label, to_key)
        rprops = self._validate_rel_properties(rel_type, properties or {})
        cypher = (
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: $fromVal}}) "
            f"MATCH (b:{self._q(to_label)} {{{self._q(to_key)}: $toVal}}) "
            f"CREATE (a)-[r:{self._q(rel_type)}]->(b) "
            f"SET r += $rprops "
            f"RETURN type(r) AS rel, properties(r) AS rprops"
        )
        rec = await self._client.execute_single(
            cypher,
            {"fromVal": from_value, "toVal": to_value, "rprops": rprops},
            readonly=False,
        )
        if not rec:
            raise KRagError("Failed to create relationship; ensure both nodes exist")
        return rec

    async def create_relationships(
        self,
        from_label: str,
        from_key: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        pairs: List[Dict[str, Any]],
    ) -> int:
        """
        批量创建关系（端点唯一且必须存在，原子化）。

        Args:
            from_label: 源端标签（唯一键）。
            from_key: 源端唯一键名。
            rel_type: 关系类型。
            to_label: 目标端标签（唯一键）。
            to_key: 目标端唯一键名。
            pairs: 列表，每项包含 {from_value, to_value, properties}。

        Returns:
            实际创建的关系条数。

        Raises:
            DataValidationError: 属性校验失败。
            KRagError: 任一对端点缺失导致整批不创建。
        Notes:
            - 保证“要么全成，要么全不成”。
        """
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        self._validate_identifier(rel_type)
        await self._ensure_unique_key(from_label, from_key)
        await self._ensure_unique_key(to_label, to_key)
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
        rec = await self._client.execute_single(cypher, {"rows": rows}, readonly=False)
        if not rec:
            raise KRagError("Batch create relationships failed; no result returned")
        missing = int(rec.get("missing") or 0)
        if missing > 0:
            raise KRagError("Some pairs reference non-existent nodes; no relationships were created")
        created = int(rec.get("created") or 0)
        return created

    async def create_relationships_by_property(
        self,
        from_label: str,
        from_key: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        pairs: List[Dict[str, Any]],
    ) -> int:
        """
        按属性批量创建关系（非唯一，允许部分成功）。

        Args:
            from_label: 源端标签。
            from_key: 源端匹配键（可非唯一）。
            rel_type: 关系类型。
            to_label: 目标端标签。
            to_key: 目标端匹配键（可非唯一）。
            pairs: 列表，每项包含 {from_value, to_value, properties}。

        Returns:
            创建成功的关系条数。

        Raises:
            DataValidationError: 属性校验失败。
            KRagError: 数据库未返回计数。

        Notes:
            - 每对 (from_value, to_value) 可匹配多节点，实际创建数为 A×B（笛卡尔乘积），存在“爆炸”风险；
            - 建议限制选择性并控制批次规模。
        """
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
        cypher = (
            f"UNWIND $rows AS row "
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: row.fv}}) "
            f"MATCH (b:{self._q(to_label)} {{{self._q(to_key)}: row.tv}}) "
            f"CREATE (a)-[rel:{self._q(rel_type)}]->(b) "
            f"SET rel += row.props "
            f"RETURN count(rel) AS created"
        )
        rec = await self._client.execute_single(cypher, {"rows": rows}, readonly=False)
        if not rec or not isinstance(rec.get("created"), int):
            raise KRagError("Batch create relationships by property failed; no count")
        return int(rec.get("created") or 0)

    async def merge_relationship(
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
        """
        合并单条关系（端点唯一）。

        Args:
            from_label/from_key/from_value: 源端（唯一）。
            rel_type: 关系类型。
            to_label/to_key/to_value: 目标端（唯一）。
            properties: 关系属性。

        Returns:
            {"rel": 类型, "rprops": 属性}；若端点不存在，返回 None（静默跳过）。

        Raises:
            ValueError/DataValidationError: 标识符或属性校验失败。
        Notes:
            - 不创建端点节点，仅在两端都存在时进行 MERGE。
        """
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        self._validate_identifier(rel_type)
        await self._ensure_unique_key(from_label, from_key)
        await self._ensure_unique_key(to_label, to_key)
        rprops = self._validate_rel_properties(rel_type, properties or {})
        cypher = (
            f"MATCH (a:{self._q(from_label)} {{{self._q(from_key)}: $fromVal}}) "
            f"MATCH (b:{self._q(to_label)} {{{self._q(to_key)}: $toVal}}) "
            f"MERGE (a)-[r:{self._q(rel_type)}]->(b) "
            f"SET r += $rprops "
            f"RETURN type(r) AS rel, properties(r) AS rprops"
        )
        rec = await self._client.execute_single(
            cypher,
            {"fromVal": from_value, "toVal": to_value, "rprops": rprops},
            readonly=False,
        )
        return rec

    async def merge_relationships(
        self,
        from_label: str,
        from_key: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        pairs: List[Dict[str, Any]],
    ) -> int:
        """
        批量合并关系（端点唯一且必须存在）。

        Args:
            from_label/from_key: 源端（唯一）。
            rel_type: 关系类型。
            to_label/to_key: 目标端（唯一）。
            pairs: 列表，每项 {from_value, to_value, properties}。

        Returns:
            upsert 的关系数量。

        Raises:
            DataValidationError: 属性校验失败。
            KRagError: 数据库未返回计数。
        """
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(rel_type)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        await self._ensure_unique_key(from_label, from_key)
        await self._ensure_unique_key(to_label, to_key)
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
        rec = await self._client.execute_single(cypher, {"rows": rows}, readonly=False)
        if not rec or not isinstance(rec.get("upserted"), int):
            raise KRagError("Batch merge relationships failed; no count")
        return int(rec.get("upserted") or 0)

    async def merge_relationships_by_property(
        self,
        from_label: str,
        from_key: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        pairs: List[Dict[str, Any]],
    ) -> int:
        """
        按属性批量合并关系（非唯一）。

        Args:
            from_label/from_key: 源端匹配键（可非唯一）。
            rel_type: 关系类型。
            to_label/to_key: 目标端匹配键（可非唯一）。
            pairs: 列表，每项 {from_value, to_value, properties}。

        Returns:
            upsert 的关系数量。

        Raises:
            DataValidationError: 属性校验失败。
            KRagError: 数据库未返回计数。

        Notes:
            - 可能出现 A×B 放大；限制选择性或分批执行。
        """
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
        rec = await self._client.execute_single(cypher, {"rows": rows}, readonly=False)
        if not rec or not isinstance(rec.get("upserted"), int):
            raise KRagError("Batch merge relationships by property failed; no count")
        return int(rec.get("upserted") or 0)

    async def delete_relationship(
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
        """
        删除单条关系（端点唯一）。

        Args:
            from_label/from_key/from_value: 源端（唯一）。
            rel_type: 关系类型。
            to_label/to_key/to_value: 目标端（唯一）。
            rel_props: 关系属性过滤（全部匹配时删除）。

        Returns:
            实际删除的数量（0/1+）。

        Raises:
            DataValidationError: 属性键非字符串。
            KRagError: 数据库未返回计数。
        """
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        self._validate_identifier(rel_type)
        await self._ensure_unique_key(from_label, from_key)
        await self._ensure_unique_key(to_label, to_key)
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
        rec = await self._client.execute_single(
            cypher,
            params,
            readonly=False,
        )
        if not rec or not isinstance(rec.get("deleted"), int):
            raise KRagError("Delete relationship failed; no count")
        return int(rec.get("deleted") or 0)
        
    async def delete_relationships(
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
        """
        批量删除关系（端点唯一）。

        Args:
            from_label/from_key: 源端（唯一）。
            rel_type: 默认关系类型；当 rel_types 为空时使用。
            to_label/to_key: 目标端（唯一）。
            pairs: 每项 {from_value, to_value}。
            rel_types: 多关系类型列表。
            rel_props: 关系属性过滤。

        Returns:
            删除的关系数量。

        Raises:
            DataValidationError: rel_types 为空或属性键非字符串。
            KRagError: 数据库未返回计数。
        """
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(rel_type)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        await self._ensure_unique_key(from_label, from_key)
        await self._ensure_unique_key(to_label, to_key)
        if not pairs:
            return 0
        if rel_types is not None:
            if not rel_types:
                raise DataValidationError("rel_types cannot be empty")
            for rt in rel_types:
                self._validate_identifier(rt)
            types_param = rel_types
        else:
            types_param = [rel_type]
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
        rec = await self._client.execute_single(cypher, params, readonly=False)
        if not rec or not isinstance(rec.get("deleted"), int):
            raise KRagError("Batch delete relationships failed; no count")
        return int(rec.get("deleted") or 0)

    async def delete_relationships_by_property(
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
        """
        按属性批量删除关系（非唯一）。

        Args:
            from_label/from_key: 源端匹配键（可非唯一）。
            rel_type: 默认关系类型；当 rel_types 为空时使用。
            to_label/to_key: 目标端匹配键（可非唯一）。
            pairs: 每项 {from_value, to_value}。
            rel_types: 多关系类型列表。
            rel_props: 关系属性过滤。

        Returns:
            删除的关系数量。

        Raises:
            DataValidationError: rel_types 为空或属性键非字符串。
            KRagError: 数据库未返回计数。

        Notes:
            - 可能命中大量关系并删除；务必控制选择性。
        """
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(rel_type)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        if not pairs:
            return 0
        if rel_types is not None:
            if not rel_types:
                raise DataValidationError("rel_types cannot be empty")
            for rt in rel_types:
                self._validate_identifier(rt)
            types_param = rel_types
        else:
            types_param = [rel_type]
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
        rec = await self._client.execute_single(cypher, params, readonly=False)
        if not rec or not isinstance(rec.get("deleted"), int):
            raise KRagError("Batch delete relationships by property failed; no count")
        return int(rec.get("deleted") or 0)

    async def delete_all_relationships(
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
        """
        删除两节点之间的所有关系（端点唯一）。

        Args:
            from_label/from_key/from_value: 源端（唯一）。
            to_label/to_key/to_value: 目标端（唯一）。
            both_directions: 为 True 时同时删除 b->a。

        Returns:
            删除的关系数量。

        Raises:
            KRagError: 数据库未返回计数。
        """
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        await self._ensure_unique_key(from_label, from_key)
        await self._ensure_unique_key(to_label, to_key)
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
        rec = await self._client.execute_single(
            cypher,
            {"fromVal": from_value, "toVal": to_value, "both": bool(both_directions)},
            readonly=False,
        )
        if not rec or not isinstance(rec.get("deleted"), int):
            raise KRagError("Delete all relationships failed; no count")
        return int(rec.get("deleted") or 0)

    async def delete_all_relationships_by_property(
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
        """
        按属性删除“所有匹配的两端对”之间的所有关系（非唯一）。

        Args:
            from_label/from_key/from_value: 源端匹配键与值（可非唯一）。
            to_label/to_key/to_value: 目标端匹配键与值（可非唯一）。
            both_directions: 为 True 时同时删除 b->a。

        Returns:
            删除的关系数量。

        Raises:
            KRagError: 数据库未返回计数。

        Notes:
            - 匹配可能产出多对节点，删除量可能远超预期；务必严格控制选择性。
        """
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
        rec = await self._client.execute_single(
            cypher,
            {"fromVal": from_value, "toVal": to_value, "both": bool(both_directions)},
            readonly=False,
        )
        if not rec or not isinstance(rec.get("deleted"), int):
            raise KRagError("Delete all relationships by property failed; no count")
        return int(rec.get("deleted") or 0)


    async def get_relationships(
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
        """
        获取两端点之间的关系列表（端点唯一）。

        Args:
            from_label/from_key/from_value: 源端（唯一）。
            to_label/to_key/to_value: 目标端（唯一）。
            rel_types: 仅返回这些类型；None 表示不限。
            include_rel_props: 是否包含关系属性。

        Returns:
            列表：每项包含 {rel, rprops, from_node, to_node}。

        Raises:
            DataValidationError: rel_types 为空或类型名不合法。
        """
        self._validate_identifier(from_label)
        self._validate_identifier(from_key)
        self._validate_identifier(to_label)
        self._validate_identifier(to_key)
        await self._ensure_unique_key(from_label, from_key)
        await self._ensure_unique_key(to_label, to_key)
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
        rows = await self._client.execute(cypher, params, readonly=True)
        return list(rows)

    async def get_all_relationships(
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
        """
        获取两端点之间的“所有类型”关系（端点唯一）的便捷封装。

        Args:
            from_label/from_key/from_value: 源端（唯一）。
            to_label/to_key/to_value: 目标端（唯一）。
            include_rel_props: 是否包含关系属性。

        Returns:
            关系列表（同 get_relationships）。
        """
        return await self.get_relationships(
            from_label,
            from_key,
            from_value,
            to_label,
            to_key,
            to_value,
            rel_types=None,
            include_rel_props=include_rel_props,
        )

    async def vector_search(
        self,
        label: str,
        vector_property: str,
        query_vector: List[float],
        top_k: int = 10,
        *,
        filters: Optional[Dict[str, Any]] = None,
        include_score: bool = True,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量相似检索（KNN on VECTOR INDEX）。

        Args:
            label: 目标节点标签（需在 schema 中为 `vector_property` 建立向量索引）。
            vector_property: 向量属性名称（与索引一致）。
            query_vector: 查询向量（长度需与索引维度一致）。
            top_k: 返回的相似结果数量上限。
            filters: 额外的属性过滤（可选，等值过滤）。
            include_score: 结果中是否包含相似度分数 `score`。
            score_threshold: 相似度下限（仅当相似度为“值越大越相似”的度量如 cosine/dot 时语义一致；若为距离度量如 euclidean，该阈值不适用）。

        Returns:
            列表：每项为 {"node_id": int, "labels": List[str], "node": dict, "score": float?}

        Raises:
            ValueError: 参数不合法（标识符、top_k、query_vector 类型）。
            KRagError: 数据库不支持向量检索或执行失败。

        Notes:
            - 需要 Neo4j 5.11+ 且开启向量索引功能；
            - 索引名按约定为 `vec_{label_lower}_{vector_property}`，需先通过 SchemaManager.apply 创建；
            - 维度不匹配会在数据库侧报错（Dimension mismatch）；
            - filters 为等值过滤，复杂过滤建议上层先筛选候选集再做 re-rank；
            - score_threshold 以“分数越高越好”的相似度为语义（例如 cosine/dot）。若索引使用 euclidean 距离，该阈值不建议使用。
        """
        self._validate_identifier(label)
        self._validate_identifier(vector_property)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(query_vector, list) or not query_vector or not all(
            isinstance(x, (int, float)) for x in query_vector
        ):
            raise ValueError("query_vector must be a non-empty list of numbers")
        index_name = f"vec_{label.lower()}_{vector_property}"

        # 构造等值过滤（静态属性键，参数值）
        where_clauses: List[str] = []
        params: Dict[str, Any] = {"index": index_name, "k": int(top_k), "vec": [float(x) for x in query_vector]}
        if filters:
            for idx, (k, v) in enumerate(filters.items()):
                if not isinstance(k, str):
                    raise DataValidationError("Filter property keys must be strings")
                pname = f"fp{idx}"
                where_clauses.append(f"node.{self._q(k)} = ${pname}")
                params[pname] = v
        if score_threshold is not None:
            if not isinstance(score_threshold, (int, float)):
                raise ValueError("score_threshold must be a number")
            params["threshold"] = float(score_threshold)
            where_clauses.append("score >= $threshold")
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        cypher = (
            "CALL db.index.vector.queryNodes($index, $k, $vec) "
            "YIELD node, score "
            f"{where_sql} "
            "RETURN id(node) AS node_id, labels(node) AS labels, properties(node) AS node, score "
            "ORDER BY score DESC "
            "LIMIT $k"
        )
        try:
            rows = await self._client.execute(cypher, params, readonly=True)
        except Exception as e:
            raise KRagError(f"Vector search failed: {e}")
        results: List[Dict[str, Any]] = []
        for r in rows:
            item = {
                "node_id": r.get("node_id"),
                "labels": r.get("labels"),
                "node": r.get("node"),
            }
            if include_score:
                item["score"] = r.get("score")
            results.append(item)
        return results

