from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re
from pydantic import BaseModel
import yaml

from .neo4j_client import Neo4jClient
from common.exceptions import SchemaValidationError, NotFoundInSchemaError


logger = logging.getLogger(__name__)


@dataclass
class SchemaConfig(BaseModel):
    labels: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]


class SchemaManager:
    """
    图模式管理器：根据配置创建/更新节点标签、关系类型、索引与唯一约束。

    生产注意事项：
    - 唯一约束：为 (Label.property) 创建唯一约束要求现存数据无重复；建议先排重/备份。
    - 索引：创建 BTREE/向量索引可能耗时并影响写入性能，应在低峰期执行。
    - 向量索引：需正确配置维度与相似度（cosine/euclidean/dotproduct）。
    - drop_missing=True：按命名规范（uniq_/idx_/vec_/ridx_）删除已不在配置中的对象，务必谨慎。
    - prune_removed_properties：可能扫描/修改大量数据，默认 dry_run=True，先预览计划再执行。
    - 幂等性：创建均使用 IF NOT EXISTS，可重复执行；删除遵循命名规范尽量避免误删。
    """

    def __init__(self, client: Neo4jClient) -> None:
        """
        初始化模式管理器。

        Args:
            client: 已连接的 Neo4jClient。
        """
        self._client = client

    _RESERVED_IDENTIFIERS = {
        "MATCH", "RETURN", "CREATE", "DELETE", "REMOVE", "SET", "MERGE", "WHERE",
        "AND", "OR", "NOT", "UNION", "OPTIONAL", "INDEX", "CONSTRAINT", "RELATIONSHIP",
        "NODE", "SHOW", "YIELD", "CALL", "DB", "TRUE", "FALSE", "NULL", "EXISTS",
        "CONTAINS", "STARTS", "ENDS", "WITH",
    }

    @staticmethod
    def _q(identifier: str) -> str:
        """将标识符用反引号包裹，避免与关键字/特殊字符冲突。"""
        return f"`{identifier}`"

    @staticmethod
    def load_yaml(path: str) -> SchemaConfig:
        """
        从 YAML 文件加载 Schema 配置。

        Args:
            path: 配置文件路径。

        Returns:
            解析后的 SchemaConfig。
        """
        with open(path, "r", encoding="utf-8") as f:
            if not os.path.exists(path):
                raise NotFoundInSchemaError(f"NOT FIND Schema on {path}")
            try:
                data = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise SchemaValidationError(exc)
        return SchemaConfig(
            labels=data.get("labels", []) or [],
            relationships=data.get("relationships", []) or [],
        )

    def apply(
        self,
        cfg: SchemaConfig,
        *,
        drop_missing: bool = False,
    ) -> None:
        """
        应用 Schema 配置至数据库（创建/更新索引与约束）。

        Args:
            cfg: Schema 配置对象。
            drop_missing: 为 True 时删除“配置中已缺失”的对象（限命名规范内）。

        Raises:
            SchemaValidationError: 配置不合法时抛出。

        Notes:
            - 会按顺序应用 labels 与 relationships；
            - drop_missing 将产生删除动作，建议先在测试/预生产验证；
            - 大库上执行会产生 DDL 负载，建议低峰期进行。
        """
        self.validate(cfg)
        if drop_missing:
            self._drop_missing(cfg)
        for label in cfg.labels:
            self._apply_label(label)
        for rel in cfg.relationships:
            self._apply_relationship(rel)
        if drop_missing:
            self.prune_removed_properties(cfg, dry_run=False)

    def validate(self, cfg: SchemaConfig) -> None:
        """
        校验配置的基本合法性（命名、重复、属性类型、索引配置、关系端点等）。

        Args:
            cfg: 待校验的 Schema 配置。

        Raises:
            SchemaValidationError: 发现不合法配置时抛出。
        """

        label_names = set()
        for label in cfg.labels:
            name = label.get("name")
            if not name or not isinstance(name, str):
                raise SchemaValidationError("Label missing valid 'name'")
            if not self._is_safe_identifier(name):
                raise SchemaValidationError(f"Invalid label name: {name}")
            if name in label_names:
                raise SchemaValidationError(f"Duplicate label name: {name}")
            label_names.add(name)

            props = label.get("properties", []) or []
            prop_names = set()
            for p in props:
                p_name = p.get("name")
                if not p_name or not isinstance(p_name, str):
                    raise SchemaValidationError(f"Label {name} has property with invalid name")
                if not self._is_safe_identifier(p_name):
                    raise SchemaValidationError(f"Invalid property name {name}.{p_name}")
                if p_name in prop_names:
                    raise SchemaValidationError(f"Duplicate property {p_name} in label {name}")
                prop_names.add(p_name)

                p_type = p.get("type", "string")
                if p_type == "vector":
                    dim = p.get("dimensions")
                    if not isinstance(dim, int) or dim <= 0:
                        raise SchemaValidationError(f"Vector property {name}.{p_name} missing positive 'dimensions'")
                    sim = p.get("similarity", "cosine")
                    if sim not in {"cosine", "euclidean", "l2", "dot"}:
                        raise SchemaValidationError(f"Vector similarity not supported: {sim}")

            for idx in label.get("indexes", []) or []:
                if idx.get("type") != "btree":
                    raise SchemaValidationError(f"Only btree indexes supported for now in {name}")
                props_list = idx.get("properties", []) or []
                if not props_list:
                    raise SchemaValidationError(f"Index in {name} must specify properties")
                for ip in props_list:
                    if not self._is_safe_identifier(ip):
                        raise SchemaValidationError(f"Invalid index property {ip} in label {name}")
                    if ip not in prop_names:
                        raise SchemaValidationError(f"Index property {ip} not found in label {name}")

        for rel in cfg.relationships:
            r_type = rel.get("type")
            r_from = rel.get("from")
            r_to = rel.get("to")
            if not all(isinstance(x, str) and x for x in [r_type, r_from, r_to]):
                raise SchemaValidationError("Relationship must have type/from/to")
            if not self._is_safe_identifier(r_type):
                raise SchemaValidationError(f"Invalid relationship type: {r_type}")
            if not self._is_safe_identifier(r_from) or not self._is_safe_identifier(r_to):
                raise SchemaValidationError(f"Invalid relationship endpoint labels: {r_from}->{r_to}")
            if r_from not in label_names or r_to not in label_names:
                raise SchemaValidationError(
                    f"Relationship {r_type} refers to unknown labels: {r_from}->{r_to}"
                )

            r_props = rel.get("properties", []) or []
            r_prop_names = set()
            for p in r_props:
                p_name = p.get("name")
                if not p_name or not isinstance(p_name, str):
                    raise SchemaValidationError(f"Relationship {r_type} has property with invalid name")
                if not self._is_safe_identifier(p_name):
                    raise SchemaValidationError(f"Invalid relationship property name {r_type}.{p_name}")
                if p_name in r_prop_names:
                    raise SchemaValidationError(f"Duplicate relationship property {p_name} in {r_type}")
                r_prop_names.add(p_name)
                if "index" in p and not isinstance(p.get("index"), bool):
                    raise SchemaValidationError(f"Relationship {r_type}.{p_name} index must be bool")
                if p.get("required") or p.get("exists"):
                    raise SchemaValidationError(
                        f"Neo4j 5+ does not support existence constraints on relationship properties: {r_type}.{p_name}"
                    )

    def _apply_label(self, label_cfg: Dict[str, Any]) -> None:
        """
        应用节点标签的唯一约束、BTREE 索引与向量索引。

        Args:
            label_cfg: 单个标签的配置字典。

        Raises:
            SchemaValidationError: 配置不合法（命名、属性、索引等）。

        Notes:
            - 唯一约束：若现有数据有重复值将导致创建失败；
            - 索引：创建/重建索引可能耗时且影响写入性能；
            - 向量索引：需指定维度与相似度，命名以 vec_ 开头。
        """
        label = label_cfg["name"]
        if not self._is_safe_identifier(label):
            raise SchemaValidationError(f"Invalid label name: {label}")

        for prop in label_cfg.get("properties", []) or []:
            if prop.get("unique"):
                pname = prop.get("name")
                if not self._is_safe_identifier(pname):
                    raise SchemaValidationError(f"Invalid property name for unique constraint: {pname}")
                name = f"uniq_{label.lower()}_{pname}"
                cypher = (
                    "CREATE CONSTRAINT %s IF NOT EXISTS FOR (n:%s) REQUIRE n.%s IS UNIQUE"
                    % (self._q(name), self._q(label), self._q(pname)) 
                )
                self._client.execute(cypher)

        for idx in label_cfg.get("indexes", []) or []:
            if idx.get("type") == "btree":
                props = idx.get("properties", [])
                if not props:
                    continue
                if len(props) == 1:
                    if not self._is_safe_identifier(props[0]):
                        raise SchemaValidationError(f"Invalid index property in {label}: {props[0]}")
                    name = f"idx_{label.lower()}_{props[0]}"
                    cypher = (
                        "CREATE INDEX %s IF NOT EXISTS FOR (n:%s) ON (n.%s)"
                        % (self._q(name), self._q(label), self._q(props[0]))
                    )
                else:
                    for p in props:
                        if not self._is_safe_identifier(p):
                            raise SchemaValidationError(f"Invalid index property in {label}: {p}")
                    name = f"idx_{label.lower()}_{'_'.join(props)}"
                    props_list = ", ".join([f"n.{self._q(p)}" for p in props])
                    cypher = (
                        "CREATE INDEX %s IF NOT EXISTS FOR (n:%s) ON (%s)"
                        % (self._q(name), self._q(label), props_list)
                    )
                self._client.execute(cypher)

        for prop in label_cfg.get("properties", []) or []:
            if prop.get("type") == "vector":
                pname = prop.get("name")
                if not self._is_safe_identifier(pname):
                    raise SchemaValidationError(f"Invalid vector property name in {label}: {pname}")
                name = f"vec_{label.lower()}_{pname}"
                dimension = int(prop.get("dimensions"))
                similarity = prop.get("similarity", "cosine")
                if similarity == "l2":
                    similarity = "euclidean"
                elif similarity == "dot":
                    similarity = "dotproduct"
                cypher = (
                    """
                    CREATE VECTOR INDEX %s IF NOT EXISTS
                    FOR (n:%s) ON (n.%s)
                    OPTIONS {indexConfig: {
                      `vector.dimensions`: $dimension,
                      `vector.similarity_function`: $similarity
                    }}
                    """
                    % (self._q(name), self._q(label), self._q(pname)) 
                )
                self._client.execute(cypher, {"dimension": dimension, "similarity": similarity})

    def _apply_relationship(self, rel_cfg: Dict[str, Any]) -> None:
        """
        应用关系属性索引（可选）。

        Args:
            rel_cfg: 单个关系类型的配置字典。

        Raises:
            SchemaValidationError: 配置不合法（命名、属性等）。

        Notes:
            - 关系属性目前不支持唯一/存在约束（Neo4j 5+），仅能创建普通索引；
            - 对低选择性属性谨慎建索引，避免计划退化与空间浪费。
        """
        rtype = rel_cfg["type"]
        if not self._is_safe_identifier(rtype):
            raise SchemaValidationError(f"Invalid relationship type: {rtype}")
        for prop in rel_cfg.get("properties", []) or []:
            pname = prop.get("name")
            if not pname:
                continue
            if not self._is_safe_identifier(pname):
                raise SchemaValidationError(f"Invalid relationship property: {pname}")
            if prop.get("index"):
                iname = f"ridx_{rtype.lower()}_{pname}"
                cypher = (
                    "CREATE INDEX %s IF NOT EXISTS FOR ()-[r:%s]-() ON (r.%s)"
                    % (self._q(iname), self._q(rtype), self._q(pname))
                )
                self._client.execute(cypher)

    def _desired_names(self, cfg: SchemaConfig) -> Dict[str, set]:
        """
        计算期望存在的对象名集合（基于命名规范）。

        Args:
            cfg: Schema 配置。

        Returns:
            包含 constraints 与 indexes 两个集合的字典。
        """
        desired_constraints = set()
        desired_indexes = set()

        for label in cfg.labels:
            lname = label["name"]
            for prop in label.get("properties", []) or []:
                if prop.get("unique"):
                    desired_constraints.add(f"uniq_{lname.lower()}_{prop['name']}")

            for idx in label.get("indexes", []) or []:
                if idx.get("type") == "btree":
                    props = idx.get("properties", []) or []
                    if not props:
                        continue
                    if len(props) == 1:
                        desired_indexes.add(f"idx_{lname.lower()}_{props[0]}")
                    else:
                        desired_indexes.add(f"idx_{lname.lower()}_{'_'.join(props)}")

            for prop in label.get("properties", []) or []:
                if prop.get("type") == "vector":
                    desired_indexes.add(f"vec_{lname.lower()}_{prop['name']}")

        for rel in cfg.relationships:
            rtype = rel.get("type")
            for prop in rel.get("properties", []) or []:
                pname = prop.get("name")
                if not pname:
                    continue
                if prop.get("index"):
                    desired_indexes.add(f"ridx_{rtype.lower()}_{pname}")

        return {"constraints": desired_constraints, "indexes": desired_indexes}

    def _existing_names(self) -> Dict[str, set]:
        """
        查询当前数据库中（按命名前缀）可见的对象名集合。

        Returns:
            包含 constraints 与 indexes 两个集合的字典。
        """
        existing_constraints = set()
        existing_indexes = set()

        rows = self._client.execute(
            "SHOW CONSTRAINTS YIELD name RETURN name",
            readonly=True,
        )
        for r in rows:
            name = r.get("name")
            if isinstance(name, str) and (
                name.startswith("uniq_") or name.startswith("constr_")
            ):
                existing_constraints.add(name)

        rows = self._client.execute(
            "SHOW INDEXES YIELD name RETURN name",
            readonly=True,
        )
        for r in rows:
            name = r.get("name")
            if isinstance(name, str) and (
                name.startswith("idx_") or name.startswith("vec_") or name.startswith("ridx_")
            ):
                existing_indexes.add(name)

        return {"constraints": existing_constraints, "indexes": existing_indexes}

    def _existing_objects(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        列出所有索引与约束的结构化信息（用于精确判断与比对）。

        Returns:
            {"constraints": {...}, "indexes": {...}} 的结构化详情。
        """
        constraints: Dict[str, Dict[str, Any]] = {}
        indexes: Dict[str, Dict[str, Any]] = {}

        c_rows = self._client.execute(
            "SHOW CONSTRAINTS YIELD name, entityType, labelsOrTypes, properties RETURN name, entityType, labelsOrTypes, properties",
            readonly=True,
        )
        for r in c_rows:
            name = r.get("name")
            if isinstance(name, str):
                constraints[name] = {
                    "entityType": r.get("entityType"),
                    "labelsOrTypes": r.get("labelsOrTypes") or [],
                    "properties": r.get("properties") or [],
                }

        i_rows = self._client.execute(
            "SHOW INDEXES YIELD name, entityType, labelsOrTypes, properties, options RETURN name, entityType, labelsOrTypes, properties, options",
            readonly=True,
        )
        for r in i_rows:
            name = r.get("name")
            if isinstance(name, str):
                indexes[name] = {
                    "entityType": r.get("entityType"),
                    "labelsOrTypes": r.get("labelsOrTypes") or [],
                    "properties": r.get("properties") or [],
                    "options": r.get("options") or {},
                }

        return {"constraints": constraints, "indexes": indexes}

    @staticmethod
    def _is_safe_identifier(value: str) -> bool:
        """
        判断标识符是否为安全命名（字母/数字/下划线，且非数字开头）。

        Args:
            value: 待校验的字符串。

        Returns:
            True 表示安全；False 表示不符合规范。
        """
        if not isinstance(value, str):
            return False
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value):
            return False
        # if value.upper() in self._RESERVED_IDENTIFIERS:
        #     return False
        return True

    def _drop_missing(
        self,
        cfg: SchemaConfig,
    ) -> None:
        """
        删除配置中未声明的索引/约束（仅限符合命名规范的对象）。

        Args:
            cfg: Schema 配置。

        Notes:
            - 仅删除名称匹配 uniq_/idx_/vec_/ridx_ 且语义与本工具一致的对象；
            - 避免误删外部对象，但仍建议在受控环境且留有备份时执行。
        """
        desired = self._desired_names(cfg)
        existing = self._existing_names()
        details = self._existing_objects()
        details_constraints = details.get("constraints", {})
        details_indexes = details.get("indexes", {})

        managed_labels = {label["name"] for label in cfg.labels}
        managed_rel_types = {rel.get("type") for rel in cfg.relationships if rel.get("type")}

        def canonical_index_name(info: Dict[str, Any]) -> Optional[str]:
            entity = info.get("entityType")
            labels = list(info.get("labelsOrTypes") or [])
            props = list(info.get("properties") or [])
            if entity == "NODE" and len(labels) == 1 and len(props) >= 1:
                lname = labels[0]
                if any(not self._is_safe_identifier(x) for x in [lname, *props]):
                    return None
                if len(props) == 1:
                    return f"idx_{lname.lower()}_{props[0]}"
                return f"idx_{lname.lower()}_{'_'.join(props)}"
            if entity == "RELATIONSHIP" and len(labels) == 1 and len(props) == 1:
                rtype = labels[0]
                pname = props[0]
                if any(not self._is_safe_identifier(x) for x in [rtype, pname]):
                    return None
                return f"ridx_{rtype.lower()}_{pname}"
            return None

        def canonical_constraint_name(info: Dict[str, Any]) -> Optional[str]:
            entity = info.get("entityType")
            labels = list(info.get("labelsOrTypes") or [])
            props = list(info.get("properties") or [])
            if entity == "NODE" and len(labels) == 1 and len(props) == 1:
                lname = labels[0]
                pname = props[0]
                if any(not self._is_safe_identifier(x) for x in [lname, pname]):
                    return None
                return f"uniq_{lname.lower()}_{pname}"
            return None


        to_drop_constraints = existing["constraints"] - desired["constraints"]
        for name in to_drop_constraints:
            if not self._is_safe_identifier(name):
                continue
            info = details_constraints.get(name)
            if not info:
                continue
            canon = canonical_constraint_name(info)
            if canon != name:
                continue
            cypher = f"DROP CONSTRAINT {self._q(name)} IF EXISTS"
            self._client.execute(cypher)

        to_drop_indexes = existing["indexes"] - desired["indexes"]
        for name in to_drop_indexes:
            if not self._is_safe_identifier(name):
                continue
            info = details_indexes.get(name)
            if not info:
                continue
            entity = info.get("entityType")
            labels = list(info.get("labelsOrTypes") or [])
            if entity == "NODE":
                canon = canonical_index_name(info)
                if canon != name and not name.startswith("vec_"):
                    continue
            elif entity == "RELATIONSHIP":
                canon = canonical_index_name(info)
                if canon != name:
                    continue
            cypher = f"DROP INDEX {self._q(name)} IF EXISTS"
            self._client.execute(cypher)

        desired_vec_options: Dict[str, Dict[str, Any]] = {}
        for label in cfg.labels:
            lname = label["name"]
            for prop in label.get("properties", []) or []:
                if prop.get("type") == "vector":
                    name = f"vec_{lname.lower()}_{prop['name']}"
                    sim = prop.get("similarity", "cosine")
                    if sim == "l2":
                        sim = "euclidean"
                    elif sim == "dot":
                        sim = "dotproduct"
                    desired_vec_options[name] = {
                        "vector.dimensions": int(prop.get("dimensions")),
                        "vector.similarity_function": sim,
                    }

        for name in existing["indexes"] & desired["indexes"]:
            if not name.startswith("vec_"):
                continue
            info = details_indexes.get(name) or {}
            opts = info.get("options") or {}
            cfg_map = (
                (opts.get("indexConfig") or {})
                if isinstance(opts, dict) else {}
            )
            current_dim = cfg_map.get("vector.dimensions") or opts.get("vector.dimensions")
            current_sim = cfg_map.get("vector.similarity_function") or opts.get("vector.similarity_function")
            want = desired_vec_options.get(name)
            if not want:
                continue
            if current_dim != want.get("vector.dimensions") or (
                isinstance(current_sim, str) and current_sim.lower() != str(want.get("vector.similarity_function")).lower()
            ):
                cypher = f"DROP INDEX {self._q(name)} IF EXISTS"
                self._client.execute(cypher)

    def prune_removed_properties(
        self,
        cfg: SchemaConfig,
        *,
        labels: Optional[List[str]] = None,
        sample_limit: Optional[int] = 1000,
        per_node_prop_limit: Optional[int] = 64,
        batch_size: Optional[int] = 500,
        dry_run: bool = True,
    ) -> Dict[str, List[str]]:
        """
        移除配置未声明的节点属性（清理“脏字段”）。

        Args:
            cfg: Schema 配置（用于计算各标签允许的属性集合）。
            labels: 仅处理这些标签；默认为配置中所有标签。
            sample_limit: 采样节点上限；None 表示全量扫描（更慢）。
            per_node_prop_limit: 单节点属性键取样上限（避免极端大节点造成单行过大）。
            batch_size: 子查询批大小（在大图上控制资源占用）。
            dry_run: True 表示仅返回计划不执行；False 表示实际执行删除。

        Returns:
            {label: [to_remove_props...]} 的计划或执行清单。

        Notes:
            - 首选 db.schema.nodeTypeProperties()，不可用时回退为采样/分批扫描；
            - 执行删除时带有跨标签保护（protect_labels），尽量避免误删其他标签需要的同名属性；
            - 该操作可能耗时较长，建议低峰期执行并先 dry_run 审阅计划。
        """

        allowed_by_label: Dict[str, set] = {}
        for label in cfg.labels:
            lname = label["name"]
            allowed = {p.get("name") for p in (label.get("properties", []) or []) if p.get("name")}
            allowed_by_label[lname] = allowed

        target_labels = labels or list(allowed_by_label.keys())
        plan: Dict[str, List[str]] = {}

        for lname in target_labels:
            allowed = allowed_by_label.get(lname, set())

            existing: set = set()
            try:
                rows = self._client.execute(
                    (
                        "CALL db.schema.nodeTypeProperties() "
                        "YIELD nodeLabels, propertyName "
                        "WITH nodeLabels, propertyName "
                        "WHERE $label IN nodeLabels "
                        "RETURN DISTINCT propertyName AS k"
                    ),
                    {"label": lname},
                    readonly=True,
                )
                existing = {r.get("k") for r in rows if isinstance(r.get("k"), str)}
            except Exception:
                logger.debug(
                    "db.schema.nodeTypeProperties unavailable; falling back to sampling for label %s",
                    lname,
                )
                existing = set()

            if not existing:
                if not self._is_safe_identifier(lname):
                    raise SchemaValidationError(f"Invalid label name: {lname}")
                try:
                    if sample_limit is None:
                        cypher = (
                            f"MATCH (n:{self._q(lname)}) "
                            f"CALL {{ WITH n UNWIND keys(n)[0..$perNodePropLimit] AS k RETURN k }} "
                            f"IN TRANSACTIONS OF $batchSize ROWS "
                            f"RETURN DISTINCT k"
                        )
                        params = {"perNodePropLimit": int(per_node_prop_limit or 64), "batchSize": int(batch_size or 500)}
                    else:
                        cypher = (
                            f"MATCH (n:{self._q(lname)}) WITH n LIMIT $limit "
                            f"CALL {{ WITH n UNWIND keys(n)[0..$perNodePropLimit] AS k RETURN k }} "
                            f"IN TRANSACTIONS OF $batchSize ROWS "
                            f"RETURN DISTINCT k"
                        )
                        params = {
                            "limit": int(sample_limit),
                            "perNodePropLimit": int(per_node_prop_limit or 64),
                            "batchSize": int(batch_size or 500),
                        }
                    rows = self._client.execute(cypher, params, readonly=True)
                except Exception:
                    if sample_limit is None:
                        cypher = (
                            f"MATCH (n:{self._q(lname)}) "
                            f"UNWIND keys(n)[0..$perNodePropLimit] AS k "
                            f"RETURN DISTINCT k"
                        )
                        params = {"perNodePropLimit": int(per_node_prop_limit or 64)}
                    else:
                        cypher = (
                            f"MATCH (n:{self._q(lname)}) WITH n LIMIT $limit "
                            f"UNWIND keys(n)[0..$perNodePropLimit] AS k "
                            f"RETURN DISTINCT k"
                        )
                        params = {"limit": int(sample_limit), "perNodePropLimit": int(per_node_prop_limit or 64)}
                    rows = self._client.execute(cypher, params, readonly=True)
            existing = {r.get("k") for r in rows if isinstance(r.get("k"), str)}
            to_remove = sorted(existing - allowed)
            plan[lname] = to_remove

            if not dry_run and to_remove:
                for prop in to_remove:
                    if not self._is_safe_identifier(lname):
                        raise SchemaValidationError(f"Invalid label name: {lname}")
                    if not self._is_safe_identifier(prop):
                        continue
                    protect_labels = [
                        other for other, allowed_set in allowed_by_label.items()
                        if other != lname and prop in allowed_set and self._is_safe_identifier(other)
                    ]
                    rm_cypher = (
                        f"MATCH (n:{self._q(lname)}) "
                        f"WHERE n.{self._q(prop)} IS NOT NULL "
                        f"AND ALL(lbl IN $protect WHERE NOT lbl IN labels(n)) "
                        f"REMOVE n.{self._q(prop)}"
                    )
                    self._client.execute(rm_cypher, {"protect": protect_labels})

        return plan


