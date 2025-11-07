import pytest

from core.graph.schema_manager import SchemaManager, SchemaConfig
from common.exceptions import SchemaValidationError

# 验证是否能成功加载schema并且删除不期望字段以及判断label是否还存在对应关系未删除
@pytest.mark.asyncio
async def test_validate_and_apply(neo4j_client, tmp_path):
    sm = SchemaManager(neo4j_client)
    cfg = sm.load_yaml("config/graph_schema.yaml")
    # 严格校验，若关系引用了被移除的 label，应当立即失败，提醒先清理 schema
    sm.validate(cfg)
    await sm.apply(cfg, drop_missing=True)

# 验证非法字符是否成功判别
def test_validate_rejects_invalid_identifiers(neo4j_client):
    sm = SchemaManager(neo4j_client)
    bad_cfg = SchemaConfig(labels=[{"name": "Bad Label", "properties": []}], relationships=[])
    with pytest.raises(SchemaValidationError):
        sm.validate(bad_cfg)


@pytest.mark.asyncio
async def test_drop_missing_does_not_drop_foreign_index(neo4j_client):
    sm = SchemaManager(neo4j_client)

    # 预建非受管标签 OtherX 的索引（使用非规范化前缀，避免被 _drop_missing 识别并删除）
    await neo4j_client.execute(
        "CREATE INDEX `user_idx_otherx_temp` IF NOT EXISTS FOR (n:`OtherX`) ON (n.`temp`)"
    )

    try:
        cfg = sm.load_yaml("config/graph_schema.yaml")
        sm.validate(cfg)
        await sm.apply(cfg, drop_missing=True)

        rows = await neo4j_client.execute(
            "SHOW INDEXES YIELD name WHERE name = 'user_idx_otherx_temp' RETURN name",
            readonly=True,
        )
        assert rows, "user_idx_otherx_temp should not be dropped"
    finally:
        await neo4j_client.execute("DROP INDEX `user_idx_otherx_temp` IF EXISTS")


#保护非唯一属性测试
@pytest.mark.asyncio
async def test_prune_multilabel_protection_integration(neo4j_client):
    sm = SchemaManager(neo4j_client)
    # 定义 A 不允许 deadprop，B 允许 deadprop
    cfg = SchemaConfig(
        labels=[
            {"name": "A", "properties": [{"name": "keep"}], "indexes": []},
            {"name": "B", "properties": [{"name": "deadprop"}], "indexes": []},
        ],
        relationships=[],
    )
    sm.validate(cfg)

    # 建一个同时有 A 和 B 的节点，包含 deadprop 属性
    await neo4j_client.execute("CREATE (n:`A`:`B` {deadprop: 'x', keep: 'y'})")
    try:
        # 从 A 的角度移除未声明属性，B 允许的属性应被保护
        await sm.prune_removed_properties(cfg, labels=["A"], dry_run=False)

        # 验证 deadprop 仍然存在（因为节点同时有 B）
        res = await neo4j_client.execute(
            "MATCH (n:`A`:`B`) RETURN n.deadprop IS NOT NULL AS ok LIMIT 1",
            readonly=True,
        )
        assert res and res[0].get("ok") is True
    finally:
        await neo4j_client.execute("MATCH (n:`A`:`B`) DETACH DELETE n")


@pytest.mark.skipif(True, reason="依赖环境是否支持 VECTOR INDEX，按需启用")
@pytest.mark.asyncio
async def test_vector_similarity_normalization_and_drift(neo4j_client):
    sm = SchemaManager(neo4j_client)
    # 期望配置：VecDrift.emb 维度 64，相似度 l2 (将被归一为 euclidean)
    cfg = SchemaConfig(
        labels=[
            {"name": "VecDrift", "properties": [{"name": "emb", "type": "vector", "dimensions": 64, "similarity": "l2"}], "indexes": []}
        ],
        relationships=[],
    )
    sm.validate(cfg)

    # 先创建一个错误配置的向量索引以触发漂移修复
    await neo4j_client.execute(
        """
        CREATE VECTOR INDEX `vec_vecdrift_emb` IF NOT EXISTS
        FOR (n:`VecDrift`) ON (n.`emb`)
        OPTIONS {indexConfig: {`vector.dimensions`: 32, `vector.similarity_function`: 'cosine'}}
        """
    )
    try:
        await sm.apply(cfg, drop_missing=True)
        # 读取索引配置，验证已被重建为维度 64、相似度 euclidean
        rows = await neo4j_client.execute(
            "SHOW INDEXES YIELD name, options WHERE name = 'vec_vecdrift_emb' RETURN name, options",
            readonly=True,
        )
        assert rows, "vector index should exist"
        opts = rows[0].get("options") or {}
        cfg_map = (opts.get("indexConfig") or {}) if isinstance(opts, dict) else {}
        dim = cfg_map.get("vector.dimensions") or opts.get("vector.dimensions")
        sim = cfg_map.get("vector.similarity_function") or opts.get("vector.similarity_function")
        assert int(dim) == 64
        assert str(sim).lower() == "euclidean"
    # except Exception as e:
    #     pass
    finally:
        await neo4j_client.execute("DROP INDEX `vec_vecdrift_emb` IF EXISTS")


