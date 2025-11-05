import logging
import os
import uuid
import pytest

from core.graph.schema_manager import SchemaManager
from core.graph.node_manager import NodeManager
from common.exceptions import DataValidationError, KRagError


@pytest.fixture(scope="module")
def nm(neo4j_client) -> NodeManager:
    # Apply schema before NodeManager tests to guarantee constraints/indexes
    sm = SchemaManager(neo4j_client)
    base_dir = os.path.dirname(os.path.dirname(__file__))  # 回到项目根目录
    cfg_path = os.path.join(base_dir, "config", "graph_schema.yaml")
    cfg = sm.load_yaml(cfg_path)
    sm.apply(cfg,drop_missing=True)
    return NodeManager(neo4j_client)


# 测试节点的完整 CRUD 操作和邻居查询功能。
# 测试内容：
# - 单个节点创建（merge_node）
# - 单个节点读取（get_node）
# - 批量节点读取（get_nodes）
# - 单个节点更新（update_node）
# - 批量节点合并（merge_nodes）
# - 批量节点更新（update_nodes）
# - 关系创建（create_relationship）
# - 邻居查询（get_node_with_neighbors）
# - 按唯一键删除（delete_node）
# - 按属性批量删除（delete_nodes_by_property）
def test_nodes_crud_and_neighbors(nm: NodeManager):
    did = f"doc_{uuid.uuid4().hex[:8]}"
    kw = f"kw_{uuid.uuid4().hex[:6]}"

    # merge nodes
    nm.merge_node("Document", key="id", properties={"id": did, "file_name": "a.pdf", "doc_type": "t1"})
    nm.merge_node("Keyword", key="text", properties={"text": kw})

    # get single
    got = nm.get_node("Document", "id", did)
    assert got and got["node"]["id"] == did and got["node"]["file_name"] == "a.pdf"

    # get multiple
    did2 = f"doc_{uuid.uuid4().hex[:8]}"
    nm.merge_node("Document", key="id", properties={"id": did2, "file_name": "b.pdf", "doc_type": "t1"})
    many = nm.get_nodes("Document", "id", [did, did2])
    assert len(many) == 2

    # update single
    ud = nm.update_node("Document", "id", did, {"doc_type": "t2"})
    assert ud and ud["node"]["doc_type"] == "t2"

    # neighbors
    nm.create_relationship("Document", "id", did, "HAS_KEYWORD", "Keyword", "text", kw, {})
    neigh = nm.get_node_with_neighbors("Document", "id", did, direction="out", neighbor_labels=["Keyword"], limit=10)
    assert neigh and any(e["rel"] == "HAS_KEYWORD" for e in neigh["edges"])

    # bulk merge and update
    items = [
        {"id": f"doc_{uuid.uuid4().hex[:6]}", "file_name": f"f{i}.pdf", "doc_type": "t3"}
        for i in range(3)
    ]
    upserted = nm.merge_nodes("Document", key="id", items=items)
    assert upserted == len(items)

    updates = [{"id": it["id"], "doc_type": "t4"} for it in items]
    updated = nm.update_nodes("Document", key="id", items=updates)
    assert updated == len(items)

    # delete by unique and by property
    assert nm.delete_node("Document", "id", did2) == 1
    nm.merge_node("Document", key="id", properties={"id": f"doc_{uuid.uuid4().hex[:6]}", "file_name": "same.pdf"})
    nm.merge_node("Document", key="id", properties={"id": f"doc_{uuid.uuid4().hex[:6]}", "file_name": "same.pdf"})
    removed_by_prop = nm.delete_nodes_by_property("Document", "file_name", ["same.pdf"])
    assert removed_by_prop >= 2

    # cleanup
    nm.delete_relationship("Document", "id", did, "HAS_KEYWORD", "Keyword", "text", kw)
    nm.delete_node("Keyword", "text", kw)
    nm.delete_node("Document", "id", did)
    for it in items:
        nm.delete_node("Document", "id", it["id"])


# 测试关系的完整操作矩阵，包括创建、合并、删除的各种模式。
# 测试内容：
# - 单个关系创建（create_relationship，带属性）
# - 批量关系合并（merge_relationships，唯一端点）
# - 按属性批量创建关系（create_relationships_by_property，A×B 笛卡尔积）
# - 按属性删除关系（delete_relationship，带属性过滤）
# - 批量删除关系（delete_relationships，带类型列表）
# - 按属性批量删除关系（delete_relationships_by_property）
# - 方向控制删除（delete_all_relationships，单向/双向）
def test_relationships_matrix(nm: NodeManager):
    # Prepare documents grouped by doc_type
    d_group1 = [f"doc_{uuid.uuid4().hex[:6]}" for _ in range(2)]
    d_group2 = [f"doc_{uuid.uuid4().hex[:6]}" for _ in range(2)]
    for did in d_group1:
        nm.merge_node("Document", key="id", properties={"id": did, "doc_type": "g1"})
    for did in d_group2:
        nm.merge_node("Document", key="id", properties={"id": did, "doc_type": "g2"})

    # create_relationship (unique endpoints present)
    did_a = d_group1[0]
    did_b = d_group2[0]
    nm.create_relationship("Document", "id", did_a, "CITES", "Document", "id", did_b, {"note": "x"})
    rels = nm.get_all_relationships("Document", "id", did_a, "Document", "id", did_b)
    assert any(r["rel"] == "CITES" for r in rels)

    # merge_relationships with unique endpoints
    pairs = [
        {"from_value": d_group1[0], "to_value": d_group2[1], "properties": {"note": "y"}},
        {"from_value": d_group1[1], "to_value": d_group2[0], "properties": {"note": "z"}},
    ]
    upserted = nm.merge_relationships("Document", "id", "CITES", "Document", "id", pairs)
    assert upserted == len(pairs)

    # create_relationships_by_property (non-unique, property-based A×B)
    created = nm.create_relationships_by_property(
        "Document", "doc_type", "CITES", "Document", "doc_type",
        pairs=[{"from_value": "g1", "to_value": "g2", "properties": {}}],
    )
    assert created >= len(d_group1) * len(d_group2)

    # delete single relationship with property filter
    deleted_one = nm.delete_relationship("Document", "id", did_a, "CITES", "Document", "id", did_b, rel_props={"note": "x"})
    assert deleted_one >= 1

    # delete batch with type list
    deleted_batch = nm.delete_relationships(
        "Document", "id", "CITES", "Document", "id",
        pairs=[{"from_value": d_group1[0], "to_value": d_group2[1]}],
        rel_types=["CITES"],
    )
    assert deleted_batch >= 1

    # delete property-based batch for remaining matrix
    deleted_prop = nm.delete_relationships_by_property(
        "Document", "doc_type", "CITES", "Document", "doc_type",
        pairs=[{"from_value": "g1", "to_value": "g2"}],
        rel_types=["CITES"],
    )
    assert deleted_prop >= 1

    # recreate some, then delete all with direction control
    nm.create_relationship("Document", "id", d_group1[0], "CITES", "Document", "id", d_group2[0], {})
    nm.create_relationship("Document", "id", d_group2[0], "CITES", "Document", "id", d_group1[0], {})
    only_out = nm.delete_all_relationships("Document", "id", d_group1[0], "Document", "id", d_group2[0], both_directions=False)
    assert only_out >= 1
    only_in = nm.delete_all_relationships("Document", "id", d_group1[0], "Document", "id", d_group2[0], both_directions=True)
    assert only_in >= 0

    # cleanup nodes
    for did in d_group1 + d_group2:
        nm.delete_node("Document", "id", did)


# 测试邻居查询的可视化输出功能。
# 测试内容：
# - 创建节点和关系
# - 查询节点的邻居（direction="both"，包含关系属性）
# - 打印节点和边的详细信息（用于调试和可视化）
def test_neighbors_visual(nm: NodeManager):
    did = f"doc_{uuid.uuid4().hex[:8]}"
    kw = f"kw_{uuid.uuid4().hex[:6]}"

    nm.merge_node("Document", key="id", properties={"id": did, "file_name": "vis.pdf", "doc_type": "vis"})
    nm.merge_node("Keyword", key="text", properties={"text": kw})
    nm.create_relationship("Document", "id", did, "HAS_KEYWORD", "Keyword", "text", kw, {})

    res = nm.get_node_with_neighbors(
        "Document", "id", did, direction="both", neighbor_labels=["Keyword"], limit=10, include_rel_props=True
    )
    assert res is not None
    # print a compact, human-friendly view (run pytest with -s to see output)
    print("NODE:", res.get("node"))
    edges = res.get("edges", [])
    print("EDGES_COUNT:", len(edges))
    for i, e in enumerate(edges):
        if i >= 5:
            print("...", len(edges) - i, "more edges omitted")
            break
        nlabels = e.get("node", {}).get("labels", [])
        nprops = e.get("node", {}).get("props", {})
        print(
            f"edge[{i}]: dir={e.get('direction')} rel={e.get('rel')} neighbor_labels={nlabels} neighbor_props_keys={list(nprops.keys())}"
        )

    # cleanup
    nm.delete_relationship("Document", "id", did, "HAS_KEYWORD", "Keyword", "text", kw)
    nm.delete_node("Keyword", "text", kw)
    nm.delete_node("Document", "id", did)


# 测试多样化的邻居查询场景，包括多种节点类型和关系类型的组合。
# 测试内容：
# - 创建多种节点类型（Document, Title, Clause, Organization, Keyword）
# - 创建多种关系类型（HAS_KEYWORD, PUBLISHED_BY, CONTAINS, CITES）
# - 测试双向关系（outbound 和 inbound）
# - 按方向过滤邻居（direction="out"/"in"/"both"）
# - 按关系类型过滤（rel_types）
# - 统计和分组邻居关系
def test_neighbors_variety(nm: NodeManager):
    # Create a document with diverse neighbor labels and relationship types
    did = f"doc_{uuid.uuid4().hex[:8]}"
    did2 = f"doc_{uuid.uuid4().hex[:8]}"
    tid = f"title_{uuid.uuid4().hex[:6]}"
    cid = f"clause_{uuid.uuid4().hex[:6]}"
    org = f"org_{uuid.uuid4().hex[:6]}"
    kw1 = f"kw_{uuid.uuid4().hex[:6]}"
    kw2 = f"kw_{uuid.uuid4().hex[:6]}"

    # nodes
    nm.merge_node("Document", key="id", properties={"id": did, "file_name": "n1.pdf"})
    nm.merge_node("Document", key="id", properties={"id": did2, "file_name": "n2.pdf"})
    nm.merge_node("Title", key="id", properties={"id": tid, "document_id": did, "title_text": "T"})
    nm.merge_node("Clause", key="id", properties={"id": cid, "summary": "S"})
    nm.merge_node("Organization", key="id", properties={"id": org, "name": "Org"})
    nm.merge_node("Keyword", key="text", properties={"text": kw1})
    nm.merge_node("Keyword", key="text", properties={"text": kw2})

    # outbound from D
    nm.create_relationship("Document", "id", did, "HAS_KEYWORD", "Keyword", "text", kw1, {})
    nm.create_relationship("Document", "id", did, "HAS_KEYWORD", "Keyword", "text", kw2, {})
    nm.create_relationship("Document", "id", did, "PUBLISHED_BY", "Organization", "id", org, {"published_at": "2024-01-01"})
    nm.create_relationship("Document", "id", did, "CONTAINS", "Title", "id", tid, {"edge_type": "heading"})
    nm.create_relationship("Document", "id", did, "CITES", "Document", "id", did2, {"target_level": "doc"})

    # inbound to D
    nm.create_relationship("Clause", "id", cid, "CITES", "Document", "id", did, {"target_level": "doc"})
    nm.create_relationship("Document", "id", did2, "CITES", "Document", "id", did, {"target_level": "doc"})

    # query
    result = nm.get_node_with_neighbors("Document", "id", did, direction="both", limit=100, include_rel_props=True)
    assert result is not None
    edges = result.get("edges", [])
    print("VARIETY_EDGES_COUNT:", len(edges))

    # group by (direction, rel, neighbor label)
    summary = {}
    for e in edges:
        labels = e.get("node", {}).get("labels", [])
        top_label = labels[0] if labels else "_unknown"
        key = (e.get("direction"), e.get("rel"), top_label)
        summary[key] = summary.get(key, 0) + 1
    for (d, r, l), c in sorted(summary.items()):
        print(f"dir={d} rel={r} neighbor_label={l} count={c}")

    # filtered views
    out_only = nm.get_node_with_neighbors("Document", "id", did, direction="out", rel_types=["HAS_KEYWORD", "PUBLISHED_BY"], limit=50)
    assert out_only is not None and len(out_only.get("edges", [])) >= 2
    print("FILTERED_OUT_EDGES:", len(out_only.get("edges", [])))

    in_only = nm.get_node_with_neighbors("Document", "id", did, direction="in", rel_types=["CITES"], limit=50)
    assert in_only is not None and len(in_only.get("edges", [])) >= 1
    print("FILTERED_IN_CITES_EDGES:", len(in_only.get("edges", [])))

    # cleanup
    nm.delete_node("Title", "id", tid)
    nm.delete_node("Clause", "id", cid)
    nm.delete_node("Organization", "id", org)
    nm.delete_node("Keyword", "text", kw1)
    nm.delete_node("Keyword", "text", kw2)
    nm.delete_node("Document", "id", did2)
    nm.delete_node("Document", "id", did)

# 测试唯一性约束和输入验证功能。
# 测试内容：
# - 验证 None 值被拒绝（get_node）
# - 验证空列表被拒绝（get_nodes）
# - 验证使用非唯一键进行单节点操作会抛出异常（merge_node, get_node_with_neighbors）
# - 确保唯一键要求被正确执行
def test_unique_and_input_validation(nm: NodeManager):
    # get_node: None value
    with pytest.raises(ValueError):
        nm.get_node("Document", "id", None)

    # get_nodes: empty list and only None
    with pytest.raises(ValueError):
        nm.get_nodes("Document", "id", [])
    with pytest.raises(ValueError):
        nm.get_nodes("Document", "id", [None, None])

    # merge_node with non-unique key should raise
    did = f"doc_{uuid.uuid4().hex[:8]}"
    with pytest.raises(KRagError):
        nm.merge_node("Document", key="file_name", properties={"id": did, "file_name": "x.pdf"})

    # get_node_with_neighbors requires unique key
    nm.merge_node("Document", key="id", properties={"id": did, "file_name": "x.pdf"})
    with pytest.raises(KRagError):
        nm.get_node_with_neighbors("Document", "file_name", "x.pdf")
    nm.delete_node("Document", "id", did)


# 测试更新操作拒绝修改主键的功能。
# 测试内容：
# - 验证在更新节点时，不能修改主键值
# - 确保数据完整性（主键不可变）
def test_update_reject_primary_key(nm: NodeManager):
    did = f"doc_{uuid.uuid4().hex[:8]}"
    nm.merge_node("Document", key="id", properties={"id": did, "file_name": "c.pdf", "doc_type": "t1"})
    # reject updating primary key in updates
    with pytest.raises(ValueError):
        nm.update_node("Document", "id", did, {"id": "other"})
    nm.delete_node("Document", "id", did)


# 测试批量删除节点时的自动去重功能。
# 测试内容：
# - 验证在删除列表中包含重复 ID 时，只删除一次
# - 确保去重逻辑正确工作
def test_delete_nodes_dedup(nm: NodeManager):
    did = f"doc_{uuid.uuid4().hex[:8]}"
    nm.merge_node("Document", key="id", properties={"id": did, "file_name": "d.pdf"})
    # repeated id should still delete once
    deleted = nm.delete_nodes("Document", "id", [did, did])
    assert deleted == 1


# 测试删除关系时的验证逻辑。
# 测试内容：
# - 验证空的 rel_types 列表会被拒绝
# - 验证关系属性键必须是字符串类型
# - 验证 get_relationships 要求使用唯一键（非唯一键会抛出异常）
def test_delete_relationships_validation(nm: NodeManager):
    # prepare endpoints
    a = f"doc_{uuid.uuid4().hex[:6]}"
    b = f"doc_{uuid.uuid4().hex[:6]}"
    nm.merge_node("Document", key="id", properties={"id": a})
    nm.merge_node("Document", key="id", properties={"id": b})
    nm.create_relationship("Document", "id", a, "CITES", "Document", "id", b, {"target_level": "t"})

    # rel_types empty should error
    with pytest.raises(DataValidationError):
        nm.delete_relationships(
            "Document", "id", "CITES", "Document", "id",
            pairs=[{"from_value": a, "to_value": b}],
            rel_types=[],
        )

    # rel_props key must be string
    with pytest.raises(DataValidationError):
        nm.delete_relationship("Document", "id", a, "CITES", "Document", "id", b, rel_props={123: "x"})

    # non-unique keys in get_relationships should be rejected
    nm.update_node("Document", "id", a, {"doc_type": "g1"})
    nm.update_node("Document", "id", b, {"doc_type": "g2"})
    with pytest.raises(KRagError):
        nm.get_relationships("Document", "doc_type", "g1", "Document", "doc_type", "g2")

    # cleanup
    nm.delete_all_relationships("Document", "id", a, "Document", "id", b, both_directions=True)
    nm.delete_node("Document", "id", a)
    nm.delete_node("Document", "id", b)
#

# 测试合并关系时端点缺失的优雅处理。
# 测试内容：
# - 验证当关系目标端点不存在时，merge_relationship 返回 None 而不是抛出异常
# - 确保优雅降级，避免因缺失端点导致程序崩溃
def test_merge_relationship_skip_missing(nm: NodeManager):
    # only from endpoint exists
    a = f"doc_{uuid.uuid4().hex[:6]}"
    nm.merge_node("Document", key="id", properties={"id": a})
    # to endpoint missing → should skip (return None)
    res = nm.merge_relationship("Document", "id", a, "CITES", "Document", "id", "no_such_doc", {})
    assert res is None
    nm.delete_node("Document", "id", a)
