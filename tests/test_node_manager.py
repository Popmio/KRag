import logging
import uuid
import pytest

from core.graph.schema_manager import SchemaManager
from core.graph.node_manager import NodeManager
from common.exceptions import DataValidationError, KRagError


@pytest.fixture(scope="module")
def nm(neo4j_client) -> NodeManager:
    # Apply schema before NodeManager tests to guarantee constraints/indexes
    sm = SchemaManager(neo4j_client)
    cfg = sm.load_yaml("config/graph_schema.yaml")
    sm.apply(cfg,drop_missing=True)
    return NodeManager(neo4j_client)


# def test_nodes_crud_and_neighbors(nm: NodeManager):
#     did = f"doc_{uuid.uuid4().hex[:8]}"
#     kw = f"kw_{uuid.uuid4().hex[:6]}"
#
#     # merge nodes
#     nm.merge_node("Document", key="id", properties={"id": did, "file_name": "a.pdf", "doc_type": "t1"})
#     nm.merge_node("Keyword", key="text", properties={"text": kw})
#
#     # get single
#     got = nm.get_node("Document", "id", did)
#     assert got and got["node"]["id"] == did and got["node"]["file_name"] == "a.pdf"
#
#     # get multiple
#     did2 = f"doc_{uuid.uuid4().hex[:8]}"
#     nm.merge_node("Document", key="id", properties={"id": did2, "file_name": "b.pdf", "doc_type": "t1"})
#     many = nm.get_nodes("Document", "id", [did, did2])
#     assert len(many) == 2
#
#     # update single
#     ud = nm.update_node("Document", "id", did, {"doc_type": "t2"})
#     assert ud and ud["node"]["doc_type"] == "t2"
#
#     # neighbors
#     nm.create_relationship("Document", "id", did, "HAS_KEYWORD", "Keyword", "text", kw, {})
#     neigh = nm.get_node_with_neighbors("Document", "id", did, direction="out", neighbor_labels=["Keyword"], limit=10)
#     assert neigh and any(e["rel"] == "HAS_KEYWORD" for e in neigh["edges"])
#
#     # bulk merge and update
#     items = [
#         {"id": f"doc_{uuid.uuid4().hex[:6]}", "file_name": f"f{i}.pdf", "doc_type": "t3"}
#         for i in range(3)
#     ]
#     upserted = nm.merge_nodes("Document", key="id", items=items)
#     assert upserted == len(items)
#
#     updates = [{"id": it["id"], "doc_type": "t4"} for it in items]
#     updated = nm.update_nodes("Document", key="id", items=updates)
#     assert updated == len(items)
#
#     # delete by unique and by property
#     assert nm.delete_node("Document", "id", did2) == 1
#     nm.merge_node("Document", key="id", properties={"id": f"doc_{uuid.uuid4().hex[:6]}", "file_name": "same.pdf"})
#     nm.merge_node("Document", key="id", properties={"id": f"doc_{uuid.uuid4().hex[:6]}", "file_name": "same.pdf"})
#     removed_by_prop = nm.delete_nodes_by_property("Document", "file_name", ["same.pdf"])
#     assert removed_by_prop >= 2
#
#     # cleanup
#     nm.delete_relationship("Document", "id", did, "HAS_KEYWORD", "Keyword", "text", kw)
#     nm.delete_node("Keyword", "text", kw)
#     nm.delete_node("Document", "id", did)
#     for it in items:
#         nm.delete_node("Document", "id", it["id"])


# def test_relationships_matrix(nm: NodeManager):
#     # Prepare documents grouped by doc_type
#     d_group1 = [f"doc_{uuid.uuid4().hex[:6]}" for _ in range(2)]
#     d_group2 = [f"doc_{uuid.uuid4().hex[:6]}" for _ in range(2)]
#     for did in d_group1:
#         nm.merge_node("Document", key="id", properties={"id": did, "doc_type": "g1"})
#     for did in d_group2:
#         nm.merge_node("Document", key="id", properties={"id": did, "doc_type": "g2"})
#
#     # create_relationship (unique endpoints present)
#     did_a = d_group1[0]
#     did_b = d_group2[0]
#     nm.create_relationship("Document", "id", did_a, "CITES", "Document", "id", did_b, {"note": "x"})
#     rels = nm.get_all_relationships("Document", "id", did_a, "Document", "id", did_b)
#     assert any(r["rel"] == "CITES" for r in rels)
#
#     # merge_relationships with unique endpoints
#     pairs = [
#         {"from_value": d_group1[0], "to_value": d_group2[1], "properties": {"note": "y"}},
#         {"from_value": d_group1[1], "to_value": d_group2[0], "properties": {"note": "z"}},
#     ]
#     upserted = nm.merge_relationships("Document", "id", "CITES", "Document", "id", pairs)
#     assert upserted == len(pairs)
#
#     # create_relationships_by_property (non-unique, property-based A×B)
#     created = nm.create_relationships_by_property(
#         "Document", "doc_type", "CITES", "Document", "doc_type",
#         pairs=[{"from_value": "g1", "to_value": "g2", "properties": {}}],
#     )
#     assert created >= len(d_group1) * len(d_group2)
#
#     # delete single relationship with property filter
#     deleted_one = nm.delete_relationship("Document", "id", did_a, "CITES", "Document", "id", did_b, rel_props={"note": "x"})
#     assert deleted_one >= 1
#
#     # delete batch with type list
#     deleted_batch = nm.delete_relationships(
#         "Document", "id", "CITES", "Document", "id",
#         pairs=[{"from_value": d_group1[0], "to_value": d_group2[1]}],
#         rel_types=["CITES"],
#     )
#     assert deleted_batch >= 1
#
#     # delete property-based batch for remaining matrix
#     deleted_prop = nm.delete_relationships_by_property(
#         "Document", "doc_type", "CITES", "Document", "doc_type",
#         pairs=[{"from_value": "g1", "to_value": "g2"}],
#         rel_types=["CITES"],
#     )
#     assert deleted_prop >= 1
#
#     # recreate some, then delete all with direction control
#     nm.create_relationship("Document", "id", d_group1[0], "CITES", "Document", "id", d_group2[0], {})
#     nm.create_relationship("Document", "id", d_group2[0], "CITES", "Document", "id", d_group1[0], {})
#     only_out = nm.delete_all_relationships("Document", "id", d_group1[0], "Document", "id", d_group2[0], both_directions=False)
#     assert only_out >= 1
#     only_in = nm.delete_all_relationships("Document", "id", d_group1[0], "Document", "id", d_group2[0], both_directions=True)
#     assert only_in >= 0
#
#     # cleanup nodes
#     for did in d_group1 + d_group2:
#         nm.delete_node("Document", "id", did)


# def test_unique_and_input_validation(nm: NodeManager):
#     # get_node: None value
#     with pytest.raises(ValueError):
#         nm.get_node("Document", "id", None)
#
#     # get_nodes: empty list and only None
#     with pytest.raises(ValueError):
#         nm.get_nodes("Document", "id", [])
#     with pytest.raises(ValueError):
#         nm.get_nodes("Document", "id", [None, None])
#
#     # merge_node with non-unique key should raise
#     did = f"doc_{uuid.uuid4().hex[:8]}"
#     with pytest.raises(KRagError):
#         nm.merge_node("Document", key="file_name", properties={"id": did, "file_name": "x.pdf"})
#
#     # get_node_with_neighbors requires unique key
#     nm.merge_node("Document", key="id", properties={"id": did, "file_name": "x.pdf"})
#     with pytest.raises(KRagError):
#         nm.get_node_with_neighbors("Document", "file_name", "x.pdf")
#     nm.delete_node("Document", "id", did)


# def test_update_reject_primary_key(nm: NodeManager):
#     did = f"doc_{uuid.uuid4().hex[:8]}"
#     nm.merge_node("Document", key="id", properties={"id": did, "file_name": "c.pdf", "doc_type": "t1"})
#     # reject updating primary key in updates
#     with pytest.raises(ValueError):
#         nm.update_node("Document", "id", did, {"id": "other"})
#     nm.delete_node("Document", "id", did)

#
# def test_delete_nodes_dedup(nm: NodeManager):
#     did = f"doc_{uuid.uuid4().hex[:8]}"
#     nm.merge_node("Document", key="id", properties={"id": did, "file_name": "d.pdf"})
#     # repeated id should still delete once
#     deleted = nm.delete_nodes("Document", "id", [did, did])
#     assert deleted == 1

#
# def test_delete_relationships_validation(nm: NodeManager):
#     # prepare endpoints
#     a = f"doc_{uuid.uuid4().hex[:6]}"
#     b = f"doc_{uuid.uuid4().hex[:6]}"
#     nm.merge_node("Document", key="id", properties={"id": a})
#     nm.merge_node("Document", key="id", properties={"id": b})
#     nm.create_relationship("Document", "id", a, "CITES", "Document", "id", b, {"target_level": "t"})
#
#     # rel_types empty should error
#     with pytest.raises(DataValidationError):
#         nm.delete_relationships(
#             "Document", "id", "CITES", "Document", "id",
#             pairs=[{"from_value": a, "to_value": b}],
#             rel_types=[],
#         )
#
#     # rel_props key must be string
#     with pytest.raises(DataValidationError):
#         nm.delete_relationship("Document", "id", a, "CITES", "Document", "id", b, rel_props={123: "x"})
#
#     # non-unique keys in get_relationships should be rejected
#     nm.update_node("Document", "id", a, {"doc_type": "g1"})
#     nm.update_node("Document", "id", b, {"doc_type": "g2"})
#     with pytest.raises(KRagError):
#         nm.get_relationships("Document", "doc_type", "g1", "Document", "doc_type", "g2")
#
#     # cleanup
#     nm.delete_all_relationships("Document", "id", a, "Document", "id", b, both_directions=True)
#     nm.delete_node("Document", "id", a)
#     nm.delete_node("Document", "id", b)
# #
#
# def test_merge_relationship_skip_missing(nm: NodeManager):
#     # only from endpoint exists
#     a = f"doc_{uuid.uuid4().hex[:6]}"
#     nm.merge_node("Document", key="id", properties={"id": a})
#     # to endpoint missing → should skip (return None)
#     res = nm.merge_relationship("Document", "id", a, "CITES", "Document", "id", "no_such_doc", {})
#     assert res is None
#     nm.delete_node("Document", "id", a)
