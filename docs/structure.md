## KRag 项目结构与文件说明

本项目目标：使用 Neo4j 管理知识图谱，结合向量嵌入与向量索引实现混合检索，对外提供 REST API 与 MCP 服务。

### 根目录
- `README.md`: 快速上手与项目简介。
- `requirements.txt`: 运行所需 Python 依赖。
- `pyproject.toml`: 可选的工具配置（ruff、mypy、pytest 等）。
- `__init__.py`: 将根目录作为包入口（便于相对导入）。
- `data/`: 本地样例或缓存数据，建议在 .gitignore 中排除大文件。

### ai/
- `ai/__init__.py`: 包初始化。
- `ai/model_registry.py`: 模型注册/选择的集中入口（选择不同 embedding/reranker）。
- `ai/embedding/`
  - `base.py`: 嵌入模型接口定义（抽象类）。
  - `local_embedder.py`: 本地嵌入模型实现。
  - `api_embedder.py`: 通过外部 API 获取向量的实现。
- `ai/reranker/`
  - `base.py`: 重排器接口定义。
  - `local_reranker.py`: 本地重排模型实现。

### cache/
- `cache/__init__.py`: 包初始化。
- `cache/redis_cache.py`: 基于 Redis 的缓存客户端与常用操作。

### common/
- `common/__init__.py`: 包初始化。
- `common/exceptions.py`: 自定义异常类型。
- `common/models/`
  - `__init__.py`
  - `KGnode.py`: 知识图谱节点模型定义。
  - `MCPToolCall.py`: MCP 工具调用的数据模型。
  - `QueryRequest.py`: 检索/查询请求数据结构。
- `common/utils/`
  - `__init__.py`
  - `cypher_builder.py`: 生成 Cypher 语句的工具方法。
  - `embedding_utils.py`: 向量归一化、相似度等嵌入相关工具函数。

### config/
- `config/models.yaml`: 模型相关配置（模型名、维度、提供方等）。
- `config/server.yaml`: 服务运行配置（端口、开关、限流等）。
- `config/graph_schema.yaml`: 图谱 schema 配置（标签、关系、属性、索引等）。

### core/
- `core/__init__.py`: 包初始化。
- `core/graph/`
  - `__init__.py`
  - `neo4j_client.py`: Neo4j 连接与会话管理（建议封装驱动与会话工厂）。
  - `node_manager.py`: 节点 CRUD 与关系写入等基础操作。
  - `schema_manager.py`: 基于配置的 schema/约束/索引管理（与 migrations 配合）。
- `core/indexing/`
  - `__init__.py`
  - `vector_index_manager.py`: 向量索引的创建/删除/检查/统计，确保维度与相似度配置一致。
- `core/retrieval/`
  - `__init__.py`
  - `async_query_engine.py`: 异步检索编排（图查询 + 向量召回 + 重排）。
  - `hybrid_retriever.py`: 关键词 + 向量的混合检索实现。
- `core/versioning/`
  - `__init__.py`
  - `snapshot_manager.py`: 图谱快照/回滚相关逻辑。
  - `version_router.py`: 基于版本选择不同 schema/索引策略。

### ingestion/
- `ingestion/__init__.py`: 包初始化。
- `ingestion/connectors.py`: 外部数据源连接（文件系统/HTTP/DB 等）。
- `ingestion/neo4j_writer.py`: 将解析后的数据写入 Neo4j，包含幂等与批量策略。
- `ingestion/pipeline/`
  - `__init__.py`
  - `embedding_generator.py`: 生成并回写节点向量（与 `ai/embedding/*`、`core/indexing/*` 一致）。

### interfaces/
- `interfaces/__init__.py`: 包初始化。
- `interfaces/api/`
  - `__init__.py`
  - `app.py`: FastAPI 入口，挂载 v1 路由并提供 `/healthz`。
  - `deps.py`: 依赖注入（会话、模型、缓存等）。
  - `start_server.py`: Uvicorn 启动入口（`python -m interfaces.api.start_server`）。
  - `v1/`
    - `__init__.py`
    - `graph_app.py`: v1 的路由与接口（示例 `/v1/ping`）。
- `interfaces/mcp/`
  - `__init__.py`
  - `start_mcp_server.py`: MCP 服务启动入口。
  - `protocol/mcp_handler.py`: MCP 协议处理。
  - `tools/get_entity.py`, `tools/kg_search.py`: MCP 工具实现。

### monitoring/
- `monitoring/__init__.py`: 包初始化。
- `monitoring/logging.py`: 日志配置。
- `monitoring/metrics.py`: 指标上报（建议 OpenTelemetry/Prometheus）。
- `monitoring/tracing.py`: 分布式追踪（建议 OpenTelemetry）。

### deployment/
- `deployment/Dockerfile`: 应用容器镜像构建。
- `deployment/docker-compose.yml`: 本地多服务启动（建议包含 app/neo4j/redis）。
- `deployment/k8s/`: Kubernetes 清单（建议后续分 base/overlays）。

### docs/
- `docs/architecture.md`: 架构说明。
- `docs/mcp_tools.md`: MCP 使用说明。
- `docs/rest_api_openapi.yaml`: REST API 的 OpenAPI 规范。
- `docs/structure.md`: 本文件，项目结构与说明。

### test(s)/
- `test/` 与 `test.py`: 现有测试样例，建议统一迁移到 `tests/` 并使用 pytest 发现。

---

## 关键流程串联
- 嵌入生成：`ingestion/pipeline/embedding_generator.py` → 写入 `neo4j_writer.py` → 节点属性保存为向量。
- 索引管理：`core/indexing/vector_index_manager.py` 依据 `config/models.yaml`/`graph_schema.yaml` 创建向量索引。
- 检索：`core/retrieval/hybrid_retriever.py` 组合关键词与向量召回，`async_query_engine.py` 编排执行并调用 `ai/reranker/*` 重排。
- 对外服务：FastAPI (`interfaces/api/app.py`) 提供 REST；MCP (`interfaces/mcp/*`) 提供工具能力。

## 运行指引（简要）
1) 安装依赖：`pip install -r requirements.txt`
2) 启动 API：`python -m interfaces.api.start_server`（默认 8000）
3) 健康检查：`GET /healthz`、`GET /v1/ping`
4) 配置更新：修改 `config/*.yaml` 后，确保索引维度/相似度与 `vector_index_manager.py` 使用一致。

## 图谱基础管理 Quickstart
- 配置 Neo4j 连接：在环境变量中设置 `NEO4J_URI/USERNAME/PASSWORD/DATABASE`（参考 `.env.example`）。
- 应用 Schema：
  ```python
  from core.graph.neo4j_client import Neo4jClient, Neo4jConfig
  from core.graph.schema_manager import SchemaManager

  client = Neo4jClient(Neo4jConfig(uri="bolt://localhost:7687", username="neo4j", password="***"))
  sm = SchemaManager(client)
  cfg = sm.load_yaml("config/graph_schema.yaml")
  sm.apply(cfg)
  ```
- 基础写入/关系：
  ```python
  from core.graph.node_manager import NodeManager
  nm = NodeManager(client)
  nm.merge_node("Document", key="id", properties={"id": "doc_1", "title": "hello"})
  nm.merge_relationship("Document", "id", "doc_1", "MENTIONS", "Entity", "id", "ent_1", {"weight": 0.9})
  ```


