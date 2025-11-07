import os
import sys

# Ensure project root is on sys.path so imports like `from core...` work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import pytest_asyncio
from core.graph.neo4j_client import Neo4jClient, Neo4jConfig


@pytest_asyncio.fixture(scope="function")
async def neo4j_client():
    """
    Neo4j 客户端 fixture
    
    如果 Neo4j 服务不可用，会跳过所有依赖此 fixture 的测试。
    可以通过环境变量配置连接信息：
    - NEO4J_URI: 连接地址（默认: bolt://localhost:7687）
    - NEO4J_USERNAME: 用户名（默认: neo4j）
    - NEO4J_PASSWORD: 密码（默认: x1234567）
    - NEO4J_DATABASE: 数据库名（默认: neo4j）
    """
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "K6c3N5p8")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    client = None
    try:
        client = Neo4jClient(Neo4jConfig(uri=uri, username=username, password=password, database=database))
        if not await client.ping():
            pytest.skip("Neo4j not reachable (ping failed)")
        yield client
    except Exception as e:
        # 连接失败时优雅跳过测试，而不是抛出错误
        error_msg = str(e)
        error_type = type(e).__name__
        
        # 提供更详细的错误信息
        if "ConnectionRefusedError" in error_type or "积极拒绝" in error_msg or "ServiceUnavailable" in error_type:
            skip_reason = (
                f"Neo4j 连接被拒绝（{error_type}）\n"
                f"配置信息：\n"
                f"  - URI: {uri}\n"
                f"  - 用户名: {username}\n"
                f"  - 密码: {'*' * len(password)}\n"
                f"请检查：\n"
                f"  1. Neo4j 服务是否已启动（查看 Neo4j Console 是否显示 'Started'）\n"
                f"  2. 端口是否正确（默认 Bolt 端口是 7687）\n"
                f"  3. 等待几秒后重试（Neo4j 可能需要时间启动）\n"
                f"  4. 访问 http://localhost:7474 确认 Neo4j Browser 可用\n"
                f"错误详情: {error_msg[:200]}"
            )
        elif "AuthError" in error_type or "authentication" in error_msg.lower():
            skip_reason = (
                f"Neo4j 认证失败（{error_type}）\n"
                f"配置信息：\n"
                f"  - URI: {uri}\n"
                f"  - 用户名: {username}\n"
                f"  - 密码: {'*' * len(password)}（实际长度: {len(password)}）\n"
                f"请检查：\n"
                f"  1. 用户名是否正确（默认: neo4j）\n"
                f"  2. 密码是否正确（您设置的密码是: x1234567）\n"
                f"  3. 在 Neo4j Browser 中确认可以登录\n"
                f"错误详情: {error_msg[:200]}"
            )
        else:
            skip_reason = (
                f"Neo4j 连接失败（{error_type}）\n"
                f"配置信息：\n"
                f"  - URI: {uri}\n"
                f"  - 用户名: {username}\n"
                f"  - 密码: {'*' * len(password)}\n"
                f"错误详情: {error_msg[:200]}"
            )
        pytest.skip(skip_reason)
    finally:
        if client is not None:
            await client.close()


