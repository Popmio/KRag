"""
API嵌入模型实现
支持OpenAI、Cohere等API服务
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
import logging

from .base import BaseEmbedder, ModelConfig, EmbeddingResult, ModelLoadError, ModelInferenceError

logger = logging.getLogger(__name__)


class APIEmbedder(BaseEmbedder):
    """API嵌入模型"""
    
    def __init__(self, config: ModelConfig, api_config: Dict[str, Any]):
        super().__init__(config)
        self.api_config = api_config
        self.base_url = api_config.get("base_url", "")
        self.api_key = api_config.get("api_key", "")
        self.timeout = api_config.get("timeout", 30)
        self.dimension = api_config.get("dimension", 1536)
        
    def load_model(self) -> None:
        """加载模型（API模式无需加载）"""
        if not self.api_key:
            raise ModelLoadError("API密钥未配置")
        
        self._is_loaded = True
        logger.info(f"API嵌入模型 {self.config.name} 准备就绪")
    
    async def _make_request(self, session: aiohttp.ClientSession, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送API请求"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with session.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise ModelInferenceError(f"API请求失败: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            raise ModelInferenceError("API请求超时")
        except Exception as e:
            raise ModelInferenceError(f"API请求异常: {e}")
    
    async def embed_texts_async(self, texts: List[str]) -> EmbeddingResult:
        """异步嵌入文本"""
        if not self._is_loaded:
            raise ModelInferenceError("模型未加载")
        
        try:
            # 根据提供商准备不同的请求数据
            provider = self._get_provider()
            
            if provider == "qwen":
                # Qwen API格式
                payload = {
                    "model": self.config.name,
                    "input": texts
                }
            else:
                # OpenAI兼容格式
                payload = {
                    "input": texts,
                    "model": self.config.name
                }
            
            # 发送请求
            async with aiohttp.ClientSession() as session:
                response = await self._make_request(session, payload)
                
                # 解析响应
                embeddings = []
                actual_dimension = None
                
                if provider == "qwen":
                    # Qwen响应格式
                    for item in response.get("data", []):
                        embeddings.append(item["embedding"])
                        if actual_dimension is None:
                            actual_dimension = len(item["embedding"])
                else:
                    # OpenAI响应格式
                    for item in response.get("data", []):
                        embeddings.append(item["embedding"])
                        if actual_dimension is None:
                            actual_dimension = len(item["embedding"])
                
                # 使用实际维度，如果获取不到则使用配置的维度
                dimension = actual_dimension or self.dimension
                
                return EmbeddingResult(
                    embeddings=embeddings,
                    model_name=self.config.name,
                    dimension=dimension,
                    metadata={
                        "provider": provider,
                        "batch_size": len(texts),
                        "usage": response.get("usage", {}),
                        "actual_dimension": actual_dimension
                    }
                )
                
        except Exception as e:
            raise ModelInferenceError(f"API嵌入失败: {e}")
    
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """同步嵌入文本"""
        return asyncio.run(self.embed_texts_async(texts))
    
    def embed_query(self, query: str) -> List[float]:
        """对查询进行嵌入"""
        result = self.embed_texts([query])
        return result.embeddings[0]
    
    def _get_provider(self) -> str:
        """获取API提供商"""
        if "openai.com" in self.base_url:
            return "openai"
        elif "cohere.ai" in self.base_url:
            return "cohere"
        elif "dashscope.aliyuncs.com" in self.base_url:
            return "qwen"
        else:
            return "unknown"
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            "dimension": self.dimension,
            "provider": self._get_provider(),
            "base_url": self.base_url,
            "timeout": self.timeout
        })
        return info


class OpenAIEmbedder(APIEmbedder):
    """OpenAI嵌入模型"""
    
    def __init__(self, config: ModelConfig, api_key: str):
        api_config = {
            "base_url": "https://api.openai.com/v1",
            "api_key": api_key,
            "timeout": 30,
            "dimension": 1536
        }
        super().__init__(config, api_config)


class CohereEmbedder(APIEmbedder):
    """Cohere嵌入模型"""
    
    def __init__(self, config: ModelConfig, api_key: str):
        api_config = {
            "base_url": "https://api.cohere.ai/v1",
            "api_key": api_key,
            "timeout": 30,
            "dimension": 1024
        }
        super().__init__(config, api_config)


class QwenEmbedder(APIEmbedder):
    """通义千问嵌入模型"""
    
    def __init__(self, config: ModelConfig, api_key: str):
        api_config = {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": api_key,
            "timeout": 30,
            "dimension": 1536
        }
        super().__init__(config, api_config)
