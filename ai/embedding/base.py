"""
AI模型基类
定义所有AI模型的通用接口和基础功能
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """模型配置基类"""
    name: str
    description: Optional[str] = None
    max_length: Optional[int] = None
    batch_size: int = 32
    device: str = "auto"
    cache_dir: Optional[str] = None
    source: Optional[str] = None


class EmbeddingResult(BaseModel):
    """嵌入结果"""
    embeddings: List[List[float]]
    model_name: str
    dimension: int
    metadata: Optional[Dict[str, Any]] = None


class RerankResult(BaseModel):
    """重排序结果"""
    scores: List[float]
    ranked_indices: List[int]
    model_name: str
    metadata: Optional[Dict[str, Any]] = None


class LLMResult(BaseModel):
    """LLM生成结果"""
    text: str
    model_name: str
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseEmbedder(ABC):
    """嵌入模型基类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """加载模型"""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """对文本进行嵌入"""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """对查询进行嵌入"""
        pass
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "device": self.config.device,
            "is_loaded": self._is_loaded
        }


class BaseReranker(ABC):
    """重排序模型基类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """加载模型"""
        pass
    
    @abstractmethod
    def rerank(self, query: str, documents: List[str]) -> RerankResult:
        """重排序文档"""
        pass
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "device": self.config.device,
            "is_loaded": self._is_loaded
        }


class BaseLLM(ABC):
    """大语言模型基类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """加载模型"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResult:
        """生成文本"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResult:
        """对话生成"""
        pass
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "max_length": self.config.max_length,
            "device": self.config.device,
            "is_loaded": self._is_loaded
        }


class ModelError(Exception):
    """模型相关异常"""
    pass


class ModelLoadError(ModelError):
    """模型加载异常"""
    pass


class ModelInferenceError(ModelError):
    """模型推理异常"""
    pass