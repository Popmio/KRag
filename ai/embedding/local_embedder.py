"""
本地嵌入模型实现
支持HuggingFace和ModelScope模型，根据配置自动选择加载方式
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

from .base import BaseEmbedder, ModelConfig, EmbeddingResult, ModelLoadError, ModelInferenceError

logger = logging.getLogger(__name__)


class LocalEmbedder(BaseEmbedder):
    """本地嵌入模型 - 支持HuggingFace和ModelScope"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = self._get_device()
        self.dimension = None
        self.model = None
        self.tokenizer = None
        self.max_length = getattr(config, 'max_length', 8192)
        self.source = getattr(config, 'source', 'huggingface')  # 获取模型来源
        
    def _get_device(self) -> str:
        """获取计算设备"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        
        # 如果明确指定了设备，检查是否可用
        if self.config.device == "cuda":
            if not torch.cuda.is_available():
                logger.warning(f"配置为cuda但CUDA不可用，将使用CPU。如需使用GPU，请安装GPU版本的PyTorch")
                return "cpu"
            return "cuda"
        
        if self.config.device == "mps":
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                logger.warning(f"配置为mps但MPS不可用，将使用CPU")
                return "cpu"
            return "mps"
        
        return self.config.device
    
    def _check_model_exists(self, model_name: str) -> bool:
        """检查模型是否存在"""
        try:
            cache_dir = self.config.cache_dir or "models/cache"
            cache_dir = str(Path(cache_dir))
            model_path = Path(cache_dir) / str(model_name).replace("/", "_")
            
            # 检查目录是否存在
            if not model_path.exists():
                return False
            
            # 检查HuggingFace格式的必需文件
            required_files = ["config.json", "pytorch_model.bin"]
            # 也检查safetensors格式
            safetensors_files = list(model_path.glob("*.safetensors"))
            
            if safetensors_files:
                # 如果有safetensors文件，认为模型存在
                return True
            
            # 检查传统的pytorch_model.bin
            for file_name in required_files:
                if not (model_path / file_name).exists():
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"检查模型存在性失败: {e}")
            return False
    
    def load_model(self) -> None:
        """加载模型"""
        try:
            model_name = self.config.name
            logger.info(f"开始加载模型: {model_name} (来源: {self.source})")
            
            # 根据来源选择加载方式
            if self.source.lower() == "modelscope":
                self._load_modelscope_model(model_name)
            else:
                self._load_huggingface_model(model_name)
            
            self._is_loaded = True
            logger.info(f"模型 {model_name} 加载成功，维度: {self.dimension}")
            
        except Exception as e:
            raise ModelLoadError(f"加载模型失败: {e}")
    
    def _load_huggingface_model(self, model_name: str) -> None:
        """加载HuggingFace模型"""
        try:
            # 设置缓存目录
            cache_dir = self.config.cache_dir or "models/cache"
            cache_dir = str(Path(cache_dir))
            os.makedirs(cache_dir, exist_ok=True)
            
            # 加载tokenizer和模型
            logger.info("加载HuggingFace tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                padding_side='left'  # 左填充，适合last token pooling
            )
            
            logger.info("加载HuggingFace模型...")
            # 根据设备选择加载方式
            if self.device == "cuda":
                # GPU加载，使用半精度以节省内存
                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # CPU加载
                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    dtype=torch.float32,
                    trust_remote_code=True
                )
                self.model.to(self.device)
            
            # 获取模型维度
            if hasattr(self.model.config, 'hidden_size'):
                self.dimension = self.model.config.hidden_size
            else:
                # 对于Qwen3-Embedding-8B，维度是1024
                self.dimension = 1024
                
        except Exception as e:
            raise ModelLoadError(f"加载HuggingFace模型失败: {e}")
    
    def _load_modelscope_model(self, model_name: str) -> None:
        """加载ModelScope模型"""
        try:
            from modelscope import AutoModel, AutoTokenizer
            
            # 设置缓存目录
            cache_dir = self.config.cache_dir or "models/cache"
            cache_dir = Path(cache_dir).absolute()
            os.makedirs(cache_dir, exist_ok=True)
            
            # 检查是否已有本地下载的模型（Registry下载后会被放在 cache_dir / model_id 下）
            local_model_dir = cache_dir / model_name.replace('/', '_')
            
            has_weights = any(local_model_dir.glob("*.safetensors"))
            has_bin = (local_model_dir / "pytorch_model.bin").exists()
            has_config = (local_model_dir / "config.json").exists()
            if local_model_dir.exists() and (has_weights or has_bin) and has_config:
                # 从本地路径加载（避免重复下载）
                logger.info(f"从本地路径加载模型: {local_model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), padding_side='left')
                if self.device == "cuda":
                    self.model = AutoModel.from_pretrained(str(local_model_dir), dtype=torch.float16, device_map="auto", trust_remote_code=True)
                else:
                    self.model = AutoModel.from_pretrained(str(local_model_dir), dtype=torch.float32, trust_remote_code=True)
                    self.model.to(self.device)
            else:
                # 从ModelScope下载并加载
                logger.info(f"从ModelScope下载并加载模型: {model_name}")
                # 设置ModelScope缓存目录环境变量
                os.environ["MODELSCOPE_CACHE"] = str(cache_dir)
                logger.info(f"设置ModelScope缓存目录: {cache_dir}")
                
                # 加载tokenizer和模型
                logger.info("加载ModelScope tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir),
                    padding_side='left'
                )
                
                logger.info("加载ModelScope模型...")
                if self.device == "cuda":
                    self.model = AutoModel.from_pretrained(
                        model_name,
                        cache_dir=str(cache_dir),
                        dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    self.model = AutoModel.from_pretrained(
                        model_name,
                        cache_dir=str(cache_dir),
                        dtype=torch.float32,
                        trust_remote_code=True
                    )
                    self.model.to(self.device)
            
            # 获取模型维度
            if hasattr(self.model.config, 'hidden_size'):
                self.dimension = self.model.config.hidden_size
            else:
                self.dimension = 1024
                
        except ImportError:
            raise ModelLoadError("ModelScope未安装，请安装: pip install modelscope")
        except Exception as e:
            raise ModelLoadError(f"加载ModelScope模型失败: {e}")
    
    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Last token pooling - 获取最后一个有效token的嵌入"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """为查询添加指令"""
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def embed_texts(self, texts: List[str], task_description: Optional[str] = None) -> EmbeddingResult:
        """对文本进行嵌入"""
        if not self._is_loaded:
            raise ModelInferenceError("模型未加载")
        
        try:
            # 如果提供了任务描述，为文本添加指令
            if task_description:
                processed_texts = [self.get_detailed_instruct(task_description, text) for text in texts]
            else:
                processed_texts = texts
            
            # 分词
            batch_dict = self.tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            # 移动到设备
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                
                # 使用last token pooling获取嵌入
                embeddings = self.last_token_pool(
                    outputs.last_hidden_state, 
                    batch_dict['attention_mask']
                )
                
                # 归一化嵌入向量
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # 转换为numpy数组
                embeddings_np = embeddings.cpu().numpy()
            
            return EmbeddingResult(
                embeddings=embeddings_np.tolist(),
                model_name=self.config.name,
                dimension=self.dimension,
                metadata={
                    "device": self.device,
                    "batch_size": len(texts),
                    "max_length": self.max_length,
                    "task_description": task_description,
                    "normalized": True
                }
            )
            
        except Exception as e:
            raise ModelInferenceError(f"文本嵌入失败: {e}")
    
    def embed_query(self, query: str, task_description: Optional[str] = None) -> List[float]:
        """对查询进行嵌入"""
        result = self.embed_texts([query], task_description)
        return result.embeddings[0]
    
    def embed_documents(self, documents: List[str]) -> EmbeddingResult:
        """对文档进行嵌入（不需要指令）"""
        return self.embed_texts(documents, task_description=None)
    
    def compute_similarity(self, queries: List[str], documents: List[str], 
                          task_description: Optional[str] = None) -> np.ndarray:
        """计算查询和文档之间的相似度"""
        # 嵌入查询（带指令）
        query_result = self.embed_texts(queries, task_description)
        query_embeddings = np.array(query_result.embeddings)
        
        # 嵌入文档（不带指令）
        doc_result = self.embed_documents(documents)
        doc_embeddings = np.array(doc_result.embeddings)
        
        # 计算相似度矩阵
        similarity_scores = query_embeddings @ doc_embeddings.T
        
        return similarity_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            "dimension": self.dimension,
            "device": self.device,
            "max_length": self.max_length,
            "source": self.source,
            "model_type": "transformer",
            "pooling_method": "last_token",
            "normalized": True
        })
        return info
    
    def set_max_length(self, max_length: int) -> None:
        """设置最大序列长度"""
        self.max_length = max_length
        logger.info(f"设置最大序列长度为: {max_length}")
    
    def enable_flash_attention(self) -> None:
        """启用Flash Attention 2（需要重新加载模型）"""
        if self.device != "cuda":
            logger.warning("Flash Attention 2 仅在CUDA设备上可用")
            return
        
        logger.info("启用Flash Attention 2需要重新加载模型...")
        # 这里可以实现重新加载逻辑，暂时只是记录
        logger.info("请重新调用load_model()以启用Flash Attention 2")