"""
模型注册器
负责管理所有AI模型的注册、加载和获取
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Type
from pathlib import Path
import logging

from .embedding.base import BaseEmbedder, BaseReranker, BaseLLM, ModelConfig, ModelLoadError
from .embedding.local_embedder import LocalEmbedder
from .embedding.api_embedder import APIEmbedder, OpenAIEmbedder, CohereEmbedder, QwenEmbedder

logger = logging.getLogger(__name__)


def _find_project_root(config_file: str = "config/models.yaml") -> Path:
    """
    查找项目根目录（包含config/models.yaml的目录）
    
    查找策略：
    1. 检查环境变量 MODEL_CONFIG_PATH，如果设置了且文件存在，使用其所在目录
    2. 如果config_file是绝对路径且文件存在，直接使用
    3. 从当前文件位置开始向上查找，直到找到包含config/models.yaml的目录
    4. 如果找不到，返回当前工作目录
    
    Args:
        config_file: 配置文件名（相对于项目根目录）
        
    Returns:
        项目根目录的Path对象
    """
    config_path = Path(config_file)
    
    # 1. 检查环境变量
    env_config_path = os.environ.get("MODEL_CONFIG_PATH")
    if env_config_path:
        env_path = Path(env_config_path)
        if env_path.exists():
            # 如果环境变量指向的是文件，返回其父目录
            if env_path.is_file():
                return env_path.parent
            # 如果是目录，检查其下是否有config/models.yaml
            config_in_env = env_path / config_file
            if config_in_env.exists():
                return env_path
    
    # 2. 如果是绝对路径且文件存在，直接使用其父目录
    if config_path.is_absolute() and config_path.exists():
        return config_path.parent
    
    # 3. 从当前文件位置开始向上查找项目根目录
    # 获取当前文件（model_registry.py）的目录
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    
    # 向上查找，直到找到包含config/models.yaml的目录
    search_dir = current_dir
    while search_dir != search_dir.parent:  # 直到根目录
        config_file_path = search_dir / config_file
        if config_file_path.exists():
            return search_dir
        search_dir = search_dir.parent
    
    # 4. 如果找不到，尝试从当前工作目录查找
    cwd = Path.cwd()
    config_file_path = cwd / config_file
    if config_file_path.exists():
        return cwd
    
    # 5. 最后返回当前文件所在目录（向后兼容）
    logger.warning(f"未找到项目根目录，使用当前文件所在目录: {current_dir}")
    return current_dir


class ModelRegistry:
    """模型注册器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模型注册器
        
        Args:
            config_path: 配置文件路径。如果为None，则：
                1. 优先使用环境变量 MODEL_CONFIG_PATH
                2. 否则自动查找项目根目录下的 config/models.yaml
                如果为相对路径，则相对于自动找到的项目根目录
                如果为绝对路径，则直接使用
        """
        if config_path is None:
            # 检查环境变量
            env_config_path = os.environ.get("MODEL_CONFIG_PATH")
            if env_config_path:
                self.config_path = Path(env_config_path)
                # 从配置文件路径推导项目根目录
                if self.config_path.is_file():
                    self.project_root = self.config_path.parent.parent  # config/models.yaml -> 项目根目录
                else:
                    self.project_root = self.config_path.parent
            else:
                # 自动查找项目根目录
                self.project_root = _find_project_root()
                self.config_path = self.project_root / "config/models.yaml"
        else:
            config_path_obj = Path(config_path)
            if config_path_obj.is_absolute():
                # 绝对路径直接使用
                self.config_path = config_path_obj
                # 从配置文件路径推导项目根目录
                if self.config_path.name == "models.yaml" and self.config_path.parent.name == "config":
                    self.project_root = self.config_path.parent.parent
                else:
                    self.project_root = _find_project_root()
            else:
                # 相对路径，相对于项目根目录
                self.project_root = _find_project_root()
                self.config_path = self.project_root / config_path
        
        self.config = self._load_config()
        self._embedders: Dict[str, BaseEmbedder] = {}
        self._rerankers: Dict[str, BaseReranker] = {}
        self._llms: Dict[str, BaseLLM] = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """加载模型配置"""
        try:
            if not self.config_path.exists():
                logger.warning(f"配置文件不存在: {self.config_path}，使用空配置")
                return {}
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                logger.info(f"成功加载配置文件: {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"加载模型配置失败: {e}，配置文件路径: {self.config_path}")
            return {}
# ... existing code ...

    def get_embedder(self, model_name: Optional[str] = None, provider: str = "local") -> BaseEmbedder:
        """获取嵌入模型"""
        if model_name is None:
            model_name = self._get_default_model("embedding", provider)
        
        cache_key = f"{provider}:{model_name}"
        
        if cache_key not in self._embedders:
            self._embedders[cache_key] = self._create_embedder(model_name, provider)
        
        return self._embedders[cache_key]
    
    def get_reranker(self, model_name: Optional[str] = None, provider: str = "local") -> BaseReranker:
        """获取重排序模型"""
        if model_name is None:
            model_name = self._get_default_model("reranker", provider)
        
        cache_key = f"{provider}:{model_name}"
        
        if cache_key not in self._rerankers:
            self._rerankers[cache_key] = self._create_reranker(model_name, provider)
        
        return self._rerankers[cache_key]
    
    def get_llm(self, model_name: Optional[str] = None, provider: str = "local") -> BaseLLM:
        """获取大语言模型"""
        if model_name is None:
            model_name = self._get_default_model("llm", provider)
        
        cache_key = f"{provider}:{model_name}"
        
        if cache_key not in self._llms:
            self._llms[cache_key] = self._create_llm(model_name, provider)
        
        return self._llms[cache_key]
    
    def _get_default_model(self, model_type: str, provider: str) -> str:
        """获取默认模型名称"""
        try:
            models_config = self.config.get("models", {})
            model_config = models_config.get(model_type, {})
            provider_config = model_config.get(provider, {})
            return provider_config.get("default", "")
        except Exception:
            return ""
    
    def _create_embedder(self, model_name: str, provider: str) -> BaseEmbedder:
        """创建嵌入模型实例（带自动下载与格式分流：transformers vs GGUF）"""
        try:
            # 获取模型配置
            model_config = self._get_model_config("embedding", provider, model_name)
            # 生成基础配置 - 优先使用模型级别的cache_dir
            cache_dir = model_config.get("cache_dir") or model_config.get("modelpath") or self._get_cache_dir()
            
            # 确保 cache_dir 是绝对路径（相对于项目根目录）
            cache_dir_path = Path(cache_dir)
            if not cache_dir_path.is_absolute():
                cache_dir = str((self.project_root / cache_dir_path).resolve())
            
            config = ModelConfig(
                name=model_name,
                description=model_config.get("description"),
                max_length=model_config.get("max_length"),
                batch_size=model_config.get("batch_size", 32),
                device=self._get_device_preference(),
                cache_dir=cache_dir,
                source=model_config.get("source", "huggingface")
            )

            # 先下载缓存
            download_root = self._download_model_if_needed(config.name, config.source, Path(config.cache_dir))
            # 检测格式
            model_format = self._detect_model_format(config.name, Path(config.cache_dir), download_root)
            logger.info(f"模型格式检测: {model_format}")

            if model_format == "gguf":
                # 返回 GGUF 嵌入器
                return _LlamaCppEmbedder(config, download_root)

            # 非 GGUF，使用现有本地嵌入器
            if provider == "local":
                return LocalEmbedder(config)
            elif provider == "api":
                return self._create_api_embedder(config, model_config)
            else:
                raise ValueError(f"不支持的提供商: {provider}")
                
        except Exception as e:
            logger.error(f"创建嵌入模型失败: {e}")
            raise
    
    def _create_api_embedder(self, config: ModelConfig, model_config: Dict[str, Any]) -> APIEmbedder:
        """创建API嵌入模型"""
        # 获取API配置
        api_config = self.config.get("models", {}).get("embedding", {}).get("api", {})
        model_name = config.name
        
        # 从模型配置中获取提供商信息
        provider = model_config.get("provider", "")
        
        if provider == "openai":
            provider_config = api_config.get("providers", {}).get("openai", {})
            api_key = provider_config.get("api_key", "")
            if not api_key or api_key == "your-openai-api-key-here":
                raise ModelLoadError("请在config/models.yaml中配置OpenAI API密钥")
            return OpenAIEmbedder(config, api_key)
        elif provider == "cohere":
            provider_config = api_config.get("providers", {}).get("cohere", {})
            api_key = provider_config.get("api_key", "")
            if not api_key or api_key == "your-cohere-api-key-here":
                raise ModelLoadError("请在config/models.yaml中配置Cohere API密钥")
            return CohereEmbedder(config, api_key)
        elif provider == "qwen":
            provider_config = api_config.get("providers", {}).get("qwen", {})
            api_key = provider_config.get("api_key", "")
            if not api_key or api_key == "your-qwen-api-key-here":
                raise ModelLoadError("请在config/models.yaml中配置通义千问API密钥")
            return QwenEmbedder(config, api_key)
        else:
            # 如果没有指定提供商，尝试通过模型名称推断
            if "openai" in model_name.lower() or "ada" in model_name.lower():
                provider_config = api_config.get("providers", {}).get("openai", {})
                api_key = provider_config.get("api_key", "")
                if not api_key or api_key == "your-openai-api-key-here":
                    raise ModelLoadError("请在config/models.yaml中配置OpenAI API密钥")
                return OpenAIEmbedder(config, api_key)
            elif "cohere" in model_name.lower():
                provider_config = api_config.get("providers", {}).get("cohere", {})
                api_key = provider_config.get("api_key", "")
                if not api_key or api_key == "your-cohere-api-key-here":
                    raise ModelLoadError("请在config/models.yaml中配置Cohere API密钥")
                return CohereEmbedder(config, api_key)
            elif "qwen" in model_name.lower() or "text-embedding-v" in model_name.lower():
                provider_config = api_config.get("providers", {}).get("qwen", {})
                api_key = provider_config.get("api_key", "")
                if not api_key or api_key == "your-qwen-api-key-here":
                    raise ModelLoadError("请在config/models.yaml中配置通义千问API密钥")
                return QwenEmbedder(config, api_key)
            else:
                # 通用API配置
                api_config_dict = {
                    "base_url": api_config.get("base_url", ""),
                    "api_key": api_config.get("api_key", ""),
                    "timeout": api_config.get("timeout", 30),
                    "dimension": model_config.get("dimension", 1536)
                }
                return APIEmbedder(config, api_config_dict)
    
    def _create_reranker(self, model_name: str, provider: str) -> BaseReranker:
        """创建重排序模型实例"""
        # TODO: 实现重排序模型创建
        raise NotImplementedError("重排序模型暂未实现")
    
    def _create_llm(self, model_name: str, provider: str) -> BaseLLM:
        """创建大语言模型实例"""
        # TODO: 实现LLM模型创建
        raise NotImplementedError("LLM模型暂未实现")
    
    def _get_model_config(self, model_type: str, provider: str, model_name: str) -> Dict[str, Any]:
        """获取模型配置"""
        try:
            models_config = self.config.get("models", {})
            model_config = models_config.get(model_type, {})
            provider_config = model_config.get(provider, {})
            
            if provider == "local":
                models = provider_config.get("models", [])
                matching_models = []
                
                # 收集所有匹配的模型
                for model in models:
                    if model["name"] == model_name:
                        matching_models.append(model)
                
                if not matching_models:
                    return {}
                
                # 如果有多个匹配的模型，按优先级选择
                if len(matching_models) > 1:
                    logger.info(f"发现多个匹配的模型: {model_name}")
                    for i, model in enumerate(matching_models):
                        logger.info(f"  选项{i+1}: {model['name']} (来源: {model.get('source', 'unknown')})")
                    
                    # 获取配置的优先级
                    source_priority = provider_config.get("source_priority", ["huggingface", "modelscope"])
                    logger.info(f"来源优先级: {source_priority}")
                    
                    # 按优先级选择模型
                    for priority_source in source_priority:
                        logger.info(f"检查优先级来源: {priority_source}")
                        priority_model = next((m for m in matching_models if m.get('source', '').lower() == priority_source.lower()), None)
                        if priority_model:
                            logger.info(f"✅ 选择{priority_source}模型: {priority_model['name']} (描述: {priority_model.get('description', 'N/A')})")
                            return priority_model
                        else:
                            logger.info(f"❌ 未找到{priority_source}来源的模型")
                    
                    # 如果没有匹配的优先级，选择第一个
                    logger.info(f"选择第一个模型: {matching_models[0]['name']}")
                    return matching_models[0]
                
                return matching_models[0]
                
            elif provider == "api":
                providers = provider_config.get("providers", {})
                for provider_name, provider_config in providers.items():
                    models = provider_config.get("models", [])
                    for model in models:
                        if model["name"] == model_name:
                            return model
            
            return {}
        except Exception as e:
            logger.error(f"获取模型配置失败: {e}")
            return {}
    
    def _get_device_preference(self) -> str:
        """获取设备偏好"""
        loading_config = self.config.get("loading", {})
        device_config = loading_config.get("device", {})
        return device_config.get("preferred", "cuda")
    
    def _get_cache_dir(self) -> str:
        """
        获取缓存目录（相对于项目根目录的绝对路径）
        
        如果配置中的 cache_dir 是相对路径，则转换为相对于项目根目录的绝对路径
        如果是绝对路径，则直接使用
        """
        loading_config = self.config.get("loading", {})
        cache_config = loading_config.get("cache", {})
        cache_dir = cache_config.get("cache_dir", "models/cache")
        
        cache_dir_path = Path(cache_dir)
        if cache_dir_path.is_absolute():
            # 已经是绝对路径，直接使用
            return str(cache_dir_path)
        else:
            # 相对路径，转换为相对于项目根目录的绝对路径
            return str((self.project_root / cache_dir_path).resolve())
    
    def preload_models(self) -> None:
        """预加载模型"""
        try:
            loading_config = self.config.get("loading", {})
            preload_config = loading_config.get("preload", {})
            
            if not preload_config.get("enabled", False):
                return
            
            models_to_preload = preload_config.get("models", [])
            
            for model_name in models_to_preload:
                try:
                    logger.info(f"预加载模型: {model_name}")
                    
                    # 判断模型类型并预加载
                    if self._is_embedding_model(model_name):
                        self.get_embedder(model_name)
                    elif self._is_reranker_model(model_name):
                        self.get_reranker(model_name)
                    elif self._is_llm_model(model_name):
                        self.get_llm(model_name)
                        
                except Exception as e:
                    logger.warning(f"预加载模型 {model_name} 失败: {e}")
                    
        except Exception as e:
            logger.error(f"预加载模型失败: {e}")
    
    def _is_embedding_model(self, model_name: str) -> bool:
        """判断是否为嵌入模型"""
        embedding_config = self.config.get("models", {}).get("embedding", {})
        
        # 检查本地模型
        local_models = embedding_config.get("local", {}).get("models", [])
        for model in local_models:
            if model["name"] == model_name:
                return True
        
        # 检查API模型
        api_providers = embedding_config.get("api", {}).get("providers", {})
        for provider_config in api_providers.values():
            models = provider_config.get("models", [])
            for model in models:
                if model["name"] == model_name:
                    return True
        
        return False
    
    def _is_reranker_model(self, model_name: str) -> bool:
        """判断是否为重排序模型"""
        # TODO: 实现重排序模型判断
        return False
    
    def _is_llm_model(self, model_name: str) -> bool:
        """判断是否为LLM模型"""
        # TODO: 实现LLM模型判断
        return False
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """获取可用模型列表"""
        available = {
            "embedding": {"local": [], "api": []},
            "reranker": {"local": [], "api": []},
            "llm": {"local": [], "api": []}
        }
        
        try:
            models_config = self.config.get("models", {})
            
            for model_type in ["embedding", "reranker", "llm"]:
                model_config = models_config.get(model_type, {})
                
                # 本地模型
                local_models = model_config.get("local", {}).get("models", [])
                available[model_type]["local"] = [model["name"] for model in local_models]
                
                # API模型
                api_providers = model_config.get("api", {}).get("providers", {})
                api_models = []
                for provider_config in api_providers.values():
                    models = provider_config.get("models", [])
                    api_models.extend([model["name"] for model in models])
                available[model_type]["api"] = api_models
                
        except Exception as e:
            logger.error(f"获取可用模型失败: {e}")
        
        return available
    
    def reload_config(self) -> None:
        """重新加载配置"""
        self.config = self._load_config()
        # 清空缓存，强制重新创建模型实例
        self._embedders.clear()
        self._rerankers.clear()
        self._llms.clear()

    # -------------------- 扩展：下载与格式分流 --------------------
    def _download_model_if_needed(self, model_name: str, source: str, cache_dir: Path) -> Optional[str]:
        """按来源下载模型，返回下载根目录（若可获取）。"""
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if source.lower() == "huggingface":
                return self._snapshot_download_hf(model_name, cache_dir)
            # 默认 modelscope
            return self._snapshot_download_modelscope(model_name, cache_dir)
        except Exception as e:
            logger.warning(f"下载模型失败: {e}")
            return None

    def _snapshot_download_modelscope(self, model_id: str, cache_dir: Path) -> Optional[str]:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            # 为每个模型单独创建子目录，避免ModelScope内部自动生成models/org/repo路径
            local_dir = str(cache_dir / model_id.replace('/', '_'))
            return snapshot_download(model_id, local_dir=local_dir)
        except Exception as e:
            logger.warning(f"ModelScope 下载失败: {e}")
            return None

    def _snapshot_download_hf(self, model_id: str, cache_dir: Path) -> Optional[str]:
        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download
            return hf_snapshot_download(repo_id=model_id, local_dir=str(cache_dir / model_id.replace('/', '_')), local_dir_use_symlinks=False)
        except Exception as e:
            logger.warning(f"HuggingFace 下载失败: {e}")
            return None

    def _detect_model_format(self, model_name: str, cache_dir: Path, download_root: Optional[str]) -> str:
        """检测格式：返回 "gguf" 或 "transformers""" 
        # 规则：名称含 gguf 或下载目录/缓存目录存在 .gguf 即认为 GGUF
        if "gguf" in model_name.lower():
            return "gguf"
        search_root = Path(download_root) if download_root else (cache_dir / model_name.replace('/', '_'))
        if not search_root.exists():
            search_root = cache_dir
        for _ in search_root.rglob("*.gguf"):
            return "gguf"
        return "transformers"


# 轻量 GGUF 嵌入器（llama-cpp-python）
class _LlamaCppEmbedder(BaseEmbedder):
    def __init__(self, config: ModelConfig, download_root: Optional[str]):
        super().__init__(config)
        self.download_root = download_root
        self.llm = None
        self._dim: Optional[int] = None

    def _find_best_gguf(self) -> Path:
        cache_dir = Path(self.config.cache_dir or "models/cache")
        search_root = Path(self.download_root) if self.download_root else (cache_dir / self.config.name.replace('/', '_'))
        if not search_root.exists():
            search_root = cache_dir
        candidates = list(search_root.rglob("*.gguf"))
        if not candidates:
            raise ModelLoadError("未找到 .gguf 文件")
        def score(p: Path) -> int:
            n = p.name.lower()
            if "q8_0" in n:
                return 3
            if "f16" in n:
                return 2
            return 1
        candidates.sort(key=score, reverse=True)
        return candidates[0]

    def load_model(self) -> None:
        try:
            from llama_cpp import Llama
        except Exception:
            raise ModelLoadError("未安装 llama-cpp-python，请安装: pip install llama-cpp-python")
        # 降噪：降低llama-cpp日志级别，并关闭verbose
        os.environ.setdefault("LLAMA_LOG_LEVEL", "ERROR")
        os.environ.setdefault("GGML_LOG_LEVEL", "ERROR")
        gguf_path = self._find_best_gguf()
        # 关闭verbose以避免加载与推理阶段的大量控制台输出
        self.llm = Llama(model_path=str(gguf_path), embedding=True, verbose=False)
        # 试算一个维度
        test_vec = self.llm.embed("test")
        self._dim = len(test_vec)
        self._is_loaded = True

    def embed_texts(self, texts: List[str]) -> Any:  # EmbeddingResult
        if not self._is_loaded or self.llm is None:
            raise ModelLoadError("模型未加载")
        vectors = [self.llm.embed(t) for t in texts]
        from .embedding.base import EmbeddingResult  # type: ignore
        return EmbeddingResult(
            embeddings=[list(map(float, v)) for v in vectors],
            model_name=self.config.name,
            dimension=self._dim or (len(vectors[0]) if vectors else 0),
            metadata={"provider": "llama_cpp", "gguf": True}
        )

    def embed_query(self, query: str) -> List[float]:
        res = self.embed_texts([query])
        return res.embeddings[0]

# -------------------- 全局便捷函数与实例 --------------------
# 全局模型注册器实例，便于外部直接调用 get_embedder()
model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """获取模型注册器实例"""
    return model_registry


def get_embedder(model_name: Optional[str] = None, provider: str = "local") -> BaseEmbedder:
    """获取嵌入模型（带自动下载与格式分流）"""
    return model_registry.get_embedder(model_name, provider)


def get_reranker(model_name: Optional[str] = None, provider: str = "local") -> BaseReranker:
    """获取重排序模型"""
    return model_registry.get_reranker(model_name, provider)


def get_llm(model_name: Optional[str] = None, provider: str = "local") -> BaseLLM:
    """获取大语言模型"""
    return model_registry.get_llm(model_name, provider)