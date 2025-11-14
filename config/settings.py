import os
from pathlib import Path
from typing import Dict, List, Any

class Settings:
    """系统配置类"""

    # 基础路径
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    VECTOR_STORE_DIR = BASE_DIR / "vector_store"
    CHAT_HISTORY_DIR = BASE_DIR / "chat_history"
    LOG_DIR = BASE_DIR / "logs"

    # 文件路径
    VECTOR_STORE_PATH = VECTOR_STORE_DIR / "faiss_index"
    CHAT_HISTORY_PATH = CHAT_HISTORY_DIR / "chat_history.json"
    LOG_FILE = LOG_DIR / "app.log"

    # ==================== LLM 提供商配置 ====================
    # 可选: "ollama" 或 "online"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "online")  # 默认使用 API 方式

    # ==================== Ollama 配置 ====================
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODELS = [
        "qwen:7b",
        "qwen:14b",
        "qwen:32b",
        "llama2:7b",
        "llama2:13b",
        "llama2:70b",
        "mistral:7b",
        "codellama:7b",
        "vicuna:7b",
        "baichuan:7b"
    ]
    OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

    # ==================== 在线 API 配置（阿里云百炼） ====================
    ONLINE_API_KEY = os.getenv("ONLINE_API_KEY", "sk-abe3417c96f6441b83efed38708bcfb6")
    ONLINE_BASE_URL = os.getenv("ONLINE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    ONLINE_MODELS = [
        "qwen-plus",
        "qwen-turbo",
        "qwen-max",
        "qwen-max-longcontext"
    ]
    ONLINE_EMBEDDING_MODEL = "text-embedding-v1"  # 阿里云百炼的嵌入模型

    # ==================== 通用模型配置 ====================
    @classmethod
    def get_available_models(cls) -> List[str]:
        """根据 LLM 提供商返回可用模型列表"""
        provider = os.getenv("LLM_PROVIDER", cls.LLM_PROVIDER)
        if provider == "ollama":
            return cls.OLLAMA_MODELS
        else:  # online
            return cls.ONLINE_MODELS

    @classmethod
    def get_default_model(cls) -> str:
        """获取默认模型"""
        provider = os.getenv("LLM_PROVIDER", cls.LLM_PROVIDER)
        if provider == "ollama":
            return "qwen:7b"
        else:  # online
            return "qwen-plus"

    @classmethod
    def get_embedding_model(cls) -> str:
        """获取嵌入模型"""
        provider = os.getenv("LLM_PROVIDER", cls.LLM_PROVIDER)
        if provider == "ollama":
            return cls.OLLAMA_EMBEDDING_MODEL
        else:  # online
            return cls.ONLINE_EMBEDDING_MODEL

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_TOP_K = 3
    DEFAULT_SEARCH_TYPE = "similarity"  # similarity 或 mmr

    # 向量存储配置
    VECTOR_DIMENSION = 768  # Ollama nomic-embed-text 和阿里云 text-embedding-v1 都是 768 维
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # 天气API配置
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "73053d990f2e27ad6e600344eee77866")  # 请替换为实际的API密钥
    WEATHER_API_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
    WEATHER_CITY_URL = "https://restapi.amap.com/v3/config/district"

    # 文档处理配置
    SUPPORTED_FILE_TYPES = [
        ".pdf",
        ".txt",
        ".md",
        ".docx"
    ]

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    CACHE_ENABLED = True
    CACHE_EXPIRE_TIME = 3600  # 1小时

    # 显示配置
    MAX_CHAT_HISTORY_DISPLAY = 100
    MESSAGE_TRUNCATE_LENGTH = 500

    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5

    # 系统配置
    ENABLE_FILE_CACHE = True
    ENABLE_CHAT_HISTORY = True
    ENABLE_VECTOR_STORE = True
    ENABLE_WEATHER_TOOL = True
    ENABLE_DOCUMENT_TOOL = True

    @classmethod
    def initialize_directories(cls):
        """初始化必要的目录"""
        directories = [
            cls.DATA_DIR,
            cls.VECTOR_STORE_DIR,
            cls.CHAT_HISTORY_DIR,
            cls.LOG_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """获取模型配置"""
        # Ollama 模型配置
        ollama_configs = {
            "qwen:7b": {
                "temperature": cls.DEFAULT_TEMPERATURE,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "qwen:14b": {
                "temperature": cls.DEFAULT_TEMPERATURE,
                "max_tokens": cls.DEFAULT_MAX_TOKENS * 2,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "llama2:7b": {
                "temperature": 0.8,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
                "top_p": 0.95,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
        }

        # 在线 API 模型配置
        online_configs = {
            "qwen-plus": {
                "temperature": cls.DEFAULT_TEMPERATURE,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
                "top_p": 0.9,
            },
            "qwen-turbo": {
                "temperature": cls.DEFAULT_TEMPERATURE,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
                "top_p": 0.9,
            },
            "qwen-max": {
                "temperature": cls.DEFAULT_TEMPERATURE,
                "max_tokens": 4096,
                "top_p": 0.9,
            }
        }

        # 根据提供商选择配置
        provider = os.getenv("LLM_PROVIDER", cls.LLM_PROVIDER)
        if provider == "ollama":
            return ollama_configs.get(model_name, {
                "temperature": cls.DEFAULT_TEMPERATURE,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            })
        else:  # online
            return online_configs.get(model_name, {
                "temperature": cls.DEFAULT_TEMPERATURE,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
                "top_p": 0.9,
            })

    @classmethod
    def get_provider_info(cls) -> Dict[str, str]:
        """获取当前提供商信息"""
        provider = os.getenv("LLM_PROVIDER", cls.LLM_PROVIDER)
        if provider == "ollama":
            return {
                "provider": "Ollama (本地)",
                "base_url": cls.OLLAMA_BASE_URL,
                "embedding": cls.OLLAMA_EMBEDDING_MODEL
            }
        else:  # online
            return {
                "provider": "阿里云百炼 (在线)",
                "base_url": cls.ONLINE_BASE_URL,
                "embedding": cls.ONLINE_EMBEDDING_MODEL
            }
