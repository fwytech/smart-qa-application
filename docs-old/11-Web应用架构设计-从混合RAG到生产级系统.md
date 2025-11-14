# 第11讲：Web应用架构设计 - 从混合RAG到生产级系统

在前面10讲中，我们构建了完整的混合RAG系统，但它只是命令行工具。真实场景中，用户需要一个**友好的Web界面**。

这一讲，我们将设计一个**生产级智能问答Web应用**，让非技术用户也能轻松使用Agentic RAG。

---

## 一、从CLI到Web应用的演进

### CLI版本的局限

**混合RAG系统（第10讲）：**

```python
# 使用方式（命令行）
hybrid_rag = HybridRAG()
hybrid_rag.add_documents(docs)
result = hybrid_rag.query("孙悟空是谁？")
print(result["answer"])
```

**问题：**
- ❌ 需要编程知识才能使用
- ❌ 无法方便地上传文档
- ❌ 没有历史对话记录
- ❌ 无法保存和分享会话
- ❌ 不支持多用户

### Web应用的优势

```
┌─────────────────────────────────────────────────┐
│         生产级智能问答Web应用                      │
│                                                 │
│  用户界面                                         │
│  ├─ 聊天对话框 →  轻松提问                         │
│  ├─ 文档上传   →  拖拽上传PDF/Word/TXT             │
│  ├─ 历史记录   →  查看和导出                       │
│  ├─ 设置面板   →  可视化配置                       │
│  └─ 实时反馈   →  进度条、状态提示                  │
│                                                 │
│  技术特性                                         │
│  ├─ 多用户支持                                    │
│  ├─ 会话管理                                      │
│  ├─ 权限控制                                      │
│  ├─ 数据持久化                                    │
│  └─ 性能监控                                      │
└─────────────────────────────────────────────────┘
```

---

## 二、技术选型

### Web框架选择

**为什么选择Streamlit？**

| 框架 | 优点 | 缺点 | 适合场景 |
|------|------|------|---------|
| **Streamlit** | 快速开发、纯Python、自动刷新 | 定制化有限 | **内部工具、MVP** |
| Flask | 灵活、轻量 | 需要前端知识 | API服务 |
| FastAPI | 高性能、自动文档 | 需要前端分离 | 微服务 |
| Django | 功能全面 | 学习曲线陡 | 大型应用 |

**Streamlit的核心优势：**

```python
# 只需几行代码就能创建UI
import streamlit as st

st.title("智能问答系统")
question = st.text_input("请输入问题")
if st.button("提问"):
    answer = generate_answer(question)
    st.write(answer)
```

- ✅ 纯Python开发，无需HTML/CSS/JavaScript
- ✅ 自动响应式设计
- ✅ 内置组件丰富（文件上传、图表、表格）
- ✅ 适合数据科学和AI应用
- ✅ 快速原型到生产（几天即可上线）

### 向量数据库选择

**为什么选择FAISS？**

| 向量库 | 优点 | 缺点 | 适合场景 |
|-------|------|------|---------|
| **FAISS** | 速度快、本地部署、免费 | 无分布式 | **中小规模** |
| ChromaDB | 简单易用、持久化 | 性能一般 | 教学演示 |
| Pinecone | 云原生、分布式 | 收费 | 大规模生产 |
| Milvus | 功能强大、分布式 | 复杂 | 企业级 |

**FAISS特点：**
- Facebook开发的高性能向量搜索库
- 支持百万级向量快速检索（<100ms）
- 本地部署，无需外部服务
- 内存高效，支持量化压缩

### LLM提供商：双模式支持

**创新设计：Ollama + 阿里云百炼双模式**

```
┌─────────────────────────────────────────────────┐
│            双模式LLM架构                          │
│                                                 │
│  模式1：Ollama（本地）                            │
│  ├─ 适用场景：开发测试、隐私要求高                │
│  ├─ 优势：免费、离线、数据隐私                    │
│  └─ 劣势：需要GPU、模型能力有限                   │
│                                                 │
│  模式2：阿里云百炼（在线）                         │
│  ├─ 适用场景：生产环境、高质量要求                │
│  ├─ 优势：模型强大、无需GPU、稳定可靠             │
│  └─ 劣势：按调用付费、需要网络                   │
│                                                 │
│  统一接口：UnifiedLLMClient                      │
│  └─ 根据配置自动切换提供商                       │
└─────────────────────────────────────────────────┘
```

**为什么双模式？**

1. **开发阶段**：使用Ollama本地测试（免费、快速迭代）
2. **生产环境**：切换到百炼在线API（高质量、稳定）
3. **私有部署**：使用Ollama保护数据隐私
4. **成本优化**：根据流量灵活切换

---

## 三、系统架构设计

### 整体架构图

```
┌──────────────────────────────────────────────────────┐
│                 用户浏览器                            │
└────────────────┬─────────────────────────────────────┘
                 │
                 │ HTTP
                 ↓
┌──────────────────────────────────────────────────────┐
│              Streamlit Web服务                        │
│                                                      │
│  app.py (主应用入口)                                  │
│     ├─ 页面路由                                       │
│     ├─ UI渲染                                         │
│     └─ 会话管理                                       │
└────────────────┬─────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ↓                 ↓
┌──────────────┐   ┌──────────────┐
│  Config层     │   │  Models层    │
│              │   │              │
│  settings.py │   │  agent.py    │
│  ├─双模式配置 │   │  └─Agent代理  │
│  ├─参数管理   │   │              │
│  └─路径配置   │   └──────────────┘
└──────────────┘
        │
        ↓
┌──────────────────────────────────────────────┐
│              Services层（服务层）              │
│                                              │
│  ┌──────────────┐  ┌──────────────┐         │
│  │ llm_client   │  │vector_store  │         │
│  │ ├─Ollama     │  │ ├─FAISS      │         │
│  │ └─百炼API    │  │ └─嵌入模型   │         │
│  └──────────────┘  └──────────────┘         │
│                                              │
│  ┌──────────────┐                            │
│  │weather_tools │  (可扩展更多工具)          │
│  │ └─天气查询   │                            │
│  └──────────────┘                            │
└──────────────────────────────────────────────┘
        │
        ↓
┌──────────────────────────────────────────────┐
│              Utils层（工具层）                 │
│                                              │
│  ┌─────────────────┐  ┌──────────────┐      │
│  │document_processor│  │chat_history  │      │
│  │ ├─文件解析       │  │ ├─记录保存    │      │
│  │ ├─文本分块       │  │ └─导出CSV     │      │
│  │ └─元数据提取     │  │              │      │
│  └─────────────────┘  └──────────────┘      │
│                                              │
│  ┌──────────────┐  ┌──────────────┐         │
│  │ui_components │  │decorators    │         │
│  │ └─UI辅助     │  │ └─错误处理   │         │
│  └──────────────┘  └──────────────┘         │
└──────────────────────────────────────────────┘
```

### 分层设计原则

**为什么要分层？**

1. **职责分离**：每一层只负责特定功能
2. **易于测试**：可以独立测试每一层
3. **便于扩展**：添加新功能不影响其他层
4. **团队协作**：不同层可由不同人开发

**各层职责：**

| 层级 | 职责 | 示例 |
|------|------|------|
| **App层** | 页面渲染、用户交互 | `app.py` |
| **Config层** | 配置管理、参数定义 | `settings.py` |
| **Models层** | 核心业务逻辑 | `agent.py` |
| **Services层** | 外部服务调用 | `llm_client.py`, `vector_store.py` |
| **Utils层** | 通用工具函数 | `document_processor.py` |

---

## 四、目录结构

### 完整的项目结构

```
agentic_rag_smart_qa_project/
├── app.py                          # 主应用入口（约365行）
│
├── config/                         # 配置层
│   ├── __init__.py
│   └── settings.py                 # 系统配置（约211行）
│       ├─ LLM提供商配置（Ollama + 百炼）
│       ├─ 模型参数配置
│       ├─ 路径配置
│       └─ 工具开关
│
├── models/                         # 模型层
│   ├── __init__.py
│   └── agent.py                    # Agent代理（约230行）
│       ├─ LangChain集成
│       ├─ ReAct框架
│       ├─ 工具编排
│       └─ 记忆管理
│
├── services/                       # 服务层
│   ├── __init__.py
│   ├── llm_client.py              # LLM客户端（统一接口）
│   ├── vector_store.py            # 向量存储服务（约310行）
│   └── weather_tools.py           # 天气工具
│
├── utils/                          # 工具层
│   ├── __init__.py
│   ├── document_processor.py      # 文档处理
│   ├── chat_history.py            # 聊天历史
│   ├── ui_components.py           # UI组件
│   └── decorators.py              # 装饰器（错误处理、日志）
│
├── data/                           # 数据目录
│   └── documents/                 # 上传的文档
│
├── vector_store/                   # 向量存储
│   └── faiss_index/               # FAISS索引文件
│
├── chat_history/                   # 聊天记录
│   └── chat_history.json          # 历史记录文件
│
├── logs/                           # 日志目录
│   └── app.log                    # 应用日志
│
├── requirements.txt                # 依赖列表
├── .env                           # 环境变量（不提交到Git）
└── README.md                      # 项目说明
```

### 为什么这么组织？

1. **扁平化设计**
   - 不超过2层嵌套
   - 便于快速定位文件
   - 符合Python习惯

2. **按职责分组**
   - config：所有配置集中管理
   - models：业务逻辑
   - services：外部调用
   - utils：通用工具

3. **数据分离**
   - 代码和数据分开
   - data/、vector_store/、chat_history/独立
   - 便于备份和迁移

---

## 五、核心配置设计

### uv 项目管理

1、初始化项目结构：

```bash
uv init 03-smart-qa-application
```

2、修改 `pyproject.toml`:

```
[project]
name = "agentic-rag-smart-qa"
version = "2.0.0"
description = "Agentic RAG智能问答系统 - 生产级Web应用，支持双模式LLM（Ollama本地 + 在线API）"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
authors = [
    {name = "fwytech", email = "fwytech@126.com"}
]

dependencies = [
    # Web 框架
    "streamlit>=1.29.0",
    # AI 框架
    "langchain>=0.2.0",
    "langchain-community>=0.2.0",
    "langchain-core>=0.1.0",
    # 向量数据库
    "faiss-cpu>=1.8.0",
    # LLM 客户端
    "ollama==0.1.7",
    "openai>=1.0.0",
    # 文档处理
    "pypdf==3.17.0",
    "python-docx==1.1.0",
    "unstructured>=0.12.0",
    # HTTP 请求
    "requests==2.31.0",
    "urllib3==2.1.0",
    # 数据处理
    "numpy>=1.27.0",
    "pandas>=2.2.0",
    # 可视化
    "plotly==5.18.0",
    # 机器学习
    "sentence-transformers>=2.0.0",
    "torch>=2.2.0",
    "transformers>=4.36.0",
    # 浏览器自动化
    "selenium>=4.15.0,<5.0.0",
    "webdriver-manager>=4.0.0,<5.0.0",
    # 其他工具
    "python-dotenv>=1.0.0,<2.0.0",
    "json5>=0.9.0,<1.0.0",
    "markdown>=3.5.0,<4.0.0",
    "tiktoken>=0.12.0",
]

[build-system]
requires = ["setuptools>=68.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["config*", "services*", "utils*"]
exclude = ["data*", "files*"]

[[tool.uv.index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

```

3、同步安装项目依赖包：

```bash
cd study-agentic-rag/03-smart-qa-application
uv sync
```

### settings.py核心内容

**代码文件：** `03-smart-qa-application/config/settings.py`

```python
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
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # 默认使用 Ollama

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

```

**为什么这么写？**

1. **为什么用 `Path` 而不是字符串？**
   - `Path` 自动处理跨平台路径分隔符（Windows用`\`，Linux用`/`）
   - 支持链式操作：`BASE_DIR / "data" / "docs"`
   - 更安全，自动转义特殊字符

2. **为什么用 `os.getenv()` 读取环境变量？**
   - 安全：敏感信息（API密钥）不写在代码里
   - 灵活：不同环境（开发/测试/生产）用不同配置
   - 标准：符合12要素应用（12-Factor App）原则

3. **为什么用 `@classmethod`？**
   - 无需实例化，直接调用：`Settings.get_default_model()`
   - 节省内存，全局共享配置
   - 便于在任何地方访问配置

4. **为什么要工具开关？**
   - 开发环境可能没有天气API密钥
   - 按需启用工具，降低复杂度
   - 便于调试和测试

### 环境变量配置（.env文件）

```bash
# .env 文件（不提交到Git，添加到.gitignore）

# LLM提供商选择
LLM_PROVIDER=ollama  # 或 online

# Ollama配置（本地部署）
OLLAMA_BASE_URL=http://localhost:11434

# 阿里云百炼配置（在线API）
ONLINE_API_KEY=sk-xxxx
ONLINE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 天气API密钥（可选）
WEATHER_API_KEY=你的高德地图API密钥

# 日志级别
LOG_LEVEL=INFO
```

**使用方式：**

```python
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 现在可以用os.getenv()读取
api_key = os.getenv("ONLINE_API_KEY")
```

---

## 六、数据流转流程

### 用户提问的完整流程

```
用户在Web界面输入问题
   ↓
app.py: 捕获用户输入
   ↓
app.py: 调用 generate_response(query)
   ↓
app.py: 创建Agent（如果未创建）
   ├─ 从settings读取配置
   ├─ 创建llm_client（根据LLM_PROVIDER选择Ollama或百炼）
   ├─ 创建vector_store（FAISS）
   └─ 注册工具（文档搜索、天气查询）
   ↓
models/agent.py: agent.generate_response(query)
   ├─ LangChain ReAct循环
   ├─ 决定是否使用工具
   │  ├─ 使用文档搜索 → services/vector_store.py
   │  └─ 使用天气查询 → services/weather_tools.py
   ├─ LLM推理和生成 → services/llm_client.py
   └─ 返回答案
   ↓
app.py: 显示答案到界面
   ↓
app.py: 保存到聊天历史 → utils/chat_history.py
   ↓
用户看到回答
```

### 文档上传的完整流程

```
用户上传PDF文件
   ↓
app.py: 捕获文件
   ↓
app.py: process_uploaded_files(files)
   ↓
utils/document_processor.py: 解析文件
   ├─ PDF → 提取文本
   ├─ Word → 提取文本
   ├─ TXT/MD → 直接读取
   ├─ 文本清洗
   └─ 文本分块（chunk_size=1000, overlap=200）
   ↓
services/vector_store.py: add_documents(chunks)
   ├─ 生成嵌入向量（使用llm_client的嵌入模型）
   ├─ 构建FAISS索引
   └─ 保存索引到磁盘
   ↓
app.py: 更新向量存储状态
   ↓
用户看到"向量存储已准备"
```

---

## 七、关键设计决策

### 决策1：为什么用LangChain？

**LangChain的价值：**

```python
# 不用LangChain（需要自己实现ReAct）
def my_react_loop(query):
    for i in range(max_iterations):
        # 自己写Think逻辑
        decision = think(query, context)
        # 自己写Act逻辑
        observation = act(decision)
        # 自己写Observe逻辑
        is_done = observe(observation)
        if is_done:
            break
    return generate(context)

# 用LangChain（框架已实现）
from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent, tools, max_iterations=5)
result = executor.invoke({"input": query})
```

**优势：**
- ✅ 成熟的ReAct实现（经过大量测试）
- ✅ 工具集成简单（`Tool`类）
- ✅ 记忆管理（`ConversationBufferMemory`）
- ✅ 丰富的集成（FAISS、OpenAI、Ollama）
- ✅ 社区活跃，问题好解决

### 决策2：为什么用会话状态管理？

**Streamlit的特点：**

```python
# Streamlit每次交互都会重新运行整个脚本
st.title("智能问答")

# 问题：每次都会重置为空列表
chat_history = []  # ❌ 用户刷新页面，历史丢失

# 解决：使用session_state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 添加消息
st.session_state.chat_history.append({"role": "user", "content": query})

# 历史会保留在整个会话中 ✅
```

**为什么需要？**
- Streamlit无状态架构，每次交互重新运行
- `session_state`保持会话数据
- 聊天历史、向量存储状态都依赖它

### 决策3：为什么要错误处理装饰器？

```python
# utils/decorators.py
def error_handler(func):
    """错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} 出错: {str(e)}")
            st.error(f"操作失败: {str(e)}")
            return None
    return wrapper

# 使用
@error_handler
def process_uploaded_files(self, files):
    # 文件处理逻辑
    ...
```

**价值：**
- 统一错误处理逻辑
- 自动记录日志
- 友好的用户错误提示
- 代码更简洁

---

## 八、总结

### 核心架构特点

1. **分层设计**
   - 4层架构：App → Config/Models → Services → Utils
   - 职责清晰，易于维护

2. **双模式LLM**
   - Ollama（本地）+ 阿里云百炼（在线）
   - 统一接口，灵活切换

3. **技术选型**
   - Streamlit：快速构建Web UI
   - FAISS：高性能向量检索
   - LangChain：成熟的Agent框架

4. **生产就绪**
   - 配置管理（环境变量）
   - 错误处理（装饰器）
   - 日志记录
   - 数据持久化

### 架构优势

| 特性 | 实现方式 | 价值 |
|------|---------|------|
| **快速开发** | Streamlit纯Python | 几天上线MVP |
| **灵活部署** | 双模式LLM | 本地+云端 |
| **易于扩展** | 分层+工具系统 | 添加新功能简单 |
| **用户友好** | Web界面 | 非技术用户可用 |
| **成本优化** | Ollama免费 | 降低开发成本 |


