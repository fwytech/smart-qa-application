# LangChain+Agentic RAG智能问答系统 - 实战教程

本系列教程将带你从零构建一个生产级的智能问答系统，支持双模式LLM（Ollama本地 + 阿里云百炼在线API）。

## 📚 教程目录

### 第01章：项目架构设计与环境搭建 - 从零构建生产级RAG系统
- Agentic RAG核心概念
- 技术栈选型与架构设计
- 项目环境搭建
- 依赖配置与验证

### 第02章：配置中心 - 双模式LLM的统一管理与灵活切换  
- 配置中心设计原则
- 双模式LLM配置管理
- Settings类完整实现（216行）
- 配置验证与测试

### 第03章：统一LLM客户端 - Ollama与在线API的无缝集成
- 统一接口设计（适配器模式）
- UnifiedLLMClient实现（155行）
- UnifiedEmbeddingClient实现（90行）
- 双模式无缝切换

### 第04章：FAISS向量存储 - 构建高性能语义检索引擎
- FAISS向量库原理
- VectorStoreService完整实现（476行）
- 多策略检索（similarity/mmr）
- 索引持久化与管理

### 第05章：辅助工具类(上) - 装饰器与文档处理器的实战应用
- 装饰器工具（error_handler/log_execution/performance_monitor）
- DocumentProcessor文档处理器（565行）
- 多格式文档支持
- 缓存机制实现

### 第06章：辅助工具类(下) - 天气查询、聊天历史与UI组件
- WeatherService天气服务（331行）
- ChatHistoryManager聊天历史（488行）
- UIComponents界面组件（139行）

### 第07章：Agentic RAG核心 - ReAct智能代理的完整实现
- Agentic RAG架构设计
- AgenticRAGAgent核心实现（629行）
- 三层路由系统（direct/tool_direct/react）
- ReAct多步推理框架

### 第08章：Streamlit Web应用 - 打造生产级智能问答界面
- Streamlit应用架构
- AgenticRAGSystem主应用（693行）
- 界面布局与交互设计
- 完整系统集成

### 第09章：系统集成测试与生产环境部署优化
- 功能测试流程
- Docker部署方案
- 性能优化建议
- 常见问题排查

## 🎯 学习目标

完成本系列教程后，你将掌握：

✅ Agentic RAG系统的完整设计与实现
✅ 双模式LLM（Ollama + 云端API）架构
✅ FAISS向量存储与语义检索
✅ LangChain Agent开发
✅ ReAct推理框架应用
✅ Streamlit Web应用开发
✅ 生产环境部署与优化

## 📊 项目统计

- **总代码量**：约4000+行
- **教程文档**：9章
- **总字数**：约8万字
- **核心模块**：10个
- **预计学习时间**：20-30小时

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/fwytech/smart-qa-application.git
cd smart-qa-application

# 按照第01章配置环境
pip install -e .

# 配置.env文件（按照第01章）
cp .env.example .env

# 运行应用（第08章）
streamlit run app.py
```

## 💡 学习建议

1. **按顺序学习**：每章都基于前面的内容
2. **动手实践**：每完成一章就运行测试
3. **代码对比**：确保代码与项目源码一致
4. **深入理解**：不仅要会写，更要懂原理

## 📞 反馈与交流

- 项目地址：https://github.com/fwytech/smart-qa-application
- 作者邮箱：fwytech@126.com

---

Happy Coding! 🎉
