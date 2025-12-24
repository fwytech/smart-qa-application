import streamlit as st
import os
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from config.settings import Settings
try:
    from langchain.globals import set_verbose
except Exception:
    def set_verbose(_: bool):
        return None
from models.agent import AgenticRAGAgent
from services.vector_store import VectorStoreService
from services.weather_tools import WeatherService
from utils.document_processor import DocumentProcessor
from utils.ui_components import UIComponents
from utils.chat_history import ChatHistoryManager
from utils.decorators import error_handler, log_execution

class AgenticRAGSystem:
    """主应用类 - Agentic RAG智能问答系统"""

    def __init__(self):
        self.settings = Settings()
        self.vector_store = VectorStoreService()
        self.weather_tools = WeatherService()
        self.doc_processor = DocumentProcessor()
        self.ui_components = UIComponents()
        self.chat_history = ChatHistoryManager()
        self.agent = None
        self._initialize_system()

    @error_handler()
    def _initialize_system(self):
        """初始化系统组件"""
        logging.basicConfig(
            level=getattr(logging, self.settings.LOG_LEVEL, logging.INFO),
            format=self.settings.LOG_FORMAT,
            force=True
        )
        st.set_page_config(
            page_title="Agentic RAG智能问答系统",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.markdown(
            """
            <style>
            /* 容器左右各5rem留白 */
            div.block-container{max-width:calc(100vw - 10rem); padding-left:5rem; padding-right:5rem;}
            /* 两列之间固定5rem间距，且保持不换行 */
            div[data-testid="stHorizontalBlock"]{gap:5rem; flex-wrap:nowrap;}
            /* 窄屏自动缩小间距，避免拥挤 */
            @media (max-width: 1280px){
              div[data-testid="stHorizontalBlock"]{gap:3rem;}
            }
            @media (max-width: 768px){
              div[data-testid="stHorizontalBlock"]{gap:1rem;}
            }
            /* 调整标题字号 */
            h2{font-size:1.5rem !important; line-height:1.25 !important;}
            h3{font-size:1.25rem !important; line-height:1.25 !important;}
            /* 主页英雄标题更大字号 */
            .hero-title{font-size:2rem !important;}
            </style>
            """,
            unsafe_allow_html=True
        )

        # 初始化会话状态
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.chat_history = []
            st.session_state.vector_store_ready = False
            st.session_state.current_model = self.settings.get_default_model()
            st.session_state.temperature = self.settings.DEFAULT_TEMPERATURE
            st.session_state.max_tokens = self.settings.DEFAULT_MAX_TOKENS
            st.session_state.top_k = self.settings.DEFAULT_TOP_K
            st.session_state.search_type = self.settings.DEFAULT_SEARCH_TYPE
            st.session_state.llm_provider = self.settings.LLM_PROVIDER
            st.session_state.processed_upload_ids = []

        # 创建必要的目录
        os.makedirs(self.settings.DATA_DIR, exist_ok=True)
        os.makedirs(self.settings.VECTOR_STORE_DIR, exist_ok=True)
        os.makedirs(self.settings.CHAT_HISTORY_DIR, exist_ok=True)

        # 自动加载已有向量存储索引
        if os.path.exists(self.settings.VECTOR_STORE_PATH) and not st.session_state.vector_store_ready:
            try:
                if self.vector_store.load_index(self.settings.VECTOR_STORE_PATH):
                    st.session_state.vector_store_ready = True
            except Exception:
                pass

    @error_handler()
    def _create_agent(self):
        """创建Agent实例"""
        tools = []

        # 始终注册文档搜索工具（内部自行处理未就绪情况）
        tools.append(self._create_document_search_tool())
        # 注册多策略检索工具（similarity / mmr / random）
        tools.append(self._create_document_search_tool_similarity())
        tools.append(self._create_document_search_tool_mmr())
        tools.append(self._create_document_search_tool_random())

        # 添加天气查询工具
        tools.append(self._create_weather_tool())

        # 添加日期时间工具
        tools.append(self._create_datetime_tool())

        self.agent = AgenticRAGAgent(
            model_name=st.session_state.current_model,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            tools=tools if tools else None,
            route_mode=st.session_state.route_mode
        )

    def _create_document_search_tool(self):
        """创建文档搜索工具"""
        def document_search(query: str, top_k: Optional[int] = None) -> str:
            """搜索文档中的相关信息"""
            try:
                if not st.session_state.vector_store_ready:
                    return "向量存储未准备好，请先上传文档。"

                top_k = top_k or st.session_state.top_k
                results = self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    search_type=st.session_state.search_type
                )

                if not results:
                    return "未找到相关文档信息。"

                # 格式化搜索结果（相关度百分比）
                formatted_results = []
                scores = [r.get('score', 0.0) for r in results]
                mn = min(scores) if scores else 0.0
                mx = max(scores) if scores else 1.0
                n = len(results)
                for i, result in enumerate(results, 1):
                    src = (result.get('metadata') or {}).get('source', '未知来源')
                    if st.session_state.search_type == "similarity":
                        raw = result.get('score', None)
                        if isinstance(raw, (int, float)):
                            if n <= 1 or mx == mn:
                                percent = 100
                            else:
                                percent = round(100 * (mx - raw) / (mx - mn))
                            percent = max(0, min(100, percent))
                        else:
                            percent = round(100 - (i - 1) * (70 / (n - 1 if n > 1 else 1)))
                    else:
                        n = len(results)
                        percent = round(100 - (i - 1) * (70 / (n - 1 if n > 1 else 1)))
                    raw = result.get('score', None)
                    raw_str = f"{raw:.3f}" if isinstance(raw, (int, float)) else "—"
                    formatted_results.append(
                        f"【文档{i}】\n内容: {result['content']}\n"
                        f"相关度: {percent}%（原始分数：{raw_str}）\n"
                        f"来源: {src}\n"
                    )

                return "\n".join(formatted_results)

            except Exception as e:
                return f"文档搜索出错: {str(e)}"

        return document_search

    def _create_document_search_tool_similarity(self):
        """创建基于 similarity 的文档搜索工具"""
        def document_search_similarity(query: str, top_k: Optional[int] = None) -> str:
            try:
                if not st.session_state.vector_store_ready:
                    return "向量存储未准备好，请先上传文档。"
                top_k = top_k or st.session_state.top_k
                results = self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    search_type="similarity"
                )
                if not results:
                    return "未找到相关文档信息。"
                formatted = []
                scores = [r.get('score', 0.0) for r in results]
                mn = min(scores) if scores else 0.0
                mx = max(scores) if scores else 1.0
                n = len(results)
                for i, r in enumerate(results, 1):
                    src = (r.get('metadata') or {}).get('source', '未知来源')
                    raw = r.get('score', None)
                    if isinstance(raw, (int, float)):
                        if n <= 1 or mx == mn:
                            percent = 100
                        else:
                            percent = round(100 * (mx - raw) / (mx - mn))
                        percent = max(0, min(100, percent))
                    else:
                        percent = round(100 - (i - 1) * (70 / (n - 1 if n > 1 else 1)))
                    raw = r.get('score', None)
                    raw_str = f"{raw:.3f}" if isinstance(raw, (int, float)) else "—"
                    formatted.append(
                        f"【文档{i}】\n内容: {r['content']}\n相关度: {percent}%（原始分数：{raw_str}）\n来源: {src}\n"
                    )
                return "\n".join(formatted)
            except Exception as e:
                return f"文档搜索出错: {str(e)}"
        return document_search_similarity

    def _create_document_search_tool_mmr(self):
        """创建基于 mmr 的文档搜索工具"""
        def document_search_mmr(query: str, top_k: Optional[int] = None) -> str:
            try:
                if not st.session_state.vector_store_ready:
                    return "向量存储未准备好，请先上传文档。"
                top_k = top_k or st.session_state.top_k
                results = self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    search_type="mmr"
                )
                if not results:
                    return "未找到相关文档信息。"
                formatted = []
                n = len(results)
                for i, r in enumerate(results, 1):
                    src = (r.get('metadata') or {}).get('source', '未知来源')
                    percent = round(100 - (i - 1) * (70 / (n - 1 if n > 1 else 1)))
                    raw = r.get('score', None)
                    raw_str = f"{raw:.3f}" if isinstance(raw, (int, float)) else "—"
                    formatted.append(
                        f"【文档{i}】\n内容: {r['content']}\n相关度: {percent}%（原始分数：{raw_str}）\n来源: {src}\n"
                    )
                return "\n".join(formatted)
            except Exception as e:
                return f"文档搜索出错: {str(e)}"
        return document_search_mmr

    def _create_document_search_tool_random(self):
        """创建随机策略的文档搜索工具（在 similarity 与 mmr 中随机）"""
        import random
        def document_search_random(query: str, top_k: Optional[int] = None) -> str:
            try:
                if not st.session_state.vector_store_ready:
                    return "向量存储未准备好，请先上传文档。"
                algo = random.choice(["similarity", "mmr"])
                top_k = top_k or st.session_state.top_k
                results = self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    search_type=algo
                )
                if not results:
                    return "未找到相关文档信息。"
                formatted = []
                # 根据实际算法选择百分比映射
                scores = [r.get('score', 0.0) for r in results]
                mn = min(scores) if scores else 0.0
                mx = max(scores) if scores else 1.0
                n = len(results)
                for i, r in enumerate(results, 1):
                    src = (r.get('metadata') or {}).get('source', '未知来源')
                    raw = r.get('score', None)
                    if isinstance(raw, (int, float)) and n > 1 and mx != mn:
                        percent = round(100 * (mx - raw) / (mx - mn))
                        percent = max(0, min(100, percent))
                    else:
                        percent = 100 if n <= 1 or mx == mn else round(100 - (i - 1) * (70 / (n - 1 if n > 1 else 1)))
                    raw = r.get('score', None)
                    raw_str = f"{raw:.3f}" if isinstance(raw, (int, float)) else "—"
                    formatted.append(
                        f"【文档{i}】\n内容: {r['content']}\n相关度: {percent}%（原始分数：{raw_str}）\n来源: {src}\n"
                    )
                return "\n".join(formatted)
            except Exception as e:
                return f"文档搜索出错: {str(e)}"
        return document_search_random

    def _create_weather_tool(self):
        """创建天气查询工具"""
        def weather_query(city: str, forecast_days: int = 1) -> str:
            """查询天气信息"""
            try:
                if forecast_days == 1:
                    return self.weather_tools.get_current_weather(city)
                else:
                    return self.weather_tools.get_weather_forecast(city, forecast_days)
            except Exception as e:
                return f"天气查询出错: {str(e)}"

        return weather_query    

    def _create_datetime_tool(self):
        """创建日期时间查询工具"""
        from datetime import datetime
        def datetime_now(_: str = "") -> str:
            """返回当前日期时间与星期信息"""
            now = datetime.now()
            week_map = {0: "星期一", 1: "星期二", 2: "星期三", 3: "星期四", 4: "星期五", 5: "星期六", 6: "星期日"}
            date_str = now.strftime("%Y年%m月%d日")
            time_str = now.strftime("%H:%M:%S")
            weekday = week_map[now.weekday()]
            return f"今天是{date_str}，{weekday}，当前时间 {time_str}"
        return datetime_now

    @error_handler()
    def process_uploaded_files(self, uploaded_files):
        """处理上传的文件"""
        if not uploaded_files:
            return 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            all_documents = []
            total_files = len(uploaded_files)

            for i, file in enumerate(uploaded_files):
                status_text.text(f"正在处理文件: {file.name} ({i+1}/{total_files})")

                # 处理文档
                documents = self.doc_processor.process_uploaded_file(file)
                if documents:
                    # 分割文档，降低嵌入长度，避免超过API限制
                    split_docs = self.doc_processor.split_documents(documents)
                    # 过滤空片段
                    split_docs = [d for d in split_docs if d.page_content and d.page_content.strip()]
                    all_documents.extend(split_docs)

                progress_bar.progress((i + 1) / total_files)

            if all_documents:
                status_text.text("正在构建向量存储...")

                # 添加到向量存储
                self.vector_store.add_documents(all_documents)

                # 保存向量存储
                self.vector_store.save_index(self.settings.VECTOR_STORE_PATH)

                st.session_state.vector_store_ready = True
                status_text.text(f"✅ 成功处理 {len(all_documents)} 个文档片段")

                return len(all_documents)
            else:
                status_text.text("⚠️ 没有有效的文档被处理")
                return 0

        except Exception as e:
            status_text.text(f"❌ 处理文件时出错: {str(e)}")
            return 0
        finally:
            progress_bar.empty()      

    @error_handler()
    def generate_response(self, query: str) -> str:
        """生成回答"""
        try:
            # 创建Agent（如果需要）
            if not self.agent:
                self._create_agent()

            # 生成回答
            response = self.agent.generate_response(query)

            return response

        except Exception as e:
            return f"生成回答时出错: {str(e)}"        

    def run(self):
        """运行应用"""
        # 标题
        # 确保 session state 已初始化
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.chat_history = []
            st.session_state.vector_store_ready = False
            st.session_state.current_model = self.settings.get_default_model()
            st.session_state.temperature = self.settings.DEFAULT_TEMPERATURE
            st.session_state.max_tokens = self.settings.DEFAULT_MAX_TOKENS
            st.session_state.top_k = self.settings.DEFAULT_TOP_K
            st.session_state.search_type = self.settings.DEFAULT_SEARCH_TYPE
            st.session_state.llm_provider = self.settings.LLM_PROVIDER
        
        st.markdown(
            "<h2 class='hero-title' style='text-align:center; font-weight:600; margin:0;'>🤖 基于LangChain+Agentic RAG技术的智能问答系统</h2>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        # 侧边栏
        with st.sidebar:
            st.header("⚙️ 系统配置")

            # LLM 提供商信息
            provider_info = self.settings.get_provider_info()
            st.info(f"🔧 **LLM 提供商**: {provider_info['provider']}\n\n"
                   f"📡 **服务地址**: {provider_info['base_url']}\n\n"
                   f"🎯 **嵌入模型**: {provider_info['embedding']}")

            st.markdown("---")

            # 日志设置
            st.subheader("日志设置")
            if "log_level" not in st.session_state:
                st.session_state.log_level = self.settings.LOG_LEVEL
            st.session_state.log_level = st.selectbox(
                "日志级别:",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG","INFO","WARNING","ERROR"].index(st.session_state.log_level)
            )
            logging.getLogger().setLevel(getattr(logging, st.session_state.log_level))
            # 外部库网络日志
            if st.session_state.log_level == "DEBUG":
                os.environ["OPENAI_LOG"] = "debug"
                os.environ["HTTPX_LOG_LEVEL"] = "debug"
                set_verbose(True)
            else:
                os.environ["OPENAI_LOG"] = ""
                os.environ["HTTPX_LOG_LEVEL"] = ""
                set_verbose(False)

            # 模型设置
            st.subheader("模型设置")

            # 获取可用模型列表
            available_models = self.settings.get_available_models()

            # 确保当前模型在列表中
            if st.session_state.current_model not in available_models:
                st.session_state.current_model = self.settings.get_default_model()

            st.session_state.current_model = st.selectbox(
                "选择模型:",
                available_models,
                index=available_models.index(st.session_state.current_model) if st.session_state.current_model in available_models else 0
            )

            st.session_state.temperature = st.slider(
                "温度系数:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1
            )

            st.session_state.max_tokens = st.slider(
                "最大token数:",
                min_value=100,
                max_value=4000,
                value=st.session_state.max_tokens,
                step=100
            )

            # RAG设置
            st.subheader("RAG设置")
            st.session_state.top_k = st.slider(
                "检索数量:",
                min_value=1,
                max_value=10,
                value=st.session_state.top_k,
                step=1
            )

            st.session_state.search_type = st.selectbox(
                "搜索类型:",
                ["similarity", "mmr"],
                index=0 if st.session_state.search_type == "similarity" else 1
            )

            # 路由模式
            st.subheader("路由模式")
            if "route_mode" not in st.session_state:
                st.session_state.route_mode = "auto"

            label_by_value = {
                "auto": "智能选择",
                "kb_first": "知识库优先",
                "react_first": "深度思考",
            }
            value_by_label = {v: k for k, v in label_by_value.items()}

            selected_label = st.selectbox(
                "选择路由模式:",
                [label_by_value[v] for v in ["auto", "kb_first", "react_first"]],
                index=["auto", "kb_first", "react_first"].index(st.session_state.route_mode)
            )
            st.session_state.route_mode = value_by_label.get(selected_label, "auto")

            # 文档上传
            st.subheader("📄 文档上传")
            uploaded_files = st.file_uploader(
                "上传文档:",
                type=['pdf', 'txt', 'md', 'docx'],
                accept_multiple_files=True
            )

            new_files = []
            if uploaded_files:
                known = set(st.session_state.processed_upload_ids or [])
                for f in uploaded_files:
                    uid = f"{getattr(f, 'name', '')}:{getattr(f, 'size', 0)}"
                    if uid and uid not in known:
                        new_files.append(f)

            if new_files:
                with st.spinner("正在自动处理上传的文档..."):
                    doc_count = self.process_uploaded_files(new_files)
                    if doc_count > 0:
                        for f in new_files:
                            uid = f"{getattr(f, 'name', '')}:{getattr(f, 'size', 0)}"
                            if uid:
                                st.session_state.processed_upload_ids.append(uid)
                        st.success(f"成功处理 {doc_count} 个文档片段")
                        st.rerun()

            # 向量存储状态
            st.subheader("📊 向量存储状态")
            if st.session_state.vector_store_ready:
                st.success("✅ 向量存储已准备")
                if st.button("🗑️ 清空向量存储"):
                    self.vector_store.clear()
                    st.session_state.vector_store_ready = False
                    st.rerun()
            else:
                st.warning("⚠️ 向量存储未准备")

                # 加载已有向量存储
                if os.path.exists(self.settings.VECTOR_STORE_PATH):
                    if st.button("📂 加载已有向量存储"):
                        try:
                            self.vector_store.load_index(self.settings.VECTOR_STORE_PATH)
                            st.session_state.vector_store_ready = True
                            st.success("✅ 向量存储加载成功")
                            st.rerun()
                        except Exception as e:
                            st.error(f"加载向量存储失败: {str(e)}")

            # 聊天记录管理
            st.subheader("💬 聊天记录")

            # 导出聊天记录
            if st.session_state.chat_history:
                if st.button("📥 导出聊天记录"):
                    csv_content = self.chat_history.export_to_csv()
                    st.download_button(
                        label="下载CSV文件",
                        data=csv_content,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            # 清空聊天记录
            if st.button("🗑️ 清空聊天记录"):
                st.session_state.chat_history = []
                self.chat_history.clear_history()
                st.rerun()

        # 主界面：两列布局，间距固定为5rem
        col1, col2 = st.columns([7,3], gap="small")

        with col1:
            # 聊天界面
            st.subheader("💬 智能问答")

            # 显示聊天记录（助手消息支持HTML以呈现引用展开）
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant":
                        st.markdown(message["content"], unsafe_allow_html=True)
                    else:
                        st.markdown(message["content"])

            # 用户输入
            if prompt := st.chat_input("请输入您的问题..."):
                # 添加用户消息
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                # 显示用户消息
                with st.chat_message("user"):
                    st.markdown(prompt)

                # 生成回答
                with st.chat_message("assistant"):
                    with st.spinner("正在思考..."):
                        response = self.generate_response(prompt)
                        st.markdown(response, unsafe_allow_html=True)

                        # 保存回答到聊天记录
                        st.session_state.chat_history.append({"role": "assistant", "content": response})

                        # 保存聊天记录到文件
                        self.chat_history.add_message("user", prompt)
                        self.chat_history.add_message("assistant", response)

        with col2:
            # 聊天统计
            st.subheader("📊 聊天统计")

            if st.session_state.chat_history:
                total_messages = len(st.session_state.chat_history)
                user_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
                assistant_messages = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])

                m_total, m_user, m_assistant = st.columns(3)
                m_total.metric("总消息数", total_messages)
                m_user.metric("用户消息", user_messages)
                m_assistant.metric("助手消息", assistant_messages)

                st.subheader("最近消息")
                recent_messages = st.session_state.chat_history[-5:]
                for msg in recent_messages:
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    plain = re.sub(r"<[^>]+>", "", msg["content"]) if isinstance(msg.get("content"), str) else ""
                    preview = plain[:100] + "..." if len(plain) > 100 else plain
                    st.text(f"{role_icon}: {preview}")
            else:
                st.subheader("最近消息")
                persisted_recent = self.chat_history.get_history(limit=2)
                if persisted_recent:
                    for msg in persisted_recent:
                        role_icon = "👤" if msg.get("role") == "user" else "🤖"
                        content = msg.get("content", "")
                        plain = re.sub(r"<[^>]+>", "", content) if isinstance(content, str) else ""
                        preview = plain[:100] + "..." if len(plain) > 100 else plain
                        st.text(f"{role_icon}: {preview}")
                else:
                    st.caption("暂无聊天记录")

            st.markdown("---")
            st.subheader("📚 知识库文档")
            if st.session_state.vector_store_ready:
                # 可选搜索框
                q = st.text_input("搜索文件名", value="", placeholder="输入关键字过滤")
                doc_list = self.vector_store.get_document_list()
                if q:
                    doc_list = [d for d in doc_list if q.lower() in d["文件名"].lower()]
                if doc_list:
                    sort_by = st.selectbox("排序字段", ["文件名", "文件类型", "片段数"], index=0)
                    sort_order = st.selectbox("排序方式", ["升序", "降序"], index=0)
                    page_size = st.selectbox("每页数量", [10, 20, 50], index=0)

                    # 排序
                    def _key(d):
                        v = d.get(sort_by)
                        if sort_by == "片段数":
                            try:
                                return int(v)
                            except Exception:
                                return 0
                        return str(v).lower() if v is not None else ""

                    doc_list = sorted(doc_list, key=_key, reverse=(sort_order == "降序"))

                    # 分页
                    total_pages = (len(doc_list) + page_size - 1) // page_size if doc_list else 1
                    if "kb_page" not in st.session_state:
                        st.session_state.kb_page = 1
                    st.session_state.kb_page = st.number_input(
                        "当前页",
                        min_value=1,
                        max_value=total_pages,
                        value=st.session_state.kb_page,
                        step=1
                    )

                    start = (st.session_state.kb_page - 1) * page_size
                    end = start + page_size
                    page_slice = doc_list[start:end]

                    st.dataframe(page_slice, width='stretch')
                    st.caption(f"共 {len(doc_list)} 条，{total_pages} 页；当前第 {st.session_state.kb_page} 页")
                else:
                    st.caption("暂无文档或未匹配到结果")
            else:
                st.caption("向量存储未准备，上传文档后将自动构建并显示列表")


if __name__ == "__main__":
    print("启动应用...")
    app = AgenticRAGSystem()
    app.run()
