import streamlit as st
from typing import List, Dict, Optional, Any
from config.settings import Settings

class UIComponents:
    """UI组件类 - 负责渲染各种Streamlit界面元素"""

    def __init__(self):
        self.settings = Settings()

    
    def render_model_selector(self, current_model: str, key_prefix: str = "") -> str:
        """渲染模型选择器"""
        """
        - 作用 ：让用户选择AI模型
        - 界面元素 ：下拉选择框 + 刷新按钮
        - 返回 ：用户选择的模型名称
        """
        col1, col2 = st.columns([2, 1])

        with col1:
            selected_model = st.selectbox(
                "🤖 选择模型",
                options=self.settings.AVAILABLE_MODELS,
                index=self.settings.AVAILABLE_MODELS.index(current_model)
                if current_model in self.settings.AVAILABLE_MODELS else 0,
                help="选择要使用的语言模型",
                key=f"{key_prefix}model_selector"
            )

        with col2:
            if st.button("🔄 刷新模型列表", key=f"{key_prefix}refresh_models"):
                st.rerun()

        return selected_model

    def render_temperature_slider(self, current_temp: float, key_prefix: str = "") -> float:
        """渲染温度系数滑块"""
        """
        - 作用 ：控制AI回答的随机性（0.0-1.0）
        - 界面元素 ：滑块 + 智能提示
        - 返回 ：温度值
        """
        temperature = st.slider(
            "🌡️ 温度系数 (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=current_temp,
            step=0.1,
            help="控制回答的随机性。值越高，回答越随机；值越低，回答越确定。",
            key=f"{key_prefix}temperature_slider"
        )

        # 显示温度解释
        temp_explanation = self._get_temperature_explanation(temperature)
        st.caption(f"💡 {temp_explanation}")

        return temperature

    def render_rag_settings(self, current_top_k: int, current_search_type: str, key_prefix: str = "") -> tuple:
        """渲染RAG设置"""
        """
        - 作用 ：配置检索增强生成参数
        - 界面元素 ：两个滑块/选择器
        - 返回 ：top_k值和搜索类型
        """
        st.subheader("🔍 RAG设置")

        col1, col2 = st.columns(2)

        with col1:
            top_k = st.slider(
                "检索数量 (Top-K)",
                min_value=1,
                max_value=10,
                value=current_top_k,
                step=1,
                help="从向量存储中检索的相关文档数量",
                key=f"{key_prefix}top_k_slider"
            )

        with col2:
            search_type = st.selectbox(
                "搜索类型",
                options=["similarity", "mmr"],
                index=0 if current_search_type == "similarity" else 1,
                help="similarity: 相似度搜索; mmr: 最大边际相关性搜索",
                key=f"{key_prefix}search_type_select"
            )

        # 显示搜索类型解释
        search_explanation = self._get_search_type_explanation(search_type)
        st.caption(f"💡 {search_explanation}")

        return top_k, search_type

    def render_vector_store_status(self, is_ready: bool, stats: Optional[Dict] = None):
        """渲染向量存储状态"""
        """
        - 作用 ：显示知识库状态
        - 界面元素 ：状态指示器 + 统计信息
        """
        if is_ready:
            st.success("✅ 向量存储已准备就绪")

            if stats:
                with st.expander("📊 向量存储统计"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("文档数量", stats.get('documents_count', 0))

                    with col2:
                        st.metric("向量总数", stats.get('total_vectors', 0))

                    with col3:
                        st.metric("向量维度", stats.get('dimension', 0))

                    if stats.get('index_path'):
                        st.caption(f"📁 索引路径: {stats['index_path']}")

        else:
            st.warning("⚠️ 向量存储未准备")

    def _get_temperature_explanation(self, temperature: float) -> str:
        """获取温度系数解释"""
        if temperature < 0.3:
            return "低温度：回答更确定、保守"
        elif temperature < 0.7:
            return "中等温度：平衡确定性和创造性"
        else:
            return "高温度：回答更随机、有创造性"

    def _get_search_type_explanation(self, search_type: str) -> str:
        """获取搜索类型解释"""
        if search_type == "similarity":
            return "相似度搜索：基于向量相似度检索最相关的文档"
        else:
            return "MMR搜索：在相关性和多样性之间取得平衡"