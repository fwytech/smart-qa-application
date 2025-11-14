import os
import json
import pickle
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import Settings
from services.llm_client import UnifiedEmbeddingClient

logger = logging.getLogger(__name__)

class VectorStoreService:
    """向量存储服务类 - 支持 Ollama 和在线 API embedding"""

    def __init__(self):
        self.settings = Settings()
        # 使用统一的嵌入客户端
        self.embedding_client = UnifiedEmbeddingClient()
        self.embeddings = self.embedding_client.get_embeddings()
        self.vector_store = None
        self.documents = []
        self.index_path = None

        logger.info(f"向量存储服务初始化成功 - 提供商: {self.settings.LLM_PROVIDER}, 嵌入模型: {self.settings.get_embedding_model()}")

        # 尝试自动加载已存在的索引，避免运行期间状态丢失
        try:
            idx_path = str(self.settings.VECTOR_STORE_PATH)
            if os.path.exists(idx_path):
                loaded = self.load_index(idx_path)
                if loaded:
                    logger.info("检测到已有向量索引，已自动加载")
        except Exception as e:
            logger.warning(f"自动加载向量索引失败: {str(e)}")

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """创建向量存储"""
        try:
            logger.info(f"创建向量存储，文档数量: {len(documents)}")

            # 清洗文档，避免空片段和超长内容
            documents = self._sanitize_documents(documents)

            # 创建向量存储
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            self.vector_store = vector_store
            self.documents = documents

            logger.info("向量存储创建成功")
            return vector_store

        except Exception as e:
            logger.error(f"创建向量存储失败: {str(e)}")
            raise

    def load_index(self, index_path: str) -> bool:
        """加载向量存储索引"""
        try:
            logger.info(f"加载向量存储索引: {index_path}")

            if not os.path.exists(index_path):
                logger.warning(f"索引路径不存在: {index_path}")
                return False

            # 加载FAISS索引
            self.vector_store = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # 加载文档（如果存在）
            docs_path = f"{index_path}_docs.pkl"
            if os.path.exists(docs_path):
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)

            self.index_path = index_path
            logger.info("向量存储索引加载成功")
            return True

        except Exception as e:
            logger.error(f"加载向量存储索引失败: {str(e)}")
            return False

    def save_index(self, index_path: str) -> bool:
        """保存向量存储索引"""
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化，无法保存")
                return False

            logger.info(f"保存向量存储索引: {index_path}")

            # 保存FAISS索引
            self.vector_store.save_local(index_path)

            # 保存文档
            docs_path = f"{index_path}_docs.pkl"
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)

            # 保存元数据
            metadata = {
                "created_at": datetime.now().isoformat(),
                "documents_count": len(self.documents),
                "embedding_model": self.embedding_model_name,
                "vector_dimension": self.settings.VECTOR_DIMENSION
            }

            metadata_path = f"{index_path}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.index_path = index_path
            logger.info("向量存储索引保存成功")
            return True

        except Exception as e:
            logger.error(f"保存向量存储索引失败: {str(e)}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "similarity",
        score_threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        """搜索向量存储"""
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []

            logger.info(f"搜索查询: {query}, top_k: {top_k}, search_type: {search_type}")

            if search_type == "similarity":
                # 相似度搜索
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k
                )
            elif search_type == "mmr":
                # MMR搜索（最大边际相关性）- 社区版不返回分数
                mmr_docs = self.vector_store.max_marginal_relevance_search(
                    query=query,
                    k=top_k,
                    fetch_k=top_k * 2
                )
                results = [(doc, 1.0) for doc in mmr_docs]
            else:
                raise ValueError(f"不支持的搜索类型: {search_type}")

            # 格式化结果
            formatted_results = []
            for doc, score in results:
                if score >= score_threshold:  # 过滤低分结果
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    })

            # 如果相似度搜索没有命中，自动回退到MMR提高召回
            if not formatted_results and search_type == "similarity":
                mmr_docs = self.vector_store.max_marginal_relevance_search(
                    query=query,
                    k=top_k,
                    fetch_k=top_k * 2
                )
                formatted_results = [{
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 1.0
                } for doc in mmr_docs]

            logger.info(f"搜索完成，找到 {len(formatted_results)} 个结果")
            return formatted_results

        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return []

    def add_documents(self, documents: List[Document]) -> bool:
        """添加文档到向量存储"""
        try:
            logger.info(f"添加文档到向量存储，数量: {len(documents)}")

            documents = self._sanitize_documents(documents)

            if not self.vector_store:
                # 如果向量存储不存在，创建新的
                self.create_vector_store(documents)
            else:
                # 添加到现有向量存储
                self.vector_store.add_documents(documents)
                self.documents.extend(documents)

            logger.info("文档添加成功")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False

    def _sanitize_documents(self, documents: List[Document]) -> List[Document]:
        """清洗文档内容，过滤空内容并裁剪过长文本以满足嵌入长度限制"""
        cleaned: List[Document] = []
        # 近似字符上限，避免超过 2048 token（中文场景下字符与token近似）
        max_chars = 2000
        for doc in documents:
            content = (doc.page_content or "").strip()
            if not content:
                continue
            if len(content) > max_chars:
                content = content[:max_chars]
            cleaned.append(Document(page_content=content, metadata=doc.metadata))
        logger.info(f"清洗后文档数量: {len(cleaned)}")
        return cleaned

    def delete_document(self, doc_id: str) -> bool:
        """从向量存储中删除文档"""
        try:
            logger.info(f"删除文档: {doc_id}")

            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return False

            # FAISS不直接支持删除操作，需要重新创建索引
            remaining_docs = [
                doc for doc in self.documents
                if doc.metadata.get("id") != doc_id
            ]

            if len(remaining_docs) < len(self.documents):
                self.create_vector_store(remaining_docs)
                logger.info(f"文档删除成功: {doc_id}")
                return True
            else:
                logger.warning(f"未找到文档: {doc_id}")
                return False

        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return False

    def clear(self):
        """清空向量存储"""
        try:
            logger.info("清空向量存储")

            # 同时清理磁盘索引文件，避免自动加载旧索引
            idx_path = self.index_path or str(self.settings.VECTOR_STORE_PATH)
            try:
                if idx_path and os.path.exists(idx_path):
                    import shutil
                    shutil.rmtree(idx_path, ignore_errors=True)
                # 删除附加的docs与metadata文件
                docs_path = f"{idx_path}_docs.pkl"
                meta_path = f"{idx_path}_metadata.json"
                if os.path.exists(docs_path):
                    os.remove(docs_path)
                if os.path.exists(meta_path):
                    os.remove(meta_path)
            except Exception as e:
                logger.warning(f"删除索引文件失败: {str(e)}")

            # 清理内存状态
            self.vector_store = None
            self.documents = []
            self.index_path = None

            logger.info("向量存储已清空（内存与磁盘）")

        except Exception as e:
            logger.error(f"清空向量存储失败: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """获取向量存储统计信息"""
        try:
            stats = {
                "documents_count": len(self.documents),
                "vector_store_initialized": self.vector_store is not None,
                "embedding_model": self.embedding_model_name,
                "index_path": self.index_path
            }

            if self.vector_store:
                # 获取索引信息
                index = self.vector_store.index
                stats.update({
                    "total_vectors": index.ntotal if hasattr(index, 'ntotal') else 0,
                    "dimension": index.d if hasattr(index, 'd') else 0
                })

            return stats

        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {"error": str(e)}

    def get_document_list(self) -> List[Dict[str, Any]]:
        """获取知识库中的文档列表及统计信息"""
        try:
            docs = self.documents or []
            by_source: Dict[str, Dict[str, Any]] = {}
            for d in docs:
                meta = d.metadata or {}
                src = meta.get("source", "未知来源")
                ftype = meta.get("file_type", "")
                by_source.setdefault(src, {"文件名": src, "文件类型": ftype, "片段数": 0})
                by_source[src]["片段数"] += 1
            # 转为列表并按文件名排序
            result = list(by_source.values())
            result.sort(key=lambda x: x["文件名"].lower())
            return result
        except Exception as e:
            logger.error(f"获取文档列表失败: {str(e)}")
            return []

    def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Document]:
        """分割文档"""
        try:
            chunk_size = chunk_size or self.settings.CHUNK_SIZE
            chunk_overlap = chunk_overlap or self.settings.CHUNK_OVERLAP

            logger.info(f"分割文档，chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
            )

            split_docs = text_splitter.split_documents(documents)

            # 添加元数据
            for i, doc in enumerate(split_docs):
                if "chunk_id" not in doc.metadata:
                    doc.metadata["chunk_id"] = i
                if "chunk_size" not in doc.metadata:
                    doc.metadata["chunk_size"] = len(doc.page_content)

            logger.info(f"文档分割完成，片段数量: {len(split_docs)}")
            return split_docs

        except Exception as e:
            logger.error(f"文档分割失败: {str(e)}")
            return documents

    def similarity_search_with_threshold(
        self,
        query: str,
        threshold: float = 0.7,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """带阈值的相似度搜索"""
        try:
            results = self.search(query, top_k * 2, "similarity")  # 获取更多结果用于过滤

            # 按阈值过滤
            filtered_results = [
                result for result in results
                if result["score"] >= threshold
            ]

            # 返回前top_k个结果
            return filtered_results[:top_k]

        except Exception as e:
            logger.error(f"阈值搜索失败: {str(e)}")
            return []

    @property
    def embedding_model_name(self) -> str:
        """获取嵌入模型名称"""
        if hasattr(self.embeddings, 'embedding_model_name'):
            return self.embeddings.embedding_model_name
        return self.settings.get_embedding_model()


# ==============================================================================
#  ONLINE-ONLY 快速测试
# ==============================================================================
def test_online_only():
    """
    纯在线模式功能测试
    强制使用 online / text-embedding-v1，无需本地 Ollama
    """
    import os, tempfile, json
    os.environ["LLM_PROVIDER"] = "online"          # 锁死 online
    os.environ["ONLINE_API_KEY"] = os.getenv(
        "ONLINE_API_KEY",
        "sk-abe3417c96f6441b83efed38708bcfb6"      # 默认 Demo key
    )

    print("===  VectorStoreService 在线模型功能测试  ===")
    print("LLM_PROVIDER =", os.environ["LLM_PROVIDER"])
    print("ONLINE_API_KEY =", os.environ["ONLINE_API_KEY"][:10] + "***")

    try:
        # 1. 初始化
        print("\n1. 初始化服务（online）...")
        vs = VectorStoreService()
        print("   ✓ 初始化完成，嵌入模型：", vs.settings.get_embedding_model())

        # 2. 构造测试文档
        docs = [
            Document(page_content="Python 是简洁强大的编程语言。", metadata={"id": "1"}),
            Document(page_content="机器学习无需显式编程即可学习。", metadata={"id": "2"}),
            Document(page_content="向量数据库支撑语义搜索。", metadata={"id": "3"}),
        ]

        # 3. 创建索引
        print("\n2. 创建向量索引...")
        vs.create_vector_store(docs)
        print("   ✓ 索引创建成功，文档数：", len(vs.documents))

        # 4. 搜索测试
        print("\n3. 相似度搜索...")
        res = vs.search("Python 语言特点", top_k=2, score_threshold=0.3)
        print("   返回结果数：", len(res))
        for r in res:
            print("   - score={:.3f}, content={}".format(r["score"], r["content"][:60]))

        print("\n4. MMR 搜索...")
        mmr = vs.search("机器学习", top_k=2, search_type="mmr", score_threshold=0.2)
        print("   MMR 结果数：", len(mmr))

        # 5. 保存 / 加载
        print("\n5. 保存 & 加载索引...")
        with tempfile.TemporaryDirectory() as tmp:
            idx_path = os.path.join(tmp, "online_index")
            assert vs.save_index(idx_path), "保存失败"
            print("   ✓ 保存成功")

            vs2 = VectorStoreService()              # 新实例
            assert vs2.load_index(idx_path), "加载失败"
            print("   ✓ 加载成功，文档数：", len(vs2.documents))

        # 6. 统计 & 清空
        print("\n6. 统计信息：", vs.get_stats())
        vs.clear()
        print("   ✓ 已清空，文档数：", len(vs.documents))

        print("\n===  在线模型测试全部通过  ===")
        return True

    except Exception as e:
        print("\n❌ 测试失败：", e)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 直接运行即走 online 测试
    test_online_only()
