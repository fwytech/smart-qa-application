import os
import hashlib
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from config.settings import Settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器类"""

    def __init__(self):
        self.settings = Settings()
        self.cache_dir = self.settings.DATA_DIR / "document_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_hash(self, file_content: bytes) -> str:
        """计算文件哈希值"""
        return hashlib.md5(file_content).hexdigest()

    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """处理上传的文件（支持 UploadedFile 对象和文件路径）"""
        try:
            # 处理不同类型的输入
            if hasattr(uploaded_file, 'size') and hasattr(uploaded_file, 'read'):
                # Streamlit UploadedFile 对象
                if uploaded_file.size > self.settings.MAX_FILE_SIZE:
                    raise ValueError(f"文件大小超过限制: {uploaded_file.size} > {self.settings.MAX_FILE_SIZE}")
                file_content = uploaded_file.read()
                file_name = uploaded_file.name
            else:
                # 文件路径
                file_path = Path(uploaded_file)
                if not file_path.exists():
                    raise ValueError(f"文件不存在: {file_path}")
                file_size = file_path.stat().st_size
                if file_size > self.settings.MAX_FILE_SIZE:
                    raise ValueError(f"文件大小超过限制: {file_size} > {self.settings.MAX_FILE_SIZE}")
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                file_name = file_path.name
            
            file_type = Path(file_name).suffix.lower()

            # 检查文件类型
            if file_type not in self.settings.SUPPORTED_FILE_TYPES:
                raise ValueError(f"不支持的文件类型: {file_type}")

            # 计算文件哈希
            file_hash = self._get_file_hash(file_content)
            cache_path = self._get_cache_path(file_hash, file_name)

            # 尝试从缓存加载
            cached_documents = self._load_from_cache(cache_path)
            if cached_documents is not None:
                return cached_documents

            # 处理文件
            documents = self._process_file_content(file_content, file_name, file_type)

            # 保存到缓存
            self._save_to_cache(cache_path, documents)

            logger.info(f"处理文件成功: {file_name}, 文档数量: {len(documents)}")
            return documents

        except Exception as e:
            logger.error(f"处理上传文件失败: {str(e)}")
            raise

    def _process_file_content(self, file_content: bytes, file_name: str, file_type: str) -> List[Document]:
        """处理文件内容"""
        try:
            # 创建临时文件
            temp_dir = self.settings.DATA_DIR / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / file_name

            # 写入临时文件
            with open(temp_path, 'wb') as f:
                f.write(file_content)

            try:
                # 根据文件类型选择加载器
                if file_type == '.pdf':
                    documents = self._load_pdf(temp_path)
                elif file_type == '.txt':
                    documents = self._load_text(temp_path)
                elif file_type == '.md':
                    documents = self._load_markdown(temp_path)
                elif file_type == '.docx':
                    documents = self._load_word(temp_path)
                else:
                    raise ValueError(f"不支持的文件类型: {file_type}")

                # 添加元数据
                for i, doc in enumerate(documents):
                    doc.metadata.update({
                        'source': file_name,
                        'file_type': file_type,
                        'chunk_index': i,
                        'total_chunks': len(documents),
                        'processing_timestamp': str(Path(temp_path).stat().st_mtime)
                    })

                return documents

            finally:
                # 清理临时文件
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as e:
            logger.error(f"处理文件内容失败: {str(e)}")
            raise

    def _load_pdf(self, file_path: Path) -> List[Document]:
        """加载PDF文件"""
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()

            # 添加页码信息
            for i, doc in enumerate(documents):
                if 'page' not in doc.metadata:
                    doc.metadata['page'] = i + 1

            logger.info(f"加载PDF成功: {file_path.name}, 页数: {len(documents)}")
            return documents

        except Exception as e:
            logger.error(f"加载PDF失败: {str(e)}")
            raise
            
    def _load_text(self, file_path: Path) -> List[Document]:
        """加载文本文件"""
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            
            logger.info(f"加载文本文件成功: {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"加载文本文件失败: {str(e)}")
            raise
    
    def _load_markdown(self, file_path: Path) -> List[Document]:
        """加载Markdown文件"""
        try:
            # Markdown文件也使用文本加载器
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            
            # 添加文件类型标识
            for doc in documents:
                doc.metadata['file_type'] = '.md'
            
            logger.info(f"加载Markdown文件成功: {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"加载Markdown文件失败: {str(e)}")
            raise
    
    def _load_word(self, file_path: Path) -> List[Document]:
        """加载Word文档"""
        try:
            loader = UnstructuredWordDocumentLoader(str(file_path))
            documents = loader.load()
            
            logger.info(f"加载Word文档成功: {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"加载Word文档失败: {str(e)}")
            raise           


    def split_documents(self, documents: List[Document], chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
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
            
            # 更新元数据
            for i, doc in enumerate(split_docs):
                doc.metadata['chunk_index'] = i
                doc.metadata['chunk_size'] = len(doc.page_content)
                doc.metadata['total_chunks'] = len(split_docs)
            
            logger.info(f"文档分割完成，片段数量: {len(split_docs)}")
            return split_docs
            
        except Exception as e:
            logger.error(f"文档分割失败: {str(e)}")
            return documents    

    def _get_cache_path(self, file_hash: str, file_name: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{file_hash}_{file_name}.json"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[List[Document]]:
        """从缓存加载文档"""
        try:
            if cache_path.exists() and self.settings.CACHE_ENABLED:
                import json
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # 检查缓存是否过期
                import time
                current_time = time.time()
                cache_time = cache_data.get('timestamp', 0)
                
                if current_time - cache_time < self.settings.CACHE_EXPIRE_TIME:
                    # 重建Document对象
                    documents = []
                    for doc_data in cache_data.get('documents', []):
                        doc = Document(
                            page_content=doc_data['page_content'],
                            metadata=doc_data['metadata']
                        )
                        documents.append(doc)
                    
                    logger.info(f"从缓存加载文档成功: {len(documents)} 个文档")
                    return documents
                else:
                    logger.info("缓存已过期")
                    
        except Exception as e:
            logger.error(f"从缓存加载失败: {str(e)}")
        
        return None
    
    def _save_to_cache(self, cache_path: Path, documents: List[Document]):
        """保存文档到缓存"""
        try:
            if not self.settings.CACHE_ENABLED:
                return
                
            import json
            import time
            
            cache_data = {
                'timestamp': time.time(),
                'documents': [
                    {
                        'page_content': doc.page_content,
                        'metadata': doc.metadata
                    }
                    for doc in documents
                ]
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存到缓存成功: {cache_path}")
            
        except Exception as e:
            logger.error(f"保存到缓存失败: {str(e)}")
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            stats = {
                'cache_enabled': self.settings.CACHE_ENABLED,
                'cache_expire_time': self.settings.CACHE_EXPIRE_TIME,
                'cache_files_count': len(cache_files),
                'cache_total_size_bytes': total_size,
                'cache_total_size_mb': round(total_size / (1024 * 1024), 2)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取缓存统计信息失败: {str(e)}")
            return {'error': str(e)}
    
    def clear_cache(self) -> bool:
        """清空缓存"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            deleted_count = 0
            
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"删除缓存文件失败: {cache_file}, 错误: {str(e)}")
            
            logger.info(f"清空缓存成功，删除文件数: {deleted_count}")
            return True
            
        except Exception as e:
            logger.error(f"清空缓存失败: {str(e)}")
            return False                

    def process_documents_batch(self, uploaded_files: List) -> Dict[str, Any]:
        """批量处理文档"""
        try:
            results = {
                'total_files': len(uploaded_files),
                'processed_files': 0,
                'failed_files': 0,
                'total_documents': 0,
                'errors': []
            }
            
            all_documents = []
            
            for file in uploaded_files:
                try:
                    documents = self.process_uploaded_file(file)
                    all_documents.extend(documents)
                    results['processed_files'] += 1
                    results['total_documents'] += len(documents)
                    
                    logger.info(f"处理文件成功: {file.name}")
                    
                except Exception as e:
                    results['failed_files'] += 1
                    error_info = {
                        'file_name': file.name,
                        'error': str(e)
                    }
                    results['errors'].append(error_info)
                    logger.error(f"处理文件失败: {file.name}, 错误: {str(e)}")
            
            results['all_documents'] = all_documents
            return results
            
        except Exception as e:
            logger.error(f"批量处理文档失败: {str(e)}")
            raise


def test_document_processor():
    """测试文档处理器核心功能"""
    import logging
    import time
    from pathlib import Path
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("📚 文档处理器核心功能测试")
    print("=" * 60)
    
    try:
        # 初始化设置和处理器
        from config.settings import Settings
        settings = Settings()
        processor = DocumentProcessor()
        
        # 测试文件路径
        test_files_dir = Path("files")
        if not test_files_dir.exists():
            print(f"❌ 测试目录不存在: {test_files_dir}")
            return
        
        # 获取测试文件
        test_files = list(test_files_dir.glob("*"))
        if not test_files:
            print("❌ 没有找到测试文件")
            return
        
        print(f"📁 找到 {len(test_files)} 个测试文件:")
        for i, file in enumerate(test_files, 1):
            file_size = file.stat().st_size / 1024
            print(f"   {i}. {file.name} ({file_size:.1f} KB)")
        
        print("\n" + "=" * 40)
        print("🔧 开始功能测试...")
        print("=" * 40)
        
        # 1. 测试单个文件处理
        print("\n📄 1. 测试单个文件处理")
        print("-" * 30)
        
        test_file = test_files[0]  # 使用第一个文件
        print(f"处理文件: {test_file.name}")
        
        try:
            documents = processor.process_uploaded_file(test_file)
            print(f"✅ 成功处理！提取了 {len(documents)} 个文档片段")
            
            if documents:
                # 显示第一个片段的信息
                first_doc = documents[0]
                content_preview = first_doc.page_content[:100] + "..." if len(first_doc.page_content) > 100 else first_doc.page_content
                print(f"   第一个片段预览: {content_preview}")
                print(f"   片段长度: {len(first_doc.page_content)} 字符")
                print(f"   元数据: {first_doc.metadata}")
                
        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 2. 测试文档分割
        print("\n✂️ 2. 测试文档分割功能")
        print("-" * 30)
        
        if 'documents' in locals() and documents:
            try:
                # 测试不同的分割参数
                test_params = [
                    (500, 50),
                    (1000, 100)
                ]
                
                for chunk_size, overlap in test_params:
                    split_docs = processor.split_documents(documents, chunk_size, overlap)
                    print(f"   chunk_size={chunk_size}, overlap={overlap}: {len(split_docs)} 个片段")
                    
                    if split_docs:
                        avg_size = sum(len(doc.page_content) for doc in split_docs) / len(split_docs)
                        print(f"   平均片段大小: {avg_size:.0f} 字符")
                        
            except Exception as e:
                print(f"❌ 分割失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 3. 测试批量处理
        print("\n📦 3. 测试批量处理")
        print("-" * 30)
        
        try:
            # 模拟Streamlit的UploadedFile对象
            class MockUploadedFile:
                def __init__(self, file_path):
                    self.name = file_path.name
                    self.type = self._get_file_type(file_path)
                    self.size = file_path.stat().st_size
                    self._file_path = file_path
                    
                def _get_file_type(self, file_path):
                    suffix = file_path.suffix.lower()
                    type_map = {
                        '.pdf': 'application/pdf',
                        '.txt': 'text/plain',
                        '.md': 'text/markdown',
                        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    }
                    return type_map.get(suffix, 'application/octet-stream')
                
                def read(self):
                    return self._file_path.read_bytes()
            
            # 创建模拟上传文件列表
            mock_files = [MockUploadedFile(f) for f in test_files]
            
            print(f"批量处理 {len(mock_files)} 个文件...")
            batch_results = processor.process_documents_batch(mock_files)
            
            print(f"✅ 批量处理完成！")
            print(f"   总文件数: {batch_results['total_files']}")
            print(f"   成功处理: {batch_results['processed_files']}")
            print(f"   处理失败: {batch_results['failed_files']}")
            print(f"   总文档片段: {batch_results['total_documents']}")
            
            # 显示失败文件信息
            if batch_results['errors']:
                print("\n   失败的文件:")
                for error in batch_results['errors']:
                    print(f"   - {error['file_name']}: {error['error']}")
                    
        except Exception as e:
            print(f"❌ 批量处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 4. 测试缓存功能
        print("\n💾 4. 测试缓存功能")
        print("-" * 30)
        
        try:
            # 获取缓存统计
            cache_stats = processor.get_cache_stats()
            print(f"✅ 缓存统计:")
            print(f"   缓存启用: {cache_stats['cache_enabled']}")
            print(f"   缓存文件数: {cache_stats['cache_files_count']}")
            print(f"   缓存大小: {cache_stats['cache_total_size_mb']} MB")
            print(f"   过期时间: {cache_stats['cache_expire_time']} 秒")
            
            # 测试再次处理相同文件（应该命中缓存）
            print(f"\n   再次处理相同文件测试缓存...")
            start_time = time.time()
            cached_docs = processor.process_uploaded_file(test_files[0])
            cache_time = time.time() - start_time
            
            print(f"   缓存处理时间: {cache_time:.3f} 秒")
            print(f"   缓存文档数: {len(cached_docs)}")
            
        except Exception as e:
            print(f"❌ 缓存测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 5. 测试文件类型识别
        print("\n🔍 5. 测试文件类型处理")
        print("-" * 30)
        
        supported_types = ['.pdf', '.txt', '.md', '.docx']
        success_count = 0
        
        for file_path in test_files:
            file_ext = file_path.suffix.lower()
            if file_ext in supported_types:
                print(f"   {file_path.name} ({file_ext}): ", end="")
                try:
                    docs = processor.process_uploaded_file(file_path)
                    print(f"✅ 成功 ({len(docs)} 片段)")
                    success_count += 1
                except Exception as e:
                    print(f"❌ 失败 ({str(e)})")
        
        print(f"\n   成功处理: {success_count}/{len(test_files)} 个文件")
        
        print("\n" + "=" * 60)
        print("🎉 测试完成！")
        print("=" * 60)
        
        # 显示总体统计
        if 'batch_results' in locals():
            print(f"📊 总体统计:")
            print(f"   处理文件类型: {len(set(f.suffix for f in test_files))}")
            print(f"   总文档片段: {batch_results['total_documents']}")
            if batch_results['total_files'] > 0:
                success_rate = (batch_results['processed_files']/batch_results['total_files']*100)
                print(f"   成功率: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_document_processor()