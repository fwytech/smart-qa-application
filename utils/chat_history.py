import json
import os
import csv
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from config.settings import Settings

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    """聊天记录管理器"""

    def __init__(self, history_file: str = None):
        """初始化聊天记录管理器
        
        方法用途：创建ChatHistoryManager实例，设置历史文件路径，加载现有记录
        
        参数解释：
            history_file (str, 可选): 历史记录文件路径，如果为None则使用配置文件中的默认路径
            
        返回值：无（构造函数）
            
        使用示例：
            >>> # 使用默认路径
            >>> manager = ChatHistoryManager()
            >>> print(manager.history_file)  # 返回配置文件中的默认路径
            >>> 
            >>> # 使用自定义路径
            >>> custom_manager = ChatHistoryManager('my_chat_history.json')
            >>> print(custom_manager.history_file)  # 返回: 'my_chat_history.json'
            >>> 
            >>> # 自动创建目录
            >>> manager = ChatHistoryManager('data/chats/history.json')
            >>> # 如果data/chats目录不存在，会自动创建
        """
        self.settings = Settings()
        self.history_file = history_file or self.settings.CHAT_HISTORY_PATH
        self.history_dir = Path(self.history_file).parent
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.chat_history = []
        self.max_history_size = 10000  # 最大历史记录数

        # 加载现有历史记录
        self.load_history()

    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """添加消息到历史记录
        
        方法用途：将一条新的聊天消息添加到历史记录中，并自动保存到文件
        
        参数解释：
            role (str): 消息角色，如 'user', 'assistant', 'system'
            content (str): 消息内容
            metadata (Dict[str, Any], 可选): 消息的元数据，如模型信息、温度等
            
        返回值：
            bool: 添加成功返回True，失败返回False
            
        使用示例：
            >>> manager = ChatHistoryManager()
            >>> success = manager.add_message('user', '你好，请介绍一下Python')
            >>> print(success)  # 返回: True
            
            >>> success = manager.add_message('assistant', 'Python是一种高级编程语言...', 
            ...                                {'model': 'gpt-4', 'temperature': 0.7})
            >>> print(success)  # 返回: True
        """
        try:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "id": self._generate_message_id()
            }

            if metadata:
                message["metadata"] = metadata

            self.chat_history.append(message)

            # 如果超过最大数量，移除最旧的记录
            if len(self.chat_history) > self.max_history_size:
                self.chat_history.pop(0)

            # 保存到文件
            self.save_history()

            logger.debug(f"添加消息成功: {role}")
            return True

        except Exception as e:
            logger.error(f"添加消息失败: {str(e)}")
            return False

    def get_history(self, limit: int = None, role_filter: str = None) -> List[Dict[str, Any]]:
        """获取历史记录
        
        方法用途：获取聊天记录，支持按角色过滤和数量限制
        
        参数解释：
            limit (int, 可选): 返回的最大记录数，None表示返回所有记录
            role_filter (str, 可选): 角色过滤器，如 'user', 'assistant'，None表示不过滤
            
        返回值：
            List[Dict[str, Any]]: 消息列表，每条消息包含role、content、timestamp、id等字段
            
        使用示例：
            >>> manager = ChatHistoryManager()
            >>> manager.add_message('user', '你好')
            >>> manager.add_message('assistant', '你好！有什么可以帮助您的吗？')
            >>> 
            >>> # 获取所有历史记录
            >>> all_history = manager.get_history()
            >>> print(len(all_history))  # 返回: 2
            >>> 
            >>> # 获取最近1条记录
            >>> recent = manager.get_history(limit=1)
            >>> print(recent[0]['role'])  # 返回: 'assistant'
            >>> 
            >>> # 只获取用户消息
            >>> user_msgs = manager.get_history(role_filter='user')
            >>> print(len(user_msgs))  # 返回: 1
        """
        try:
            history = self.chat_history.copy()

            # 角色过滤
            if role_filter:
                history = [msg for msg in history if msg.get("role") == role_filter]

            # 数量限制
            if limit and limit > 0:
                history = history[-limit:]

            return history

        except Exception as e:
            logger.error(f"获取历史记录失败: {str(e)}")
            return []

    def clear_history(self) -> bool:
        """清空历史记录
        
        方法用途：清空所有聊天记录，并删除保存的历史文件
        
        参数：无
        
        返回值：
            bool: 清空成功返回True，失败返回False
            
        使用示例：
            >>> manager = ChatHistoryManager()
            >>> manager.add_message('user', '你好')
            >>> manager.add_message('assistant', '你好！')
            >>> print(len(manager.get_history()))  # 返回: 2
            >>> 
            >>> # 清空历史记录
            >>> success = manager.clear_history()
            >>> print(success)  # 返回: True
            >>> print(len(manager.get_history()))  # 返回: 0
        """
        try:
            self.chat_history = []
            self.save_history()

            logger.info("清空历史记录成功")
            return True

        except Exception as e:
            logger.error(f"清空历史记录失败: {str(e)}")
            return False

    def export_to_csv(self, output_file: str = None) -> str:
        """导出为CSV格式
        
        方法用途：将聊天记录导出为CSV格式，可以返回CSV字符串或保存到文件
        
        参数解释：
            output_file (str, 可选): 输出文件路径，如果为None则返回CSV字符串
            
        返回值：
            str: 如果output_file为None，返回CSV字符串；否则返回文件路径
            
        使用示例：
            >>> manager = ChatHistoryManager()
            >>> manager.add_message('user', '你好', {'model': 'gpt-4'})
            >>> manager.add_message('assistant', '你好！有什么可以帮助您的吗？')
            >>> 
            >>> # 导出为CSV字符串
            >>> csv_content = manager.export_to_csv()
            >>> print(csv_content[:50])  # 返回: 'role,content,timestamp,id,metadata\\r\\nuser,你好,...'
            >>> 
            >>> # 保存到文件
            >>> file_path = manager.export_to_csv('chat_history.csv')
            >>> print(file_path)  # 返回: 'chat_history.csv'
        """
        try:
            if not output_file:
                # 返回CSV字符串
                import io
                output = io.StringIO()

                if self.chat_history:
                    # 定义标准CSV字段，排除复杂的metadata字段
                    standard_fields = ['role', 'content', 'timestamp', 'id']
                    writer = csv.DictWriter(output, fieldnames=standard_fields)
                    writer.writeheader()
                    
                    # 只导出标准字段
                    for message in self.chat_history:
                        row = {field: message.get(field, '') for field in standard_fields}
                        writer.writerow(row)

                csv_content = output.getvalue()
                output.close()

                logger.info("导出CSV成功")
                return csv_content
            else:
                # 保存到文件
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    if self.chat_history:
                        # 定义标准CSV字段，排除复杂的metadata字段
                        standard_fields = ['role', 'content', 'timestamp', 'id']
                        writer = csv.DictWriter(f, fieldnames=standard_fields)
                        writer.writeheader()
                        
                        # 只导出标准字段
                        for message in self.chat_history:
                            row = {field: message.get(field, '') for field in standard_fields}
                            writer.writerow(row)

                logger.info(f"导出CSV文件成功: {output_file}")
                return output_file

        except Exception as e:
            logger.error(f"导出CSV失败: {str(e)}")
            return ""

    def search_history(self, keyword: str, role_filter: str = None) -> List[Dict[str, Any]]:
        """搜索历史记录
        
        方法用途：根据关键词搜索聊天记录，支持按角色过滤
        
        参数解释：
            keyword (str): 搜索关键词，不区分大小写
            role_filter (str, 可选): 角色过滤器，如 'user', 'assistant'，None表示不过滤
            
        返回值：
            List[Dict[str, Any]]: 匹配的消息列表
            
        使用示例：
            >>> manager = ChatHistoryManager()
            >>> manager.add_message('user', 'Python是什么？')
            >>> manager.add_message('assistant', 'Python是一种高级编程语言')
            >>> manager.add_message('user', 'Java是什么？')
            >>> 
            >>> # 搜索包含"Python"的记录
            >>> results = manager.search_history('python')
            >>> print(len(results))  # 返回: 2（用户问题和助手回答）
            >>> 
            >>> # 只搜索用户消息中包含"是什么"的记录
            >>> user_results = manager.search_history('是什么', role_filter='user')
            >>> print(len(user_results))  # 返回: 2
            >>> 
            >>> # 搜索不存在的关键词
            >>> empty_results = manager.search_history('不存在的词')
            >>> print(len(empty_results))  # 返回: 0
        """
        try:
            results = []
            keyword = keyword.lower()

            for message in self.chat_history:
                # 角色过滤
                if role_filter and message.get("role") != role_filter:
                    continue

                # 内容搜索
                content = message.get("content", "").lower()
                if keyword in content:
                    results.append(message)

            logger.info(f"搜索历史记录: '{keyword}' - 找到 {len(results)} 条结果")
            return results

        except Exception as e:
            logger.error(f"搜索历史记录失败: {str(e)}")
            return []

    def load_history(self) -> bool:
        """加载历史记录
        
        方法用途：从JSON文件加载聊天记录到内存中
        
        参数：无
        
        返回值：
            bool: 加载成功返回True，失败返回False
            
        使用示例：
            >>> manager = ChatHistoryManager()
            >>> success = manager.load_history()
            >>> print(success)  # 返回: True（如果文件存在且格式正确）
            >>> 
            >>> # 加载后可以在内存中访问历史记录
            >>> history = manager.get_history()
            >>> print(len(history))  # 返回历史记录数量
        """
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
                logger.info(f"加载历史记录成功: {len(self.chat_history)} 条记录")
            else:
                self.chat_history = []
                logger.info("历史记录文件不存在，初始化为空列表")
            return True
        except Exception as e:
            logger.error(f"加载历史记录失败: {str(e)}")
            self.chat_history = []
            return False

    def save_history(self) -> bool:
        """保存历史记录
        
        方法用途：将内存中的聊天记录保存到JSON文件
        
        参数：无
        
        返回值：
            bool: 保存成功返回True，失败返回False
            
        使用示例：
            >>> manager = ChatHistoryManager()
            >>> manager.add_message('user', '测试消息')
            >>> success = manager.save_history()
            >>> print(success)  # 返回: True
            >>> 
            >>> # 保存后可以在文件系统中看到历史文件
            >>> import os
            >>> print(os.path.exists(manager.history_file))  # 返回: True
        """
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
            logger.debug(f"保存历史记录成功: {len(self.chat_history)} 条记录")
            return True
        except Exception as e:
            logger.error(f"保存历史记录失败: {str(e)}")
            return False

    def _generate_message_id(self) -> str:
        """生成消息ID
        
        方法用途：为每条消息生成唯一标识符（UUID）
        
        参数：无
        
        返回值：
            str: 36位的UUID字符串
            
        使用示例：
            >>> manager = ChatHistoryManager()
            >>> msg_id = manager._generate_message_id()
            >>> print(msg_id)  # 返回: '550e8400-e29b-41d4-a716-446655440000'
            >>> print(len(msg_id))  # 返回: 36
            >>> 
            >>> # 每次调用都会生成不同的ID
            >>> id1 = manager._generate_message_id()
            >>> id2 = manager._generate_message_id()
            >>> print(id1 == id2)  # 返回: False
        """
        import uuid
        return str(uuid.uuid4())


if __name__ == "__main__":
    """ChatHistoryManager 类的完整测试"""
    
    import tempfile
    import os
    
    print("🚀 开始测试 ChatHistoryManager 类...\n")
    
    # 创建临时文件用于测试
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        temp_history_file = tmp_file.name
    
    try:
        # 1. 测试初始化
        print("📋 测试1: 初始化管理器")
        manager = ChatHistoryManager(history_file=temp_history_file)
        print(f"   ✅ 初始化成功，历史文件: {manager.history_file}")
        
        # 2. 测试添加消息
        print("\n💬 测试2: 添加消息")
        manager.add_message("user", "你好，请介绍一下Python")
        manager.add_message("assistant", "Python是一种高级编程语言，具有简洁易读的语法特点。", 
                          metadata={"model": "gpt-4", "temperature": 0.7})
        manager.add_message("user", "Python适合做什么类型的项目？")
        manager.add_message("assistant", "Python适合数据分析、Web开发、人工智能、自动化脚本等多种应用场景。",
                          metadata={"model": "gpt-5", "confidence": 0.95})
        print(f"   ✅ 成功添加 {len(manager.get_history())} 条消息")
        
        # 3. 测试获取历史记录
        print("\n📖 测试3: 获取历史记录")
        full_history = manager.get_history()
        print(f"   ✅ 获取完整历史: {len(full_history)} 条记录")
        
        user_messages = manager.get_history(role_filter="user")
        print(f"   ✅ 获取用户消息: {len(user_messages)} 条记录")
        
        assistant_messages = manager.get_history(role_filter="assistant", limit=1)
        print(f"   ✅ 获取最新助手消息: {len(assistant_messages)} 条记录")
        
        # 4. 测试搜索功能
        print("\n🔍 测试4: 搜索功能")
        python_results = manager.search_history("Python")
        print(f"   ✅ 搜索 'Python': 找到 {len(python_results)} 条结果")
        
        project_results = manager.search_history("项目", role_filter="user")
        print(f"   ✅ 搜索用户消息中的 '项目': 找到 {len(project_results)} 条结果")
        
        no_results = manager.search_history("JavaScript")
        print(f"   ✅ 搜索 'JavaScript': 找到 {len(no_results)} 条结果")
        
        # 5. 测试导出功能
        print("\n📊 测试5: 导出功能")
        csv_content = manager.export_to_csv()
        print(f"   ✅ 导出CSV字符串: {len(csv_content)} 字符")
        
        csv_file = temp_history_file.replace('.json', '.csv')
        csv_path = manager.export_to_csv(csv_file)
        print(f"   ✅ 保存CSV文件: {csv_path}")
        
        # 6. 测试保存和加载
        print("\n💾 测试6: 保存和加载")
        save_success = manager.save_history()
        print(f"   ✅ 保存历史记录: {save_success}")
        
        # 创建新管理器实例并加载
        new_manager = ChatHistoryManager(history_file=temp_history_file)
        load_success = new_manager.load_history()
        print(f"   ✅ 加载历史记录: {load_success}")
        print(f"   ✅ 加载后记录数: {len(new_manager.get_history())}")
        
        # 7. 测试清空历史
        print("\n🗑️ 测试7: 清空历史记录")
        clear_success = new_manager.clear_history()
        print(f"   ✅ 清空历史记录: {clear_success}")
        print(f"   ✅ 清空后记录数: {len(new_manager.get_history())}")
        
        # 8. 测试消息ID生成
        print("\n🆔 测试8: 消息ID生成")
        msg_id = manager._generate_message_id()
        print(f"   ✅ 生成消息ID: {msg_id}")
        print(f"   ✅ ID长度: {len(msg_id)} 字符")
        
        # 9. 测试边界情况
        print("\n⚡ 测试9: 边界情况")
        empty_manager = ChatHistoryManager(history_file="non_existent_file.json")
        empty_manager.load_history()  # 应该能处理不存在的文件
        print(f"   ✅ 处理不存在的文件: {len(empty_manager.get_history())} 条记录")
        
        empty_results = empty_manager.search_history("任何内容")
        print(f"   ✅ 搜索空历史: {len(empty_results)} 条结果")
        
        empty_csv = empty_manager.export_to_csv()
        print(f"   ✅ 导出空历史: '{empty_csv}'")
        
        print("\n🎉 所有测试完成！ChatHistoryManager 类工作正常！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_history_file):
            os.unlink(temp_history_file)
        csv_file = temp_history_file.replace('.json', '.csv')
        if os.path.exists(csv_file):
            os.unlink(csv_file)
        print(f"\n🧹 清理临时文件完成")