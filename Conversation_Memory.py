from typing import Dict, List, Any
from datetime import datetime
import json
# Conversation_Memory.py
from datetime import datetime
from typing import List, Dict, Any, Optional


class ConversationMemory:
    """管理对话历史的类"""

    def __init__(self, max_turns: int = 10):
        self.memory = {}
        self.max_turns = max_turns

    def get_session_key(self, request) -> str:
        """从请求中获取会话ID"""
        return request.headers.get("X-Session-ID", "default")

    def add_message(self, session_key: str, role: str, content: str, **kwargs):
        """添加消息到对话历史"""
        if session_key not in self.memory:
            self.memory[session_key] = []

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        # 添加额外的元数据
        if kwargs:
            message.update(kwargs)

        self.memory[session_key].append(message)

        # 保持最近N轮对话
        if len(self.memory[session_key]) > self.max_turns * 2:
            self.memory[session_key] = self.memory[session_key][-self.max_turns * 2:]

    def get_recent_history(self, session_key: str, max_turns: Optional[int] = None) -> List[Dict]:
        """获取最近的对话历史"""
        if session_key not in self.memory:
            return []

        if max_turns is None:
            max_turns = self.max_turns

        return self.memory[session_key][-max_turns * 2:]

    def get_formatted_history(self, session_key: str) -> str:
        """将对话历史格式化为文本"""
        history = self.get_recent_history(session_key)
        formatted = []
        for msg in history:
            formatted.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(formatted)

    def clear_history(self, session_key: str):
        """清空指定会话的历史"""
        if session_key in self.memory:
            self.memory[session_key] = []

