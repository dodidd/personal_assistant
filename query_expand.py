# query_expander.py
from typing import List, Dict, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLLM
import re


class QueryExpander:
    """基于LangChain的智能查询扩展器"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.setup_chains()

    def setup_chains(self):
        """设置查询扩展和重写链"""

        # 1. 查询扩展链 - 用于基于对话历史优化查询
        self.expansion_prompt = PromptTemplate(
            input_variables=["query", "conversation_history"],
            template="""
            你是一个查询优化助手。基于对话历史和当前问题，生成一个更完整的查询用于信息检索。

            对话历史（最近的对话）：
            {conversation_history}

            用户当前问题：{query}

            请优化这个查询用于从知识库检索：
            1. 如果问题中有代词（如"它"、"这个"、"那个"、"他们"），请明确指代的内容
            2. 如果问题不完整，请基于对话历史补充必要的上下文
            3. 保持查询简洁但信息完整
            4. 如果需要多角度检索，可以包含相关关键词

            优化后的查询：
            """
        )

        self.expansion_chain = LLMChain(
            llm=self.llm,
            prompt=self.expansion_prompt,
            output_parser=StrOutputParser(),
            verbose=False
        )

        # 2. 检索决策链 - 判断是否需要检索
        self.retrieval_decision_prompt = PromptTemplate(
            input_variables=["query", "conversation_history"],
            template="""
            请分析以下对话，判断是否需要从知识库检索信息来回答问题：

            对话历史：
            {conversation_history}

            当前问题：{query}

            请考虑：
            1. 这个问题是否需要具体的事实性信息、数据或专业知识？
            2. 对话历史中是否有足够的信息来直接回答？
            3. 这个问题是否涉及外部知识或特定领域的文档？

            请回答"需要检索"或"不需要检索"，并简要说明原因。

            决策和原因：
            """
        )

        self.retrieval_decision_chain = LLMChain(
            llm=self.llm,
            prompt=self.retrieval_decision_prompt,
            output_parser=StrOutputParser(),
            verbose=False
        )

        # 3. 查询重写链 - 生成更好的检索查询
        self.rewrite_prompt = PromptTemplate(
            input_variables=["original_query", "context"],
            template="""
            请重写以下查询，使其更适合从文档库中检索相关信息：

            原始查询：{original_query}

            上下文信息：{context}

            重写要求：
            1. 保持查询的核心意图
            2. 包含具体的检索关键词
            3. 如果可能，将通用问题转化为更具体的检索查询
            4. 考虑可能的同义词和相关术语

            重写后的查询：
            """
        )

        self.rewrite_chain = LLMChain(
            llm=self.llm,
            prompt=self.rewrite_prompt,
            output_parser=StrOutputParser(),
            verbose=False
        )

    def format_conversation_history(self, history: List[Dict], max_turns: int = 3) -> str:
        """格式化对话历史"""
        if not history:
            return "无对话历史"

        # 只取最近几轮对话
        recent_history = history[-max_turns * 2:] if len(history) > max_turns * 2 else history

        formatted = []
        for msg in recent_history:
            role = "用户" if msg["role"] == "user" else "助手"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    async def expand_query(self, query: str, conversation_history: List[Dict]) -> str:
        """扩展查询以包含上下文信息"""
        if not conversation_history:
            return query

        try:
            history_text = self.format_conversation_history(conversation_history, max_turns=2)

            expanded = await self.expansion_chain.arun(
                query=query,
                conversation_history=history_text
            )

            # 清理结果
            expanded = expanded.strip()
            if expanded.startswith('"') and expanded.endswith('"'):
                expanded = expanded[1:-1]

            return expanded if expanded and expanded != query else query

        except Exception as e:
            print(f"查询扩展失败: {e}")
            return query

    async def should_retrieve(self, query: str, conversation_history: List[Dict]) -> bool:
        """智能判断是否需要检索"""
        # 基础规则过滤
        simple_patterns = [
            r"^你好$", r"^谢谢$", r"^再见$", r"^在吗$",
            r"^hi$", r"^hello$", r"^ok$", r"^好的$"
        ]

        if any(re.match(pattern, query.lower()) for pattern in simple_patterns):
            return False

        if len(query.strip()) < 2:
            return False

        # 使用LLM进行更智能的判断
        try:
            history_text = self.format_conversation_history(conversation_history, max_turns=2)

            decision = await self.retrieval_decision_chain.arun(
                query=query,
                conversation_history=history_text
            )

            # 从LLM响应中提取决策
            decision_text = decision.lower()
            if "需要检索" in decision_text:
                return True
            elif "不需要检索" in decision_text:
                return False

            # 如果LLM没有明确说明，使用启发式规则
            question_words = ["什么", "如何", "为什么", "怎样", "何时", "哪里", "谁", "多少", "哪个"]
            retrieval_keywords = ["文档", "文件", "资料", "知识库", "报告", "数据", "信息", "查询"]

            if any(word in query for word in question_words + retrieval_keywords):
                return True

            return len(query) > 10  # 较长的问题默认需要检索

        except Exception as e:
            print(f"检索决策失败: {e}")
            # 失败时使用保守策略
            question_words = ["什么", "如何", "为什么"]
            return any(word in query for word in question_words)

    async def rewrite_for_retrieval(self, query: str, context: str = "") -> str:
        """为检索优化重写查询"""
        try:
            rewritten = await self.rewrite_chain.arun(
                original_query=query,
                context=context if context else "无额外上下文"
            )

            rewritten = rewritten.strip()
            return rewritten if rewritten else query

        except Exception as e:
            print(f"查询重写失败: {e}")
            return query

    def extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词（简单实现）"""
        # 移除常见停用词
        stopwords = {"的", "了", "在", "是", "我", "有", "和", "就",
                     "不", "人", "都", "一", "一个", "上", "也", "很",
                     "到", "说", "要", "去", "你", "会", "着", "没有",
                     "看", "好", "自己", "这"}

        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', query)
        keywords = [word for word in words if word not in stopwords and len(word) > 1]

        return keywords[:5]  # 最多返回5个关键词