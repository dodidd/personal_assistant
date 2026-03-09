# rag_pipeline.py
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from CHLoader import ConversationHistoryLoader
from utils import rerank
from langchain_core.documents import Document  # 确保导入Document类型
import json
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
# 加载 .env 文件中的环境变量
load_dotenv()

# 获取 DeepSeek API 密钥和基础 URL
deepseek_api_key = os.getenv("SILICONFLOW_API_KEY")
deepseek_api_base = "https://api.siliconflow.cn/v1"
llm=None
def get_llm():
    global llm
    if llm is None:
        llm = ChatOpenAI(
            model_name="deepseek-ai/DeepSeek-V3.1",
            api_key=deepseek_api_key,
            base_url=deepseek_api_base,
            temperature=0.1,
            max_tokens=2048,  # 限制生成长度
            timeout=40,  # 超时设置
        )
    return llm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser

# 查询优化链
optimize_prompt = PromptTemplate(
    input_variables=["user_query", "conversation_history"],
    template="""
基于对话历史优化用户查询，使其更适合检索。

对话历史：
{conversation_history}

用户当前查询：{user_query}

要求：
- 如果查询中有指代词（如“它”“那个”），结合历史替换为具体实体。
- 如果历史中有相关上下文，补充完整。
- 只输出优化后的查询，不要解释。

优化查询：
"""
)
optimize_chain = LLMChain(llm=get_llm(), prompt=optimize_prompt, output_key="optimized_query")

# 检索必要性判断链（输出JSON）
retrieval_prompt = PromptTemplate(
    input_variables=["user_query", "conversation_history"],
    template="""
判断用户查询是否需要检索外部知识库。

对话历史：
{conversation_history}

用户查询：{user_query}

请以JSON格式输出：
{{
    "needs_retrieval": true/false,
    "reason": "简要原因"
}}

判断规则：
- 如果是问候、感谢、闲聊等，无需检索 → false
- 如果涉及事实、数据、文档内容，或含有“什么”“如何”“为什么”等疑问词 → true
- 如果对话历史中已提供足够信息，可判断为false
"""
)
retrieval_chain = LLMChain(llm=get_llm(), prompt=retrieval_prompt, output_key="retrieval_json")

from langchain_core.runnables import RunnableParallel, RunnablePassthrough


def build_preprocessing_chain(memory):
    history_loader = ConversationHistoryLoader(memory,use_conversation=True)

    # 并行执行两个子链
    parallel_tasks = RunnableParallel(
        optimized_query=optimize_chain,  # 输出 "optimized_query"
        retrieval_json=retrieval_chain  # 输出 "retrieval_json"
    )

    # 完整预处理链：先加载历史，再并行执行，最后解析retrieval_json为布尔值
    preprocessing_chain = (
            history_loader
            | parallel_tasks
            | RunnablePassthrough.assign(
        needs_retrieval=lambda x: json.loads(x["retrieval_json"]).get("needs_retrieval", True)
    )
    )
    return preprocessing_chain

class RAGPipeline:
    """RAG处理管道"""

    def __init__(self, retriever, full_chain, conversation_memory):
        """
        初始化RAG管道
        :param retriever: 混合检索器 (ensemble_retriever)
        :param full_chain: 生成链 (prompt | llm)
        :param conversation_memory: 对话记忆管理器
        """
        self.retriever = retriever
        self.full_chain = full_chain
        self.conversation_memory = conversation_memory

        self.preprocessing_chain = build_preprocessing_chain(conversation_memory)

    async def process_query(
            self,
            user_query: str,
            session_key: str,
            use_conversation: bool = True
    ) -> Dict[str, Any]:
        """
        主处理流程
        """
        # 阶段1: 准备阶段
        preparation_result = await self._prepare_query(
            user_query, session_key, use_conversation
        )

        # 阶段2: 检索阶段
        retrieval_result = await self._retrieve_documents(
            preparation_result["optimized_query"],
            preparation_result["should_retrieve"]
        )

        # 阶段3: 构建阶段
        context_data = self._build_context(
            user_query=user_query,
            optimized_query=preparation_result["optimized_query"],
            conversation_history=preparation_result["conversation_history"],
            retrieved_docs=retrieval_result["documents"],
            retrieval_used=retrieval_result["retrieval_used"]
        )
        retrieval_result_str = self._format_retrieved_docs_to_str(retrieval_result["documents"])
        # 阶段4: 生成阶段
        try:
            # print("context_data", context_data)

            print("retrieval_result_str", retrieval_result_str)
            generation_result = await self.full_chain.ainvoke({
                "query": user_query,
                "retrieved_docs": retrieval_result_str,
                "time_analysis_json": "{}"
            })
            # 修复3: 兼容不同返回格式，提取最终答案
            if isinstance(generation_result, dict):
                answer = generation_result.get("final_answer", str(generation_result))
            else:
                answer = str(generation_result)  # 直接是字符串的情况

        except Exception as e:
            print(f"生成答案失败: {str(e)}")
            answer = f"生成答案时出错: {str(e)}"

        # 阶段5: 后处理阶段
        final_result = self._post_process(
            user_query=user_query,
            answer=answer,  # 使用提取后的答案
            retrieved_docs=retrieval_result["documents"],
            retrieval_used=retrieval_result["retrieval_used"],
            metadata={
                "optimized_query": preparation_result["optimized_query"],
                "session_key": session_key,
                "conversation_history": preparation_result["conversation_history"]
            }
        )

        # 阶段6: 更新对话记忆
        self._update_conversation_memory(
            session_key=session_key,
            user_query=user_query,
            assistant_answer=answer,  # 使用提取后的答案
            retrieval_used=retrieval_result["retrieval_used"]
        )

        return final_result

    # ==================== 阶段1: 准备阶段 ====================
    async def _prepare_query(
            self,
            user_query: str,
            session_key: str,
            use_conversation: bool
    ) -> Dict[str, Any]:
        """准备查询：获取对话历史、优化查询、决定是否检索"""
        print(f"[阶段1] 准备查询: '{user_query}'")

        # 1.1 获取对话历史
        conversation_history = []
        if use_conversation:
            conversation_history = self.conversation_memory.get_recent_history(session_key)
            print(f"  对话历史: {len(conversation_history)} 条消息")

        # 1.2 优化查询（基于对话历史）
        optimized_query = self._optimize_query_with_history(user_query, conversation_history)

        # 1.3 决定是否需要检索
        should_retrieve = self._should_retrieve_for_query(user_query, conversation_history)

        return {
            "conversation_history": conversation_history,
            "optimized_query": optimized_query,
            "should_retrieve": should_retrieve,
            "original_query": user_query
        }

    def _optimize_query_with_history(self, user_query: str, conversation_history: List[Dict]) -> str:
        """基于对话历史优化查询"""
        if not conversation_history:
            return user_query

        # 检查是否需要澄清指代
        if any(word in user_query for word in ["这个", "那个", "它", "他们"]):
            last_user_question = None
            for msg in reversed(conversation_history):
                if msg['role'] == 'user':
                    last_user_question = msg['content']
                    break

            if last_user_question:
                return f"{user_query} (上下文: {last_user_question})"

        return user_query

    def _should_retrieve_for_query(self, user_query: str, conversation_history: List[Dict]) -> bool:
        """判断是否需要检索"""
        simple_queries = ["你好", "谢谢", "再见", "在吗", "hi", "hello",""]
        if user_query.lower() in [q.lower() for q in simple_queries]:
            return False

        retrieval_keywords = ["文档", "文件", "资料", "知识", "查询", "搜索", "什么", "如何", "为什么"]
        if any(keyword in user_query for keyword in retrieval_keywords):
            return True

        return len(user_query) > 3

    # ==================== 阶段2: 检索阶段 ====================
    async def _retrieve_documents(self, query: str, should_retrieve: bool) -> Dict[str, Any]:
        """检索文档：执行检索和重排序"""
        print(f"[阶段2] 检索文档: 需要检索={should_retrieve}")

        if not should_retrieve:
            return {
                "documents": [],
                "retrieval_used": False,
                "retrieved_count": 0
            }

        try:
            # 2.1 初始检索
            initial_docs = self.retriever.invoke(query)[:8]
            retrieved_count = len(initial_docs)
            print(f"  初始检索到 {retrieved_count} 个文档")

            if not initial_docs:
                return {
                    "documents": [],
                    "retrieval_used": True,
                    "retrieved_count": 0
                }

            # 2.2 重排序
            reranked_docs = rerank.rerank_documents_siliconflow(
                query, initial_docs, top_n=5
            )
            print(f"  重排序后保留 {len(reranked_docs)} 个文档")

            return {
                "documents": reranked_docs,
                "retrieval_used": True,
                "retrieved_count": retrieved_count
            }

        except Exception as e:
            print(f"  检索失败: {e}")
            return {
                "documents": [],
                "retrieval_used": False,
                "retrieved_count": 0
            }

    # ==================== 阶段3: 构建阶段 ====================
    def _format_retrieved_docs_to_str(self, retrieved_docs: List[Document]) -> str:

        if not retrieved_docs:
            return "无相关参考文档"

        doc_str_list = []
        for idx, doc in enumerate(retrieved_docs, 1):
            # 修复：Document对象用.访问属性，而非[]
            doc_content = doc.page_content.strip()  # 文档内容
            # 从metadata中提取来源信息
            doc_source = doc.metadata.get('source', doc.metadata.get('filename', f"文档{idx}"))
            # 提取时间信息（可选）
            doc_time = doc.metadata.get('added_time', doc.metadata.get('timestamp', ''))

            # 构建单篇文档的格式化字符串
            doc_str = f"【参考文档{idx} - {doc_source}】"
            if doc_time:
                doc_str += f"（添加时间：{doc_time}）"
            doc_str += f"\n{doc_content}\n"

            doc_str_list.append(doc_str)

        return "\n".join(doc_str_list)

    def _build_context(
            self,
            user_query: str,
            optimized_query: str,
            conversation_history: List[Dict],
            retrieved_docs: List[Document],
            retrieval_used: bool
    ) -> Dict[str, Any]:
        """构建上下文：准备给full_chain的输入"""
        print(f"[阶段3] 构建上下文: 使用检索={retrieval_used}")

        if retrieval_used and retrieved_docs:
            # 使用检索结果的上下文
            context_text = self._build_enhanced_context(
                user_query=user_query,
                conversation_history=conversation_history,
                retrieved_docs=retrieved_docs
            )
            context_type = "enhanced_with_retrieval"
        else:
            # 纯对话上下文
            context_text = self._build_conversation_only_context(
                user_query=user_query,
                conversation_history=conversation_history
            )
            context_type = "conversation_only"

        print(f"  上下文类型: {context_type}, 长度: {len(context_text)} 字符")

        return {
            "context_text": context_text,
            "context_type": context_type,
            "optimized_query": optimized_query
        }

    def _build_enhanced_context(self, user_query: str, conversation_history: List[Dict],
                                retrieved_docs: List[Document]) -> str:
        """构建增强上下文"""
        parts = []
        parts.append("你是一个智能助手，可以结合对话历史和知识库内容回答问题。")

        if conversation_history:
            parts.append("\n### 对话历史：")
            for msg in conversation_history[-6:]:
                role_name = "用户" if msg['role'] == 'user' else "助手"
                parts.append(f"{role_name}: {msg['content']}")

        parts.append("\n### 相关知识：")
        # 修复：适配Document对象的page_content属性
        parts.append("\n\n".join([doc.page_content for doc in retrieved_docs]))

        parts.append(f"\n### 当前问题：{user_query}")
        parts.append("\n### 回答要求：")
        parts.append("1. 基于以上信息回答问题")
        parts.append("2. 如果知识库中没有相关信息，请说明")
        parts.append("3. 保持回答自然，就像在对话中一样")

        return "\n".join(parts)

    def _build_conversation_only_context(self, user_query: str, conversation_history: List[Dict]) -> str:
        """构建纯对话上下文"""
        parts = []
        parts.append("你是一个智能助手，正在与用户对话。")

        if conversation_history:
            parts.append("\n以下是对话历史：")
            for msg in conversation_history:
                role_name = "用户" if msg['role'] == 'user' else "助手"
                parts.append(f"{role_name}: {msg['content']}")

        parts.append(f"\n用户的最新问题是：{user_query}")
        parts.append("\n请给出自然、有帮助的回答。")

        return "\n".join(parts)

    # ==================== 阶段4: 生成阶段（备用） ====================
    async def _generate_answer(self, context_text: str) -> Dict[str, Any]:
        """生成答案：备用方法"""
        print(f"[阶段4] 生成答案")

        try:
            # 异步包装同步调用
            answer_response = await asyncio.to_thread(
                self.full_chain.invoke, {"input": context_text}
            )

            # 提取答案内容
            answer_text = answer_response
            if hasattr(answer_response, 'content'):
                answer_text = answer_response.content
            elif isinstance(answer_response, dict):
                answer_text = answer_response.get('answer', str(answer_response))

            print(f"  答案生成完成，长度: {len(answer_text)} 字符")

            return {
                "answer": answer_text,
                "raw_response": answer_response
            }

        except Exception as e:
            print(f"  生成失败: {e}")
            return {
                "answer": f"生成答案时出错: {str(e)}",
                "raw_response": None
            }

    # ==================== 阶段5: 后处理阶段 ====================
    def _post_process(
            self,
            user_query: str,
            answer: str,
            retrieved_docs: List[Document],
            retrieval_used: bool,
            metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """后处理：生成摘要、时间线等"""
        print(f"[阶段5] 后处理")

        # 5.1 生成摘要
        summary = self._generate_summary(retrieved_docs, retrieval_used)

        # 5.2 构建时间线
        timeline = self._build_timeline(retrieved_docs)

        # 5.3 格式化来源
        formatted_content = self._format_content(retrieved_docs)

        return {
            "question": user_query,
            "answer": answer,
            "summary": summary,
            "timeline": timeline,
            "success": True,
            "retrieval_used": retrieval_used,
            "retrieved_count": len(retrieved_docs),
            "optimized_query": metadata.get("optimized_query", user_query),
            "conversation_turns": len(metadata.get("conversation_history", [])) // 2,
            "content": formatted_content,
            "session_key": metadata.get("session_key", "default"),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_summary(self, docs: List[Document], retrieval_used: bool) -> str:
        """生成摘要"""
        if not retrieval_used or not docs:
            return "本次回答基于对话上下文（未使用知识库检索）"

        sources = set()
        dates = []

        for doc in docs[:3]:
            if 'source' in doc.metadata:
                sources.add(doc.metadata['source'])
            if 'timestamp' in doc.metadata:
                dates.append(doc.metadata['timestamp'])

        summary_parts = []
        if sources:
            summary_parts.append(f"来源：{', '.join(list(sources)[:2])}")
        if dates:
            date_str = sorted(set(dates))
            if len(date_str) > 1:
                summary_parts.append(f"时间范围：{date_str[0]} 到 {date_str[-1]}")
            else:
                summary_parts.append(f"时间：{date_str[0]}")

        return " | ".join(summary_parts) if summary_parts else "检索到相关内容"

    def _build_timeline(self, docs: List[Document]) -> Dict:
        """构建时间线"""
        timeline = {}
        for doc in docs:
            if 'timestamp' in doc.metadata:
                date = doc.metadata['timestamp']
                if date not in timeline:
                    timeline[date] = []
                timeline[date].append({
                    'preview': doc.page_content[:80] + '...',
                    'source': doc.metadata.get('source', '')
                })
        return dict(sorted(timeline.items(), reverse=True))

    def _format_content(self, docs: List[Document]) -> List[Dict]:
        """格式化内容"""
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, 'score', 0.0)
            }
            for doc in docs
        ]

    # ==================== 阶段6: 记忆更新 ====================
    def _update_conversation_memory(
            self,
            session_key: str,
            user_query: str,
            assistant_answer: str,
            retrieval_used: bool
    ):
        """更新对话记忆"""
        print(f"[阶段6] 更新对话记忆")

        # 添加用户消息
        self.conversation_memory.add_message(
            session_key,
            "user",
            user_query
        )

        # 添加助手消息
        self.conversation_memory.add_message(
            session_key,
            "assistant",
            assistant_answer,
            retrieval_used=retrieval_used
        )

        print(f"  对话记忆已更新，当前轮数: {len(self.conversation_memory.get_recent_history(session_key)) // 2}")