# time_aware_retriever.py
import re
import json
from datetime import datetime, timedelta
from dateutil import parser
from typing import List, Optional, Tuple, Dict, Any
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import BaseModel, Field, ConfigDict

class TimeAwareRetriever(BaseRetriever, BaseModel):  # 双重继承
    """
    时间感知检索器
    """
    # Pydantic 字段定义
    base_retriever: BaseRetriever
    time_metadata_dict: Optional[Dict[str, Dict]] = Field(default_factory=dict)
    time_weight: float = Field(default=0.3, ge=0.0, le=1.0)  # 0-1范围

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow"  # 允许额外属性
    )

    def __init__(self, **kwargs):
        # 先初始化 Pydantic 模型
        super().__init__(**kwargs)

        # 然后初始化其他属性
        self.time_patterns = {
            "last_week": re.compile(r"上周|过去七天|last week|past week", re.IGNORECASE),
            "last_month": re.compile(r"上月|上个月|last month|past month", re.IGNORECASE),
            "yesterday": re.compile(r"昨天|yesterday", re.IGNORECASE),
            "recent": re.compile(r"最近|近期|recently|recent", re.IGNORECASE),
            "this_month": re.compile(r"本月|这个月|this month", re.IGNORECASE),
            "this_year": re.compile(r"今年|this year", re.IGNORECASE),
        }

        self.date_patterns = [
            re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日"),
            re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})"),
            re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
        ]

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
            time_range: Optional[Tuple[datetime, datetime]] = None,
            **kwargs
    ) -> List[Document]:
        """
        核心检索方法
        """
        # 1. 解析查询中的时间意图
        time_intent = self._extract_time_intent(query)
        if time_range:
            time_intent["time_range"] = time_range

        # 2. 调整召回数量（如果有时间过滤，需要更多候选）
        original_k = kwargs.get("k", 10)
        if time_intent["time_aware"]:
            kwargs["k"] = min(original_k * 3, 50)  # 最多召回50个

        # 3. 使用基础检索器获取候选文档
        candidates = self.base_retriever.get_relevant_documents(query, **kwargs)

        # 4. 应用时间感知处理
        if time_intent["time_aware"] and time_intent["time_range"]:
            processed_docs = self._process_with_time_filter(
                candidates, time_intent, original_k
            )
        else:
            # 没有时间意图，返回原始结果
            processed_docs = candidates[:original_k]

        return processed_docs

    def _extract_time_intent(self, query: str) -> Dict[str, Any]:
        """从查询中提取时间意图"""
        intent = {
            "time_aware": False,
            "time_range": None,
            "time_keywords": [],
            "query_without_time": query  # 移除时间关键词后的查询
        }

        query_lower = query.lower()
        modified_query = query

        # 检查预定义时间模式
        for pattern_name, pattern in self.time_patterns.items():
            match = pattern.search(query_lower)
            if match:
                intent["time_aware"] = True
                intent["time_keywords"].append(pattern_name)
                intent["time_range"] = self._parse_time_keyword(pattern_name)
                # 从查询中移除时间关键词
                modified_query = pattern.sub("", modified_query).strip()
                break

        # 检查具体日期格式
        if not intent["time_range"]:
            for date_pattern in self.date_patterns:
                match = date_pattern.search(query)
                if match:
                    intent["time_aware"] = True
                    intent["time_keywords"].append("specific_date")

                    # 解析日期
                    groups = match.groups()
                    if len(groups) >= 3:
                        year, month, day = groups[:3]
                        try:
                            specific_date = datetime(int(year), int(month), int(day))
                            # 设置日期前后3天的范围
                            intent["time_range"] = (
                                specific_date - timedelta(days=3),
                                specific_date + timedelta(days=3)
                            )
                            # 从查询中移除日期
                            modified_query = date_pattern.sub("", modified_query).strip()
                        except ValueError:
                            pass
                    break

        intent["query_without_time"] = modified_query
        return intent

    def _parse_time_keyword(self, keyword: str) -> Optional[Tuple[datetime, datetime]]:
        """将时间关键词转换为具体时间范围"""
        now = datetime.now()

        if keyword == "last_week":
            end_date = now - timedelta(days=now.weekday() + 7)
            start_date = end_date - timedelta(days=7)
        elif keyword == "last_month":
            if now.month == 1:
                start_date = datetime(now.year - 1, 12, 1)
            else:
                start_date = datetime(now.year, now.month - 1, 1)
            end_date = (start_date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        elif keyword == "yesterday":
            yesterday = now - timedelta(days=1)
            start_date = datetime(yesterday.year, yesterday.month, yesterday.day)
            end_date = start_date + timedelta(days=1)
        elif keyword == "recent":
            start_date = now - timedelta(days=14)  # 最近两周
            end_date = now
        elif keyword == "this_month":
            start_date = datetime(now.year, now.month, 1)
            end_date = now
        elif keyword == "this_year":
            start_date = datetime(now.year, 1, 1)
            end_date = now
        else:
            return None

        return (start_date, end_date)

    def _process_with_time_filter(
            self,
            candidates: List[Document],
            time_intent: Dict[str, Any],
            k: int
    ) -> List[Document]:
        """应用时间过滤和重排序"""
        if not candidates:
            return []

        # 1. 过滤文档
        filtered_docs = []
        for doc in candidates:
            doc_time = self._extract_document_time(doc)
            if doc_time is None:
                # 没有时间信息的文档，先保留
                filtered_docs.append((doc, doc_time))
            elif time_intent["time_range"]:
                start_date, end_date = time_intent["time_range"]
                if start_date <= doc_time <= end_date:
                    filtered_docs.append((doc, doc_time))

        # 2. 如果过滤后结果太少，放宽条件（包含无时间信息的文档）
        if len(filtered_docs) < k // 2:
            # 重新添加所有无时间信息的文档
            for doc in candidates:
                doc_time = self._extract_document_time(doc)
                if doc_time is None and (doc, None) not in filtered_docs:
                    filtered_docs.append((doc, None))

        # 3. 重排序
        if time_intent["time_range"]:
            reranked_docs = self._rerank_by_time_relevance(
                filtered_docs, time_intent["time_range"]
            )
        else:
            # 只有时间关键词但没有具体范围，按原始顺序
            reranked_docs = [doc for doc, _ in filtered_docs]

        return reranked_docs[:k]

    def _extract_document_time(self, document: Document) -> Optional[datetime]:
        """从文档元数据中提取时间"""
        metadata = document.metadata

        # 方法1：从timestamp字段提取
        if 'timestamp' in metadata and metadata['timestamp']:
            try:
                return parser.parse(str(metadata['timestamp']))
            except (ValueError, TypeError):
                pass

        # 方法2：从处理时间提取
        if 'processed_at' in metadata and metadata['processed_at']:
            # 尝试从文件名解析日期
            source_str = str(metadata['source'])
            filename_patterns = [
                r'(\d{4})-(\d{2})-(\d{2})',
                r'(\d{4})\.(\d{2})\.(\d{2})',
                r'(\d{4})(\d{2})(\d{2})'
            ]

            for pattern_str in filename_patterns:
                match = re.search(pattern_str, source_str)
                if match:
                    try:
                        year, month, day = match.groups()[:3]
                        return datetime(int(year), int(month), int(day))
                    except (ValueError, TypeError):
                        continue

        # 方法3：从page_content中寻找日期
        content = document.page_content
        date_patterns = [
            r'Date[:：]\s*(\d{4}[年\.\-/]\d{1,2}[月\.\-/]\d{1,2}[日]?)',
            r'日期[:：]\s*(\d{4}[年\.\-/]\d{1,2}[月\.\-/]\d{1,2}[日]?)',
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
        ]

        for pattern_str in date_patterns:
            match = re.search(pattern_str, content)
            if match:
                try:
                    date_str = match.group(1)
                    return parser.parse(date_str, fuzzy=True)
                except (ValueError, TypeError):
                    continue

        return None

    def _rerank_by_time_relevance(
            self,
            doc_time_pairs: List[Tuple[Document, Optional[datetime]]],
            time_range: Tuple[datetime, datetime]
    ) -> List[Document]:
        """按时间相关性重排序文档"""
        if not doc_time_pairs:
            return []

        start_date, end_date = time_range
        time_span = (end_date - start_date).total_seconds()

        # 计算中心时间点
        center_time = start_date + (end_date - start_date) / 2

        scored_docs = []
        for doc, doc_time in doc_time_pairs:
            if doc_time is None:
                # 没有时间信息的文档给中等分数
                time_score = 0.5
            else:
                # 计算时间差异（越小越好）
                time_diff = abs((doc_time - center_time).total_seconds())
                if time_span > 0:
                    # 归一化到0-1范围
                    normalized_diff = min(time_diff / time_span, 1.0)
                    time_score = 1.0 - normalized_diff
                else:
                    # 时间范围为零（具体某一天）
                    if time_diff < 24 * 3600:  # 一天内
                        time_score = 1.0
                    else:
                        time_score = 1.0 / (1.0 + time_diff / (24 * 3600))

            # 结合原始排序（假设输入已按相关性排序）
            # 这里简化处理：保留原始顺序信息
            base_score = 1.0 - (len(scored_docs) / len(doc_time_pairs)) * 0.5

            # 综合得分
            combined_score = (1 - self.time_weight) * base_score + self.time_weight * time_score

            scored_docs.append((combined_score, doc))

        # 按综合得分降序排序
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs]