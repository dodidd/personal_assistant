# time_parser_chain.py
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing import Dict, Any
import json
from datetime import datetime, timedelta
import re


class TimeRangeParser:
    """基于LangChain的时间范围解析器"""

    def __init__(self, llm):
        self.llm = llm
        self.setup_chains()

    def setup_chains(self):
        """设置时间解析链"""

        # 链1: 识别是否为时间相关查询
        self.time_detection_template = """
        请分析用户查询是否包含时间范围或时间相关的意图。

        用户查询: {query}

        请判断:
        1. 查询是否明确提到时间（如"昨天"、"上周"、"去年"等）
        2. 查询是否隐含时间概念（如"最近"、"之前"、"过去的"等）
        3. 查询是否需要按时间筛选结果

        请以JSON格式回答，包含以下字段：
        - has_time: true/false
        - time_keywords: 查询中的时间关键词列表
        - time_intent: 描述时间意图
        - needs_time_filter: true/false

        回答:
        """

        self.time_detection_prompt = PromptTemplate(
            template=self.time_detection_template,
            input_variables=["query"]
        )

        self.time_detection_chain = LLMChain(
            llm=self.llm,
            prompt=self.time_detection_prompt,
            output_parser=JsonOutputParser(),
            output_key="time_analysis"
        )

        # 链2: 提取具体时间范围
        self.time_extraction_template = """
        基于查询和时间分析，提取具体的时间范围。

        当前时间: {current_time}
        用户查询: {query}
        时间分析: {time_analysis}

        请提取具体的时间范围参数，以JSON格式返回：
        - time_range_type: "absolute"或"relative"或"none"
        - start_date: 开始日期（YYYY-MM-DD格式），如果没有则为null
        - end_date: 结束日期（YYYY-MM-DD格式），如果没有则为null
        - relative_days: 相对天数（如"最近7天"则为7）
        - relative_months: 相对月数
        - time_field: 应用的时间字段（"added_time"或"content_time"或"both"）
        - description: 时间范围的自然语言描述

        示例1: 查询"昨天添加的文件"
        返回: {{
            "time_range_type": "relative",
            "start_date": null,
            "end_date": null,
            "relative_days": 1,
            "relative_months": 0,
            "time_field": "added_time",
            "description": "昨天（24小时内）添加的文件"
        }}

        示例2: 查询"2023年的项目文档"
        返回: {{
            "time_range_type": "absolute",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "relative_days": null,
            "relative_months": null,
            "time_field": "content_time",
            "description": "2023年内的内容"
        }}

        请根据实际查询提取：
        """

        self.time_extraction_prompt = PromptTemplate(
            template=self.time_extraction_template,
            input_variables=["query", "time_analysis", "current_time"]
        )

        self.time_extraction_chain = LLMChain(
            llm=self.llm,
            prompt=self.time_extraction_prompt,
            output_parser=JsonOutputParser(),
            output_key="time_range"
        )

        # 链3: 重写查询（可选）
        self.query_rewrite_template = """
        基于时间分析，优化查询用于检索。

        原始查询: {query}
        时间分析: {time_analysis}
        时间范围: {time_range}

        请生成一个优化后的查询，用于向量检索系统：
        1. 保留核心语义
        2. 明确时间范围（如果有时）
        3. 添加相关关键词

        优化后的查询:
        """

        self.query_rewrite_prompt = PromptTemplate(
            template=self.query_rewrite_template,
            input_variables=["query", "time_analysis", "time_range"]
        )

        self.query_rewrite_chain = LLMChain(
            llm=self.llm,
            prompt=self.query_rewrite_prompt,
            output_parser=StrOutputParser(),
            output_key="optimized_query"
        )

    async def parse_time_range(self, query: str) -> Dict[str, Any]:
        """解析查询中的时间范围"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 执行时间分析
            time_analysis = await self.time_detection_chain.arun(
                query=query,
                current_time=current_time
            )

            # 如果不需要时间过滤，直接返回
            if not time_analysis.get('needs_time_filter', False):
                return {
                    "has_time": False,
                    "time_range": None,
                    "optimized_query": query,
                    "time_analysis": time_analysis
                }

            # 提取时间范围
            time_range = await self.time_extraction_chain.arun(
                query=query,
                time_analysis=json.dumps(time_analysis, ensure_ascii=False),
                current_time=current_time
            )

            # 重写查询
            optimized_query = await self.query_rewrite_chain.arun(
                query=query,
                time_analysis=json.dumps(time_analysis, ensure_ascii=False),
                time_range=json.dumps(time_range, ensure_ascii=False)
            )

            return {
                "has_time": True,
                "time_range": time_range,
                "optimized_query": optimized_query,
                "time_analysis": time_analysis,
                "original_query": query
            }

        except Exception as e:
            print(f"时间解析失败: {e}")
            return {
                "has_time": False,
                "time_range": None,
                "optimized_query": query,
                "time_analysis": {"error": str(e)},
                "original_query": query
            }

    def build_metadata_filter(self, time_range: Dict) -> Dict:
        """根据时间范围构建元数据过滤器"""
        if not time_range or time_range.get('time_range_type') == 'none':
            return {}

        filter_dict = {}
        time_field = time_range.get('time_field', 'added_time')

        if time_range['time_range_type'] == 'absolute':
            # 绝对时间范围
            if time_range.get('start_date'):
                filter_dict[f'{time_field}_gte'] = time_range['start_date']
            if time_range.get('end_date'):
                filter_dict[f'{time_field}_lte'] = time_range['end_date']

        elif time_range['time_range_type'] == 'relative':
            # 相对时间范围
            days = time_range.get('relative_days')
            months = time_range.get('relative_months')

            if days:
                # 计算相对日期
                from datetime import datetime, timedelta
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days)

                filter_dict[f'{time_field}_gte'] = start_date.isoformat()
                filter_dict[f'{time_field}_lte'] = end_date.isoformat()

        return filter_dict