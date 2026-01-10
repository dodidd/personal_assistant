import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取 DeepSeek API 密钥和基础 URL
deepseek_api_key = os.getenv("SILICONFLOW_API_KEY")
deepseek_api_base = "https://api.siliconflow.cn/v1"


# ===================== 1. 时间解析链（time_parser_chain） =====================
def build_time_parser_chain():
    time_parser_prompt = PromptTemplate(
        template="""
        你是一个时间解析助手，需要从用户的查询语句中提取时间信息，输出格式要求：
        1. 若有明确时间，输出标准化时间字符串（如YYYY-MM、YYYY-MM-DD、YYYY）；
        2. 若无明确时间，输出"无"；
        3. 只输出结果，不要额外解释。

        用户查询：{query}
        """,
        input_variables=["query"]
    )

    llm = ChatOpenAI(
        model_name="deepseek-ai/DeepSeek-V3.1", # 也可以尝试 "deepseek-coder"
        api_key=deepseek_api_key,
        base_url=deepseek_api_base,
        temperature=0.1 # 降低温度，让模型生成更确定、更少发散的代码
    )

    time_parser_chain = LLMChain(
        llm=llm,
        prompt=time_parser_prompt,
        output_parser=StrOutputParser(),
        output_key="time_info"
    )
    return time_parser_chain


# ===================== 2. RAG核心组件（向量库+检索+生成） =====================
def init_vector_db():
    sample_documents = [
        "2024-05的销售数据：总销售额100万元，环比增长10%",
        "2024-04的销售数据：总销售额90.9万元，环比增长5%",
        "2023年全年销售数据：总销售额1000万元，同比增长8%",
        "产品A的定价策略：基础价99元，5月促销价89元"
    ]

    embeddings = OpenAIEmbeddings()

    vector_db = Chroma.from_texts(
        texts=sample_documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    vector_db.persist()
    return vector_db


# ===================== 3. RAG查询管道（rag_pipeline） =====================
def build_rag_pipeline():
    # 初始化组件
    time_parser_chain = build_time_parser_chain()  # 原有逻辑，保留
    vector_db = init_vector_db()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # 检索器（原有逻辑，无改动）
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # ========== 核心改动1：优化RAG生成提示词，增加时间信息标注引导 ==========
    rag_prompt = PromptTemplate(
        template="""
        请根据以下信息回答用户问题：
        1. 时间信息标注：{time_info}  # 新增：时间标注字段
        2. 相关文档：{context}
        3. 用户问题：{query}

        要求：
        - 回答必须基于提供的文档，不编造信息；
        - 优先结合【时间信息标注】筛选相关内容作答，若无匹配时间则忽略该标注；  # 新增：引导LLM使用时间标注
        - 回答简洁、准确，符合中文表达习惯。
        """,
        input_variables=["time_info", "context", "query"]
    )

    # ========== 核心改动2：并行执行时间解析（仅标注，不影响检索） ==========
    preprocess_chain = RunnableParallel(
        time_info=time_parser_chain,  # 新增：执行时间解析，但仅用于标注
        context=retriever,  # 原有逻辑：检索流程完全不变
        query=RunnablePassthrough()  # 原有逻辑：透传用户查询
    )

    # 生成链逻辑（原有结构不变，仅传入新增的time_info参数）
    rag_chain = preprocess_chain | rag_prompt | llm | StrOutputParser()

    return rag_chain


# ===================== 4. 测试RAG管道 =====================
if __name__ == "__main__":
    rag_pipeline = build_rag_pipeline()

    # 测试查询（含时间/无时间场景）
    test_queries = [
        "2024年5月的销售数据是多少？",
        "产品A5月的促销价是多少？",
        "2023年全年销售额同比增长多少？",
        "产品A的基础定价是多少？"  # 无时间查询，验证标注不干扰结果
    ]

    # 执行查询并输出结果
    for query in test_queries:
        print(f"\n=== 用户查询：{query} ===")
        result = rag_pipeline.invoke(query)
        print(f"回答：{result}")