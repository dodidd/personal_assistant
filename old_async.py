import pickle
import tempfile
import time
from datetime import datetime

import uvicorn
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
from starlette.requests import Request
from utils.load_split_temp import load_split
from utils import rerank
import json
from time_aware_retriever import TimeAwareRetriever
from utils.rerank import append_to_metadata_file

# 全局变量，避免每次请求都重新加载
loaded_embeddings_model = None
loaded_faiss_db = None
bm25_retriever = None
ensemble_retriever = None
PDF_PATH = r"./金融数据集-报表"
EMBEDDING_MODEL_NAME_OR_PATH = ''
FAISS_DB_PATH = r"./faiss_index_bge_m3"
METADATA_FILE_NAME = "documents_metadata.json"
BM25_INDEX_PATH = r"./bm25_index"

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate  # 导入 ChatPromptTemplate

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取 DeepSeek API 密钥和基础 URL
deepseek_api_key = os.getenv("SILICONFLOW_API_KEY")
deepseek_api_base = "https://api.siliconflow.cn/v1"

# 初始化 LangChain 的 ChatOpenAI 模型，用于调用 DeepSeek API
llm = ChatOpenAI(
    model_name="deepseek-ai/DeepSeek-V3.1", # 也可以尝试 "deepseek-coder"
    api_key=deepseek_api_key,
    base_url=deepseek_api_base,
    temperature=0.1 # 降低温度，让模型生成更确定、更少发散的代码
)

chainA_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个关注数据时间的个人记忆智能助手。你将收到可能基于时间的数据文本和问题。"),
        ("human", "若问题中有时间要素，优先按时间来查询检索并在开头注'需要构建时间线'：{input}"),
    ]
)
from langchain.chains import LLMChain # LLMChain类在0.1.17版本中已被弃用，并将在1.0版本中移除

chainA_chains = LLMChain(llm=llm,
                         prompt=chainA_template,
                         verbose=True
                        )

chainC_template = ChatPromptTemplate.from_messages(
    [

        ("system", "你非常善于提取文本中的重要信息，并做出一段话或分点总结。你会收到可能需要构建时间线的文本，按时间线和内容进行简要的总结。如果没有时间线或找不到时间明确的文本，则总结最相关内容"),
        ("human", "这是针对一个提问完整的内容：{input}"),
    ]
)

chainC_chains = LLMChain(llm=llm,
                         prompt=chainC_template,
                         verbose=True
                        )

# 导入SimpleSequentialChain
from langchain.chains import SimpleSequentialChain

# 在chains参数中，按顺序传入LLMChain A 和LLMChain C
# full_chain = SimpleSequentialChain(chains=[chainA_chains,chainC_chains], verbose=True)
prompt = PromptTemplate(
    input_variables=["input"],
    template="你非常善于提取文本中的重要信息，并做出一段话或分点总结。你会收到可能需要构建时间线的文本，按时间线和上下文进行简要的总结。如果没有时间线或找不到时间明确的文本，则总结最相关内容：\n{input}"
)

full_chain = prompt | llm


async def initialize_retrievers():
    """初始化检索器 - 修正版"""
    global loaded_embeddings_model, loaded_faiss_db, bm25_retriever, ensemble_retriever

    # ================= 第一步：检查数据库状态 =================
    metadata_path = os.path.join(FAISS_DB_PATH, METADATA_FILE_NAME)
    faiss_index_path = os.path.join(FAISS_DB_PATH, "index.faiss")

    has_metadata = os.path.exists(metadata_path) and os.path.getsize(metadata_path) > 10
    has_faiss = os.path.exists(faiss_index_path)

    print(f"数据库状态: metadata={has_metadata}, faiss={has_faiss}")

    # ================= 第二步：准备文档数据 =================
    document_chunks = []

    if has_metadata:
        # 情况1：数据库已存在，加载已有数据
        print("加载现有数据库...")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                raw_documents = json.load(f)

            for doc_dict in raw_documents:
                page_content = doc_dict.get('page_content_preview', '')
                metadata = {
                    'chunk_id': doc_dict.get('chunk_id', ''),
                    'source': doc_dict.get('source', ''),
                    'page': doc_dict.get('page', 0),
                    'chunk_index': doc_dict.get('chunk_index', 0),
                    'timestamp': doc_dict.get('timestamp', ''),
                    'processed_at': doc_dict.get('processed_at', ''),
                }
                document_chunks.append(Document(
                    page_content=page_content,
                    metadata=metadata
                ))

            print(f"成功加载 {len(document_chunks)} 个文档块")

        except Exception as e:
            print(f"加载metadata失败: {e}")
            document_chunks = []

    # elif os.path.exists(PDF_PATH) and PDF_PATH.endswith('.pdf'):
    #     # 情况2：有默认PDF，但没有数据库（第一次运行）
    #     print(f"检测到默认PDF: {PDF_PATH}，创建新数据库...")
    #     try:
    #         document_chunks = load_split(PDF_PATH)
    #         if document_chunks:
    #             print(f"从PDF加载了 {len(document_chunks)} 个文档块")
    #     except Exception as e:
    #         print(f"加载PDF失败: {e}")
    #         document_chunks = []

    else:
        # 情况3：既没有数据库也没有默认PDF
        print("数据库为空，创建空检索器...")
        document_chunks = []

    from langchain_community.vectorstores import FAISS
    # ================= 第三步：初始化检索器组件 =================
    if loaded_embeddings_model is None:
        start_time = time.time()

        # 1. 加载嵌入模型
        loaded_embeddings_model = rerank.get_embeddings_model(rerank.EMBEDDING_MODEL_NAME_OR_PATH)

        # 2. 处理FAISS数据库
        if has_faiss and document_chunks:
            # 已有FAISS，直接加载
            loaded_faiss_db = FAISS.load_local(
                FAISS_DB_PATH,
                loaded_embeddings_model,
                allow_dangerous_deserialization=True
            )
            print(f"已从 {FAISS_DB_PATH} 加载FAISS数据库")
        else:
            # 没有FAISS，需要创建
            if document_chunks:
                # 有文档数据，创建新数据库
                rerank.create_and_save_faiss_db(
                    document_chunks,
                    loaded_embeddings_model,
                    FAISS_DB_PATH
                )
                # 保存metadata（如果还没有）
                if not has_metadata:
                    rerank.create_and_save_metadata(
                        document_chunks,
                        FAISS_DB_PATH,
                        METADATA_FILE_NAME
                    )

                loaded_faiss_db = FAISS.load_local(
                    FAISS_DB_PATH,
                    loaded_embeddings_model,
                    allow_dangerous_deserialization=True
                )
                print(f"已创建新的FAISS数据库，包含 {len(document_chunks)} 个文档")
            else:
                # 空数据库，创建空的FAISS
                from langchain_community.vectorstores import FAISS
                loaded_faiss_db = FAISS.from_documents(
                    [],  # 空文档列表
                    loaded_embeddings_model
                )
                loaded_faiss_db.save_local(FAISS_DB_PATH)
                print("已创建空的FAISS数据库")

        # 3. 初始化BM25检索器
        if os.path.exists(BM25_INDEX_PATH):
            with open(BM25_INDEX_PATH, 'rb') as f:
                bm25_data = pickle.load(f)
                bm25_retriever = bm25_data['retriever']
            print("已加载BM25索引")
        else:
            bm25_retriever = BM25Retriever.from_documents(document_chunks or [])
            bm25_retriever.k = 6

            # 保存到文件
            with open(BM25_INDEX_PATH, 'wb') as f:
                pickle.dump({'retriever': bm25_retriever}, f)
            print("已创建BM25索引")

        # 4. 创建时间感知检索器和混合检索器
        print("正在初始化时间感知检索器...")

        time_aware_bm25 = TimeAwareRetriever(
            base_retriever=bm25_retriever,
            time_weight=0.4
        )

        vector_retriever = loaded_faiss_db.as_retriever(search_kwargs={"k": bm25_retriever.k})
        time_aware_faiss = TimeAwareRetriever(
            base_retriever=vector_retriever,
            time_weight=0.4
        )

        # 组合检索器
        ensemble_retriever = EnsembleRetriever(
            retrievers=[time_aware_bm25, time_aware_faiss],
            weights=[0.5, 0.5]
        )

        elapsed_time = time.time() - start_time
        print(f"检索器初始化完成，耗时 {elapsed_time:.2f} 秒")
        print("包含：BM25检索器、FAISS检索器、时间感知检索器")


# 异步处理函数
async def homepage(request):
    return JSONResponse({"message": "Hello, Starlette!"})


async def rag_query(request):
    """处理RAG查询请求"""
    try:
        # 初始化检索器（如果尚未初始化）
        await initialize_retrievers()

        # 获取请求体中的JSON数据
        body = await request.json()
        user_query = body.get("question",'None')

        if not user_query:
            return JSONResponse({"error": "缺少question参数"}, status_code=400)


        # 执行检索和重排序
        initial_retrieved_docs = ensemble_retriever.invoke(user_query)[:8]
        top_reranked_docs = rerank.rerank_documents_siliconflow(user_query, initial_retrieved_docs, top_n=5)
        context_text ="question:"+user_query + "context:"+"\n\n".join([doc.page_content for doc in top_reranked_docs ])
        #answer = full_chain.invoke({"input": context_text })
        print("top_reranked_docs :",top_reranked_docs )
        # 准备上下文
        context_text = "question:" + user_query + "context:" + "\n\n".join(
            [doc.page_content for doc in top_reranked_docs])

        # 1. 生成主要答案
        answer = full_chain.invoke({"input": context_text})

        # 2. 新增：生成摘要（只有一行）
        summary = generate_simple_summary(top_reranked_docs)

        # 3. 新增：生成时间线（只有一行）
        timeline = build_simple_timeline(top_reranked_docs)

        # 格式化响应
        response_data = {
            "question": user_query,
            "answer": answer.content,
            "summary": summary,  # 新增
            "timeline": timeline,  # 新增
            "success": True,
            "retrieved_count": len(initial_retrieved_docs),
            "content": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, 'score', 0.0)
                }
                for doc in top_reranked_docs
            ]
        }

        return JSONResponse(response_data)

    except Exception as e:
        return JSONResponse({"error": f"处理请求时出错: {str(e)}"}, status_code=500)



def generate_simple_summary(docs):
    """生成简单摘要"""
    if not docs:
        return ""

    # 提取关键信息
    sources = set()
    dates = []
    contents = []

    for doc in docs[:3]:  # 只取前3个
        if 'source' in doc.metadata:
            sources.add(doc.metadata['source'])
        if 'timestamp' in doc.metadata:
            dates.append(doc.metadata['timestamp'])
        contents.append(doc.page_content[:100] + "...")

    # 构建摘要
    summary_parts = []
    if sources:
        summary_parts.append(f"来源：{', '.join(list(sources)[:2])}")
    if dates:
        date_str = sorted(set(dates))
        if len(date_str) > 1:
            summary_parts.append(f"时间范围：{date_str[0]} 到 {date_str[-1]}")
        else:
            summary_parts.append(f"时间：{date_str[0]}")

    summary_parts.append("主要内容：")
    summary_parts.extend([f"- {content}" for content in contents[:2]])

    return "\n".join(summary_parts)


def build_simple_timeline(docs):
    """构建简单时间线"""
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

    # 按日期排序
    return dict(sorted(timeline.items(), reverse=True))

def generate_summary_with_llm(query, docs):
    """用LLM生成摘要"""
    if len(docs) < 1:  # 至少有文档才生成
        return ""

    # 准备上下文
    context = "\n---\n".join([
        f"文档 {i + 1}: {doc.page_content[:200]}"
        for i, doc in enumerate(docs[:3])  # 只取前3个
    ])

    prompt = f"""
    请根据以下文档生成简洁摘要：

    {context}

    摘要要求：
    1. 概括核心内容
    2. 不超过100字

    摘要：
    """

    try:
        response = full_chain.invoke({"input": prompt})
        return response.content if hasattr(response, 'content') else str(response)
    except:
        return generate_simple_summary(docs)  # LLM失败时回退


def rebuild_bm25_index():
    """重建BM25索引（基于当前metadata）"""
    try:
        metadata_path = os.path.join(FAISS_DB_PATH, METADATA_FILE_NAME)

        if not os.path.exists(metadata_path):
            print("❌ metadata文件不存在")
            return create_minimal_bm25()

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # 处理空metadata情况
        if not metadata:
            print("📭 metadata为空，创建最小BM25索引")
            return create_minimal_bm25()

        # 从metadata重建Document列表
        documents = []

        for item in metadata:
            # 获取内容（优先使用预览，没有则用完整内容）
            content = item.get('page_content_preview', '') or item.get('page_content', '')
            if not content.strip():
                continue  # 跳过空内容

            doc = Document(
                page_content=content,
                metadata={
                    'chunk_id': item.get('chunk_id', ''),
                    'source': item.get('source', ''),
                    'timestamp': item.get('timestamp', ''),
                    'page': item.get('page', 0)
                }
            )
            documents.append(doc)

        # 必须有至少一个文档才能创建BM25
        if not documents:
            print("⚠️ 没有有效文档，创建最小BM25索引")
            return create_minimal_bm25()

        print(f"🔄 重建BM25索引: {len(documents)} 个文档")

        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = 6

        # 保存到缓存文件
        import pickle
        os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
        with open(BM25_INDEX_PATH, 'wb') as f:
            pickle.dump({'retriever': bm25}, f)

        print("✅ BM25索引重建完成")
        return bm25

    except Exception as e:
        print(f"❌ 重建BM25失败: {e}")
        return create_minimal_bm25()


def create_minimal_bm25():
    """创建最小BM25索引（避免空列表错误）"""
    try:
        # 创建一个虚拟文档，确保BM25能正常工作
        dummy_doc = Document(
            page_content="系统初始化",
            metadata={'chunk_id': 'dummy_001', 'source': 'system'}
        )

        bm25 = BM25Retriever.from_documents([dummy_doc])
        bm25.k = 1  # 最小化返回结果

        print("⚠️ 使用最小BM25索引（等待上传文档）")
        return bm25
    except Exception as e:
        print(f"💥 创建最小BM25也失败: {e}")
        return None

# 添加文件上传接口
async def upload_file(request):
    """简化版文件上传"""
    try:
        # 1. 接收文件
        form = await request.form()
        file = form["file"]
        filename = file.filename

        # 2. 保存临时文件
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # 3. 处理文档
        chunks = load_split(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if not chunks:
            return JSONResponse(
                {"error": "无法提取文本内容"},
                status_code=400
            )

        print(f"✅ 处理完成: {filename} -> {len(chunks)} 个块")

        # 4. 确保目录存在
        os.makedirs(FAISS_DB_PATH, exist_ok=True)

        # 5. 检查FAISS文件是否存在
        embedding_model = rerank.get_embeddings_model('')
        faiss_index_path = os.path.join(FAISS_DB_PATH, "index.faiss")

        if os.path.exists(faiss_index_path):
            # 更新现有FAISS
            update_faiss_directly(chunks, embedding_model)
        else:
            # 创建新FAISS
            create_faiss_directly(chunks, embedding_model)

        # 6. 更新metadata（使用现有函数）
        metadata_path = os.path.join(FAISS_DB_PATH, METADATA_FILE_NAME)
        append_to_metadata_file(chunks, metadata_path)

        # 7. 重置检索器
        global loaded_faiss_db, ensemble_retriever, bm25_retriever
        loaded_faiss_db = None
        ensemble_retriever = None
        bm25_retriever = None

        bm25_retriever=rebuild_bm25_index()

        return JSONResponse({
            "success": True,
            "filename": filename,
            "chunks_count": len(chunks)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": f"上传失败: {str(e)}"},
            status_code=500
        )


def update_faiss_directly(chunks, embedding_model):
    """直接更新FAISS数据库"""
    try:
        from langchain_community.vectorstores import FAISS

        db = FAISS.load_local(
            FAISS_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        db.add_documents(chunks)
        db.save_local(FAISS_DB_PATH)
        print(f"✅ 更新FAISS数据库")
    except Exception as e:
        print(f"❌ 更新FAISS失败: {e}")
        # 失败时创建新的
        create_faiss_directly(chunks, embedding_model)


def create_faiss_directly(chunks, embedding_model):
    """直接创建FAISS数据库"""
    from langchain_community.vectorstores import FAISS

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(FAISS_DB_PATH)
    print(f"✅ 创建FAISS数据库")


async def delete_document(request: Request):
    """删除文档（处理各种文件名格式）"""
    try:
        # 获取原始参数
        raw_doc_id = request.path_params.get("doc_id", "")
        print(f"🔍 删除请求 - 原始参数: {repr(raw_doc_id)}")

        # URL解码
        import urllib.parse
        decoded_doc_id = urllib.parse.unquote(raw_doc_id)
        print(f"🔍 解码后: {repr(decoded_doc_id)}")

        # 提取纯文件名（去除路径）
        import os
        filename = os.path.basename(decoded_doc_id)
        print(f"🔍 提取文件名: {repr(filename)}")

        # 读取metadata
        metadata_path = os.path.join(FAISS_DB_PATH, METADATA_FILE_NAME)
        if not os.path.exists(metadata_path):
            return JSONResponse(
                {"error": "数据库不存在"},
                status_code=404
            )

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # 查找匹配的文档（支持多种匹配方式）
        deleted_count = 0
        new_metadata = []
        matched_sources = []

        for item in metadata:
            source = item.get('source', '')

            # 多种匹配策略
            matches = False

            # 1. 精确匹配文件名
            if source == filename:
                matches = True

            # 2. 源文件包含文件名
            elif filename in source:
                matches = True

            # 3. 文件名包含源文件
            elif source in filename:
                matches = True

            # 4. 去除扩展名后匹配
            elif os.path.splitext(source)[0] == os.path.splitext(filename)[0]:
                matches = True

            # 5. 只比较基本名（去除路径）
            elif os.path.basename(source) == filename:
                matches = True

            if matches:
                deleted_count += 1
                matched_sources.append(source)
                print(f"🗑️ 匹配删除: {source}")
                continue  # 跳过这个条目

            new_metadata.append(item)

        if deleted_count == 0:
            print(f"❌ 未找到匹配的文档")
            print(f"  请求的文件名: {filename}")
            print(f"  可用文档: {list(set(item.get('source', '') for item in metadata))}")

            return JSONResponse(
                {
                    "error": f"未找到匹配的文档: {filename}",
                    "available_files": list(set(item.get('source', '') for item in metadata))
                },
                status_code=404
            )

        # 保存更新后的metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(new_metadata, f, ensure_ascii=False, indent=2)

        print(f"✅ 删除成功: {deleted_count} 个文档块")
        print(f"✅ 匹配的文件: {matched_sources}")

        # TODO: 需要重建FAISS数据库
        await rebuild_faiss_from_metadata(new_metadata)
        global bm25_retriever
        bm25_retriever = rebuild_bm25_index()

        return JSONResponse({
            "success": True,
            "message": f"成功删除 {deleted_count} 个文档块",
            "deleted_chunks": deleted_count,
            "deleted_files": matched_sources
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": f"删除失败: {str(e)}"},
            status_code=500
        )


async def rebuild_faiss_from_metadata(metadata):
    """根据metadata重建FAISS数据库"""
    global loaded_faiss_db, ensemble_retriever
    try:
        if not metadata:
            print("📭 metadata为空，创建空数据库")
            # 创建空的FAISS
            embedding_model = rerank.get_embeddings_model(rerank.EMBEDDING_MODEL_NAME_OR_PATH)

            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document

            # 创建空数据库
            empty_db = FAISS.from_documents(
                [Document(page_content="", metadata={})],
                embedding_model
            )
            empty_db.save_local(FAISS_DB_PATH)

            # 重置全局变量
            loaded_faiss_db = empty_db
            ensemble_retriever = None

            return

        # 从metadata重建文档
        from langchain_core.documents import Document
        documents = []

        for item in metadata:
            doc = Document(
                page_content=item.get('page_content_preview', ''),
                metadata={
                    'chunk_id': item.get('chunk_id', ''),
                    'source': item.get('source', ''),
                    'page': item.get('page', 0),
                    'timestamp': item.get('timestamp', '')
                }
            )
            documents.append(doc)

        print(f"🔄 重建FAISS: {len(documents)} 个文档")

        # 重建FAISS
        embedding_model = rerank.get_embeddings_model(rerank.EMBEDDING_MODEL_NAME_OR_PATH)

        from langchain_community.vectorstores import FAISS
        db = FAISS.from_documents(documents, embedding_model)
        db.save_local(FAISS_DB_PATH)

        loaded_faiss_db = db
        ensemble_retriever = None

        print("✅ FAISS重建完成")

    except Exception as e:
        print(f"❌ 重建FAISS失败: {e}")
        import traceback
        traceback.print_exc()


async def list_documents(request:Request):
    """列出所有文档"""
    try:
        metadata_path = os.path.join(FAISS_DB_PATH, METADATA_FILE_NAME)

        if not os.path.exists(metadata_path):
            return JSONResponse({"documents": [], "total": 0})

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # 按文件分组统计
        file_stats = {}
        for item in metadata:
            source = item.get('source', '未知文件')
            if source not in file_stats:
                file_stats[source] = {
                    "filename": source,
                    "chunks_count": 0,
                    "last_updated": item.get('timestamp', ''),
                    "sample_preview": item.get('page_content_preview', '')[:100]
                }
            file_stats[source]["chunks_count"] += 1

        documents = list(file_stats.values())

        return JSONResponse({
            "success": True,
            "documents": documents,
            "total_files": len(documents),
            "total_chunks": len(metadata)
        })

    except Exception as e:
        return JSONResponse({"error": f"获取文档列表失败: {str(e)}"}, status_code=500)


async def clear_database(request:Request):
    """清空整个数据库"""
    try:
        # 1. 删除metadata文件
        metadata_path = os.path.join(FAISS_DB_PATH, METADATA_FILE_NAME)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        # 2. 删除FAISS文件
        if os.path.exists(FAISS_DB_PATH):
            for file in os.listdir(FAISS_DB_PATH):
                file_path = os.path.join(FAISS_DB_PATH, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # 3. 清空内存中的检索器
        global ensemble_retriever, loaded_faiss_db, bm25_retriever
        ensemble_retriever = None
        loaded_faiss_db = None

        # 4. 重新初始化空数据库
        await initialize_retrievers()
        bm25_retriever=rebuild_bm25_index()
        return JSONResponse({
            "success": True,
            "message": "数据库已清空"
        })

    except Exception as e:
        return JSONResponse({"error": f"清空数据库失败: {str(e)}"}, status_code=500)
# 路由配置
routes = [
    Route("/", endpoint=homepage, methods=["GET"]),
    Route("/upload", endpoint=upload_file, methods=["POST"]),
    Route("/rag_query", endpoint=rag_query, methods=["POST"]),  # 修正：使用POST方法
    Route("/api/documents", endpoint=list_documents, methods=["GET"]),
    Route("/api/documents/{doc_id}", endpoint=delete_document, methods=["DELETE"]),
    Route("/api/database", endpoint=clear_database, methods=["DELETE"]),
]

# 创建应用
app = Starlette(routes=routes)

if __name__ == "__main__":
    print("启动RAG检索服务...")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5000,
        reload=False  # 开发时启用热重载
    )