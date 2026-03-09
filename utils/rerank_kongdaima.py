import os
from os.path import isfile, isdir

import json
import requests

from langchain.retrievers import EnsembleRetriever         # EnsembleRetriever 仍在 langchain.retrievers
from langchain_community.retrievers import BM25Retriever # BM25Retriever 已移至 langchain_community.retrievers
from langchain_core.documents import Document # 确保这行已存在或添加
# 从 dotenv 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# --- 配置 ---

PDF_PATH = r"D:/class/pythonproject/RAG/金融数据集-报表"
EMBEDDING_MODEL_NAME_OR_PATH = ''
CHUNK_SIZE = 800 # 块大小
CHUNK_OVERLAP = 200 # 块重叠大小（滑块）
FAISS_DB_PATH = r"./faiss_index_bge_m3"
METADATA_FILE_NAME = "documents_metadata.json"

# --- SiliconFlow Reranker 配置 ---
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

def find_pdf_files(directory):
    pdf_files = []

    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误：目录 '{directory}' 不存在")
        return pdf_files

    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)

    return pdf_files

def load_and_split_pdf(pdf_path: str) -> list[Document]:
    """加载 PDF 文档并进行文本切分。"""
    chunks=[]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    if isfile(pdf_path):
        print(f"正在加载 PDF 文件: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"原始文档页数: {len(documents)}")
        chunks = text_splitter.split_documents(documents)
    elif isdir(pdf_path):
        all_documents = []
        pdf_paths = find_pdf_files(pdf_path)
        for pdf_path in pdf_paths:
            try:
                # 检查文件是否存在
                if not os.path.exists(pdf_path):
                    print(f"警告: 文件不存在 {pdf_path}")
                    continue

                # 检查是否是PDF文件
                if not pdf_path.lower().endswith('.pdf'):
                    print(f"警告: 不是PDF文件 {pdf_path}")
                    continue

                print(f"正在加载: {pdf_path}")

                # 加载单个PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                # 为每个文档添加源文件信息
                for doc in documents:
                    doc.metadata['source_file'] = pdf_path

                all_documents.extend(documents)
                print(f"  已加载 {len(documents)} 个页面")
                chunks = text_splitter.split_documents(all_documents)

            except Exception as e:
                print(f"加载文件 {pdf_path} 时出错: {e}")


        print(f"文档切分完成，生成 {len(chunks)} 个文本块。")
    return chunks

def get_embeddings_model(model_name_or_path: str):
    """获取嵌入模型。"""
    # print(f"正在加载嵌入模型: {model_name_or_path} (device: {device})")
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=model_name_or_path,
    #     model_kwargs={'device': device}
    # )

    # from langchain_openai import OpenAIEmbeddings
    # from dotenv import load_dotenv
    # # 加载 .env 文件中的环境变量
    # load_dotenv()
    # api_key = os.getenv("SILICONFLOW_API_KEY")
    # # 获取 DeepSeek API 密钥和基础 URL
    # embedding_model = OpenAIEmbeddings(
    #     model="BAAI/bge-large-zh-v1.5",  # 指定 BGE 模型 ID
    #     api_key=api_key,  # 您的 SiliconFlow API Key
    #     base_url="https://api.siliconflow.cn/v1",  # SiliconFlow API 端点
    # )
    from langchain_community.embeddings import OllamaEmbeddings

    # 初始化 Ollama 嵌入模型
    embedding_model= OllamaEmbeddings(
        model="bge-m3",  # 您在 Ollama 中部署的模型名称
        base_url="http://localhost:11434",  # Ollama 的默认地址
    )

    print("嵌入模型加载完成。")
    return embedding_model

def create_and_save_faiss_db(chunks: list[Document], embeddings_model, db_path: str) -> FAISS:
    """创建 FAISS 向量数据库并保存到本地。"""
    print("正在创建 FAISS 向量数据库...")
    faiss_db = FAISS.from_documents(chunks, embeddings_model)
    print(f"FAISS 向量数据库创建完成。正在保存到: {db_path}")
    faiss_db.save_local(db_path)
    print("FAISS 向量数据库保存成功。")
    return faiss_db

def create_and_save_metadata(chunks: list[Document], output_dir: str, metadata_file_name: str):
    """创建并保存文档块的元数据。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata_list = []
    file_count = len([f for f in os.listdir('./'+output_dir) if os.path.isfile(f)])
    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id": f"chunk_{i+file_count}",
            "page_content_preview": chunk.page_content,
            "source": chunk.metadata.get('source', '未知文件'),
            "page": chunk.metadata.get('page', '未知页码'),
            "start_index": chunk.metadata.get('start_index', '未知索引')
        }
        metadata_list.append(metadata)

    metadata_file_path = os.path.join(output_dir, metadata_file_name)
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)
    print(f"元数据已保存到：{metadata_file_path}")

# --- 新增：SiliconFlow Reranker 函数 ---
def rerank_documents_siliconflow(
    query: str,
    documents: list[Document], # 接收 LangChain Document 对象列表
    model: str = RERANKER_MODEL,
    api_key: str = SILICONFLOW_API_KEY,
    top_n: int = 5 # 重排后返回的顶部文档数量
) -> list[Document]:
    """
    使用 SiliconFlow Rerank API 对文档进行重排。

    Args:
        query: 用户查询字符串。
        documents: 待重排的 LangChain Document 对象列表。
        model: 重排模型名称。
        api_key: SiliconFlow API 密钥。
        top_n: 重排后返回的顶部文档数量。

    Returns:
        重排后的 LangChain Document 对象列表，按相关性分数降序排列。
    """
    if not api_key:
        print("错误：未设置 SILICONFLOW_API_KEY 环境变量，无法调用 Reranker API。")
        return []

    doc_contents = [doc.page_content for doc in documents] # 提取文档内容发送给API

    payload = {
        "model": model,
        "query": query,
        "documents": doc_contents
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print(f"\n正在调用 SiliconFlow Reranker ({model})...")
    try:
        response = requests.post(SILICONFLOW_RERANK_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status() # 对 4xx 或 5xx HTTP 状态码抛出异常
        result = response.json()

        if "results" not in result:
            print(f"Reranker API 响应格式不正确: {result}")
            return []

        # 将 rerank 结果与原始 Document 对象关联并排序
        # 创建一个带有分数和原始文档引用的列表
        reranked_items = []
        # SiliconFlow API 的 results 数组顺序与传入的 documents 数组顺序一致
        for i, res in enumerate(result["results"]):
            # 找到对应的原始 Document 对象。这里假设 res['document'] 直接是原始内容
            # 更严谨的话，可以在传入 documents 时给 Document 对象添加唯一 ID，然后通过 ID 匹配
            # 为了简化，我们假设 res['document'] 与 doc_contents[i] 相同
            reranked_items.append({
                "document": documents[i], # 原始 LangChain Document 对象
                "score": res['relevance_score']
            })

        # 按相关性分数降序排序
        reranked_items.sort(key=lambda x: x['score'], reverse=True)

        # 提取 top_n 文档
        top_reranked_docs = [item['document'] for item in reranked_items[:top_n]]

        print(f"Rerank 成功，返回 Top {len(top_reranked_docs)} 文档。")
        for i, doc_item in enumerate(reranked_items[:top_n]):
            print(f"  - Top {i+1} (Score: {doc_item['score']:.4f}): {doc_item['document'].page_content[:100]}...")
        return top_reranked_docs

    except requests.exceptions.RequestException as e:
        print(f"调用 SiliconFlow Reranker API 失败: 网络或请求错误 - {e}")
        return []
    except json.JSONDecodeError:
        print(f"Reranker API 响应不是有效的 JSON: {response.text}")
        return []
    except Exception as e:
        print(f"Reranker 发生未知错误: {e}")
        return []


def add_to_faiss(new_chunks, embedding_model, faiss_db_path):
    """向现有FAISS数据库添加新文档"""
    try:
        # 加载现有索引
        existing_db = FAISS.load_local(
            faiss_db_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        # 准备文本列表
        texts = [chunk.page_content for chunk in new_chunks]
        metadatas = [chunk.metadata for chunk in new_chunks]

        # 生成新文档的嵌入
        embeddings = embedding_model.embed_documents(texts)

        # 创建 text_embeddings 列表，格式为 [(text, embedding), ...]
        text_embeddings = list(zip(texts, embeddings))

        # 正确调用 add_embeddings
        existing_db.add_embeddings(text_embeddings, metadatas=metadatas)

        # 保存更新后的索引
        existing_db.save_local(faiss_db_path)
        print(f"成功添加 {len(new_chunks)} 个新文档到FAISS数据库")

    except Exception as e:
        print(f"添加文档到FAISS失败: {e}")
        raise


# --- 主执行 ---
if __name__ == "__main__":
    try:
        # 步骤 1: 加载并切分 PDF
        # document_chunks 在这里生成，后续用于 BM25 和 FAISS
        document_chunks = load_and_split_pdf(PDF_PATH)

        # 步骤 2: 初始化嵌入模型
        embeddings_model = get_embeddings_model(EMBEDDING_MODEL_NAME_OR_PATH)

        # 步骤 3: 创建并保存 FAISS 向量数据库
        faiss_db = create_and_save_faiss_db(document_chunks, embeddings_model, FAISS_DB_PATH)

        # 步骤 4: 创建并保存元数据文件
        create_and_save_metadata(document_chunks, FAISS_DB_PATH, METADATA_FILE_NAME)

        print("\n进程成功完成！")
        print(f"FAISS 索引和元数据存储在：{os.path.abspath(FAISS_DB_PATH)}")


        print("\n--- 混合检索功能示例 ---")

        # 1. 载入之前保存的 FAISS 数据库
        # 确保加载嵌入模型
        loaded_embeddings_model = get_embeddings_model(EMBEDDING_MODEL_NAME_OR_PATH)
        loaded_faiss_db = FAISS.load_local(FAISS_DB_PATH, loaded_embeddings_model, allow_dangerous_deserialization=True)
        print(f"已从 {FAISS_DB_PATH} 加载 FAISS 数据库。")

        # 2. 初始化 BM25 关键词检索器
        # 注意：BM25Retriever.from_documents 需要原始的 Document 对象列表
        bm25_retriever = BM25Retriever.from_documents(document_chunks)
        bm25_retriever.k = 3 # 设置关键词检索召回数量

        # 3. 初始化 FAISS 向量检索器
        vector_retriever = loaded_faiss_db.as_retriever(search_kwargs={"k": 3}) # 设置向量检索召回数量

        # 4. 组合成 EnsembleRetriever 混合检索器
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5] # 可以调整两种检索器的权重
        )
        print("\nBM25 和 FAISS 检索器已组合成 EnsembleRetriever。")


        # 5. 定义一个用户查询
        user_query = "ACME研发有限公司" # 示例查询

        # 6. 使用混合检索器进行初步检索
        initial_retrieved_docs = ensemble_retriever.invoke(user_query)

        print(f"\n--- 混合检索初步召回 {len(initial_retrieved_docs)} 篇文档 ---")
        for i, doc in enumerate(initial_retrieved_docs):
            print(f"  - 初始 Top {i+1}: {doc.page_content[:100]}...")


        # 7. 调用 Reranker 对初步检索到的文档进行重排
        # 假设我们只想获取重排后的前 3 个最相关文档
        top_reranked_docs = rerank_documents_siliconflow(user_query, initial_retrieved_docs, top_n=3)

        # 8. 现在你可以将 top_reranked_docs 传递给 LLM 进行答案生成
        # (这部分通常在完整的 RAG 链中完成，这里只是演示 rerank 的输出)
        if top_reranked_docs:
            print("\n--- 重排后的文档（用于最终答案生成）---")
            for i, doc in enumerate(top_reranked_docs):
                print(f"  - 最终 Top {i+1}: {doc.page_content[:200]}...")
        else:
            print("\n未能成功重排文档，请检查 API 密钥和网络连接。")

    except Exception as e:
        print(f"脚本执行出错: {e}")
