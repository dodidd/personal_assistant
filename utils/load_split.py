from datetime import datetime
import re
import os
from os.path import isfile, isdir
from langchain_core.documents import Document # 确保这行已存在或添加
# 从 dotenv 加载环境变量
from dotenv import load_dotenv
from nltk.corpus.reader import documents

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


CHUNK_SIZE = 600 # 块大小
CHUNK_OVERLAP = 120 # 块重叠大小（滑块）


def load_split(file_path):

    if is_markdown_file(file_path):
        chunks = load_and_split_md(file_path)
        return chunks
    else:
        chunks = load_and_split_pdf(file_path)
        return chunks



def is_markdown_file(file_path):
    """通过文件扩展名判断是否为Markdown文件"""
    # 常见的Markdown文件扩展名
    md_extensions = {'.md', '.markdown', '.mdown', '.mkd', '.mkdn', '.mdwn', '.mdtxt', '.mdtext'}

    # 获取文件扩展名并转换为小写
    file_ext = os.path.splitext(file_path)[1].lower()

    return file_ext in md_extensions


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


from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def load_and_split_md(md_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """使用LangChain的MarkdownLoader加载并分割文件"""
    try:
        print(f"使用MarkdownLoader处理文件: {md_path}")

        # 检查文件是否存在
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"文件不存在: {md_path}")

        # 使用UnstructuredMarkdownLoader加载文档
        loader = UnstructuredMarkdownLoader(md_path)
        documents = loader.load()

        print(f"成功加载文档，原始文档数: {len(documents)}")

        # 检查加载的文档
        for i, doc in enumerate(documents):
            print(f"文档 {i} 类型: {type(doc)}")
            print(f"文档 {i} 内容长度: {len(doc.page_content)}")
            print(f"文档 {i} 元数据: {doc.metadata}")

        # 初始化文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # 分割文档
        chunks = text_splitter.split_documents(documents)
        # 关键：为每个块添加时间戳
        for i, chunk in enumerate(chunks):
            # 方法1: 从文件名提取日期（如2024-03-20_report.pdf）
            filename = os.path.basename(md_path)
            timestamp = extract_timestamp_from_filename(filename)

            # 方法2: 从内容中提取（如果第一页有日期）
            if not timestamp:
                timestamp = extract_timestamp_from_content(chunk.page_content)

            # 方法3: 使用文件修改时间
            if not timestamp:
                mtime = os.path.getmtime(md_path)
                timestamp = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')

            # 添加到metadata
            chunk.metadata['timestamp'] = timestamp
            chunk.metadata['chunk_id'] = f"chunk_{i}"
        print(f"成功分割Markdown文件，生成 {len(chunks)} 个块")
        return chunks

    except Exception as e:
        print(f"使用MarkdownLoader处理失败: {e}")
        # 回退到方案2的方法
        return load_and_split_md_fallback(md_path, chunk_size, chunk_overlap)


def load_and_split_md_fallback(md_path, chunk_size=1000, chunk_overlap=200):
    """备用方案：手动处理Markdown文件"""
    from langchain.schema import Document
    try:
        print(f"使用备用方案处理文件: {md_path}")

        with open(md_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 创建Document对象
        documents = [Document(
            page_content=content,
            metadata={"source": md_path}
        )]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents)
        print(f"备用方案成功，生成 {len(chunks)} 个块")
        return chunks

    except Exception as e:
        print(f"备用方案也失败: {e}")
        raise

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

    for i, chunk in enumerate(chunks):
        # 方法1: 从文件名提取日期（如2024-03-20_report.pdf）
        filename = os.path.basename(pdf_path)
        timestamp = extract_timestamp_from_filename(filename)

        # 方法2: 从内容中提取（如果第一页有日期）
        if not timestamp:
            timestamp = extract_timestamp_from_content(chunk.page_content)

        # 方法3: 使用文件修改时间
        if not timestamp:
            mtime = os.path.getmtime(pdf_path)
            timestamp = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')

        # 添加到metadata
        chunk.metadata['timestamp'] = timestamp
        chunk.metadata['chunk_id'] = f"chunk_{i}"
    print(f"文档切分完成，生成 {len(chunks)} 个文本块。")
    return chunks


def extract_timestamp_from_filename(filename):
    """从文件名提取日期"""
    import re

    patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',  # 2024-03-20
        r'(\d{8})',  # 20240320
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            if len(match.groups()) == 3:
                year, month, day = match.groups()
                return f"{year}-{month}-{day}"
            elif len(match.groups()) == 1:
                date_str = match.group(1)
                if len(date_str) == 8:
                    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return None


def extract_timestamp_from_content(content):
    """从内容中提取日期"""
    import re

    # 在开头100个字符中找日期
    first_100 = content[:100]
    patterns = [
        r'日期[:：]\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        r'Date[:：]\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        r'(\d{4}年\d{1,2}月\d{1,2}日)',
    ]

    for pattern in patterns:
        match = re.search(pattern, first_100)
        if match:
            return match.group(1)

    return None