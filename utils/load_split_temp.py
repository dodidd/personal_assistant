"""
文档加载与分割模块
支持PDF和Markdown文件，自动添加时间戳元数据
"""
from datetime import datetime
import re
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 加载环境变量
load_dotenv()

# 配置常量
CHUNK_SIZE = 600  # 块大小
CHUNK_OVERLAP = 120  # 块重叠大小
MD_EXTENSIONS = {'.md', '.markdown', '.mdown', '.mkd', '.mkdn', '.mdwn', '.mdtxt', '.mdtext'}

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


class DocumentProcessor:
    """精简版文档处理器 - 只添加必要的时间元数据"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """初始化文档处理器"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def process_file(self, file_path: str) -> List[Document]:
        """处理单个文件"""
        print(f"📄 处理文件: {os.path.basename(file_path)}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 根据文件类型处理
        if file_path.lower().endswith('.md'):
            chunks = self._process_markdown(file_path)
        elif file_path.lower().endswith('.pdf'):
            chunks = self._process_pdf(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_path}")

        # 增强元数据
        chunks = self._enhance_chunks_metadata_simple(chunks, file_path)

        print(f"  生成 {len(chunks)} 个文本块")
        return chunks

    def _process_markdown(self, md_path: str) -> List[Document]:
        """处理Markdown文件"""
        try:
            # 尝试使用UnstructuredMarkdownLoader
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            loader = UnstructuredMarkdownLoader(md_path)
            documents = loader.load()
        except Exception as e:
            print(f"  MarkdownLoader失败，使用备用方法: {e}")
            # 备用方法
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [Document(page_content=content, metadata={"source": md_path})]

        # 分割文档
        return self.text_splitter.split_documents(documents)

    def _process_pdf(self, pdf_path: str) -> List[Document]:
        """处理PDF文件"""
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 分割文档
        return self.text_splitter.split_documents(documents)

    def _processs_txt(self,txt_path: str) -> List[Document]:
        """<UNK>TXT<UNK>"""
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(txt_path)
        documents = loader.load()

        return self.text_splitter.split_documents(documents)

    def _enhance_chunks_metadata_simple(self, chunks: List[Document], file_path: str) -> List[Document]:
        """精简版元数据增强 - 只添加必要字段"""
        filename = os.path.basename(file_path)
        current_time = datetime.now()

        enhanced_chunks = []

        for i, chunk in enumerate(chunks):
            # 复制原有元数据
            metadata = chunk.metadata.copy()

            # ========== 核心字段 ==========
            # 1. 标识信息
            metadata.update({
                'chunk_id': f"{Path(filename).stem}_{i:04d}",  # 使用文件名主干
                'chunk_index': i,
                'total_chunks': len(chunks),
            })

            # 2. 文件信息（简化路径）
            metadata.update({
                'source': filename,  # 只存文件名
                'filename': filename,
                'file_type': Path(file_path).suffix.lower().lstrip('.'),
            })

            # 3. 核心时间字段（最重要的！）
            metadata.update({
                'added_time': current_time.isoformat(),  # ISO格式，用于显示
                'added_timestamp': int(current_time.timestamp()),  # 时间戳，用于比较
                'added_date': current_time.strftime('%Y-%m-%d'),  # 日期，用于过滤
            })

            # 4. 时间分类（一个字段代替多个）
            metadata['time_category'] = self._get_time_category(current_time)

            # 5. 内容特征
            metadata.update({
                'content_length': len(chunk.page_content),
                'has_time_in_content': self._has_time_in_content(chunk.page_content),
            })

            # 6. 预览（保持一个就行）
            preview_length = min(200, len(chunk.page_content))
            metadata['preview'] = chunk.page_content[:preview_length]

            enhanced_chunks.append(Document(
                page_content=chunk.page_content,
                metadata=metadata
            ))

        return enhanced_chunks

    def _get_time_category(self, timestamp: datetime) -> str:
        """获取时间分类"""
        now = datetime.now()
        delta = now - timestamp

        if delta.days == 0:
            return "today"
        elif delta.days == 1:
            return "yesterday"
        elif delta.days <= 7:
            return "this_week"
        elif delta.days <= 30:
            return "this_month"
        elif delta.days <= 365:
            return "this_year"
        else:
            return "older"

    def _has_time_in_content(self, content: str) -> bool:
        """检查内容是否包含时间信息"""
        # 简单模式匹配
        patterns = [
            r'\d{4}年',  # 2024年
            r'\d{4}-\d{2}',  # 2024-01
            r'\d{1,2}月\d{1,2}日',  # 1月1日
            r'\d{4}/\d{1,2}/\d{1,2}',  # 2024/1/1
        ]

        for pattern in patterns:
            if re.search(pattern, content[:1000]):  # 只检查前1000字符
                return True

        return False

    def process_directory(self, directory_path: str) -> List[Document]:
        """处理目录"""
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"目录不存在: {directory_path}")

        all_chunks = []
        processed_files = 0

        print(f"📁 扫描目录: {directory_path}")

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)

                # 只处理支持的文件
                if file.lower().endswith(('.pdf', '.md', '.txt')):
                    try:
                        chunks = self.process_file(file_path)
                        all_chunks.extend(chunks)
                        processed_files += 1
                    except Exception as e:
                        print(f"  ✗ 处理失败: {file} - {e}")

        print(f"✅ 完成: {processed_files} 个文件, {len(all_chunks)} 个文本块")
        return all_chunks


# 保持向后兼容
def load_split(file_path: str) -> List[Document]:
    """向后兼容的加载分割函数"""
    processor = DocumentProcessor()

    if os.path.isfile(file_path):
        return processor.process_file(file_path)
    elif os.path.isdir(file_path):
        return processor.process_directory(file_path)
    else:
        raise ValueError(f"路径不存在: {file_path}")

if __name__ == "__main__":
    # 示例1: 处理单个文件
    print("=" * 50)
    print("示例1: 处理单个Markdown文件")
    print("=" * 50)

    # 假设有一个测试文件
    test_md = "example_2024-12-24_notes.md"  # 修改为实际文件路径
    if os.path.exists(test_md):
        processor = DocumentProcessor()
        chunks = processor.process_file(test_md)

        # 显示第一个块的元数据
        if chunks:
            print(f"\n第一个块的元数据:")
            for key, value in chunks[0].metadata.items():
                print(f"  {key}: {value}")

    # 示例2: 处理目录
    print("\n" + "=" * 50)
    print("示例2: 处理目录")
    print("=" * 50)

    test_dir = "../金融数据集-报表"
    if os.path.exists(test_dir) and os.path.isdir(test_dir):
        chunks = load_split(test_dir)
        print(chunks)
        print(f"总文档块数: {len(chunks)}")