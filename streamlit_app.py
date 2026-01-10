# streamlit_app.py
import streamlit as st
import requests
import json
import os

# --- 页面配置 ---
st.set_page_config(
    page_title="智能数据问答助手???",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 应用常量 ---
BACKEND_BASE_URL = "http://127.0.0.1:5000"
RAG_QUERY_URL = f"{BACKEND_BASE_URL}/rag_query"
UPLOAD_URL = f"{BACKEND_BASE_URL}/upload"
DOCUMENTS_API = f"{BACKEND_BASE_URL}/api/documents"
DATABASE_API = f"{BACKEND_BASE_URL}/api/database"
FAISS_DB_PATH = r"./faiss_index_bge_m3"


# --- 辅助函数 ---
def upload_file_to_backend(file):
    """上传文件到后端"""
    try:
        files = {'file': (file.name, file.getvalue(), file.type)}
        response = requests.post(UPLOAD_URL, files=files, timeout=60)
        return response.status_code == 200
    except Exception as e:
        st.error(f"上传失败: {e}")
        return False


def get_document_list():
    """获取文档列表"""
    try:
        response = requests.get(DOCUMENTS_API, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def delete_document(filename):
    """安全删除文档（处理路径和编码问题）"""
    try:
        import os
        import urllib.parse

        # 1. 只取文件名（不要完整路径）
        basename = os.path.basename(filename)
        print(f"原始文件名: {filename}")
        print(f"提取的文件名: {basename}")

        # 2. URL编码
        encoded_name = urllib.parse.quote(basename, safe='')

        # 3. 发送请求
        url = f"{BACKEND_BASE_URL}/api/documents/{encoded_name}"
        print(f"删除请求URL: {url}")

        response = requests.delete(url, timeout=10)

        if response.status_code == 200:
            result = response.json()
            return result.get("success", False)
        else:
            print(f"删除失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False

    except Exception as e:
        print(f"删除异常: {e}")
        return False



def clear_database():
    """清空数据库"""
    try:
        response = requests.delete(DATABASE_API, timeout=10)
        return response.status_code == 200
    except:
        return False


# --- 数据库管理页面 ---
def show_database_management():
    """显示数据库管理页面"""
    st.title("📚 数据库管理")

    # 刷新文档列表
    if st.button("🔄 刷新文档列表"):
        with st.spinner("加载中..."):
            result = get_document_list()
            if result and result.get("success"):
                st.session_state.documents = result["documents"]
                st.session_state.total_chunks = result["total_chunks"]
                st.success(f"已加载 {len(result['documents'])} 个文件")
            else:
                st.error("加载失败，请检查后端服务")

    # 显示文档列表
    if "documents" in st.session_state and st.session_state.documents:
        st.subheader(f"📁 文档列表 (共 {len(st.session_state.documents)} 个文件)")

        for doc in st.session_state.documents:
            with st.container():
                col1, col2, col3 = st.columns([4, 2, 1])

                with col1:
                    st.markdown(f"**{doc['filename']}**")
                    st.caption(f"📝 {doc['chunks_count']} 个文本块")
                    if doc.get('last_updated'):
                        st.caption(f"📅 {doc['last_updated']}")

                with col2:
                    # 预览按钮
                    if st.button("👁️ 预览", key=f"preview_{doc['filename']}"):
                        st.info(doc.get('sample_preview', '无预览内容'))

                with col3:
                    # 删除按钮 - 使用会话状态记录确认状态
                    delete_key = f"delete_confirm_{doc['filename']}"
                    if delete_key not in st.session_state:
                        st.session_state[delete_key] = False

                    if st.session_state[delete_key]:
                        # 确认删除状态
                        st.warning(f"确认删除 {doc['filename']}?")
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("✅ 确认", key=f"yes_{doc['filename']}"):
                                if delete_document(doc['filename']):
                                    st.success(f"已删除 {doc['filename']}")
                                    del st.session_state[delete_key]
                                    st.rerun()
                                else:
                                    st.error("删除失败")
                                    st.session_state[delete_key] = False
                        with col_no:
                            if st.button("❌ 取消", key=f"no_{doc['filename']}"):
                                st.session_state[delete_key] = False
                                st.rerun()
                    else:
                        if st.button("🗑️", key=f"delete_btn_{doc['filename']}"):
                            st.session_state[delete_key] = True
                            st.rerun()

            st.divider()

    # 清空数据库
    st.divider()
    st.subheader("⚠️ 危险操作")

    if 'confirm_clear' not in st.session_state:
        st.session_state.confirm_clear = False

    if not st.session_state.confirm_clear:
        if st.button("🧹 清空整个数据库", type="secondary"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("确认清空整个数据库吗？此操作不可恢复！")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 确认清空", type="primary"):
                if clear_database():
                    st.success("数据库已清空")
                    st.session_state.clear()  # 清空会话状态
                    st.rerun()
                else:
                    st.error("清空失败")
                    st.session_state.confirm_clear = False
        with col2:
            if st.button("❌ 取消"):
                st.session_state.confirm_clear = False
                st.rerun()

    # 数据库统计
    st.divider()
    st.subheader("📊 数据库统计")

    col1, col2, col3 = st.columns(3)
    with col1:
        count = len(st.session_state.get('documents', []))
        st.metric("文件数量", count)
    with col2:
        st.metric("文本块总数", st.session_state.get('total_chunks', 0))
    with col3:
        if os.path.exists(FAISS_DB_PATH):
            size = sum(
                os.path.getsize(os.path.join(FAISS_DB_PATH, f))
                for f in os.listdir(FAISS_DB_PATH)
                if os.path.isfile(os.path.join(FAISS_DB_PATH, f))
            )
            st.metric("数据库大小", f"{size / 1024 / 1024:.2f} MB")
        else:
            st.metric("数据库大小", "0 MB")


# --- 主页面 ---
st.title("🤖 智能数据问答助手???")
st.markdown("欢迎使用基于RAG（检索增强生成）技术的智能问答系统。您可以就数据相关问题进行提问。")
st.divider()

# --- 侧边栏 ---
with st.sidebar:
    st.header("💡 系统信息")
    st.info(
        "本项目结合了 **检索(Retrieval)**、**重排(Rerank)** 和 **生成(Generation)** "
        "技术，为您提供更精准的回答。"
    )

    st.subheader("技术栈:")
    st.markdown("""
    - **前端:** Streamlit
    - **后端:** FastAPI
    - **向量检索:** FAISS + BGE-Embeddings
    - **重排序:** BGE-Reranker (via SiliconFlow)
    - **大模型:** DeepSeek-V3.1(via SiliconFlow)
    """)

    # 上传文件
    st.divider()
    st.subheader("📤 上传文件")

    uploaded_file = st.file_uploader("选择PDF/Markdown文件", type=["pdf", "md", "txt"])
    if uploaded_file:
        if st.button("🚀 上传到数据库", use_container_width=True):
            with st.spinner("上传中..."):
                if upload_file_to_backend(uploaded_file):
                    st.success("文件上传成功！")
                    st.balloons()
                else:
                    st.error("文件上传失败")

    # 页面切换
    st.divider()
    page_options = ["💬 智能问答", "📚 数据库管理"]
    selected_page = st.radio("页面选择", page_options, index=0)

# --- 页面路由 ---
if selected_page == "📚 数据库管理":
    show_database_management()
else:
    # --- 聊天界面 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史聊天记录
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("查看引用来源"):
                    for i, source in enumerate(message["sources"]):
                        st.info(f"**来源 {i + 1}**")
                        st.text(source.get('content', ''))

    # --- 处理用户输入 ---
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用后端API
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("正在思考中... 🤔")

            try:
                payload = {"question": prompt}

                # 修复：使用 json 参数而不是 data
                response = requests.post(RAG_QUERY_URL, json=payload, timeout=120)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        # 提取数据
                        answer = result.get("answer", "抱歉，未能生成回答。")
                        summary = result.get("summary", "")
                        #timeline = result.get("timeline", {})
                        sources = result.get("content", [])  # 修复：从result直接获取

                        # 1. 显示答案
                        st.markdown(answer)

                        # 2. 摘要
                        if summary and summary not in ["<UNK>", ""]:
                            with st.expander("📋 内容摘要", expanded=False):
                                st.info(summary)

                        ##3. 时间线
                        # if timeline and isinstance(timeline, dict) and len(timeline) > 0:
                        #     dates = list(timeline.keys())[:3]
                        #     if dates:
                        #         date_chips = " | ".join([f"`{d}`" for d in dates])
                        #         st.caption(f"📅 相关时间点: {date_chips}")

                        # 4. 引用来源
                        if sources:
                            with st.expander(f"📚 引用来源 ({len(sources)}个)"):
                                for i, source in enumerate(sources):
                                    metadata = source.get('metadata', {})
                                    st.info(
                                        f"**来源 {i + 1}** | "
                                        f"文档: {metadata.get('source', 'N/A')} | "
                                        f"页码: {metadata.get('page', 'N/A')}"
                                    )
                                    if 'timestamp' in metadata:
                                        st.caption(f"时间: {metadata['timestamp']}")
                                    st.text(source.get('content', '')[:300] + "...")
                                    st.divider()

                        # 保存到历史
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })

                    else:
                        error_msg = result.get("error", "未知错误")
                        message_placeholder.error(f"处理失败: {error_msg}")
                else:
                    message_placeholder.error(f"请求失败: {response.status_code}")

            except requests.exceptions.Timeout:
                message_placeholder.error("请求超时，请稍后重试")
            except Exception as e:
                message_placeholder.error(f"请求异常: {e}")

# streamlit run .\streamlit_app.py