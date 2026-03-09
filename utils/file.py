import streamlit as st
import requests
import io


def file_upload_modal():
    """文件上传模态框"""
    # 使用 columns 和 expander 模拟模态框
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.expander("📁 上传文件到后端", expanded=True):
            st.markdown("### 选择要上传的文件")

            # 文件上传组件
            uploaded_file = st.file_uploader(
                "选择PDF或Markdown文件",
                type=["pdf", "md", "markdown", "txt"],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                # 显示文件信息
                file_details = {
                    "文件名": uploaded_file.name,
                    "类型": uploaded_file.type,
                    "大小": f"{uploaded_file.size / 1024:.1f} KB"
                }
                st.json(file_details)

                # 上传按钮
                if st.button("🚀 上传到后端", use_container_width=True):
                    success = send_file_to_backend(uploaded_file)
                    if success:
                        st.success("文件上传成功！")
                    else:
                        st.error("文件上传失败")

            # 关闭按钮
            if st.button("❌ 关闭", use_container_width=True):
                st.session_state.show_upload_modal = False
                st.rerun()


def send_file_to_backend(uploaded_file):
    """发送文件到后端API"""
    try:
        # 准备文件数据
        files = {
            'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        # 发送到后端API
        backend_url = "http://127.0.0.1:5000/upload"  # 你的后端地址
        response = requests.post(backend_url, files=files)

        if response.status_code == 200:
            return True
        else:
            st.error(f"后端返回错误: {response.status_code}")
            return False

    except Exception as e:
        st.error(f"上传失败: {str(e)}")
        return False
