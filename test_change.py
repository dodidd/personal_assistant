# test_change.py - 放在同一目录
import streamlit as st

st.title("测试页面 - 修改时间: 2024-03-21 15:30")
st.write("如果看到这个时间，说明代码已更新")

# 检查文件修改时间
import os
import time
file_time = time.ctime(os.path.getmtime(__file__))
st.write(f"文件最后修改: {file_time}")