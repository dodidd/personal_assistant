# scripts/start_dev.py
import subprocess
import sys
import time
from threading import Thread


def run_backend():
    """启动 Uvicorn 服务"""
    try:
        result = subprocess.run([
            sys.executable,  # 使用当前 Python 解释器
            "-m", "uvicorn",  # 以模块方式运行 uvicorn
            "async:app",  # 指定应用对象（假设文件名为 main.py）
            "--host", "127.0.0.1",
            "--port", "5000"
        ], check=True, capture_output=True, text=True)

        print("Uvicorn 服务启动成功")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Uvicorn 服务启动失败: {e}")
        print(f"错误输出: {e.stderr}")
    except FileNotFoundError:
        print("未找到 uvicorn，请先安装: pip install uvicorn")

def run_frontend():
    subprocess.run([sys.executable, "-m", "streamlit", "run", ".\streamlit_app.py"])


if __name__ == "__main__":
    backend_thread = Thread(target=run_backend, daemon=True)
    frontend_thread = Thread(target=run_frontend, daemon=True)

    backend_thread.start()
    time.sleep(2)
    frontend_thread.start()

    backend_thread.join()
    frontend_thread.join()