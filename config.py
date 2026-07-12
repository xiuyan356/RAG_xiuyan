"""
config -

Author : 王聪
Date :2026/7/8
Version :0.0.1
"""
import os
from pathlib import Path
from dotenv import load_dotenv


# 强制加载 .env 配置文件
load_dotenv(override=True)

class Config:
    # 1. Ollama 配置
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
    OLLAMA_OPENAI_BASE_URL = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434")

    # 2. DeepSeek 配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

    raw_path = os.getenv("CROSSENCODER_PATH")
    CrossEncoder_PATH = str(Path(raw_path).resolve())


# 实例化，方便其他模块直接导入 config
config = Config()