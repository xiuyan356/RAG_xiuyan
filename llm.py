"""
llm -

Author : 王聪
Date :2026/7/18
Version :0.0.1
"""
from langchain_openai import ChatOpenAI
from config import config

# 1. 初始化 Ollama 本地轻量级模型作为首选驱动
ollama_llm = ChatOpenAI(
    model="qwen2.5:3b",
    temperature=0,
    base_url=config.OLLAMA_OPENAI_BASE_URL,
    api_key=config.OLLAMA_API_KEY,
    timeout=90.0,
    streaming=True
)

# 2. 初始化 DeepSeek 云端大模型作为强力后备容灾
deepseek_llm = ChatOpenAI(
    base_url=config.DEEPSEEK_BASE_URL,
    api_key=config.DEEPSEEK_API_KEY,
    model='deepseek-v4-flash',
    max_tokens=1024,
    temperature=0.0,
    streaming=True
)

# 3. 绑定双保险降级策略，对外暴露出唯一的最终 LLM 实例
final_llm = (ollama_llm.with_fallbacks([deepseek_llm]))