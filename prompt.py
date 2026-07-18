"""
prompt -

Author : 王聪
Date :2026/7/18
Version :0.0.1
"""
from langchain_core.prompts import ChatPromptTemplate

# 统一配置大模型的系统人设与用户上下文组装模板
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业的 FastAPI 技术文档专家。请严格根据以下参考资料回答用户的技术问题。如果资料中没有提及相关实现，请直接回答不知道，严禁虚构 API 或参数。"),
    ("user", "参考资料：\n{context}\n\n用户问题：{question}")
])