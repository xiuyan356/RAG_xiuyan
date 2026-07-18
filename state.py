"""
state -

Author : 王聪
Date :2026/7/17
Version :0.0.1
"""

from typing import TypedDict, List
from langchain_core.documents import Document

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    raw_documents: List[Document]    # 用于存放 RRF 混合检索出来的粗筛文档列表
    final_documents: List[Document]  # 用于存放 Reranker 精排挑出来的 Top 5 文档列表