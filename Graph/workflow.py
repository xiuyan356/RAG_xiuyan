"""
wrokflow -

Author : 王聪
Date :2026/7/8
Version :0.0.1
"""
from rag import AgentState
from rag import RAG_answer_node
from langgraph.graph import START, END, StateGraph

# 初始化图并组装
workflow = StateGraph(AgentState)
workflow.add_node('rag_node', RAG_answer_node)

# 设置起点和终点
workflow.add_edge(START, 'rag_node')
workflow.add_edge('rag_node', END)

# 编译图
app = workflow.compile()
