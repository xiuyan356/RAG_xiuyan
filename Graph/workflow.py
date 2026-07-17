"""
wrokflow -

Author : 王聪
Date :2026/7/8
Version :0.0.1
"""
from rag import AgentState, retrieve_node, rerank_node, generate_node
from langgraph.graph import START, END, StateGraph

# 初始化图并组装
workflow = StateGraph(AgentState)

# 1. 注册三个拆分后的高内聚原子节点
workflow.add_node('retrieve_node', retrieve_node)
workflow.add_node('rerank_node', rerank_node)
workflow.add_node('generate_node', generate_node)

# 2. 重新编排节点之间的流转路径
workflow.add_edge(START, 'retrieve_node')          # 从起点进入检索
workflow.add_edge('retrieve_node', 'rerank_node')    # 检索完成后进入精排
workflow.add_edge('rerank_node', 'generate_node')    # 精排完成后进入LLM生成
workflow.add_edge('generate_node', END)             # 生成完成后落地结束

# 编译图
app = workflow.compile()