"""
wrokflow -

Author : 王聪
Date :2026/7/8
Version :0.0.1
"""
import sys
import os

# 动态将项目根目录加入到 Python 查找路径中，确保能顺利导入根目录下的扁平单文件
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import AgentState
from retrieve import retrieve_node
from rerank import rerank_node
from generate import generate_node
from langgraph.graph import START, END, StateGraph

# 1. 初始化图并绑定我们的状态字典结构
workflow = StateGraph(AgentState)

# 2. 注册三个拆分后、并排在根目录下的高内聚原子节点
workflow.add_node('retrieve_node', retrieve_node)
workflow.add_node('rerank_node', rerank_node)
workflow.add_node('generate_node', generate_node)

# 3. 重新编排节点之间的流转路径
workflow.add_edge(START, 'retrieve_node')          # 从起点进入检索
workflow.add_edge('retrieve_node', 'rerank_node')    # 检索完成后进入精排
workflow.add_edge('rerank_node', 'generate_node')    # 精排完成后进入LLM生成
workflow.add_edge('generate_node', END)             # 生成全量答案后落地结束

# 4. 编译图，对外暴露出唯一的最终可执行 App 实例
app = workflow.compile()