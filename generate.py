"""
generate -

Author : 王聪
Date :2026/7/18
Version :0.0.1
"""
from state import AgentState
from llm import final_llm
from prompt import rag_prompt
from langchain_core.output_parsers import StrOutputParser


def generate_node(data: AgentState) -> dict:
    """
    生成节点：从状态中读取精排文档，组装上下文提示词，
    调用大模型生成最终技术回答，并写回状态机的 context 和 answer 字段。
    """
    # 1. 从状态中读取问题与重排后的 Top 5 文档
    user_query = data.get("question")
    final_docs = data.get("final_documents", [])

    # 2. 拼接核心上下文背景墙
    context_str = "\n\n".join([doc.page_content for doc in final_docs])

    # 3. 构建链条（直接复用从外部导入的 rag_prompt 和 final_llm）
    chain = rag_prompt | final_llm | StrOutputParser()

    # 4. 执行推理生成
    response = chain.invoke({
        'context': context_str,
        'question': user_query
    })

    # 5. 将结果回传状态机
    return {
        "context": context_str,
        "answer": response
    }