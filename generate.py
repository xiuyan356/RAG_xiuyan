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

    # 3. 构建链条（移除 StrOutputParser 以保留响应元数据）
    chain = rag_prompt | final_llm

    # 4. 执行推理生成（此时 response 是一个完整的带有元数据的对象）
    response = chain.invoke({
        'context': context_str,
        'question': user_query
    })

    # 5. 【核心探针逻辑】在 return 之前抓取模型名称并打印
    meta = getattr(response, "response_metadata", {})
    model_used = meta.get("model_name", meta.get("model", getattr(response, "model", "qwen2.5:3b")))
    print(f"\n [链路探针] 当前响应生成模型: 【{model_used}】")

    # 6. 将结果回传状态机（使用 response.content 提取纯文本答案）
    return {
        "context": context_str,
        "answer": response.content
    }