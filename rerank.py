"""
rerank -

Author : 王聪
Date :2026/7/18
Version :0.0.1
"""
from config import config
from state import AgentState
from sentence_transformers import CrossEncoder

# 实例化重排模型对象
reranker = CrossEncoder(config.CrossEncoder_PATH)


def rerank_documents(query, candidate_docs, top_n=5):
    """
    对 RRF 候选文档进行 Reranker 精排，返回 Top N
    """
    if not candidate_docs:
        return []

    # 安全去重
    unique_docs = list({doc.page_content: doc for doc in candidate_docs}.values())

    pairs = [[query, doc.page_content] for doc in unique_docs]
    scores = reranker.predict(pairs)

    doc_score_pairs = list(zip(unique_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    final_docs = [doc for doc, _ in doc_score_pairs[:top_n]]

    print(f"\n========== Reranker 精排 (从 {len(unique_docs)} 个候选精选 Top {top_n}) ==========")
    for i, doc in enumerate(final_docs):
        score = doc_score_pairs[i][1]
        print(f"Rank {i + 1} [Score: {score:.4f}] -> {doc.metadata.get('source', '未知')}")
        print(f"内容摘要: {doc.page_content[:120].replace(chr(10), ' ')}...")
        print("-" * 40)

    return final_docs


def rerank_node(data: AgentState) -> dict:
    """
    Reranker 节点：从状态中获取问题和粗筛文档，调用精排算法，
    将精排后的 Top 5 文档写入状态机的 final_documents 字段。
    """
    user_query = data.get("question")
    candidate_docs = data.get("raw_documents", [])

    final_docs = rerank_documents(query=user_query, candidate_docs=candidate_docs, top_n=5)

    return {"final_documents": final_docs}