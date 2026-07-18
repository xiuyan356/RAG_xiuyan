"""
retrieve -

Author : 王聪
Date :2026/7/18
Version :0.0.1
"""
from collections import defaultdict
from state import AgentState
from database import bm25_retriever, vec_retriever

def rrf_retrieve(query, k=20, rrf_k=60, weight_bm25=0.2, weight_vector=0.8):
    """
    RRF 融合：BM25 Top 20 + Vector Top 20 → 融合排序 → 返回 Top k（默认 20）
    """
    bm25_docs = bm25_retriever.invoke(query)
    vec_docs = vec_retriever.invoke(query)

    scores = defaultdict(float)
    doc_map = {}

    for rank, doc in enumerate(bm25_docs, start=1):
        key = (doc.page_content, tuple(sorted(doc.metadata.items())))
        doc_map[key] = doc
        scores[key] += weight_bm25 / (rrf_k + rank)

    for rank, doc in enumerate(vec_docs, start=1):
        key = (doc.page_content, tuple(sorted(doc.metadata.items())))
        doc_map[key] = doc
        scores[key] += weight_vector / (rrf_k + rank)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    result_docs = [doc_map[key] for key, _ in sorted_docs]

    print(f"\n========== RRF 融合结果 (Top {k}) ==========")
    for i, doc in enumerate(result_docs):
        print(f"Rank {i+1} -> {doc.metadata.get('source', '未知')}")
        print(f"内容预览: {doc.page_content[:150].replace(chr(10), ' ')}...")
        print("-" * 40)

    return result_docs

def retrieve_node(data: AgentState) -> dict:
    """
    RRF 检索节点：从状态中获取用户问题，执行双路混合检索与 RRF 分数融合，
    将生成的 20 个粗筛候选文档存入状态机的 raw_documents 字段。
    """
    user_query = data.get("question")
    candidate_docs = rrf_retrieve(query=user_query, k=20)
    return {"raw_documents": candidate_docs}