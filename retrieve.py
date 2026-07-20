"""
retrieve - 基于最新混血双路检索与 RRF 融合的 LangGraph 原子节点

Author : 王聪
Date : 2026/7/19
Version : 0.0.3
"""
from typing import List
from langchain_core.documents import Document
from state import AgentState
from database import vec_retriever, bm25_retriever

def reciprocal_rank_fusion(vec_docs: List[Document], bm25_docs: List[Document], k: int = 60, top_n: int = 20) -> List[Document]:
    """
    RRF (倒数排名融合) 算法：完美融合向量语义检索与 BM25 关键词检索的结果
    """
    rrf_scores = {}
    doc_map = {}

    # 1. 累加向量检索的排名分
    for rank, doc in enumerate(vec_docs):
        # 使用 source 相对路径作为文档唯一标识，防止对象地址不同导致去重失败
        doc_id = doc.metadata.get("source", doc.page_content)
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + k)

    # 2. 累加关键词检索的排名分
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.metadata.get("source", doc.page_content)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + k)

    # 3. 按最终融合分数从高到低排序，切片取出 top_n 送给下游 Reranker
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    final_docs = [doc_map[doc_id] for doc_id, _ in sorted_docs[:top_n]]

    return final_docs


def retrieve_node(state: AgentState) -> dict:
    """
    LangGraph 检索原子节点：
    从输入状态获取 question，并发动 Chroma 与 BM25 双路拦截，最终通过 RRF 沉淀至 raw_documents
    """
    question = state.get("question", "")
    print(f"\n[Node: Retrieve] 正在为提问发起全框架混血双路检索: '{question}'")

    # 如果提问里包含特定框架名字，我们可以窄化范围；如果不包含，则默认全量库盲搜
    search_kwargs = {"k": 20}

    question_lower = question.lower()

    # 核心优化：改用并集收集器，收集问题中所有被提及的框架标签
    detected_categories = []
    frameworks = ["fastapi", "chroma", "ollama", "langchain", "langgraph"]

    for framework in frameworks:
        if framework in question_lower:
            # 兼容处理文件名和标签的细微差异
            category_tag = "fast_api" if framework == "fastapi" else framework
            detected_categories.append(category_tag)

    # 如果发现了相关框架标签，实施定向爆破过滤
    if detected_categories:
        print(f"  -> 检测到强相关框架关键词 {detected_categories}，开启元数据定向爆破过滤！")
        if len(detected_categories) == 1:
            # 单标签检索，依然使用你原本的直接映射结构
            search_kwargs["filter"] = {"category": detected_categories[0]}
        else:
            # 跨标签检索，利用 Chroma 官方支持的 $in 语法进行多组件并集过滤
            search_kwargs["filter"] = {"category": {"$in": detected_categories}}

    # 动态调整向量检索的过滤参数
    vec_retriever.search_kwargs.update(search_kwargs)

    # 执行双路并发召回
    vec_results = vec_retriever.invoke(question)
    bm25_results = bm25_retriever.invoke(question)

    print(f"  -> Chroma 稠密向量路召回: {len(vec_results)} 条")
    print(f"  -> BM25 稀疏关键词路召回: {len(bm25_results)} 条")

    # 实施倒数排名融合算法，过滤出 20 条最优质文档交给 Reranker 节点
    fused_documents = reciprocal_rank_fusion(vec_results, bm25_results, top_n=20)
    print(f"  -> RRF 混血融合去重完成，成功粗筛出 {len(fused_documents)} 条技术文档送入精排区。")

    # 完美对齐 AgentState 结构中的 raw_documents 字段
    return {"raw_documents": fused_documents}