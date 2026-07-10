"""
new_reranker -

Author : 王聪
Date :2026/7/9
Version :0.0.1
"""


def rerank_retrieve(query, top_n=5):
    """
    第一步：利用 BM25 和 Vector 扩大召回（k=20）
    第二步：去重合并
    第三步：利用本地 CrossEncoder 交叉编码器进行精准精排，彻底消除高频词噪声
    """
    # 1. 双路粗筛：利用你之前调好的参数，各自拿回 20 个高嫌疑候选文档
    bm25_docs = bm25_retriever.invoke(query)
    vec_docs = vec_retriever.invoke(query)

    # 2. 去重合并：通过 page_content 作为唯一 Key，去掉两路结果里重复的片段
    unique_docs = {doc.page_content: doc for doc in (bm25_docs + vec_docs)}.values()
    unique_docs = list(unique_docs)

    # 3. 构造 Reranker 的输入对：[[问题, 文档1], [问题, 文档2], ...]
    pairs = [[query, doc.page_content] for doc in unique_docs]

    # 4. 让 Reranker 交叉编码器现场打分（给出 0 到 1 之间的绝对相关度）
    scores = reranker.predict(pairs)

    # 5. 将分数与文档绑定，并按照分数从高到低进行降序排列
    doc_score_pairs = list(zip(unique_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # 6. 切取经过高精度筛选后的前 top_n 个干货文档
    final_docs = [doc for doc, score in doc_score_pairs[:top_n]]

    # 打印精排日志，方便我们今晚观察 middleware 等噪声有没有被压下去
    print(f"\n========== 🎯 Reranker 智能精排 (从 {len(unique_docs)} 个去重候选精选前 {top_n} 个) ==========")
    for i, doc in enumerate(final_docs):
        score = doc_score_pairs[i][1]
        print(f"Rank {i + 1} [Score: {score:.4f}] -> {doc.metadata.get('source', '未知来源')}")
        print(f"内容摘要: {doc.page_content[:100]}...")
        print("-" * 30)

    return final_docs
