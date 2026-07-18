"""
test_retrieval -

Author : 王聪
Date :2026/7/9
Version :0.0.1
"""
from old_rag import (
    bm25_retriever,
    vec_retriever,
    rrf_retrieve
)


query = "FastAPI 如何处理路径参数"


print("\n========== BM25 ==========")

bm25_docs = bm25_retriever.invoke(query)

for i, doc in enumerate(bm25_docs):
    print(i+1, doc.metadata)
    print(doc.page_content[:200])


print("\n========== Vector ==========")

vec_docs = vec_retriever.invoke(query)

for i, doc in enumerate(vec_docs):
    print(i+1, doc.metadata)
    print(doc.page_content[:200])


print("\n========== RRF ==========")

rrf_docs = rrf_retrieve(query)

for i, doc in enumerate(rrf_docs):
    print(i+1, doc.metadata)
    print(doc.page_content[:200])