"""
my_semantic_cache -

Author : 王聪
Date :2026/7/18
Version :0.0.1
"""
import os
import sys
import numpy as np

# 确保能正常引入根目录的其他驱动
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 动态引入全局 embeddings 模型
try:
    from database import embeddings as project_embedding
except ImportError:
    project_embedding = None


def get_embedding(text: str) -> list:
    """统一的文本向量化工具函数"""
    if project_embedding and hasattr(project_embedding, 'embed_query'):
        return project_embedding.embed_query(text)
    raise RuntimeError("未成功从 database.py 导入全局 embeddings 模型，请检查变量名是否正确。")


# 在内存中维护一个轻量级语义缓存仓库 (避免对 Redis 向量插件的硬依赖)
# 结构：[{"question": str, "answer": str, "vector": np.array}]
_GLOBAL_SEMANTIC_MEMORY = []


def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


def get_semantic_cache(question: str, threshold: float = 0.96) -> dict:
    """
    纯 Python 高性能内存级语义缓存检索网关
    """
    cleaned_q = question.strip()
    if not cleaned_q or not _GLOBAL_SEMANTIC_MEMORY:
        return None

    try:
        # 将当前提问转化为向量
        query_vector = np.array(get_embedding(cleaned_q), dtype=np.float32)

        best_match = None
        max_sim = -1.0

        # 遍历内存中的历史缓存进行余弦相似度比对
        for item in _GLOBAL_SEMANTIC_MEMORY:
            sim = cosine_similarity(query_vector, item["vector"])
            if sim > max_sim:
                max_sim = sim
                best_match = item

        if best_match:
            print(f"[语义雷达扫描] 最近历史问题: '{best_match['question']}', 语义相似度: {max_sim:.4f}")
            if max_sim >= threshold:
                return {
                    "answer": best_match["answer"],
                    "similarity": max_sim
                }
    except Exception as e:
        print(f"[语义缓存查询异常] {str(e)}")

    return None


def set_semantic_cache(question: str, answer: str):
    """
    同步沉淀新知识到内存语义缓存空间
    """
    cleaned_q = question.strip()
    if not cleaned_q or not answer:
        return

    try:
        # 防重复写入
        for item in _GLOBAL_SEMANTIC_MEMORY:
            if item["question"] == cleaned_q:
                item["answer"] = answer
                return

        query_vector = np.array(get_embedding(cleaned_q), dtype=np.float32)
        _GLOBAL_SEMANTIC_MEMORY.append({
            "question": cleaned_q,
            "answer": answer,
            "vector": query_vector
        })
    except Exception as e:
        print(f"[语义缓存写入异常] {str(e)}")