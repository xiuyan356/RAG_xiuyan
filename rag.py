"""
rag -

Author : 王聪
Date :2026/7/8
Version :0.0.1
"""
import os #操作系统相关功能。
import jieba # 中文分词器
import pickle # 文档切分持久化依赖
from config import config
from typing import TypedDict, List #定义 AgentState 这个图状态结构体的
from langchain_core.documents import Document
from langchain_chroma import Chroma # 向量数据库+ 相似度检索引擎
from langchain_openai import ChatOpenAI # LangChain 提供的通用客户端，兼容各种格式调用任何兼容 OpenAI API协议的/chat/completions 接口
from collections import defaultdict #让字典在访问不存在的 key 时自动返回 0.0，让RRF不用判断key是否存在
from langchain_ollama import OllamaEmbeddings # LangChain 提供的专用嵌入客户端
from langchain_text_splitters import MarkdownTextSplitter # Markdown格式分割器
from langchain_core.prompts import ChatPromptTemplate # 构造对话模板（system/user 角色消息） 提示词
from langchain_community.retrievers import BM25Retriever # BM25 关键词检索器（基于词频统计召回文档）
from langchain_core.output_parsers import StrOutputParser #将 LLM 输出解析为纯字符串
from langchain_community.document_loaders import UnstructuredMarkdownLoader # 读取 .md 文件依赖，保留原始结构
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
# RunnableLambda: 把普通函数包装成可链式调用的 Runnable
# RunnablePassthrough: 透传输入，不改变数据（常用来保留原始 question）
# RunnableParallel: 并行执行多个 Runnable，合并输出（用于同时生成 context 和 question）


CHROMA_DIR = "chromadb" # 向量索引
KNOWLEDGE_DIR = "knowledge" # 原始知识文件
TEXTS_CACHE_FILE = "texts_cache.pkl"  # 切割后的Document

from sentence_transformers import CrossEncoder # 导入 transformers的交叉编码器：CrossEncoder
reranker = CrossEncoder(config.CrossEncoder_PATH) # 实例化模型对象

def get_all_fastapi_texts(knowledge_dir=KNOWLEDGE_DIR, force_reload=False):
    """
        加载并切分所有 Markdown 文件。
        如果存在缓存文件且 force_reload=False，直接加载缓存，跳过文件系统扫描。
    """
    if not force_reload and os.path.exists(TEXTS_CACHE_FILE):
        try:
            with open(TEXTS_CACHE_FILE, 'rb') as f:
                print(f" 从缓存快速加载文本切片: {TEXTS_CACHE_FILE}")
                return pickle.load(f)
        except Exception as e:
            print(f" 缓存读取失败（可能版本不兼容），重新生成: {e}")

    print(" 正在扫描并切分所有 Markdown 文件（首次或强制重建）...")
    all_chunks = []
    splitter = MarkdownTextSplitter(chunk_size=800, chunk_overlap=100)

    for root, dirs, files in os.walk(knowledge_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, knowledge_dir).replace("\\", "/")

                try:
                    loader = UnstructuredMarkdownLoader(file_path, mode="single")
                    documents = loader.load()

                    for doc in documents:
                        doc.metadata["source"] = relative_path

                    chunks = splitter.split_documents(documents)
                    all_chunks.extend(chunks)
                    print(f"成功加载并切分: {relative_path} (生成 {len(chunks)} 个片段)")
                except Exception as e:
                    print(f"读取文件失败 {relative_path}: {str(e)}")
                    continue
        # 新增：将结果写入缓存
        try:
            with open(TEXTS_CACHE_FILE, 'wb') as f:
                pickle.dump(all_chunks, f)
                print(f"文本切片已缓存至 {TEXTS_CACHE_FILE} (共 {len(all_chunks)} 个片段)")
        except Exception as e:
            print(f"缓存保存失败（不影响运行）: {e}")


    return all_chunks

# 执行多文件加载
texts = get_all_fastapi_texts()

def cut_zh_words(text):
    return list(jieba.cut(text))

embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)

if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    print(" 正在初始化 Chroma 向量数据库，请稍候...")
    vs = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DIR)
else:
    print("检测到已有数据库，正在直接加载本地 Chroma 索引...")
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

vec_retriever = vs.as_retriever(search_kwargs={'k': 20})

# BM25关键词检索器
bm25_retriever = BM25Retriever.from_documents(
    documents=texts,
    preprocess_func=cut_zh_words,
    bm25_params={
        'k1': 1.5,
        'b': 0.75
    }
)

# BM25 默认 k=4，需要扩大候选集参与RRF融合
bm25_retriever.k = 20

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    raw_documents: List[Document]    # 用于存放 RRF 混合检索出来的 20 个粗筛文档列表
    final_documents: List[Document]  # 用于存放 Reranker 精排挑出来的 Top 5 文档列表

# ==================== RRF 融合 ====================
def retrieve_node(query, k=20, rrf_k=60, weight_bm25=0.2, weight_vector=0.8):
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


# ==================== Reranker 精排 ====================
def rerank_documents(query, candidate_docs, top_n=5):
    """
    对 RRF 候选文档进行 Reranker 精排，返回 Top N
    """
    if not candidate_docs:
        return []

    # 安全去重（RRF 已去重，这里兜底）
    unique_docs = list({doc.page_content: doc for doc in candidate_docs}.values())

    pairs = [[query, doc.page_content] for doc in unique_docs]
    scores = reranker.predict(pairs)

    doc_score_pairs = list(zip(unique_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    final_docs = [doc for doc, _ in doc_score_pairs[:top_n]]

    print(f"\n========== Reranker 精排 (从 {len(unique_docs)} 个候选精选 Top {top_n}) ==========")
    for i, doc in enumerate(final_docs):
        score = doc_score_pairs[i][1]
        print(f"Rank {i+1} [Score: {score:.4f}] -> {doc.metadata.get('source', '未知')}")
        print(f"内容摘要: {doc.page_content[:120].replace(chr(10), ' ')}...")
        print("-" * 40)

    return final_docs

# 组装输入映射字典
# handler = RunnableParallel({
#     "context": RunnableLambda(rrf_retrieve),
#     "question": RunnablePassthrough()
# })


def RAG_answer_node(data: AgentState) -> AgentState:
    user_query = data['question']

    # 第一步：RRF 融合，拿 20 个候选
    candidate_docs = retrieve_node(user_query, k=20) # 调用RRF函数

    # 第二步：Reranker 精排，取 Top 5
    final_docs = rerank_documents(user_query, candidate_docs, top_n=5) # 调用Reranker函数

    # 拼接上下文
    context_str = "\n\n".join([doc.page_content for doc in final_docs])

    # 提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个专业的 FastAPI 技术文档专家。请严格根据以下参考资料回答用户的技术问题。如果资料中没有提及相关实现，请直接回答不知道，严禁虚构 API 或参数。"),
        ("user", "参考资料：\n{context}\n\n用户问题：{question}")
    ])

    # 3. 双保险 LLM 降级策略
    ollama_llm = ChatOpenAI(
        model="qwen2.5:1.5b",
        temperature=0,
        base_url=config.OLLAMA_OPENAI_BASE_URL,
        api_key=config.OLLAMA_API_KEY,
        timeout=60.0
    )

    deepseek_llm = ChatOpenAI(
        base_url=config.DEEPSEEK_BASE_URL,
        api_key=config.DEEPSEEK_API_KEY,
        model='deepseek-v4-flash',
        max_tokens=1024,
        temperature=0.0,
    )

    final_llm = ollama_llm.with_fallbacks([deepseek_llm])

    # chain = handler | prompt | final_llm | StrOutputParser()
    chain = prompt | final_llm | StrOutputParser()

    # response = chain.invoke({'context': context_str, 'question': user_query})
    response = chain.invoke({
        'context': context_str,
        'question': user_query
    })

    # 直接原地更新原字典
    data['context'] = context_str
    data['answer'] = response

    return data


if __name__ == "__main__":
    test_query = "FastAPI 怎么定义路径参数？"
    initial_state = {"question": test_query, "context": "", "answer": ""}
    final_state = RAG_answer_node(initial_state)
    print("\n💡 回答:", final_state['answer'])