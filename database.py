"""
database -

Author : 王聪
Date :2026/7/18
Version :0.0.1
"""
import os
import jieba
import pickle
from config import config
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 常量配置
CHROMA_DIR = "chromadb"
KNOWLEDGE_DIR = "knowledge"
TEXTS_CACHE_FILE = "texts_cache.pkl"


def get_all_fastapi_texts(knowledge_dir=KNOWLEDGE_DIR, force_reload=False):
    """加载并切分所有 Markdown 文件，支持 pickle 缓存机制"""
    if not force_reload and os.path.exists(TEXTS_CACHE_FILE):
        try:
            with open(TEXTS_CACHE_FILE, 'rb') as f:
                print(f"从缓存快速加载文本切片: {TEXTS_CACHE_FILE}")
                return pickle.load(f)
        except Exception as e:
            print(f"缓存读取失败（可能版本不兼容），重新生成: {e}")

    print("正在扫描并切分所有 Markdown 文件（首次或强制重建）...")
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

        try:
            with open(TEXTS_CACHE_FILE, 'wb') as f:
                pickle.dump(all_chunks, f)
                print(f"文本切片已缓存至 {TEXTS_CACHE_FILE} (共 {len(all_chunks)} 个片段)")
        except Exception as e:
            print(f"缓存保存失败（不影响运行）: {e}")

    return all_chunks


def cut_zh_words(text):
    """中文分词器，供 BM25 使用"""
    return list(jieba.cut(text))


# 1. 执行多文件加载/缓存读取获取基础文本
texts = get_all_fastapi_texts()

# 2. 初始化嵌入模型驱动
embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)

# 3. 初始化并构建本地 Chroma 向量数据库检索器
if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    print("正在初始化 Chroma 向量数据库，请稍候...")
    vs = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DIR)
else:
    print("检测到已有数据库，正在直接加载本地 Chroma 索引...")
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

vec_retriever = vs.as_retriever(search_kwargs={'k': 20})

# 4. 初始化并构建 BM25 关键词检索器
bm25_retriever = BM25Retriever.from_documents(
    documents=texts,
    preprocess_func=cut_zh_words,
    bm25_params={
        'k1': 1.5,
        'b': 0.75
    }
)
bm25_retriever.k = 20  # 扩大候选集参与 RRF 融合