import os
import re
import pickle
from config import config
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 常量配置
BASE_DIR = r"D:\dev\RAG_Graph"
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
TEXTS_CACHE_FILE = os.path.join(BASE_DIR, "texts_cache.pkl")


def load_stopwords():
    """从 config 配置的路径动态加载停用词文件"""
    # 优先尝试从 config 获取属性，若没有则走默认路径
    stopwords_path = getattr(config, "STOPWORDS_FILE", os.path.join(BASE_DIR, "en_stopwords.txt"))

    if os.path.exists(stopwords_path):
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                # 读取每行，去掉首尾空格，过滤空行
                words = {line.strip().lower() for line in f if line.strip()}
                print(f"成功从配置路径加载了 {len(words)} 个英文停用词。")
                return words
        except Exception as e:
            print(f"读取停用词文件失败: {e}，将启用兜底内建停用词。")
    else:
        print(f"未在 {stopwords_path} 探测到停用词文件，将启用兜底内建停用词。")

    # 兜底内建停用词，防止文件丢失导致系统瘫痪
    return {'a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'the', 'to', 'with', 'you'}


# 在模块初始化时一次性加载停用词，避免分词时重复读取 I/O
GLOBAL_STOPWORDS = load_stopwords()


def get_all_knowledge_texts(knowledge_dir=KNOWLEDGE_DIR, force_reload=False):
    """加载并切分所有子目录下的纯净 Markdown 文件，支持分类标签注入与 pickle 缓存机制"""
    if not force_reload and os.path.exists(TEXTS_CACHE_FILE):
        try:
            with open(TEXTS_CACHE_FILE, 'rb') as f:
                print(f"从缓存快速加载文本切片: {TEXTS_CACHE_FILE}")
                return pickle.load(f)
        except Exception as e:
            print(f"缓存读取失败，将重新生成: {e}")

    print(f"未检测到有效缓存，开始全量扫描知识库目录: {knowledge_dir}")
    all_chunks = []
    splitter = MarkdownTextSplitter(chunk_size=800, chunk_overlap=100)

    if not os.path.exists(knowledge_dir):
        print(f"错误：知识库根目录不存在: {knowledge_dir}")
        return all_chunks

    categories = [c for c in os.listdir(knowledge_dir) if
                  os.path.isdir(os.path.join(knowledge_dir, c)) and not c.startswith('.')]
    print(f"探测到组件库列表: {categories}")

    for category in categories:
        category_path = os.path.join(knowledge_dir, category)
        print(f"正在扫描组件库: [{category}]")
        for root, dirs, files in os.walk(category_path):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, knowledge_dir).replace("\\", "/")
                    try:
                        loader = UnstructuredMarkdownLoader(file_path, mode="single")
                        documents = loader.load()
                        for doc in documents:
                            doc.metadata["source"] = relative_path
                            doc.metadata["category"] = category
                            doc.metadata["source_file"] = file
                        chunks = splitter.split_documents(documents)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        print(f"读取文件失败 {relative_path}: {str(e)}")
                        continue
    try:
        with open(TEXTS_CACHE_FILE, 'wb') as f:
            pickle.dump(all_chunks, f)
            print(f"所有技术切片已成功持久化至缓存: {TEXTS_CACHE_FILE} (共 {len(all_chunks)} 个片段)")
    except Exception as e:
        print(f"缓存文件保存失败: {e}")
    return all_chunks


def cut_en_words(text):
    """
    工业级纯英文技术分词器：
    1. 过滤所有标点，只保留英文单词、数字和特定连接符（如 bge-m3）
    2. 统一转为小写以消除大小写对 BM25 词频的影响
    3. 过滤掉全局配置的停用词，保护核心技术词的权重
    """
    if not text:
        return []

    # 使用正则提取标准的英文 Token，允许包含下划线和连字符（精准保护 APIRouter, fast_api 等术语）
    tokens = re.findall(r'[a-zA-Z0-9_-]+', text.lower())

    # 过滤停用词并排除极短的单个干扰字符
    final_tokens = [token for token in tokens if token not in GLOBAL_STOPWORDS and len(token) > 1]

    return final_tokens


# 1. 直接从已经生成好的缓存中秒读 4625 个片段
texts = get_all_knowledge_texts(force_reload=False)

# 2. 初始化嵌入模型驱动
embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)

# 3. 分批次安全注入本地 Chroma 向量数据库
if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    print("正在初始化本地 Chroma 向量数据库...")
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    batch_size = 100
    total_texts = len(texts)
    print(f"触发防假死分批写入机制，总计 {total_texts} 个片段，每批 {batch_size} 个...")

    for i in range(0, total_texts, batch_size):
        batch = texts[i:i + batch_size]
        vs.add_documents(batch)
        print(f"  已成功向硬盘注入并落盘: {min(i + batch_size, total_texts)} / {total_texts}")

    print("Chroma 向量数据库全量分批沉淀完毕。")
else:
    print("检测到已有数据库，正在直接同步本地 Chroma 索引...")
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

vec_retriever = vs.as_retriever(search_kwargs={'k': 20})

# 初始化并构建针对纯英文优化的 BM25 关键词检索器
print("正在构建纯英文 BM25 关键词倒排索引...")
bm25_retriever = BM25Retriever.from_documents(
    documents=texts,
    preprocess_func=cut_en_words,
    bm25_params={'k1': 1.5, 'b': 0.75}
)
bm25_retriever.k = 20
print("混血双路检索器初始化完毕，系统准备就绪。")