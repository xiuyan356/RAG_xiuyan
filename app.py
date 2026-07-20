"""
api_server_fastapi -

Author : 王聪
Date :2026/7/8
Version :0.0.1
"""
import sys
import uvicorn
import traceback
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# 确保能正确导入根目录下的模块
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入图工作流、SQLite 磁盘缓存以及我们刚刚写好的高级向量语义缓存
from Graph.workflow import app as rag_app
from cache_manager import get_cached_answer, set_cached_answer
from my_semantic_cache import get_semantic_cache, set_semantic_cache

app = FastAPI(
    title="FastAPI 智能知识库 RAG 后端服务",
    version="0.7.0",
    description="基于 LangGraph + 混合检索 + CrossEncoder 重排 + 本地 Qwen/云端 DeepSeek 双重容灾 + RedisVL 向量语义缓存的工业级分布式系统"
)
# 配置跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)


# 统一定义请求体模型
class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        user_question = request.question.strip()
        if not user_question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="问题不能为空"
            )

        print(f"\n[收到前端请求] 提问: {user_question}")

        # ==========================================
        # 核心逻辑：升级为向量语义缓存拦截控制流
        # ==========================================

        # 1. 第一级拦截：L1 RedisVL 向量语义雷达匹配
        semantic_hit = get_semantic_cache(user_question, threshold=0.96)
        if semantic_hit:
            print(f"[L1 语义缓存命中] 极致速读！检测到相似历史意图，秒回响应！")
            return {
                "answer": semantic_hit["answer"],
                "context": f"[从 Redis 向量语义缓存中秒回，相似度: {semantic_hit['similarity']:.4f}]",
                "cache_hit": "L1_Redis"  # 保持前端组件的判定兼容
            }

        # 2. 第二级拦截：L2 SQLite 磁盘持久层精准匹配（作为兜底账本）
        cached_ans = get_cached_answer(user_question)
        if cached_ans:
            print("[L2 SQL 磁盘命中] 从持久化数据库提取成功，触发热回填至语义向量库...")
            set_semantic_cache(user_question, cached_ans)
            return {
                "answer": cached_ans,
                "context": "[从本地 SQLite 数据库中快速加载]",
                "cache_hit": "L2_SQLite"
            }

        # 3. 穿透级：缓存均未命中，执行昂贵的 RAG 图状态机计算
        print("[缓存全未命中] 正在触发 RAG 穿透计算（双路检索 -> 重排 -> LLM）...")

        initial_state = {
            "question": user_question,
            "context": "",
            "answer": ""
        }

        # 运行 LangGraph 工作流
        final_state = rag_app.invoke(initial_state)
        final_answer = final_state.get('answer', '未生成回答')
        final_context = final_state.get('context', '')

        # 4. 写入缓存：同步沉淀新知识到两级缓存空间
        if final_answer and final_answer != '未生成回答':
            print("[缓存双写同步] 新知识已成功同步写入 Redis 向量库与 SQLite 数据库。")
            set_semantic_cache(user_question, final_answer)
            set_cached_answer(user_question, final_answer)

        return {
            "answer": final_answer,
            "context": final_context,
            "cache_hit": "None"
        }

    except HTTPException:
        raise
    except Exception as e:
        print("\n======== 捕获到系统运行期或缓存网关崩溃 ========")
        print(f"异常类型: {type(e)}")
        print(f"异常信息: {str(e)}")
        traceback.print_exc()
        print("======================================================")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"内部服务器错误: {str(e)}"
        )


if __name__ == '__main__':
    uvicorn.run("app:app", host='127.0.0.1', port=5000, reload=True)