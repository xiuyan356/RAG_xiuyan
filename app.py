"""
api_server_fastapi -

Author : 王聪
Date :2026/7/8
Version :0.0.1
"""
import sys
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Graph.workflow import app as rag_app

app = FastAPI(title="《三体》RAG 后端服务", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

class ChatRequest(BaseModel):
    question: str

@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        user_question = request.question.strip()
        if not user_question:
            raise HTTPException(status_code=400, detail="问题不能为空")

        # 1. 构造 LangGraph 的初始状态
        initial_state = {
            "question": user_question,
            "context": "",
            "answer": ""
        }

        # 2. 运行图工作流
        final_state = rag_app.invoke(initial_state)
        return {
            "answer": final_state.get('answer', '未生成回答'),
            "context": final_state.get('context', '')
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
    # except HTTPException:
    #     raise
    # except Exception as e:
    #     import traceback
    #     print("======== 🚨 捕获到 LangGraph 运行期崩溃 🚨 ========")
    #     print(f"异常类型: {type(e)}")
    #     print(f"异常信息: {str(e)}")
    #     traceback.print_exc()  # 🎯 让这行代码生效，吐出红字堆栈！
    #     print("==================================================")
    #     raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("app:app", host='127.0.0.1', port=5000, reload=True)