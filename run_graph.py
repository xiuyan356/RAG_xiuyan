"""
run_graph - 验证全链路 LangGraph 工作流

Author : 王聪
Date : 2026/7/19
"""
from Graph.workflow import app

def run_rag_test(query: str):
    print(f"\n🚀 启动 LangGraph 智能体，收到提问: '{query}'")

    # 初始化状态输入
    initial_state = {"question": query}

    # 执行图流转
    final_state = app.invoke(initial_state)

    print("\n" + "="*50)
    print("🤖 智能体最终给出的满血版回答：")
    print("="*50)
    print(final_state.get("answer"))
    print("="*50)

    # 打印精排后的参考源，检查纯英文 FastAPI 是否被完美采用
    print("\n📚 本次回答参考的 Top 文档源：")
    for i, doc in enumerate(final_state.get("final_documents", [])):
        print(f"  [{i+1}] 来源: {doc.metadata.get('source')} | 框架: {doc.metadata.get('category')}")

if __name__ == "__main__":
    # 测试高难度混血问题：既包含 FastAPI 又包含 Ollama/Chroma
    run_rag_test("How to use APIRouter in FastAPI and save something into Chroma database?")