"""
test - 纯本地稳健版

Author : 王聪
Date : 2026/7/9
Version : 0.0.2
"""
import sys
import os

# 确保 Python 能够正确识别当前根目录和子目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Graph.workflow import app


def run_test_workflow(query: str):
    """
    运行完整的 LangGraph RAG 工作流测试，并打印各个环节的输出结果
    """
    print(f"\n================ 🚀 开始测试 RAG 图工作流 ================")
    print(f"用户输入问题: {query}\n")

    # 1. 初始化进入图的初始状态字典
    initial_state = {"question": query}

    try:
        # 2. 执行图的 invoke，触发流程线：START -> retrieve -> rerank -> generate -> END
        final_output = app.invoke(initial_state)

        # 3. 打印检索与重排的统计结果，验证各节点数据是否正常流转
        raw_docs = final_output.get("raw_documents", [])
        final_docs = final_output.get("final_documents", [])

        print("================ 📊 各节点状态流转检查 ================")
        print(f"1. [retrieve_node] 粗筛阶段总共检索出: {len(raw_docs)} 个文档片段。")
        print(f"2. [rerank_node]   精排阶段过滤保留了: {len(final_docs)} 个核心片段。")

        # 4. 打印最终生成的技术答案
        print("\n================ 🤖 最终大模型生成的回答 ================")
        print(final_output.get("answer", "❌ 未能生成有效答案"))
        print("=======================================================\n")

    except Exception as e:
        print(f"\n❌ 工作流执行过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 可以将此处的问题替换为你 knowledge 目录下 Markdown 文件里实际包含的 FastAPI 知识点
    test_query = "如何在 FastAPI 中配置跨域请求(CORS)？"

    run_test_workflow(test_query)