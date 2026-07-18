"""
main -

Author : 王聪
Date :2026/7/8
Version :0.0.1
"""
import sys
import os

# 保证包路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Graph.workflow import app


def query_rag_flow(question: str) -> str:
    """
    统一的 RAG 图工作流调用核心函数，方便后续控制台和 Web API 复用
    """
    if not question.strip():
        return "提示：输入的问题不能为空。"

    # 初始化符合 AgentState 要求的状态字典
    initial_state = {
        "question": question
    }

    try:
        # 启动工作流线
        final_state = app.invoke(initial_state)
        return final_state.get('answer', '未生成有效回答')
    except Exception as e:
        return f"系统执行异常: {str(e)}"


def main():
    print("=========================================")
    print("     欢迎使用 FastAPI 智能知识库系统       ")
    print("=========================================")

    while True:
        try:
            user_input = input('\n请输入问题（输入 q 或 exit 退出）：').strip()

            if user_input.lower() in ['q', 'exit']:
                print("感谢使用，再见！")
                break

            if not user_input:
                continue

            # 调用图工作流
            answer = query_rag_flow(user_input)

            print('\n================  最终回答 ================')
            print(answer)
            print('============================================')

        except KeyboardInterrupt:
            print("\n程序已被用户强制终止。")
            break


if __name__ == "__main__":
    main()