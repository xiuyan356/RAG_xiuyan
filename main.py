"""
main -

Author : 王聪
Date :2026/7/8
Version :0.0.1
"""
from Graph.workflow import app

def main():
    user_input = input('请输入关于《三体》的问题：')
    # 初始化状态
    initial_state = {
        "question": user_input,
        "context": "",
        "answer": ""
    }

    # 启动工作流
    final_state = app.invoke(initial_state)

    print('\n================ 最终回答 ================')
    print(final_state['answer'])

if __name__ == "__main__":
    main()
