"""
main -

Author : 王聪
Date :2026/7/8
Version :0.0.1
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Graph.workflow import app
from cache_manager import get_cached_answer, set_cached_answer
from my_redis_cache import get_cached_answer,set_cached_answer


def query_rag_flow(question: str) -> str:
    """
    统一的 RAG 图工作流调用核心函数，集成了 [Redis 内存] + [SQLite 磁盘] 双层缓存机制
    """
    cleaned_q = question.strip()
    if not cleaned_q:
        return "提示：输入的问题不能为空。"

    # 1. 第一层防御：Redis 内存缓存拦截 (L1 Cache)
    redis_answer = get_cached_answer(cleaned_q)
    if redis_answer:
        print("\n[L1 Redis 内存命中] 极致速读！直接从 Redis 内存高频存取区秒回答案！")
        return redis_answer

    # 2. 第二层防御：SQLite 磁盘缓存检查 (L2 Cache)
    sqlite_answer = get_cached_answer(cleaned_q)
    if sqlite_answer:
        print("\n[L2 SQL 磁盘命中] 从本地 SQLite 捞回数据，并顺手同步回填至 Redis 内存...")
        set_cached_answer(cleaned_q, sqlite_answer)
        return sqlite_answer

    # 3. 缓存全穿透：走真实的 LangGraph 重度流水线
    print("\n[缓存全未命中] 正在触发 RAG 穿透计算（双路检索 -> 重排 -> LLM）...")
    initial_state = {"question": cleaned_q}

    try:
        final_state = app.invoke(initial_state)
        answer = final_state.get('answer', '未生成有效回答')

        # 4. 成功生成有效答案后，同步写入双层缓存
        if answer and "未生成有效回答" not in answer and "系统执行异常" not in answer:
            set_cached_answer(cleaned_q, answer)
            set_cached_answer(cleaned_q, answer)
            print("[缓存双写同步] 新知识已成功同步写入 Redis 内存与 SQLite 数据库。")

        return answer
    except Exception as e:
        return f"系统执行异常: {str(e)}"


def main():
    print("=========================================")
    print("  欢迎使用多级缓存 (Redis+SQL) 智能知识库  ")
    print("=========================================")

    while True:
        try:
            user_input = input('\n请输入问题（输入 q 或 exit 退出）：').strip()

            if user_input.lower() in ['q', 'exit']:
                print("感谢使用，再见！")
                break

            if not user_input:
                continue

            answer = query_rag_flow(user_input)

            print('\n================ 最终回答 ================')
            print(answer)
            print('============================================')

        except KeyboardInterrupt:
            print("\n程序已被用户强制终止。")
            break


if __name__ == "__main__":
    main()