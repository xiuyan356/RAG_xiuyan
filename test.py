"""
test - 纯本地稳健版

Author : 王聪
Date : 2026/7/9
Version : 0.0.2
"""
import os
# 强力清除可能干扰本地 127.0.0.1 Ollama 连接的代理环境变量
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

import json
import time
from typing import List, Dict, Any
from rag import RAG_answer_node


def load_test_queries(file_path: str = "test_queries.json") -> List[Dict[str, Any]]:
    """加载测试集 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_single_test(query: str) -> Dict[str, Any]:
    """运行单个查询，返回结果"""
    initial_state = {
        "question": query,
        "context": "",
        "answer": ""
    }

    start_time = time.time()
    final_state = RAG_answer_node(initial_state)
    elapsed = time.time() - start_time

    return {
        "question": query,
        "context": final_state.get("context", ""),
        "answer": final_state.get("answer", ""),
        "elapsed_seconds": round(elapsed, 2)
    }


def print_result_summary(results: List[Dict[str, Any]]):
    """打印测试结果摘要"""
    print("\n" + "=" * 80)
    print(f"📊 测试完成！共 {len(results)} 条查询")
    print("=" * 80)

    total_time = sum(r['elapsed_seconds'] for r in results)
    avg_time = total_time / len(results)

    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均耗时: {avg_time:.2f} 秒/条")

    print("\n" + "-" * 80)
    print("📝 详细结果（前 5 条预览）:")
    print("-" * 80)

    for i, r in enumerate(results[:5], 1):
        print(f"\n【{i}】问题: {r['question']}")
        print(f"  ⏱ 耗时: {r['elapsed_seconds']} 秒")
        context_preview = r['context'][:200] + "..." if len(r['context']) > 200 else r['context']
        answer_preview = r['answer'][:200] + "..." if len(r['answer']) > 200 else r['answer']
        print(f"  📄 上下文片段: {context_preview}")
        print(f"  💡 回答片段: {answer_preview}")
        print("-" * 40)


def save_results(results: List[Dict[str, Any]], file_path: str = "test_results.json"):
    """将详细结果保存到 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 完整结果已保存到 {file_path}")


def main():
    # 加载测试集
    print("📂 正在加载测试集...")
    test_queries = load_test_queries()
    print(f"✅ 已加载 {len(test_queries)} 条测试问题")

    # 运行测试
    print("\n🚀 开始运行测试...")
    results = []

    for i, item in enumerate(test_queries, 1):
        print(f"  处理中 ({i}/{len(test_queries)}): {item['question'][:30]}...")
        result = run_single_test(item['question'])
        results.append({
            "id": item['id'],
            "category": item['category'],
            "question": item['question'],
            "expected_source": item.get('expected_source', ''),
            "context": result['context'],
            "answer": result['answer'],
            "elapsed_seconds": result['elapsed_seconds']
        })

    # 打印摘要
    print_result_summary(results)

    # 保存详细结果
    save_results(results)


if __name__ == "__main__":
    main()