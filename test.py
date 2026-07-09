"""
test -

Author : 王聪
Date :2026/7/9
Version :0.0.1
"""
# test_fallback.py
import os
from config import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def test_fallback_mechanism():
    """测试 fallback 是否正常工作"""

    # 1. 创建两个 LLM
    # 主 LLM - 配置正确的连接（会成功）
    ollama_llm = ChatOpenAI(
        model="qwen2.5:1.5b",
        temperature=0,
        base_url=config.OLLAMA_OPENAI_BASE_URL,
        api_key=config.OLLAMA_API_KEY,
        timeout=60.0
    )

    # 备用 LLM
    deepseek_llm = ChatOpenAI(
        base_url=config.DEEPSEEK_BASE_URL,
        api_key=config.DEEPSEEK_API_KEY,
        model='deepseek-v4-flash',
        max_tokens=1024,
        temperature=0.0,
    )

    # 2. 创建 prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手，请简洁回答。"),
        ("user", "{question}")
    ])

    # ===== 测试1: 正常情况（Ollama 运行正常）=====
    print("=" * 50)
    print("测试1: Ollama 正常运行")
    print("=" * 50)

    final_llm = ollama_llm.with_fallbacks([deepseek_llm])
    chain = prompt | final_llm | StrOutputParser()

    try:
        result = chain.invoke({"question": "你好，请用一句话介绍自己"})
        print(f"✅ 主 LLM 响应: {result}\n")
        print("注意：此时使用的是 Ollama，不是 fallback")
    except Exception as e:
        print(f"❌ 失败: {e}\n")

    # ===== 测试2: 强制触发 fallback（错误端口）=====
    print("=" * 50)
    print("测试2: Ollama 连接失败 - 强制触发 fallback")
    print("=" * 50)

    # 故意使用错误的端口
    broken_ollama = ChatOpenAI(
        model="qwen2.5:1.5b",
        temperature=0,
        base_url="http://localhost:9999",  # 错误端口
        api_key="dummy",
        timeout=2.0  # 快速超时
    ).with_fallbacks([deepseek_llm])

    chain2 = prompt | broken_ollama | StrOutputParser()

    try:
        result = chain2.invoke({"question": "你好，请用一句话介绍自己"})
        print(f"✅ 使用了 fallback (DeepSeek) 响应: {result}\n")
    except Exception as e:
        print(f"❌ 所有尝试都失败: {e}\n")

    # ===== 测试3: 超时触发 fallback =====
    print("=" * 50)
    print("测试3: 模拟超时 - 强制触发 fallback")
    print("=" * 50)

    slow_ollama = ChatOpenAI(
        model="qwen2.5:1.5b",
        base_url=config.OLLAMA_OPENAI_BASE_URL,
        api_key=config.OLLAMA_API_KEY,
        timeout=0.001  # 极短的超时
    ).with_fallbacks([deepseek_llm])

    chain3 = prompt | slow_ollama | StrOutputParser()

    try:
        result = chain3.invoke({"question": "你好"})
        print(f"✅ 使用了 fallback (DeepSeek) 响应: {result}\n")
    except Exception as e:
        print(f"❌ 所有尝试都失败: {e}\n")

    # ===== 测试4: 验证 fallback 返回的内容 =====
    print("=" * 50)
    print("测试4: 验证 fallback 是否真的返回了 DeepSeek 的回答")
    print("=" * 50)

    # 直接调用 DeepSeek 作为对比
    print("直接调用 DeepSeek:")
    direct_ds = prompt | deepseek_llm | StrOutputParser()
    ds_result = direct_ds.invoke({"question": "你是谁开发的？"})
    print(f"DeepSeek: {ds_result}\n")

    # 通过 fallback 调用（强制触发）
    print("通过 fallback 调用:")
    try:
        fallback_result = chain2.invoke({"question": "你是谁开发的？"})
        print(f"Fallback 结果: {fallback_result}\n")
    except Exception as e:
        print(f"失败: {e}\n")


def test_fallback_with_rag_context():
    """测试在 RAG 场景下的 fallback"""
    print("=" * 50)
    print("测试5: RAG 场景下的 fallback")
    print("=" * 50)

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "根据参考资料回答问题。"),
        ("user", "参考资料：{context}\n\n问题：{question}")
    ])

    # 配置
    ollama_llm = ChatOpenAI(
        model="qwen2.5:1.5b",
        temperature=0,
        base_url=config.OLLAMA_OPENAI_BASE_URL,
        api_key=config.OLLAMA_API_KEY,
        timeout=60.0
    )

    deepseek_llm = ChatOpenAI(
        base_url=config.DEEPSEEK_BASE_URL,
        api_key=config.DEEPSEEK_API_KEY,
        model='deepseek-v4-flash',
        max_tokens=1024,
        temperature=0.0,
    )

    # 故意使用错误的端口触发 fallback
    broken_ollama = ChatOpenAI(
        model="qwen2.5:1.5b",
        base_url="http://localhost:9999",
        api_key="dummy",
        timeout=1.0
    ).with_fallbacks([deepseek_llm])

    chain = prompt | broken_ollama | StrOutputParser()

    try:
        result = chain.invoke({
            "context": "三体是一部科幻小说，作者是刘慈欣。",
            "question": "三体的作者是谁？"
        })
        print(f"✅ Fallback 在 RAG 场景下工作: {result}\n")
    except Exception as e:
        print(f"❌ RAG 场景下失败: {e}\n")


if __name__ == "__main__":
    # 运行所有测试
    test_fallback_mechanism()
    test_fallback_with_rag_context()

    print("=" * 50)
    print("测试完成！")
    print("=" * 50)