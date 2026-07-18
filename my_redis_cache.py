"""
cache_manager -

Author : 王聪
Date :2026/7/18
Version :0.0.1
"""
import sqlite3
import os

DB_PATH = "rag_cache.db"


def init_cache_db():
    """初始化 SQLite 缓存表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 创建缓存表：将用户问题设为唯一主键
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS qa_cache (
            question TEXT PRIMARY KEY,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def get_cached_answer(question: str) -> str:
    """精准匹配：根据问题查询是否有缓存的答案"""
    cleaned_q = question.strip()
    if not cleaned_q:
        return None

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT answer FROM qa_cache WHERE question = ?", (cleaned_q,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return row[0]
    return None


def set_cached_answer(question: str, answer: str):
    """写入或更新缓存"""
    cleaned_q = question.strip()
    if not cleaned_q or not answer:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 使用 INSERT OR REPLACE，如果提问重复则覆盖更新答案
    cursor.execute(
        "INSERT OR REPLACE INTO qa_cache (question, answer) VALUES (?, ?)",
        (cleaned_q, answer)
    )
    conn.commit()
    conn.close()


# 脚本加载时自动检查并初始化本地数据库文件
init_cache_db()