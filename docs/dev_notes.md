[# 项目开发日志 (CHANGELOG)

## [0.3.0]

### 变更类型：功能新增 / 性能优化

#### 变更详情
- 接入 `BAAI/bge-reranker-base` 本地 CrossEncoder 模型，作为检索后精排模块
- 重构 `RAG_answer_node` 流程：`BM25 + Vector → RRF 融合 → Reranker 精排 → Top 5 → LLM`
- 调整 BM25 和 Vector 召回数从 `k=5` 扩至 `k=20`，为 Reranker 提供充足候选空间
- 新增 `rerank_documents()` 函数，对 RRF 候选文档进行交叉编码重排序

#### 影响范围
- `rag.py`：核心检索流程重构
- `test_retrieval.py`：30 条 FastAPI 技术问题测试集验证通过
- 精排有效压制 BM25 高频词噪声（如 `middleware.md` 干扰）

#### 测试结果摘要
| 指标 | 结果 |
|------|------|
| RRF 候选数 | 20 条 |
| Reranker 精排后 Top 5 | 相关文档占比 > 60% |
| 平均单条响应时间 | 约 24 秒（纯 CPU） |

---

## [0.2.0]

### 变更类型：算法优化

#### 变更详情
- 将原版 RRF（等权重融合）升级为 **Weighted RRF**
- 新增 `weight_bm25` 和 `weight_vector` 参数，独立控制两路检索器的贡献比例
- 当前配置：`weight_bm25=0.3`，`weight_vector=0.7`，有效降低 BM25 高频词带来的噪声

#### 影响范围
- `rrf_retrieve()` 函数签名和内部计算逻辑调整
- 为进一步调优（如 `0.2/0.8`、`0.15/0.85`）预留参数接口

---

## [0.1.0]

### 变更类型：功能新增 / 基础设施优化

#### 变更详情
- 将知识库从《三体》时间线替换为 **FastAPI 官方文档**（`.md` 格式，约 100+ 文件）
- 引入 `pickle` 序列化缓存（`texts_cache.pkl`），避免每次启动重复进行 Markdown 切分
- 配置 `MarkdownTextSplitter`（`chunk_size=800`，`chunk_overlap=100`）适配技术文档结构

#### 影响范围
- `knowledge/` 目录内容全部替换
- `get_all_fastapi_texts()` 函数新增缓存读写逻辑
- 启动速度从每次 3~5 秒降至 **0.5 秒以内**（第二次起）

---

## 待办 / 后续优化方向

- [ ] 尝试 `weight_bm25=0.2, weight_vector=0.8`，观察 Reranker Top 1 是否更稳定
- [ ] 考虑替换分词器，jieba分词在多英文专业文档表现欠佳

]()