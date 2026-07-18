# 项目开发日志 (CHANGELOG)

## [0.6.0] - 2026-07-18

### 变更类型：功能新增 / 性能优化

#### 变更详情
- **引入 Redis 内存层（L1 Cache）**：集成 Redis 键值对高频存储，利用全内存特性实现 O(1) 复杂度的极速缓存匹配。
- **构建 `my_redis_cache.py`**：基于连接池（ConnectionPool）机制封装高并发安全连接，并引入 `setex` 原子操作为缓存配置 24 小时自动过期策略（TTL=86400秒）。
- **重构 `main.py` 为双层冷热联动缓存架构**：
  - **L1 级**：优先拦截并查询 Redis 内存，命中则极致秒回。
  - **L2 级**：内存未命中时穿透至本地 SQLite 磁盘，命中则回填（Cache Aside 模式）至 Redis 并返回。
  - **穿透级**：双层缓存均穿透时，触发真实的 LangGraph 重度图流水线计算，生成有效答案后执行“双写同步”持久化。
- **解决命名空间冲突**：避免与系统第三方环境库重名，将本地驱动模块重命名为 `my_redis_cache.py`，修正 IDE 路由路径索引。

#### 影响范围
- `my_redis_cache.py`：新增 Redis 缓存核心管理组件。
- `main.py`：升级多级拦截控制流。
- `CHANGELOG.md`：追溯体系更新。

#### 测试结果摘要
| 指标 | 结果 |
|------|------|
| 缓存全穿透首次计算 | 约 24 秒（完整走检索、Reranker 与大模型） |
| **L1 Redis 内存命中** | **< 0.001 秒（极致高并发响应，完全免除 CPU 算力与 Token 消耗）** |
| **L2 SQL 磁盘命中恢复** | **约 0.01 秒（具备冷数据热回填能力）** |

---

## [0.5.0] - 2026-07-18

### 变更类型：功能新增 / 基础设施优化

#### 变更详情
- **引入 SQL Cache 机制**：使用轻量级本地 SQLite 数据库（`rag_cache.db`）作为持久化问答缓存层，精准拦截重复提问。
- **构建 `cache_manager.py`**：专门负责缓存表的初始化（`qa_cache`）、数据读（`get_cached_answer`）与数据写（`set_cached_answer`）。
- **优化控制台交互**：重构 `main.py` 升级为 `while True` 持续对话模式，避免单次提问重复拉起数据库索引。

#### 影响范围
- `cache_manager.py`：新增缓存管理模块。
- `main.py`：作为系统统一入口，在 `query_rag_flow` 中织入缓存拦截逻辑。

#### 测试结果摘要
| 指标 | 结果 |
|------|------|
| 缓存未命中响应时间 | 约 24 秒（完整走双路检索 + 重排 + 大模型） |
| **缓存命中响应时间** | **< 0.01 秒（零 Token 消耗，内存级秒回）** |

---

## [0.4.0] - 2026-07-18

### 变更类型：架构重构（工程化解耦）

#### 变更详情
- **彻底废弃并下线老旧臃肿的 `rag.py`**，按工业级规范进行原子级单文件解耦。
- 抽离数据与检索驱动至 `database.py`（统一收拢 Markdown 扫描、Chroma 和 BM25 初始化）。
- 将大模型初始化与灾备逻辑移至 `llm.py`，提示词管理移至 `prompt.py`。
- 将 LangGraph 状态机的三大核心算子剥离为根目录下的独立节点文件：`retrieve.py`、`rerank.py`、`generate.py`
- 重新穿线 `Graph/workflow.py`，采用扁平化导入，实现业务逻辑与工作流拓扑的分离。
- 参数微调：根据此前规划，正式将混合检索权重调整为 `weight_bm25=0.2, weight_vector=0.8`。

#### 影响范围
- 项目根目录全套新原子服务文件（`database.py`, `llm.py`, `prompt.py`, `retrieve.py`, `rerank.py`, `generate.py`）。
- `Graph/workflow.py`：图编译逻辑更新。
- `test.py`：全新的图节点流转全链路自动化测试脚本。

---

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

- [x] ~~尝试 `weight_bm25=0.2, weight_vector=0.8`，观察 Reranker Top 1 是否更稳定~~ (已在 0.4.0 落地)
- [x] ~~引入本地持久化 SQL 缓存层与 Redis 内存极速层（多级联动拦截机制）~~ (已在 0.6.0 落地)
- [ ] 考虑替换分词器，jieba 分词在多英文专业文档表现欠佳
- **[当前目标 - 路线图 ⑦]** 引入 FastAPI 正式将 `query_rag_flow` 包装为 Web 接口，准备进行前端 UI 联调与前后端解耦交互
- **[进阶探索 - 路线图 ⑨]** 基于 RedisVL 探索更高阶的向量语义相似度缓存（Semantic Cache），解除字面量精准匹配的限制