# DocuAsk API

本文档记录 DocuAsk 当前 FastAPI 后端已经验证的接口。

当前后端定位：为本地文档 Agentic RAG 问答系统提供可复用的文档入库、文件上传、检索问答、Agentic ask、LLM answer 生成和检索评测能力。

## 启动后端

在仓库根目录运行：

```powershell
python -m uvicorn backend.app:app --app-dir "." --host 127.0.0.1 --port 8000
```

健康检查地址：

```text
http://127.0.0.1:8000/health
```

## GET /health

用途：检查后端服务是否启动。

响应示例：

```json
{
  "status": "ok",
  "service": "docuask-api",
  "version": "0.1.0"
}
```

## POST /documents

用途：把文档文本切分成 chunks，计算 embedding，并写入本地 Chroma 持久化库。

请求体：

```json
{
  "text": "# Python 学习 FAQ\n\n## RAG 的基本流程是什么？\nRAG 会先检索资料，再让 LLM 根据资料回答。",
  "embedding_mode": "Teaching keyword embedding",
  "chunk_size": 350,
  "chunk_overlap": 50
}
```

响应字段：

```text
document_id
embedding_mode
collection_name
chunk_count
stored_chunk_count
```

说明：

- `collection_name` 是后续 `/qa` 查询的关键参数。
- 当前支持 `Teaching keyword embedding` 和 `BGE Chinese embedding`。
- Chroma 数据保存在 `backend/storage/chroma_db_v2/`，该目录不提交到 Git。

## POST /documents/upload

用途：上传 `.txt`、`.md`、`.pdf` 或 `.docx` 文件，并复用文档入库流程写入 Chroma。

请求类型：

```text
multipart/form-data
```

字段：

| Field | Type | Required | 说明 |
|---|---|---|---|
| `file` | file | yes | `.txt`、`.md`、`.pdf` 或 `.docx` 文件 |
| `embedding_mode` | form string | no | 默认 `Teaching keyword embedding` |
| `chunk_size` | form int | no | 默认 `350` |
| `chunk_overlap` | form int | no | 默认 `50` |

响应字段与 `POST /documents` 相同：

```text
document_id
embedding_mode
collection_name
chunk_count
stored_chunk_count
```

说明：

- 当前接受 `.txt`、`.md`、`.pdf` 和 `.docx`。
- 当前按 `utf-8`、`gbk` 顺序尝试解码。
- PDF / Word 只支持可提取文本，不支持扫描版 PDF OCR。

## POST /qa

用途：基于已入库的 `collection_name` 检索相关 chunks，并返回可供 LLM 使用的上下文。

请求体：

```json
{
  "collection_name": "uploaded_document_chunks_keyword_v3_xxxxxxxxxxxx",
  "question": "RAG 的基本流程是什么？",
  "embedding_mode": "Teaching keyword embedding",
  "top_k": 3,
  "retrieval_mode": "vector"
}
```

`retrieval_mode` 支持：

```text
vector
bm25
rrf
rerank
```

响应字段：

```text
question
embedding_mode
collection_name
retrieval_mode
top_k
retrieved_chunks
context
```

`retrieved_chunks` 中的分数字段按模式不同而不同：

| Mode | Score field | 含义 |
|---|---|---|
| `vector` | `distance` | Chroma cosine distance，越低越相似 |
| `bm25` | `score` | BM25 关键词得分，越高越相关 |
| `rrf` | `rrf_score` | RRF 融合排序得分，越高排序越靠前 |
| `rerank` | `rerank_score` | 本地二阶段重排得分，越高排序越靠前 |

## POST /answer

用途：先按指定 `retrieval_mode` 检索相关 chunks，再调用 DeepSeek OpenAI-compatible API 生成最终回答。

请求体：

```json
{
  "collection_name": "uploaded_document_chunks_keyword_v3_xxxxxxxxxxxx",
  "question": "RAG 的基本流程是什么？",
  "embedding_mode": "Teaching keyword embedding",
  "top_k": 3,
  "retrieval_mode": "rrf"
}
```

响应字段：

```text
question
embedding_mode
collection_name
retrieval_mode
top_k
retrieved_chunks
context
answer
sources
```

说明：

- `answer` 由 DeepSeek 基于 `context` 生成。
- `sources` 记录本次回答使用的 chunk 编号。
- 如果没有设置 `DEEPSEEK_API_KEY`，接口返回 503。
- 自动化测试不调用真实 LLM API，只验证无 key 分支。

## POST /agent/ask

用途：在回答前执行轻量 Agentic RAG 流程，包括 intent routing、tool selection、retrieval/rerank、fallback/refusal 和 trace 输出。

请求体：

```json
{
  "collection_name": "uploaded_document_chunks_keyword_v3_xxxxxxxxxxxx",
  "question": "RAG 的基本流程是什么？",
  "embedding_mode": "Teaching keyword embedding",
  "top_k": 3,
  "retrieval_mode": "auto",
  "use_llm": false,
  "conversation_history": [
    {
      "question": "什么是 embedding？",
      "answer": "Embedding 是把文本转换成向量。"
    }
  ]
}
```

`retrieval_mode` 支持：

```text
auto
vector
bm25
rrf
rerank
```

响应字段：

```text
question
effective_question
intent
embedding_mode
collection_name
retrieval_mode
selected_tools
top_k
retrieved_chunks
context
sources
final_answer
trace
confidence
fallback_reason
```

当前 intent 包括：

```text
document_qa
summary
compare
metadata_query
out_of_scope
```

说明：

- `auto` 模式下，普通文档问答默认选择 `rerank`，总结类问题选择 `scan`，来源/片段类问题倾向 `bm25`。
- `trace` 记录 `classify_intent`、`plan`、`retrieve`、`answer/refuse` 等执行步骤。
- `conversation_history` 可用于轻量追问改写，例如把“它有什么作用？”结合上一轮问题重写成更完整的问题。
- `use_llm=false` 时接口返回基于上下文的本地回答，便于测试和无 API Key 环境演示。
- 文档外问题会触发拒答，返回 `fallback_reason=out_of_scope`。

## POST /agent/evaluation

用途：评估 Agentic RAG 行为是否可靠，不只看检索命中，还检查 intent、tool selection、refusal 和 source citation。

请求体：

```json
{
  "text": "# Python 学习 FAQ\n\n## RAG 的基本流程是什么？\nRAG 会先检索资料，再让 LLM 根据资料回答。",
  "embedding_mode": "Teaching keyword embedding",
  "chunk_size": 350,
  "chunk_overlap": 50,
  "top_k": 3
}
```

响应字段：

```text
embedding_mode
chunk_count
case_count
intent_accuracy
tool_selection_accuracy
refusal_accuracy
source_citation_rate
rows
failure_cases
```

当前默认 Agent evaluation cases 覆盖：

```text
document_qa
summary
compare
metadata_query
out_of_scope
```

当前 FAQ 样例验证结果：

| Metric | Value |
|---|---:|
| intent_accuracy | 1.0 |
| tool_selection_accuracy | 1.0 |
| refusal_accuracy | 1.0 |
| source_citation_rate | 1.0 |

## POST /evaluation

用途：用固定 15 题评测当前文档切分和检索模式的召回效果。

如果请求体传入 `evaluation_cases`，接口会使用自定义问题集；如果不传，使用默认 FAQ 15 题。

请求体：

```json
{
  "text": "# Python 学习 FAQ\n\n## RAG 的基本流程是什么？\nRAG 会先检索资料，再让 LLM 根据资料回答。",
  "embedding_mode": "Teaching keyword embedding",
  "chunk_size": 350,
  "chunk_overlap": 50,
  "top_k": 3,
  "retrieval_mode": "rrf",
  "evaluation_cases": [
    {
      "question": "RRF 混合检索有什么作用？",
      "expected_top_chunk": 4
    }
  ]
}
```

响应字段：

```text
embedding_mode
retrieval_mode
chunk_count
case_count
top_1_hit_rate
top_k_recall
rows
failure_cases
```

`evaluation_cases` 字段说明：

| Field | Type | Required | 说明 |
|---|---|---|---|
| `question` | string | yes | 评测问题 |
| `expected_top_chunk` | int | yes | 期望排第一的 chunk 编号 |

当前 FAQ 样例文档的验证结果：

| Retrieval mode | Top-1 hit | Top-k recall |
|---|---:|---:|
| `vector` | 0.733 | 1.0 |
| `bm25` | 0.867 | 0.933 |
| `rrf` | 0.8 | 0.933 |
| `rerank` | 0.867 | 1.0 |

结论：在当前小型 FAQ 测试集上，BM25 和 rerank 的 Top-1 表现更好，rerank 同时保持 100% Top-k recall；但这只是 15 题固定评测结果，不能直接推断到所有文档场景。

## 错误处理

当前已覆盖的错误分支：

| 场景 | 状态码 |
|---|---:|
| 不支持的 embedding mode | 400 |
| 不支持的 retrieval mode | 400 |
| `/documents/upload` 不支持的文件后缀 | 400 |
| `/documents/upload` 文件解析失败 | 400 |
| 空文档 | 400 |
| 文档没有可入库 chunk | 400 |
| 查询不存在的 collection | 404 |
| `/answer` 缺少 `DEEPSEEK_API_KEY` | 503 |
| `/answer` LLM quota / rate limit | 429 |
| `/answer` LLM 请求失败 | 502 |
| `/agent/ask` 不支持的 retrieval mode | 400 |
| `/agent/ask` 查询不存在的 collection | 404 |
| `/agent/ask` LLM quota / request failed | 429 / 502 |
| `/agent/evaluation` 空文档或非法 chunk 配置 | 400 |

## 自动化验证

运行：

```powershell
python -m pytest -q
```

当前测试覆盖：

```text
POST /documents
POST /documents/upload: markdown upload, docx upload, unsupported file type, and invalid PDF
POST /qa: vector / bm25 / rrf / rerank
POST /answer: missing API key and unknown retrieval mode
POST /agent/ask: intent routing, tool selection, trace, refusal, and no-key local answer
POST /agent/ask: conversation history follow-up rewrite
POST /agent/evaluation: intent/tool/refusal/source reliability metrics
POST /evaluation: vector / bm25 / rrf / rerank
POST /evaluation: custom evaluation cases
POST /evaluation: failure cases
missing collection -> 404
unknown retrieval mode -> 400
Teaching keyword retrieval metrics
```
