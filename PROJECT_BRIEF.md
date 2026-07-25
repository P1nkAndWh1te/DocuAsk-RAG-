# DocuAsk Agentic RAG 项目展示说明

## 项目定位

DocuAsk 是一个本地文档 Agentic RAG 问答原型，用于验证 RAG 与轻量 Agentic 执行链路：

```text
文档上传 -> 文档切分 -> embedding -> intent routing -> tool selection -> 检索 / rerank -> 回答 / 拒答 -> 来源引用
```

项目重点不是做一个复杂平台，而是用最小可运行系统证明：

- 能从用户上传文档中检索相关片段。
- 能把检索上下文交给 LLM 生成回答。
- 能展示来源 chunk，降低幻觉风险。
- 能用固定问题集评估检索质量。
- 能通过 Agent trace 解释 intent、工具选择、检索和 fallback/refusal 过程。

## 核心功能

- TXT / Markdown / PDF / Word 文档上传。
- Markdown `##` 标题优先切分。
- Chroma 向量检索。
- Chroma 本地持久化存储。
- FastAPI 文档入库、检索问答和检索评测接口。
- FastAPI multipart 文件上传接口，支持 `.txt` / `.md` / `.pdf` / `.docx`。
- FastAPI answer 接口，支持基于检索上下文生成最终回答。
- FastAPI `/agent/ask` 接口，支持 intent routing、tool selection、trace 和拒答边界。
- FastAPI `/agent/evaluation` 接口，评估 intent、工具选择、拒答和来源引用。
- 双 embedding 模式：
  - `Teaching keyword embedding`
  - `BGE Chinese embedding`
- `vector`、`bm25`、`rrf`、`rerank` 四种检索模式。
- Agentic RAG 模式，支持 `document_qa`、`summary`、`compare`、`metadata_query`、`out_of_scope` intent。
- 轻量 conversation history，用于处理“它/这个/刚才”等追问。
- DeepSeek OpenAI-compatible LLM API 生成回答。
- 来源 chunk 展示。
- 15 题固定检索评测。
- 自定义 evaluation cases 和多文档小样本评测。
- failure cases 记录，用于定位 Top-1 排序失败和 Top-k 未召回问题。
- Top-1 hit 和 Top-k recall 指标展示。
- pytest 自动化测试覆盖核心接口和检索指标。

## 技术栈

| Layer | Choice |
|---|---|
| UI | Streamlit |
| Vector DB | Chroma |
| Backend API | FastAPI / Uvicorn |
| File upload | python-multipart |
| PDF / Word parsing | pypdf / python-docx |
| Teaching embedding | 手写关键词向量 |
| Real embedding | BAAI/bge-small-zh-v1.5 |
| Retrieval | Vector / BM25 / RRF / Rerank |
| Agentic layer | Intent routing / Tool selection / Trace / Fallback |
| Agent evaluation | Intent accuracy / Tool accuracy / Refusal accuracy / Source citation rate |
| LLM API | DeepSeek OpenAI-compatible API |
| SDK | OpenAI Python SDK |
| Test | pytest |

## 当前评测结果

测试文档：

```text
examples/rag_faq.md
```

评测问题：15 个固定问题。

| Retrieval mode | Top-1 hit | Top-k recall |
|---|---:|---:|
| Vector retrieval | 73.3% | 100% |
| BM25 retrieval | 86.7% | 93.3% |
| RRF retrieval | 80% | 93.3% |
| Rerank retrieval | 86.7% | 100% |

结论：

```text
Rerank 在当前 FAQ 样例上保持 100% Top-k recall，并把 Top-1 hit 提升到 86.7%。
failure cases 可用于继续分析排序失败原因。
```

Agentic 行为评测：

| Metric | Value |
|---|---:|
| Intent accuracy | 100% |
| Tool selection accuracy | 100% |
| Refusal accuracy | 100% |
| Source citation rate | 100% |

## 关键工程决策

### 1. API Key 使用环境变量

DeepSeek API Key 不写入代码，而是从环境变量读取：

```text
DEEPSEEK_API_KEY
```

这样可以避免把密钥提交到 GitHub。

### 2. 保留教学版 embedding

教学版关键词 embedding 不是真实语义 embedding，但它便于解释向量检索的基本原理。

所以项目没有直接删除它，而是保留为 baseline。

### 3. BGE 与关键词 embedding 分 collection

教学版关键词向量是 12 维，BGE 向量是 512 维。

不同维度不能写入同一个 Chroma collection，因此主应用使用不同 collection：

```text
uploaded_document_chunks_keyword
uploaded_document_chunks_bge
```

### 4. 评测分 Top-1 和 Top-k

Top-1 hit 评估第一名排序是否正确。

Top-k recall 评估正确资料是否进入候选集合。

这比单一命中率更能解释 RAG 检索问题。

### 5. Rerank 作为二阶段排序

`rerank` 模式先使用 RRF 召回候选 chunk，再用本地轻量重排逻辑重新排序。

它的作用不是替代向量检索或 BM25，而是在候选集已经召回正确资料后，提高第一名排序质量。

### 6. Agentic RAG 采用轻量规则型 planner

第一版 Agentic RAG 没有直接引入复杂多 Agent 框架，而是先实现可测试、可解释的规则型 planner：

```text
classify_intent -> select_tools -> retrieve / scan / rerank -> answer / refuse -> trace
```

这样可以稳定验证 Agent 行为，并通过 `trace` 展示每一步为什么执行。

### 7. Agent evaluation 验证行为可靠性

阶段二增加 `/agent/evaluation`，不只看检索命中，还检查：

- intent 是否判断正确。
- retrieval / scan / rerank 工具是否选对。
- 文档外问题是否拒答。
- 可回答问题是否返回来源。

## 如何运行

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Docker 方式：

```powershell
docker compose up --build
```

Docker 默认使用轻量 `requirements-docker.txt`，适合快速复现 Teaching keyword embedding 演示；如需在 Docker 内使用 BGE，需要安装完整依赖。

如需 LLM 生成回答：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
python -m streamlit run app.py
```

## 可展示点

- 上传 FAQ 文档后，页面会展示文档切分结果。
- 侧边栏可以切换 embedding 模式。
- Retrieval evaluation 区域显示 Top-1 hit 和 Top-k recall。
- Agentic RAG 模式下可以看到 intent、selected tools、trace、confidence、fallback reason、effective question、来源 chunk 和最终回答。
- Agent evaluation 区域显示 intent accuracy、tool selection accuracy、refusal accuracy 和 source citation rate。
- BM25 和 RRF 模式可用于对比语义检索与关键词检索的排序差异。

## 当前限制

- PDF / Word 解析只支持可提取文本，不支持扫描版 PDF OCR。
- Chroma 持久化目录是本地运行数据，不提交到 GitHub。
- 当前只验证了小型 FAQ 文档。
- 当前多文档评测仍是小样本，不是大规模 benchmark。
- 当前 rerank 是本地轻量重排，不是外部 cross-encoder 模型。
- 当前 Agentic RAG 是轻量规则型 planner，不是复杂多 Agent Runtime。
- BGE 首次加载需要等待。
- 自动化测试不调用真实 LLM API，避免依赖 API Key、网络和额度。

## 下一步

- 扩大真实文档评测集。
- 评估是否接入外部 cross-encoder rerank 模型。
