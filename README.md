# DocuAsk

DocuAsk is a local Agentic RAG document QA system built around a Retrieval-Augmented Generation pipeline:

```text
document upload -> chunking -> embedding -> intent routing -> tool selection -> retrieval/rerank -> answer/fallback -> source citations
```

The application uses Streamlit for the UI, FastAPI for reusable backend endpoints, Chroma for local vector storage, BGE for Chinese embedding, BM25/RRF/rerank for retrieval comparison, a lightweight Agentic RAG planner for intent routing and tool selection, and DeepSeek through an OpenAI-compatible LLM API.

## Highlights

- Upload `.txt`, `.md`, `.pdf`, and `.docx` documents.
- Split documents by Markdown sections or fixed-size chunks with overlap.
- Store document chunks in a local Chroma persistent collection.
- Retrieve chunks with vector search, BM25, RRF ranking fusion, or a local rerank stage.
- Route questions through Agentic RAG intent classification, tool selection, trace logging, and fallback/refusal logic.
- Support lightweight conversation history for follow-up questions.
- Generate grounded answers with source chunk citations.
- Evaluate retrieval quality with Top-1 hit and Top-k recall.
- Evaluate Agentic behavior with intent accuracy, tool selection accuracy, refusal accuracy, and source citation rate.
- Expose reusable FastAPI endpoints for ingestion, QA, answer generation, and evaluation.
- Record retrieval failure cases during evaluation.
- Cover core API behavior, document parsing, retrieval metrics, and error branches with pytest.
- Run with Docker Compose for reproducible local deployment.

## Project Materials

- 项目展示说明：`PROJECT_BRIEF.md`
- 项目详细介绍：`PROJECT_DETAILED_GUIDE.md`
- 简历与面试摘要：`PORTFOLIO_SUMMARY.md`
- 交付验收清单：`DELIVERY_CHECKLIST.md`
- API 文档：`API.md`
- 架构说明：`ARCHITECTURE.md`

## 技术栈

- Python
- Streamlit
- Chroma
- Sentence Transformers
- OpenAI Python SDK
- DeepSeek API
- FastAPI
- Uvicorn
- pytest
- python-multipart
- pypdf
- python-docx
- BM25
- RRF

## 运行方式

在仓库根目录运行：

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

打开：

```text
http://localhost:8501
```

Docker 方式：

```powershell
docker compose up --build
```

Docker uses `requirements-docker.txt` for a lighter reproducible demo path. It supports the default Teaching keyword embedding mode. For BGE mode inside Docker, install the full `requirements.txt` dependencies because BGE requires `sentence-transformers`.

## 验证方式

```powershell
python -m pytest -q
```

也可以运行：

```powershell
.\scripts\validate.ps1
```

当前验证结果：

```text
32 passed, 1 warning
```

## DeepSeek API Key

生成回答需要在启动 Streamlit 的同一个 PowerShell 中设置环境变量：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
python -m streamlit run app.py
```

注意：

- 不要把 API Key 写进代码。
- 不要把 API Key 提交到 GitHub。
- 如果页面提示 key 未设置，先确认启动 Streamlit 的当前终端能读到该环境变量。

检查命令：

```powershell
python -c "import os; print('set' if os.environ.get('DEEPSEEK_API_KEY') else 'missing')"
```

## 测试文档

推荐使用：

```text
examples/rag_faq.md
```

可测试问题：

```text
RAG 的基本流程是什么？
API Key 为什么不能写进代码？
什么是 embedding？
向量数据库有什么作用？
DeepSeek API 怎么配置？
总结这份文档
今天上海天气怎么样？
```

## 检索评测结果

当前 FAQ 样例文档的检索结果：

```text
RAG 的基本流程是什么？        -> Chunk 6
API Key 为什么不能写进代码？  -> Chunk 3
什么是 embedding？            -> Chunk 4
向量数据库有什么作用？        -> Chunk 5
DeepSeek API 怎么配置？       -> Chunk 2
```

## Embedding 模式

当前支持两种模式：

```text
Teaching keyword embedding
BGE Chinese embedding
```

教学版关键词 embedding 适合理解向量检索原理，速度快、可解释。

BGE 中文 embedding 使用 `BAAI/bge-small-zh-v1.5`，适合验证真实中文语义检索效果。首次运行可能需要加载本地模型。

当前 FAQ 的评测结果：

```text
Vector retrieval: Top-1 73.3%, Top-k 100%
BM25 retrieval: Top-1 86.7%, Top-k 93.3%
RRF retrieval: Top-1 80%, Top-k 93.3%
Rerank retrieval: Top-1 86.7%, Top-k 100%
```

Agentic RAG reliability checks:

```text
Intent accuracy: 100%
Tool selection accuracy: 100%
Refusal accuracy: 100%
Source citation rate: 100%
```

## 当前限制

- PDF / Word 解析依赖文档本身是否包含可提取文本，扫描版 PDF 仍无法识别。
- Chroma 持久化目录是本地运行数据，不提交到 GitHub。
- BGE 模型首次加载需要等待。
- 当前 rerank 是本地轻量重排，不是外部 cross-encoder rerank 模型。
- 当前 Agentic RAG planner 是规则型轻量 planner，不是多 Agent 框架。
- BGE 首次加载会比教学版关键词 embedding 慢。
- 自动化测试不调用真实 LLM API，避免依赖 API Key、网络和额度。

## 下一步

- 扩大真实文档评测集。
- 评估是否接入外部 cross-encoder rerank 模型。
- 增加 OCR 以支持扫描版 PDF。
