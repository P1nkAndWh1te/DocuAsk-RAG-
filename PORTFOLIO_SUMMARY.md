# DocuAsk Portfolio Summary

## 一句话定位

DocuAsk 是一个面向本地文档问答场景的 Agentic RAG 系统，覆盖文档解析、chunking、向量入库、多策略检索、rerank、intent routing、tool selection、拒答、来源引用、评测和 Docker 复现运行。

## 简历版项目描述

**DocuAsk｜本地文档 Agentic RAG 问答系统**  
**技术栈：** Python / FastAPI / Streamlit / Chroma / BGE / BM25 / RRF / Rerank / DeepSeek API / pypdf / python-docx / Docker / pytest

面向本地知识文档问答场景，设计并重构 Agentic RAG 链路，支持 `.txt/.md/.pdf/.docx` 文档上传、文本切分、Chroma 向量入库、多策略检索、LLM 回答生成和来源引用，提升回答可追溯性、检索可评测性和部署可复现性。

- 拆分 `chunking`、`embedding`、`retrieval`、`generation`、`evaluation`、`agent`、`document_parser`、`rerank` 等核心模块，并通过 FastAPI 暴露 `/documents`、`/qa`、`/answer`、`/agent/ask`、`/agent/evaluation` 等接口，提升 RAG 逻辑复用性和可测试性。
- 构建 Chroma 本地持久化知识库，按 embedding mode、schema version 和 document hash 管理 collection，并在重新入库时重建同名 collection，避免索引重启丢失、不同 embedding 维度混用和本地旧索引状态污染。
- 设计 vector、BM25、RRF、rerank 四种检索模式；其中 rerank 先召回候选 chunk，再做本地轻量重排。当前固定评测中，Top-1 hit 从 vector 的 `73.3%` 提升到 rerank 的 `86.7%`，Top-k recall 保持 `100%`。
- 引入轻量 Agentic planner，支持 intent routing、tool selection、trace、fallback/refusal 和 conversation history 追问改写；通过 `/agent/evaluation` 评估 intent、工具选择、拒答和来源引用，当前小样本评测四项指标均为 `100%`。
- 使用 pytest 覆盖文档入库、文件上传、PDF/Word 解析、检索问答、Agentic ask、Agent evaluation、回答生成、异常分支和检索指标；新增 Docker Compose 配置和轻量 Docker 依赖，提升项目可复现部署能力。

## 面试讲解版本

这个项目不是简单调用大模型 API，而是围绕本地文档问答实现了一条完整的 RAG 和 Agentic RAG 链路。文档上传后会先解析、切分成 chunk，再根据 embedding mode 写入 Chroma。用户提问时，普通 RAG 模式可以选择 vector、BM25、RRF 或 rerank；Agentic 模式会先判断问题 intent，再选择检索、扫描、rerank、引用或拒答等工具，并把每一步记录到 trace 中。

我重点做了三类工程化工作：

1. 把页面里的 RAG 逻辑拆成可复用后端模块，并通过 FastAPI 暴露接口。
2. 用 Top-1 hit、Top-k recall、failure cases 和 Agent reliability metrics 验证系统行为。
3. 增加 Docker、pytest、API 文档和架构文档，让项目可以被别人复现和审查。

当前项目仍然是本地原型，不夸大为生产系统。它还不支持扫描版 PDF OCR、多用户权限、复杂多 Agent Runtime 和大规模线上压测。

## 可展示证据

- `README.md`：项目入口、运行方式、核心指标。
- `ARCHITECTURE.md`：架构、服务边界、Agentic RAG 流程。
- `API.md`：FastAPI 接口说明。
- `PROJECT_BRIEF.md`：项目展示版说明。
- `PROJECT_DETAILED_GUIDE.md`：完整项目讲解。
- `tests/`：自动化测试。
- `Dockerfile` / `docker-compose.yml`：本地可复现运行。

## 推荐演示路径

1. 运行 `python -m streamlit run app.py` 或 `docker compose up --build`。
2. 上传 `examples/rag_faq.md`。
3. 观察 Retrieval evaluation 和 Agent evaluation 指标。
4. 提问 `RAG 的基本流程是什么？`，展示来源 chunk 和 Agent trace。
5. 提问 `今天上海天气怎么样？`，展示 out-of-scope 拒答。
6. 先问 `什么是 embedding？`，再追问 `它在 RAG 中起什么作用？`，展示 `effective_question` 追问改写。
