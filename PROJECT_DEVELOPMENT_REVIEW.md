# DocuAsk 项目开发过程复盘

这份文档用于从头梳理 DocuAsk 的开发过程，包括每个部分是怎么构建的、为什么这样构建、做过哪些优化、踩过什么坑以及怎么解决。

## 1. 项目起点

DocuAsk 最开始不是直接做 Agentic RAG，而是从 Python 学习计划里的 API、Embedding、向量数据库、RAG 基础一步步搭出来的。

最早目标很简单：理解一个程序怎么调用模型服务，以及 AI 为什么能根据文档回答问题。所以项目先从 API 调用开始，再到手写 embedding，再到 Chroma 检索，最后逐步升级成完整的本地文档 RAG 问答系统。

最核心的主线一直是：

```text
文档 -> 切分 -> embedding -> 向量数据库 -> 检索相关片段 -> LLM 根据片段回答 -> 展示来源
```

这就是 RAG 的基本链路。

## 2. API 调用阶段

一开始尝试使用 OpenAI API，但遇到了：

```text
429 insufficient_quota
```

这个错误不是代码错误，而是 API 账户没有可用额度。后来确认 ChatGPT Plus 和 OpenAI API 额度不互通，所以项目改用 DeepSeek API。

DeepSeek API 兼容 OpenAI SDK，核心区别是：

```python
api_key=os.environ.get("DEEPSEEK_API_KEY")
base_url="https://api.deepseek.com"
```

这样设计的原因是：继续使用 OpenAI-compatible SDK 可以保留通用 API 调用方式，同时避开 OpenAI API 充值问题。

这个阶段踩过两个坑：

- API Key 不能写进代码，否则提交到 GitHub 后会泄露。
- 环境变量必须设置在启动 Streamlit 的同一个 PowerShell 里，否则页面进程读不到。

## 3. Embedding 学习阶段

最早没有直接使用真实 embedding，而是先做了教学版关键词 embedding。

例如定义几个概念维度：

```text
["python", "api", "git", "rag", "data"]
```

然后一句话里命中了几个关键词，就生成一个向量：

```text
Python 怎么读取文件并保存数据？
-> [1.0, 0.0, 0.0, 0.0, 1.0]
```

这样做的原因是：真实 embedding 是黑盒，新手很难理解；手写版虽然粗糙，但能看清楚文本如何变成向量、相似度为什么能排序。

这个阶段明确了一个关键点：余弦相似度主要看方向，不是单纯看大小。关键词重复会改变向量大小，也可能影响方向；如果两个向量方向一致，余弦相似度就接近 1。

## 4. Chroma 向量数据库阶段

手写 embedding 后，项目接入了 Chroma。

Chroma 的作用是：保存每个 chunk 的 embedding，并在用户提问时根据问题 embedding 找最相似的 chunk。

早期检索结果类似：

```text
Query: 怎么让 Python 程序调用接口获取数据？
Top 1: doc_api
Distance: 1.0000
```

后来进一步明确：Chroma 返回的是 distance，通常 distance 越低越相似。这个地方容易踩坑，因为很多人会把 score 和 distance 混着理解。

所以页面里专门加了提示：

```text
Distance is returned by Chroma. Lower distance means more similar.
```

## 5. Dify 体验阶段

中间体验过 Dify，目的是理解低代码 RAG 平台怎么组织知识库。

体验过程中发现两个问题：

- 问题里多了空格，例如 `Day 8 学了什么`，可能影响倒排索引检索效果。
- Dify 如果只用倒排索引，本质偏关键词匹配；要做 Hybrid Search，需要 embedding 模型。

DeepSeek 主要是聊天和推理模型，不提供 embedding 能力。因此项目最终没有依赖 Dify，而是自己实现可控的检索、评测和页面展示。

Dify 的价值是体验 RAG 产品形态，但最终项目需要更可控、更能展示工程能力的自实现链路。

## 6. Streamlit 页面阶段

项目后面开始做自己的本地页面，也就是根目录的 `app.py`。

Streamlit 负责用户交互：

```text
上传文档
选择 embedding mode
输入问题
展示 answer
展示 retrieved chunks
展示 sources
展示 evaluation metrics
```

选择 Streamlit 的原因是：这个项目重点不是前端工程，而是 RAG 链路。Streamlit 能快速做出可交互演示，适合本地项目、简历展示和面试讲解。

这个阶段也踩过一个小坑：第一次运行 Streamlit 会提示输入邮箱。这个不是项目要求，可以直接留空跳过。

## 7. 文档切分阶段

文档进入系统后，先由 `backend/services/chunking.py` 处理。

最早切分策略比较粗：

```text
chunk size: 500
overlap: 100
```

后来观察到一个问题：chunk 太大，一个 chunk 里会包含多个主题，检索回来后会混入无关信息；chunk 太小，又可能把完整语义切断。

所以最终优化为：

```text
优先按 Markdown 标题切分
否则按固定长度 + overlap 切分
```

这样构建的原因是：Markdown 标题天然代表主题边界，例如：

```markdown
## 什么是 embedding？
## RAG 的基本流程是什么？
```

按标题切分比盲目按字符切分更容易得到语义完整的 chunk。

## 8. 文档解析阶段

后续项目升级支持了多种文件：

```text
.txt
.md
.pdf
.docx
```

对应逻辑在 `backend/services/document_parser.py`。

增加 PDF / Word 的原因是：只支持 txt/md 会显得像学习 demo，不像真实文档问答系统。PDF 和 Word 是更常见的本地知识文档格式。

当前边界是：扫描版 PDF 暂不支持，因为它没有可提取文本层，需要 OCR。

## 9. Embedding 模式升级

系统现在支持两种 embedding：

```text
Teaching keyword embedding
BGE Chinese embedding
```

对应逻辑在 `backend/services/embeddings.py`。

教学版 embedding 的作用是解释原理，速度快、可控、可 debug。

BGE 中文 embedding 使用：

```text
BAAI/bge-small-zh-v1.5
```

它更接近真实中文语义检索，适合作为项目展示能力。

保留两种模式的原因是：教学版让原理可解释，BGE 让项目更接近真实应用。两个都保留，比单纯堆一个模型更有解释力。

## 10. Chroma 持久化设计

Chroma 逻辑在 `backend/services/retrieval.py`。

collection name 使用：

```text
embedding mode prefix + schema version + document hash
```

原因有三个：

```text
不同 embedding 维度不能混进同一个 collection
schema version 可以隔离历史实验数据
document hash 可以让同一份文档生成稳定 collection 名称
```

最后阶段修了一个重要坑：本地 Chroma/HNSW 旧索引偶尔会出现：

```text
Error creating hnsw segment reader: Nothing found on disk
```

解决方式是：如果同名 collection 已存在，重新入库时重建它，避免旧索引污染验证结果。

## 11. 检索模式升级

一开始只有 vector retrieval，也就是 Chroma 向量检索。

后来增加了：

```text
vector
bm25
rrf
rerank
```

对应文件：

```text
backend/services/bm25.py
backend/services/rrf.py
backend/services/rerank.py
```

这样设计的原因是：真实 RAG 不应该只有向量检索。向量检索擅长语义相似，但对 API Key、DeepSeek、Chroma 这种关键词明确的问题，BM25 可能更稳。

RRF 是融合方案，把 vector 和 BM25 的排序合并。rerank 是二阶段方案：先召回候选 chunk，再重新排序，让 Top-1 更准。

当前固定评测结果：

```text
vector: Top-1 73.3%, Top-k 100%
bm25:   Top-1 86.7%, Top-k 93.3%
rrf:    Top-1 80%,   Top-k 93.3%
rerank: Top-1 86.7%, Top-k 100%
```

这组数据适合简历表达，因为它不是空喊优化检索，而是有前后指标。

## 12. LLM 生成阶段

生成回答逻辑在 `backend/services/generation.py`。

它的原则是：

```text
只根据 retrieved chunks 回答
回答后附 sources
如果资料不足，不要编
```

这就是 RAG 抑制幻觉的核心：LLM 不再完全靠模型记忆，而是先拿到资料，再基于资料回答。

这里做了工程取舍：pytest 不调用真实 DeepSeek。因为真实 API 依赖网络、key 和额度，不适合放进自动化测试。测试重点放在本地可重复的检索链路。

## 13. FastAPI 后端阶段

后面项目把页面里的逻辑拆到了 backend，并加入 `backend/app.py`。

主要接口包括：

```text
GET  /health
POST /documents
POST /documents/upload
POST /qa
POST /answer
POST /evaluation
POST /agent/ask
POST /agent/evaluation
```

拆 FastAPI 的原因是：如果所有逻辑都堆在 Streamlit 里，项目看起来更像课程作业；拆成后端服务后，RAG 能力可以被页面、测试、脚本、未来前端共同复用。

这是项目从能跑升级到有工程结构的关键一步。

## 14. Evaluation 评测阶段

评测逻辑在 `backend/services/evaluation.py`。

项目设计了固定问题集，用两个指标看检索效果：

```text
Top-1 hit：第一名是不是期望 chunk
Top-k recall：前 k 个结果里是否包含期望 chunk
```

做 evaluation 的原因是：RAG 项目最容易犯的错误是页面能回答，就以为系统好用。但真正的问题是检索有没有找对资料。所以项目先评测检索，再看生成。

后面还加了 failure cases，用来记录失败问题。这样项目就有了：

```text
发现问题 -> 调整策略 -> 复测指标
```

的闭环。

## 15. Agentic RAG 阶段

最后项目升级成轻量 Agentic RAG，核心逻辑在 `backend/services/agent.py`。

它不是复杂多 Agent 框架，而是一个规则型 planner：

```text
question -> classify_intent -> choose tools -> retrieve / scan / rerank -> answer or refuse -> trace
```

当前 intent 包括：

```text
document_qa
summary
compare
metadata_query
out_of_scope
```

这样构建的原因是：普通 RAG 是来了问题就检索；Agentic RAG 多了一层决策，先判断用户到底要问文档、总结、对比、查来源，还是问了文档外问题。

这让系统更像会选择工具的问答助手，而不是单一检索函数。

Agentic 阶段还加入了：

```text
tool selection
trace
fallback/refusal
lightweight conversation history
effective_question
```

这使得面试时可以讲清楚：系统为什么这么答、用了哪些工具、为什么拒答。

## 16. Agent Evaluation

Agent 不是加了就算完，还要评测。

当前 `/agent/evaluation` 检查四项：

```text
intent_accuracy
tool_selection_accuracy
refusal_accuracy
source_citation_rate
```

当前小样本结果是：

```text
100%
100%
100%
100%
```

注意，这不能夸大成生产环境泛化能力，只能说在当前固定 Agent evaluation cases 上通过。

这个边界很重要，面试时讲出来反而更可信。

## 17. Docker 阶段

最后加入了 Docker：

```text
Dockerfile
docker-compose.yml
requirements-docker.txt
```

增加 Docker 的原因是：项目交给别人时，不能只说我电脑能跑。Docker Compose 可以证明它有可复现部署路径。

这里也做了取舍：Docker 版使用轻量依赖，默认支持 Teaching keyword embedding。BGE 依赖 sentence-transformers，镜像更重，所以 README 里明确说明：Docker 演示走轻量路径，BGE 本地完整环境运行。

最终验证结果：

```text
docker compose up --build -d
http://localhost:8501 -> 200
```

## 18. 测试阶段

测试目录在 `tests/`。

当前覆盖：

```text
后端 API
文档上传
PDF / Word 解析
检索问答
检索评测
Agentic ask
Agent evaluation
异常分支
```

最终结果：

```text
32 passed, 1 warning
```

这个测试策略是合理的：本地稳定逻辑必须测，真实 LLM 调用不放进自动化测试。

## 19. 项目最终架构

现在项目可以概括成：

```text
Streamlit UI
  -> FastAPI backend
    -> document_parser
    -> chunking
    -> embeddings
    -> Chroma retrieval
    -> BM25 / RRF / rerank
    -> Agent planner
    -> DeepSeek generation
    -> evaluation
```

一句话说就是：

```text
DocuAsk 是一个支持本地文档上传、多策略检索、Agentic 路由、来源引用和评测闭环的本地 RAG 问答系统。
```

## 20. 主要优化点

项目的主要优化包括：

- 从只会调用 API，升级成完整 RAG 链路。
- 从手写 embedding，升级到 BGE 中文 embedding。
- 从单一 vector 检索，升级到 BM25、RRF、rerank 对比。
- 从能回答，升级到可评测、可解释、可追溯。
- 从 Streamlit 单文件 demo，升级到 FastAPI + services 模块化结构。
- 从普通 RAG，升级到轻量 Agentic RAG。
- 从本地运行，升级到 Docker 可复现部署。

## 21. 主要踩坑和解决

| 问题 | 原因 | 解决方式 |
|---|---|---|
| OpenAI API `429 insufficient_quota` | API 账户没有可用额度，Plus 和 API 不互通 | 改用 DeepSeek OpenAI-compatible API |
| API Key 读不到 | 环境变量不在启动 Streamlit 的同一个 PowerShell 中 | 在同一个终端设置 `DEEPSEEK_API_KEY` 后再启动 |
| DeepSeek key invalid | 使用了旧 key 或复制错误 | 更换新 key 并重新启动应用 |
| Dify 检索不稳定 | 倒排索引偏关键词匹配，对空格和表述敏感 | 自己实现可控检索链路 |
| chunk 粒度不合理 | 太大会混入无关信息，太小会切断语义 | 优先 Markdown 标题切分，否则固定长度加 overlap |
| Chroma distance 容易误解 | distance 不是普通 score | 页面提示 lower distance means more similar |
| embedding 维度混用风险 | 不同 embedding 模型维度不同 | 按 embedding mode 分 collection |
| 历史索引污染 | 本地 Chroma/HNSW 旧状态异常 | 重新入库时重建同名 collection |
| Docker 依赖过重 | BGE 依赖 sentence-transformers，镜像较重 | 拆出 `requirements-docker.txt` 走轻量演示路径 |
| 测试依赖真实 API 不稳定 | 网络、额度和 key 都会影响测试 | pytest 只测本地可重复逻辑 |

## 22. 面试推荐讲法

不要从我用了哪些库开始讲，而是从问题开始：

```text
我做这个项目时想解决三个问题：
第一，本地文档问答需要来源可追溯；
第二，RAG 检索质量不能只靠肉眼看，需要指标评测；
第三，普通 RAG 缺少问题类型判断，所以我加了轻量 Agentic planner。
```

然后讲实现：

```text
文档上传后先解析和切分，使用 BGE 或教学版 embedding 写入 Chroma。
用户提问后可以选择 vector、BM25、RRF 或 rerank 检索。
Agentic 模式会先做 intent routing 和 tool selection，再决定检索、总结、对比、引用或拒答。
最后把 retrieved chunks 交给 DeepSeek 生成回答，并返回 sources。
```

最后讲优化数据：

```text
在固定 FAQ 评测集中，vector Top-1 是 73.3%，rerank 提升到 86.7%，Top-k recall 保持 100%。
Agent evaluation 中 intent、工具选择、拒答、来源引用四项小样本指标均为 100%。
```

这就是 DocuAsk 从学习练习变成简历项目的完整过程。
