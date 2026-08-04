# DocuAsk OnCall Agent 升级报告

本文档记录 DocuAsk 从“本地文档 Agentic RAG”升级到“运维知识库 + OnCall 诊断 Agent 原型”的过程。重点说明每一步做了什么、为什么做、用到了什么、怎么实现、如何验收，以及当前边界。

## 1. 升级目标

原项目 DocuAsk 已经具备：

```text
文档上传 -> 文档切分 -> embedding -> Chroma 入库 -> 多策略检索 -> Agentic RAG -> 来源引用 -> 评测
```

本次升级的目标是把它朝 AI OnCall Agent 方向推进，但不直接重写项目。

新的目标链路是：

```text
结构化告警 -> mock 指标/日志/历史事件 -> runbook 检索 -> 根因排序 -> 安全处理建议 -> trace -> evaluation
```

这样做的原因是：OnCall Agent 的核心不是单纯聊天，而是围绕告警收集证据、选择工具、检索 runbook、判断可能原因，并给出有边界的处理建议。

## 2. 阶段一：运维知识库 RAG

### 做了什么

新增运维 runbook 文档：

```text
examples/oncall_runbook.md
```

内容覆盖：

```text
API 5xx 错误率升高
Redis 连接超时
MySQL 慢查询
CPU 使用率持续升高
磁盘空间不足
告警处理通用原则
```

### 为什么这样做

OnCall Agent 不能一开始就诊断告警，它必须先有可检索的运维知识来源。runbook 是 OnCall 场景中最核心的知识载体，里面包含排查步骤、安全边界和处理原则。

### 用到了什么

- Markdown 文档
- 原有 `chunking.py`
- 原有 `retrieval.py`
- 原有 `BM25 / RRF / rerank`
- 原有 `Streamlit file_uploader`

### 怎么做的

把常见告警拆成 Markdown 二级标题，每个标题下面写：

```text
告警现象
排查步骤
安全边界
```

这样可以复用原来的 Markdown 标题切分逻辑，让每类告警尽量成为独立 chunk。

### 验收方式

上传：

```text
examples/oncall_runbook.md
```

测试问题：

```text
API 5xx 错误率升高应该怎么排查？
Redis 连接超时要看哪些指标？
MySQL 慢查询告警怎么定位？
CPU 使用率持续升高可能有哪些原因？
磁盘空间不足应该先检查什么？
OnCall 处理告警的通用原则是什么？
```

系统应检索到对应的 OnCall runbook chunk。

## 3. 阶段二：OnCall 专属检索评测

### 做了什么

在 `backend/services/evaluation.py` 中新增：

```text
ONCALL_EVALUATION_CASES
select_evaluation_cases()
```

页面新增：

```text
Evaluation set:
Auto
DocuAsk FAQ
OnCall runbook
```

### 为什么这样做

原来的 evaluation cases 是 Python/RAG FAQ 的问题。如果上传 OnCall 文档还继续用 FAQ 问题评测，指标没有意义。

因此需要根据文档类型选择不同评测集。

### 用到了什么

- 原有 `evaluate_retrieval`
- 原有 `calculate_hit_rate`
- 原有 `calculate_top_k_hit_rate`
- Streamlit `selectbox`

### 怎么做的

`select_evaluation_cases()` 会根据用户选择或文档内容判断使用哪个评测集：

```text
Auto -> 检测 oncall / runbook / 告警
DocuAsk FAQ -> 使用原 FAQ 评测集
OnCall runbook -> 使用 OnCall 评测集
```

### 踩坑和解决

一开始预期 OnCall 文档会按标题切成 7 个 chunk，但实际部分章节超过 350 字，被二次切分成 9 个 chunk。

解决方式：先打印实际 chunk 编号，再修正 OnCall evaluation cases 的 `expected_top_chunk`，不凭空猜测。

### 验收结果

新增测试验证：

```text
OnCall 文档会自动选择 OnCall evaluation cases
OnCall runbook 在 rerank 模式下 Top-1 命中稳定
```

## 4. 阶段三：结构化告警输入

### 做了什么

新增结构化告警字段：

```text
service
alert_name
severity
metric
value
duration
```

对应示例：

```json
{
  "service": "order-service",
  "alert_name": "HighErrorRate",
  "severity": "P1",
  "metric": "http_5xx_rate",
  "value": "12%",
  "duration": "5m"
}
```

### 为什么这样做

真实 OnCall 场景中，输入通常不是一句自然语言问题，而是 Alertmanager、Prometheus 或监控平台产生的结构化告警。结构化字段能让 Agent 明确知道：

```text
哪个服务出问题
什么告警
什么级别
哪个指标异常
当前值是多少
持续多久
```

### 用到了什么

- FastAPI
- Pydantic BaseModel
- 本地 dict 数据结构
- Streamlit text_input / selectbox

### 怎么做的

后端新增 `OnCallAlert` 模型，页面新增结构化输入表单。Agent 收到告警后先执行：

```text
parse_alert
```

并把结构化字段写入 trace。

## 5. 阶段四：Mock 可观测数据工具

### 做了什么

新增：

```text
backend/services/oncall.py
```

其中实现本地 mock 工具：

```text
query_metrics(service)
query_logs(service)
query_incidents(service)
```

### 为什么这样做

如果现在直接接 Prometheus、Loki、Alertmanager，项目复杂度会快速上升，还需要真实服务、指标和日志环境。

本阶段先用 mock 数据验证 Agent 工具编排和诊断流程，保留后续替换为真实工具的接口形状。

### 用到了什么

- Python dict / list
- 本地工具函数
- Agent trace

### 怎么做的

在 `oncall.py` 中定义：

```text
MOCK_METRICS
MOCK_LOGS
MOCK_INCIDENTS
```

不同服务对应不同证据，例如：

```text
order-service -> http_5xx_rate, latency_p95, deploy log, database unavailable
cache-client -> Redis timeout, pool exhausted
worker-service -> disk_usage, debug log, app.log growth
```

Agent 在诊断时依次调用：

```text
query_metrics
query_logs
query_incidents
retrieve_runbook
rank_possible_causes
```

每一步都会写入 trace。

## 6. 阶段五：Runbook 检索和根因排序

### 做了什么

结构化告警会被转换成 runbook 检索问题：

```text
HighErrorRate 告警怎么排查？服务 order-service 指标 http_5xx_rate 当前值 12% 持续 5m
```

然后复用原有检索链路：

```text
retrieve_for_mode(..., retrieval_mode="rerank")
```

同时新增本地原因排序：

```text
recent_deploy_or_dependency_failure
redis_pool_or_server_pressure
missing_index_or_full_scan
retry_storm_or_cpu_hotspot
log_or_temp_file_growth
```

### 为什么这样做

OnCall 诊断不能只返回 runbook 原文，还需要把告警字段、指标、日志和历史事件组合起来，给出可能原因排序。

### 用到了什么

- 原有 RRF / rerank 检索
- runbook chunks
- 本地规则打分
- mock 指标/日志/历史事件

### 怎么做的

`rank_possible_causes()` 会把告警字段、指标值、日志和历史事件合成证据文本，再根据关键词信号计分。

例如：

```text
deploy + 5xx + database unavailable -> recent_deploy_or_dependency_failure
redis + pool exhausted + timeout -> redis_pool_or_server_pressure
full table scan + index + slow query -> missing_index_or_full_scan
```

### 当前边界

当前原因排序是本地规则，不是机器学习模型，也不是外部 AIOps 根因分析系统。

准确说法：

```text
当前实现的是本地可解释 cause ranker，用于验证 OnCall Agent 的诊断闭环。
```

## 7. 阶段六：OnCall API

### 做了什么

新增两个接口：

```text
POST /oncall/diagnose
POST /oncall/evaluation
```

### 为什么这样做

页面只是演示层，真正的能力应该沉到 FastAPI 后端，方便测试、复用和后续接入真实告警平台。

### 用到了什么

- FastAPI
- Pydantic
- 原有 chunking / retrieval
- 新增 oncall service

### 怎么做的

`/oncall/diagnose` 输入：

```text
runbook text + structured alert
```

输出：

```text
selected_tools
metrics
logs
incidents
retrieved_chunks
sources
possible_causes
primary_cause
confidence
final_answer
trace
```

`/oncall/evaluation` 输出：

```text
root_cause_hit_rate
tool_selection_accuracy
evidence_citation_rate
safe_action_rate
```

## 8. 阶段七：Streamlit OnCall 诊断实验区

### 做了什么

页面新增：

```text
OnCall diagnosis lab
Structured alert input
OnCall evaluation
```

### 为什么这样做

面试演示时不能只展示 API。页面里直接输入告警字段、点击 Diagnose alert，更容易让面试官看到系统从“文档问答”升级到了“告警诊断”。

### 用到了什么

- Streamlit `text_input`
- Streamlit `selectbox`
- Streamlit `button`
- Streamlit `metric`
- Streamlit `json`
- Streamlit `dataframe`

### 怎么演示

上传：

```text
examples/oncall_runbook.md
```

左侧选择：

```text
Embedding mode: Teaching keyword embedding
Retrieval mode: rerank
Evaluation set: Auto
Agentic RAG mode: enabled
Agent retrieval: auto
```

在 OnCall diagnosis lab 里输入：

```text
Service: order-service
Alert name: HighErrorRate
Severity: P1
Metric: http_5xx_rate
Value: 12%
Duration: 5m
```

点击：

```text
Diagnose alert
```

页面应展示：

```text
Primary cause
Confidence
Sources
Selected tools
Mock evidence
OnCall trace
Runbook chunks
OnCall evaluation metrics
```

## 9. 阶段八：自动化测试

### 做了什么

新增测试覆盖：

```text
OnCall runbook evaluation cases auto selection
OnCall runbook retrieval metrics
POST /oncall/diagnose
POST /oncall/evaluation
```

### 为什么这样做

项目升级不能只靠页面肉眼看。后端诊断链路、工具选择、来源引用、安全建议都需要测试证明。

### 验收结果

当前测试结果：

```text
36 passed, 1 warning
```

## 10. 当前能力总结

当前 DocuAsk 已经从普通本地文档 RAG 升级为：

```text
本地文档 Agentic RAG + 运维 runbook OnCall 诊断 Agent 原型
```

已支持：

```text
txt / md / pdf / docx 上传
Markdown / fixed chunking
Teaching keyword embedding
BGE Chinese embedding
Chroma 向量检索
BM25
RRF
rerank
Agentic RAG intent routing
tool selection
trace
fallback / refusal
conversation history
OnCall structured alert input
mock metrics
mock logs
mock incidents
runbook retrieval
local cause ranking
safe action suggestion
retrieval evaluation
agent evaluation
oncall evaluation
FastAPI
Streamlit
pytest
Docker
```

## 11. 当前限制

当前不能夸大为生产级 OnCall 平台。

仍然不支持：

```text
真实 Prometheus 查询
真实 Loki 日志查询
真实 Alertmanager Webhook
真实工单系统
自动执行修复动作
多用户权限
复杂多 Agent Runtime
大规模故障评测集
外部 root cause analysis 模型
```

准确说法是：

```text
我实现的是一个本地 OnCall Agent 原型，用 mock 可观测数据和 runbook 检索验证告警诊断流程，重点展示工具编排、证据引用、根因排序、安全边界和可评测性。
```

## 12. 面试推荐讲法

可以这样讲：

```text
DocuAsk 最初是一个本地文档 Agentic RAG 系统，支持文档解析、chunking、embedding、Chroma、BM25/RRF/rerank、来源引用和评测。

后续我把它升级到 OnCall 场景：用户上传运维 runbook 后，可以输入结构化告警字段，例如 service、alert_name、severity、metric、value 和 duration。系统会先查询 mock 指标、日志和历史事件，再检索 runbook，最后通过本地 cause ranker 给出可能根因、证据来源、处理建议和 trace。

我没有把它包装成生产级 AIOps 系统。当前阶段重点是验证 OnCall Agent 的诊断闭环和可评测性。评测指标包括 root cause hit rate、tool selection accuracy、evidence citation rate 和 safe action rate。
```

## 13. 下一步建议

下一阶段可以继续做：

```text
1. 接入 Alertmanager webhook 格式。
2. 用 FastAPI 增加 /oncall/webhook 接口。
3. 把 mock metrics 替换成 Prometheus HTTP API 查询。
4. 把 mock logs 替换成 Loki 查询。
5. 扩大故障案例集，增加失败案例记录。
6. 增加告警状态流转：new -> investigating -> mitigated -> escalated。
```

推荐优先级：

```text
Alertmanager webhook mock -> Prometheus mock API shape -> Loki mock API shape -> 扩大 evaluation cases
```

原因是：继续保持“先跑通闭环，再替换真实外部系统”的节奏，风险最低，也最容易解释清楚。
