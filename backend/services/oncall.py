from backend.services.embeddings import KEYWORD_EMBEDDING_MODE
from backend.services.evaluation import retrieve_for_mode
from backend.services.generation import format_sources


DEFAULT_ONCALL_EVALUATION_CASES = [
    {
        "alert": {
            "service": "order-service",
            "alert_name": "HighErrorRate",
            "severity": "P1",
            "metric": "http_5xx_rate",
            "value": "12%",
            "duration": "5m",
        },
        "expected_primary_cause": "recent_deploy_or_dependency_failure",
        "expected_tools": ["query_metrics", "query_logs", "retrieve_runbook"],
    },
    {
        "alert": {
            "service": "cache-client",
            "alert_name": "RedisTimeout",
            "severity": "P2",
            "metric": "redis_timeout_count",
            "value": "240",
            "duration": "10m",
        },
        "expected_primary_cause": "redis_pool_or_server_pressure",
        "expected_tools": ["query_metrics", "query_logs", "retrieve_runbook"],
    },
    {
        "alert": {
            "service": "order-service",
            "alert_name": "SlowQueryHigh",
            "severity": "P2",
            "metric": "mysql_slow_query_count",
            "value": "80",
            "duration": "15m",
        },
        "expected_primary_cause": "missing_index_or_full_scan",
        "expected_tools": ["query_metrics", "query_logs", "retrieve_runbook"],
    },
    {
        "alert": {
            "service": "worker-service",
            "alert_name": "DiskAlmostFull",
            "severity": "P2",
            "metric": "disk_usage",
            "value": "91%",
            "duration": "20m",
        },
        "expected_primary_cause": "log_or_temp_file_growth",
        "expected_tools": ["query_metrics", "query_logs", "retrieve_runbook"],
    },
]


MOCK_METRICS = {
    "order-service": {
        "http_5xx_rate": "12% for 5m, started after the last deployment",
        "request_total": "normal traffic, no major QPS spike",
        "latency_p95": "1800ms, higher than the normal 320ms baseline",
        "mysql_slow_query_count": "80 queries in 15m, concentrated on order list API",
    },
    "payment-service": {
        "http_5xx_rate": "8% for 6m, downstream payment gateway timeout increased",
        "latency_p95": "2100ms, correlated with gateway timeout logs",
    },
    "cache-client": {
        "redis_timeout_count": "240 timeouts in 10m",
        "connected_clients": "close to max connection pool size",
        "blocked_clients": "0, no server-side blocking",
        "used_memory": "stable, no memory spike",
    },
    "worker-service": {
        "disk_usage": "91%, log directory grew by 18GB in 2h",
        "inode_usage": "62%, inode is not the bottleneck",
        "error_rate": "normal",
    },
}


MOCK_LOGS = {
    "order-service": [
        "ERROR order-service database unavailable on /api/orders",
        "ERROR SQL full table scan detected for order list query",
        "WARN deploy version 2026.08.04-2 started before alert window",
    ],
    "payment-service": [
        "ERROR payment gateway timeout after 3000ms",
        "WARN retry attempt reached max retry count",
    ],
    "cache-client": [
        "ERROR Redis timeout while waiting for pooled connection",
        "WARN redis pool exhausted, active=100 idle=0",
    ],
    "worker-service": [
        "ERROR repeated task failure wrote stack trace to app.log",
        "WARN debug log enabled for batch worker",
    ],
}


MOCK_INCIDENTS = {
    "order-service": [
        "2026-07-21 HighErrorRate caused by missing DB index after deploy",
        "2026-07-03 payment dependency timeout caused checkout failures",
    ],
    "cache-client": [
        "2026-07-18 RedisTimeout caused by connection pool exhaustion",
    ],
    "worker-service": [
        "2026-07-29 DiskAlmostFull caused by repeated error logs",
    ],
}


def run_oncall_diagnosis(
    alert: dict,
    chunks: list[str],
    embedding_mode: str = KEYWORD_EMBEDDING_MODE,
    top_k: int = 3,
    retrieval_mode: str = "rerank",
) -> dict:
    normalized_alert = normalize_alert(alert)
    trace = []

    trace.append(
        build_trace_step(
            "parse_alert",
            tool="alert_payload",
            result=normalized_alert["alert_name"],
            details=format_alert(normalized_alert),
            reason="turn structured alert fields into an OnCall diagnosis task",
        )
    )

    selected_tools = select_oncall_tools(normalized_alert)
    trace.append(
        build_trace_step(
            "plan",
            tool="oncall_planner",
            result=", ".join(selected_tools),
            details=f"severity={normalized_alert['severity']}; metric={normalized_alert['metric']}",
            reason="collect metrics, logs, incidents, and runbook evidence before diagnosis",
        )
    )

    metrics = query_metrics(normalized_alert["service"])
    logs = query_logs(normalized_alert["service"])
    incidents = query_incidents(normalized_alert["service"])
    trace.append(
        build_trace_step(
            "query_metrics",
            tool="mock_metrics",
            result=f"{len(metrics)} metrics",
            details="; ".join(f"{key}={value}" for key, value in metrics.items()),
            reason="inspect numeric signals related to the alert",
        )
    )
    trace.append(
        build_trace_step(
            "query_logs",
            tool="mock_logs",
            result=f"{len(logs)} log lines",
            details=" | ".join(logs[:3]),
            reason="look for error signatures and recent changes",
        )
    )
    trace.append(
        build_trace_step(
            "query_incidents",
            tool="mock_incidents",
            result=f"{len(incidents)} incidents",
            details=" | ".join(incidents[:2]),
            reason="compare the alert with previous failure patterns",
        )
    )

    runbook_question = build_runbook_question(normalized_alert)
    retrieved_chunks = retrieve_for_mode(
        question=runbook_question,
        chunks=chunks,
        top_k=top_k,
        embedding_mode=embedding_mode,
        retrieval_mode=retrieval_mode,
    )
    trace.append(
        build_trace_step(
            "retrieve_runbook",
            tool=f"retrieve_{retrieval_mode}",
            result=format_chunk_indexes(retrieved_chunks),
            details=runbook_question,
            reason="retrieve operational runbook steps for the alert type",
        )
    )

    cause_candidates = rank_possible_causes(normalized_alert, metrics, logs, incidents)
    primary_cause = cause_candidates[0]["cause"] if cause_candidates else "unknown"
    confidence = calculate_confidence(cause_candidates, retrieved_chunks)
    trace.append(
        build_trace_step(
            "rank_causes",
            tool="local_cause_ranker",
            result=primary_cause,
            details="; ".join(
                f"{item['cause']}:{item['score']}" for item in cause_candidates
            ),
            reason="rank likely causes from alert fields and mock evidence",
        )
    )

    final_answer = build_oncall_answer(
        alert=normalized_alert,
        metrics=metrics,
        logs=logs,
        incidents=incidents,
        retrieved_chunks=retrieved_chunks,
        cause_candidates=cause_candidates,
        confidence=confidence,
    )

    return {
        "alert": normalized_alert,
        "selected_tools": selected_tools,
        "metrics": metrics,
        "logs": logs,
        "incidents": incidents,
        "retrieved_chunks": retrieved_chunks,
        "sources": format_sources(retrieved_chunks) if retrieved_chunks else "",
        "possible_causes": cause_candidates,
        "primary_cause": primary_cause,
        "confidence": confidence,
        "final_answer": final_answer,
        "trace": trace,
    }


def normalize_alert(alert: dict) -> dict:
    return {
        "service": str(alert.get("service", "")).strip() or "unknown-service",
        "alert_name": str(alert.get("alert_name", "")).strip() or "UnknownAlert",
        "severity": str(alert.get("severity", "")).strip() or "P3",
        "metric": str(alert.get("metric", "")).strip() or "unknown_metric",
        "value": str(alert.get("value", "")).strip() or "unknown",
        "duration": str(alert.get("duration", "")).strip() or "unknown",
    }


def select_oncall_tools(alert: dict) -> list[str]:
    tools = ["query_metrics", "query_logs", "retrieve_runbook"]
    if alert["severity"].upper() in {"P0", "P1", "P2"}:
        tools.append("query_incidents")
    tools.append("rank_possible_causes")
    return tools


def query_metrics(service: str) -> dict[str, str]:
    return MOCK_METRICS.get(service, {})


def query_logs(service: str) -> list[str]:
    return MOCK_LOGS.get(service, [])


def query_incidents(service: str) -> list[str]:
    return MOCK_INCIDENTS.get(service, [])


def build_runbook_question(alert: dict) -> str:
    return (
        f"{alert['alert_name']} 告警怎么排查？"
        f"服务 {alert['service']} 指标 {alert['metric']} "
        f"当前值 {alert['value']} 持续 {alert['duration']}"
    )


def rank_possible_causes(
    alert: dict,
    metrics: dict[str, str],
    logs: list[str],
    incidents: list[str],
) -> list[dict]:
    evidence_text = " ".join(
        [
            alert["alert_name"],
            alert["metric"],
            alert["value"],
            " ".join(metrics.values()),
            " ".join(logs),
            " ".join(incidents),
        ]
    ).lower()
    candidates = [
        {
            "cause": "recent_deploy_or_dependency_failure",
            "signals": ["deploy", "5xx", "gateway", "dependency", "database unavailable"],
        },
        {
            "cause": "redis_pool_or_server_pressure",
            "signals": ["redis", "pool exhausted", "connected_clients", "timeout"],
        },
        {
            "cause": "missing_index_or_full_scan",
            "signals": ["slowquery", "slow query", "full table scan", "explain", "index", "索引"],
        },
        {
            "cause": "retry_storm_or_cpu_hotspot",
            "signals": ["cpu", "retry", "qps", "latency", "gc"],
        },
        {
            "cause": "log_or_temp_file_growth",
            "signals": ["disk", "log directory", "debug log", "disk_usage", "app.log"],
        },
    ]

    ranked = []
    for candidate in candidates:
        score = sum(
            1
            for signal in candidate["signals"]
            if signal.lower() in evidence_text
        )
        if score > 0:
            ranked.append(
                {
                    "cause": candidate["cause"],
                    "score": score,
                    "matched_signals": [
                        signal
                        for signal in candidate["signals"]
                        if signal.lower() in evidence_text
                    ],
                }
            )

    ranked.sort(key=lambda item: (-item["score"], item["cause"]))
    return ranked


def calculate_confidence(cause_candidates: list[dict], retrieved_chunks: list[dict]) -> str:
    if not cause_candidates or not retrieved_chunks:
        return "low"

    if cause_candidates[0]["score"] >= 3:
        return "high"

    return "medium"


def build_oncall_answer(
    alert: dict,
    metrics: dict[str, str],
    logs: list[str],
    incidents: list[str],
    retrieved_chunks: list[dict],
    cause_candidates: list[dict],
    confidence: str,
) -> str:
    primary_cause = cause_candidates[0]["cause"] if cause_candidates else "unknown"
    sources = format_sources(retrieved_chunks) if retrieved_chunks else "none"
    top_log = logs[0] if logs else "no matched log line"
    top_metric = (
        "; ".join(f"{key}={value}" for key, value in list(metrics.items())[:3])
        if metrics
        else "no matched metric"
    )
    previous_incident = incidents[0] if incidents else "no similar incident"

    return (
        f"告警 {alert['alert_name']} 影响服务 {alert['service']}，"
        f"级别 {alert['severity']}，指标 {alert['metric']}={alert['value']}，"
        f"持续 {alert['duration']}。\n\n"
        f"初步判断：{primary_cause}，置信度 {confidence}。\n\n"
        f"关键证据：\n"
        f"- 指标：{top_metric}\n"
        f"- 日志：{top_log}\n"
        f"- 历史事件：{previous_incident}\n"
        f"- Runbook 来源：{sources}\n\n"
        f"建议动作：先保留指标和日志证据，再按 Runbook 排查。"
        f"如果涉及发布后错误率上升，优先确认发布记录并准备回滚；"
        f"如果证据不足，应升级给人工值班同学处理。"
    )


def evaluate_oncall_diagnosis(
    evaluation_cases: list[dict],
    chunks: list[str],
    embedding_mode: str = KEYWORD_EMBEDDING_MODE,
    top_k: int = 3,
    retrieval_mode: str = "rerank",
) -> dict:
    rows = []
    for case in evaluation_cases:
        result = run_oncall_diagnosis(
            alert=case["alert"],
            chunks=chunks,
            embedding_mode=embedding_mode,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        expected_tools = case["expected_tools"]
        selected_tools = result["selected_tools"]
        expected_primary_cause = case["expected_primary_cause"]
        cause_hit = result["primary_cause"] == expected_primary_cause
        tool_hit = all(tool in selected_tools for tool in expected_tools)
        has_sources = bool(result["sources"])
        safe_action = "不能自动执行高风险操作" in result["final_answer"] or "升级给人工" in result["final_answer"]

        rows.append(
            {
                "alert_name": result["alert"]["alert_name"],
                "service": result["alert"]["service"],
                "expected_primary_cause": expected_primary_cause,
                "actual_primary_cause": result["primary_cause"],
                "cause_hit": cause_hit,
                "tool_hit": tool_hit,
                "has_sources": has_sources,
                "safe_action": safe_action,
                "confidence": result["confidence"],
                "selected_tools": ", ".join(selected_tools),
            }
        )

    return {
        "case_count": len(rows),
        "root_cause_hit_rate": calculate_rate(rows, "cause_hit"),
        "tool_selection_accuracy": calculate_rate(rows, "tool_hit"),
        "evidence_citation_rate": calculate_rate(rows, "has_sources"),
        "safe_action_rate": calculate_rate(rows, "safe_action"),
        "rows": rows,
        "failure_cases": [
            row for row in rows
            if not row["cause_hit"]
            or not row["tool_hit"]
            or not row["has_sources"]
            or not row["safe_action"]
        ],
    }


def calculate_rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0

    return sum(1 for row in rows if row[key]) / len(rows)


def format_alert(alert: dict) -> str:
    return (
        f"service={alert['service']}; alert_name={alert['alert_name']}; "
        f"severity={alert['severity']}; metric={alert['metric']}; "
        f"value={alert['value']}; duration={alert['duration']}"
    )


def format_chunk_indexes(retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return "none"

    return ", ".join(f"Chunk {item['chunk_index']}" for item in retrieved_chunks)


def build_trace_step(
    step: str,
    result: str,
    tool: str = "",
    details: str = "",
    reason: str = "",
) -> dict[str, str]:
    return {
        "step": step,
        "tool": tool,
        "result": result,
        "details": details,
        "reason": reason,
    }
