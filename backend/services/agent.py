from backend.services.embeddings import (
    KEYWORD_EMBEDDING_MODE,
    get_matched_concepts,
)
from backend.services.evaluation import retrieve_for_mode
from backend.services.generation import (
    MissingApiKeyError,
    format_sources,
    generate_answer_with_deepseek,
)
from backend.services.retrieval import format_retrieved_context


AGENT_RETRIEVAL_MODES = {"auto", "vector", "bm25", "rrf", "rerank"}
DOCUMENT_QA_INTENT = "document_qa"
SUMMARY_INTENT = "summary"
COMPARE_INTENT = "compare"
METADATA_QUERY_INTENT = "metadata_query"
OUT_OF_SCOPE_INTENT = "out_of_scope"
AGENT_EVALUATION_CASES = [
    {
        "question": "RAG 的基本流程是什么？",
        "expected_intent": DOCUMENT_QA_INTENT,
        "expected_retrieval_mode": "rerank",
        "should_refuse": False,
    },
    {
        "question": "总结这份文档",
        "expected_intent": SUMMARY_INTENT,
        "expected_retrieval_mode": "scan",
        "should_refuse": False,
    },
    {
        "question": "比较 BM25 和 RRF 的区别",
        "expected_intent": COMPARE_INTENT,
        "expected_retrieval_mode": "rerank",
        "should_refuse": False,
    },
    {
        "question": "本次回答用了哪些来源 chunk？",
        "expected_intent": METADATA_QUERY_INTENT,
        "expected_retrieval_mode": "bm25",
        "should_refuse": False,
    },
    {
        "question": "今天上海天气怎么样？",
        "expected_intent": OUT_OF_SCOPE_INTENT,
        "expected_retrieval_mode": "none",
        "should_refuse": True,
    },
]


def run_agentic_rag(
    question: str,
    chunks: list[str],
    embedding_mode: str = KEYWORD_EMBEDDING_MODE,
    top_k: int = 3,
    retrieval_mode: str = "auto",
    use_llm: bool = True,
    conversation_history: list[dict] | None = None,
) -> dict:
    trace = []
    effective_question = rewrite_follow_up_question(question, conversation_history or [])
    if effective_question != question:
        trace.append(
            build_trace_step(
                "rewrite_question",
                result="rewritten",
                details=effective_question,
                reason="follow-up question used previous turn context",
            )
        )

    intent = classify_intent(effective_question)
    trace.append(
        build_trace_step(
            "classify_intent",
            result=intent,
            details=f"matched_concepts={format_matched_concepts(effective_question)}",
            reason="route question to an agent intent",
        )
    )

    if intent == OUT_OF_SCOPE_INTENT:
        trace.append(
            build_trace_step(
                "refuse",
                result="out_of_scope",
                details="question does not look answerable from the local document",
                reason="intent classifier marked the question as outside local document scope",
            )
        )
        return build_agent_result(
            question=question,
            effective_question=effective_question,
            intent=intent,
            retrieval_mode="none",
            selected_tools=[],
            retrieved_chunks=[],
            final_answer="当前文档中没有足够依据回答该问题。",
            trace=trace,
            confidence="low",
            fallback_reason="out_of_scope",
        )

    selected_retrieval_mode = choose_retrieval_mode(intent, retrieval_mode)
    selected_tools = select_tools(intent, selected_retrieval_mode, use_llm)
    trace.append(
        build_trace_step(
            "plan",
            result="tools_selected",
            details=", ".join(selected_tools),
            reason=f"intent={intent}; retrieval_mode={selected_retrieval_mode}",
        )
    )

    retrieved_chunks = retrieve_for_intent(
        intent=intent,
        question=question,
        effective_question=effective_question,
        chunks=chunks,
        top_k=top_k,
        embedding_mode=embedding_mode,
        retrieval_mode=selected_retrieval_mode,
    )
    trace.append(
        build_trace_step(
            "retrieve",
            tool=f"retrieve_{selected_retrieval_mode}",
            result=format_chunk_indexes(retrieved_chunks),
            details=f"top_k={top_k}",
            reason="collect local evidence before answering",
        )
    )

    if should_refuse(effective_question, retrieved_chunks):
        trace.append(
            build_trace_step(
                "refuse",
                result="low_confidence",
                details="retrieval returned no reliable local context",
                reason="retrieved chunks are empty or below confidence signals",
            )
        )
        return build_agent_result(
            question=question,
            effective_question=effective_question,
            intent=intent,
            retrieval_mode=selected_retrieval_mode,
            selected_tools=selected_tools,
            retrieved_chunks=retrieved_chunks,
            final_answer="当前文档中没有足够依据回答该问题。",
            trace=trace,
            confidence="low",
            fallback_reason="low_retrieval_confidence",
        )

    final_answer, fallback_reason = build_final_answer(
        intent=intent,
        question=effective_question,
        retrieved_chunks=retrieved_chunks,
        use_llm=use_llm,
    )
    trace.append(
        build_trace_step(
            "answer",
            tool="generate_answer" if use_llm and not fallback_reason else "context_answer",
            result="done",
            details=f"sources={format_sources(retrieved_chunks)}",
            reason="produce grounded answer from selected context",
        )
    )

    return build_agent_result(
        question=question,
        effective_question=effective_question,
        intent=intent,
        retrieval_mode=selected_retrieval_mode,
        selected_tools=selected_tools,
        retrieved_chunks=retrieved_chunks,
        final_answer=final_answer,
        trace=trace,
        confidence="medium" if fallback_reason else "high",
        fallback_reason=fallback_reason,
    )


def classify_intent(question: str) -> str:
    normalized = question.lower().strip()

    if contains_any(normalized, ["天气", "股票", "彩票", "新闻", "汇率", "今天几号"]):
        return OUT_OF_SCOPE_INTENT

    if contains_any(normalized, ["总结", "概括", "summarize", "摘要"]):
        return SUMMARY_INTENT

    if contains_any(normalized, ["对比", "比较", "区别", "差异", "compare"]):
        return COMPARE_INTENT

    if contains_any(normalized, ["来源", "引用", "chunk", "片段", "文档里有哪些"]):
        return METADATA_QUERY_INTENT

    return DOCUMENT_QA_INTENT


def choose_retrieval_mode(intent: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode

    if intent == METADATA_QUERY_INTENT:
        return "bm25"

    if intent == SUMMARY_INTENT:
        return "scan"

    return "rerank"


def select_tools(intent: str, retrieval_mode: str, use_llm: bool) -> list[str]:
    tools = []

    if retrieval_mode == "scan":
        tools.append("scan_document_chunks")
    else:
        tools.append(f"retrieve_{retrieval_mode}")

    if retrieval_mode == "rerank":
        tools.append("rerank_chunks")

    if intent == SUMMARY_INTENT:
        tools.append("summarize_context")
    elif intent == METADATA_QUERY_INTENT:
        tools.append("cite_sources")
    else:
        tools.append("generate_answer" if use_llm else "context_answer")

    return tools


def retrieve_for_intent(
    intent: str,
    question: str,
    effective_question: str,
    chunks: list[str],
    top_k: int,
    embedding_mode: str,
    retrieval_mode: str,
) -> list[dict]:
    if intent == SUMMARY_INTENT or retrieval_mode == "scan":
        return [
            {
                "chunk_index": index,
                "text": chunk,
                "scan_score": 1.0 / index,
            }
            for index, chunk in enumerate(chunks[:top_k], start=1)
        ]

    return retrieve_for_mode(
        question=effective_question,
        chunks=chunks,
        top_k=top_k,
        embedding_mode=embedding_mode,
        retrieval_mode=retrieval_mode,
    )


def rewrite_follow_up_question(
    question: str,
    conversation_history: list[dict],
) -> str:
    if not conversation_history:
        return question

    normalized = question.strip()
    if not is_follow_up_question(normalized):
        return question

    previous_turn = conversation_history[-1]
    previous_question = str(previous_turn.get("question", "")).strip()
    if not previous_question:
        return question

    return f"{previous_question}。追问：{question}"


def is_follow_up_question(question: str) -> bool:
    lowered = question.lower()
    return contains_any(
        lowered,
        [
            "它",
            "这个",
            "这点",
            "刚才",
            "上面",
            "前面",
            "that",
            "it",
            "this",
        ],
    )


def should_refuse(question: str, retrieved_chunks: list[dict]) -> bool:
    if not retrieved_chunks:
        return True

    if get_matched_concepts(question):
        return False

    return not any(has_positive_score(item) for item in retrieved_chunks)


def has_positive_score(item: dict) -> bool:
    for key in ("rerank_score", "rrf_score", "score", "scan_score"):
        if item.get(key, 0) > 0:
            return True

    if "distance" in item and item["distance"] is not None:
        return item["distance"] < 1.0

    return False


def build_final_answer(
    intent: str,
    question: str,
    retrieved_chunks: list[dict],
    use_llm: bool,
) -> tuple[str, str]:
    if intent == METADATA_QUERY_INTENT:
        return build_metadata_answer(retrieved_chunks), ""

    if not use_llm:
        return build_context_answer(intent, retrieved_chunks), ""

    try:
        return generate_answer_with_deepseek(question, retrieved_chunks), ""
    except MissingApiKeyError:
        return build_context_answer(intent, retrieved_chunks), "missing_api_key_used_context_answer"


def build_context_answer(intent: str, retrieved_chunks: list[dict]) -> str:
    sources = format_sources(retrieved_chunks)

    if intent == SUMMARY_INTENT:
        previews = [
            compact_text(item["text"], max_length=120)
            for item in retrieved_chunks
        ]
        return "根据文档片段，主要内容包括：" + "；".join(previews) + f"\n来源：{sources}"

    if intent == COMPARE_INTENT:
        previews = [
            f"Chunk {item['chunk_index']}：{compact_text(item['text'], max_length=100)}"
            for item in retrieved_chunks
        ]
        return "可对比的相关资料如下：" + "；".join(previews) + f"\n来源：{sources}"

    best_chunk = retrieved_chunks[0]
    return (
        "根据检索到的资料："
        f"{compact_text(best_chunk['text'], max_length=180)}\n"
        f"来源：{sources}"
    )


def build_metadata_answer(retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return "当前没有检索到可引用的文档片段。"

    lines = [
        f"- Chunk {item['chunk_index']}：{compact_text(item['text'], max_length=80)}"
        for item in retrieved_chunks
    ]
    return "本次使用的来源片段包括：\n" + "\n".join(lines)


def compact_text(text: str, max_length: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized

    return normalized[:max_length].rstrip() + "..."


def build_agent_result(
    question: str,
    effective_question: str,
    intent: str,
    retrieval_mode: str,
    selected_tools: list[str],
    retrieved_chunks: list[dict],
    final_answer: str,
    trace: list[dict],
    confidence: str,
    fallback_reason: str,
) -> dict:
    return {
        "question": question,
        "effective_question": effective_question,
        "intent": intent,
        "retrieval_mode": retrieval_mode,
        "selected_tools": selected_tools,
        "retrieved_chunks": retrieved_chunks,
        "context": format_retrieved_context(retrieved_chunks),
        "sources": format_sources(retrieved_chunks) if retrieved_chunks else "",
        "final_answer": final_answer,
        "trace": trace,
        "confidence": confidence,
        "fallback_reason": fallback_reason,
    }


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


def evaluate_agentic_rag(
    evaluation_cases: list[dict],
    chunks: list[str],
    embedding_mode: str = KEYWORD_EMBEDDING_MODE,
    top_k: int = 3,
) -> dict:
    rows = []
    for case in evaluation_cases:
        result = run_agentic_rag(
            question=case["question"],
            chunks=chunks,
            embedding_mode=embedding_mode,
            top_k=top_k,
            retrieval_mode="auto",
            use_llm=False,
        )
        expected_intent = case["expected_intent"]
        expected_retrieval_mode = case["expected_retrieval_mode"]
        should_refuse = case["should_refuse"]
        refused = result["confidence"] == "low" and bool(result["fallback_reason"])
        has_sources = bool(result["sources"])

        rows.append(
            {
                "question": case["question"],
                "expected_intent": expected_intent,
                "actual_intent": result["intent"],
                "intent_hit": result["intent"] == expected_intent,
                "expected_retrieval_mode": expected_retrieval_mode,
                "actual_retrieval_mode": result["retrieval_mode"],
                "tool_hit": result["retrieval_mode"] == expected_retrieval_mode,
                "should_refuse": should_refuse,
                "refused": refused,
                "refusal_hit": refused == should_refuse,
                "has_sources": has_sources,
                "fallback_reason": result["fallback_reason"] or "none",
                "trace_steps": ", ".join(step["step"] for step in result["trace"]),
            }
        )

    return {
        "case_count": len(rows),
        "intent_accuracy": calculate_boolean_rate(rows, "intent_hit"),
        "tool_selection_accuracy": calculate_boolean_rate(rows, "tool_hit"),
        "refusal_accuracy": calculate_boolean_rate(rows, "refusal_hit"),
        "source_citation_rate": calculate_source_citation_rate(rows),
        "rows": rows,
        "failure_cases": [
            row for row in rows
            if not row["intent_hit"] or not row["tool_hit"] or not row["refusal_hit"]
        ],
    }


def calculate_boolean_rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0

    return sum(1 for row in rows if row[key]) / len(rows)


def calculate_source_citation_rate(rows: list[dict]) -> float:
    answerable_rows = [row for row in rows if not row["should_refuse"]]
    if not answerable_rows:
        return 0.0

    return sum(1 for row in answerable_rows if row["has_sources"]) / len(answerable_rows)


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def format_matched_concepts(question: str) -> str:
    return ", ".join(get_matched_concepts(question)) or "none"


def format_chunk_indexes(retrieved_chunks: list[dict]) -> str:
    return ", ".join(
        f"Chunk {item['chunk_index']}"
        for item in retrieved_chunks
    ) or "none"
