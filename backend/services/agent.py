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


def run_agentic_rag(
    question: str,
    chunks: list[str],
    embedding_mode: str = KEYWORD_EMBEDDING_MODE,
    top_k: int = 3,
    retrieval_mode: str = "auto",
    use_llm: bool = True,
) -> dict:
    trace = []
    intent = classify_intent(question)
    trace.append(
        build_trace_step(
            "classify_intent",
            result=intent,
            details=f"matched_concepts={format_matched_concepts(question)}",
        )
    )

    if intent == OUT_OF_SCOPE_INTENT:
        trace.append(
            build_trace_step(
                "refuse",
                result="out_of_scope",
                details="question does not look answerable from the local document",
            )
        )
        return build_agent_result(
            question=question,
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
        )
    )

    retrieved_chunks = retrieve_for_intent(
        intent=intent,
        question=question,
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
        )
    )

    if should_refuse(question, retrieved_chunks):
        trace.append(
            build_trace_step(
                "refuse",
                result="low_confidence",
                details="retrieval returned no reliable local context",
            )
        )
        return build_agent_result(
            question=question,
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
        question=question,
        retrieved_chunks=retrieved_chunks,
        use_llm=use_llm,
    )
    trace.append(
        build_trace_step(
            "answer",
            tool="generate_answer" if use_llm and not fallback_reason else "context_answer",
            result="done",
            details=f"sources={format_sources(retrieved_chunks)}",
        )
    )

    return build_agent_result(
        question=question,
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
        question=question,
        chunks=chunks,
        top_k=top_k,
        embedding_mode=embedding_mode,
        retrieval_mode=retrieval_mode,
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
) -> dict[str, str]:
    return {
        "step": step,
        "tool": tool,
        "result": result,
        "details": details,
    }


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def format_matched_concepts(question: str) -> str:
    return ", ".join(get_matched_concepts(question)) or "none"


def format_chunk_indexes(retrieved_chunks: list[dict]) -> str:
    return ", ".join(
        f"Chunk {item['chunk_index']}"
        for item in retrieved_chunks
    ) or "none"
