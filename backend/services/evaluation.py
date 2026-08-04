from backend.services.bm25 import retrieve_relevant_chunks_bm25
from backend.services.embeddings import KEYWORD_EMBEDDING_MODE, get_matched_concepts
from backend.services.rerank import rerank_chunks
from backend.services.retrieval import retrieve_relevant_chunks
from backend.services.rrf import retrieve_relevant_chunks_rrf


EVALUATION_CASES = [
    {"question": "LLM API 如何配置？", "expected_top_chunk": 2},
    {"question": "DeepSeek API 怎么配置？", "expected_top_chunk": 2},
    {"question": "OpenAI-compatible API 和 ChatGPT Plus 额度互通吗？", "expected_top_chunk": 2},
    {"question": "DeepSeek 使用哪个环境变量？", "expected_top_chunk": 2},
    {"question": "API Key 为什么不能写进代码？", "expected_top_chunk": 3},
    {"question": "怎么从环境变量读取 DEEPSEEK_API_KEY？", "expected_top_chunk": 3},
    {"question": "密钥泄露有什么风险？", "expected_top_chunk": 3},
    {"question": "什么是 embedding？", "expected_top_chunk": 4},
    {"question": "embedding 如何用于检索？", "expected_top_chunk": 4},
    {"question": "文本为什么要转成数字向量？", "expected_top_chunk": 4},
    {"question": "向量数据库有什么作用？", "expected_top_chunk": 5},
    {"question": "当前项目使用哪个向量数据库？", "expected_top_chunk": 5},
    {"question": "Chroma 保存了什么？", "expected_top_chunk": 5},
    {"question": "RAG 的基本流程是什么？", "expected_top_chunk": 6},
    {"question": "RAG 为什么要先检索资料？", "expected_top_chunk": 6},
]
ONCALL_EVALUATION_CASES = [
    {"question": "API 5xx 错误率升高应该怎么排查？", "expected_top_chunk": 2},
    {"question": "Redis 连接超时要看哪些指标？", "expected_top_chunk": 4},
    {"question": "MySQL 慢查询告警怎么定位？", "expected_top_chunk": 6},
    {"question": "CPU 使用率持续升高可能有哪些原因？", "expected_top_chunk": 7},
    {"question": "磁盘空间不足应该先检查什么？", "expected_top_chunk": 8},
    {"question": "OnCall 处理告警的通用原则是什么？", "expected_top_chunk": 9},
]
RETRIEVAL_MODES = {"vector", "bm25", "rrf", "rerank"}


def select_evaluation_cases(document_text: str, requested_set: str) -> list[dict]:
    if requested_set == "OnCall runbook":
        return ONCALL_EVALUATION_CASES

    if requested_set == "DocuAsk FAQ":
        return EVALUATION_CASES

    lowered_text = document_text.lower()
    if (
        "oncall" in lowered_text
        or "runbook" in lowered_text
        or "告警" in document_text
    ):
        return ONCALL_EVALUATION_CASES

    return EVALUATION_CASES


def evaluate_retrieval(
    evaluation_cases: list[dict],
    chunks: list[str],
    top_k: int,
    embedding_mode: str = KEYWORD_EMBEDDING_MODE,
    retrieval_mode: str = "vector",
) -> list[dict]:
    rows = []

    for case in evaluation_cases:
        question = case["question"]
        expected_top_chunk = case["expected_top_chunk"]
        retrieved_chunks = retrieve_for_mode(
            question=question,
            chunks=chunks,
            top_k=top_k,
            embedding_mode=embedding_mode,
            retrieval_mode=retrieval_mode,
        )
        matched_concepts = get_matched_concepts(question)
        actual_top_chunk = (
            retrieved_chunks[0]["chunk_index"]
            if retrieved_chunks
            else None
        )
        retrieved_chunk_indexes = [
            item["chunk_index"]
            for item in retrieved_chunks
        ]

        rows.append(
            {
                "question": question,
                "embedding_mode": embedding_mode,
                "retrieval_mode": retrieval_mode,
                "expected_top_chunk": f"Chunk {expected_top_chunk}",
                "matched_concepts": ", ".join(matched_concepts) or "none",
                "top_chunks": ", ".join(
                    f"Chunk {item['chunk_index']}" for item in retrieved_chunks
                ) or "none",
                "best_score": format_best_score(retrieved_chunks),
                "hit": actual_top_chunk == expected_top_chunk,
                "top_k_hit": expected_top_chunk in retrieved_chunk_indexes,
                "failure_reason": build_failure_reason(
                    actual_top_chunk,
                    expected_top_chunk,
                    retrieved_chunk_indexes,
                ),
            }
        )

    return rows


def get_failure_cases(evaluation_rows: list[dict]) -> list[dict]:
    return [
        row for row in evaluation_rows
        if not row["hit"] or not row["top_k_hit"]
    ]


def build_failure_reason(
    actual_top_chunk: int | None,
    expected_top_chunk: int,
    retrieved_chunk_indexes: list[int],
) -> str:
    if actual_top_chunk == expected_top_chunk:
        return "none"

    if expected_top_chunk in retrieved_chunk_indexes:
        return "expected chunk was retrieved but not ranked first"

    return "expected chunk was not retrieved in top_k"


def retrieve_for_mode(
    question: str,
    chunks: list[str],
    top_k: int,
    embedding_mode: str,
    retrieval_mode: str,
) -> list[dict]:
    if retrieval_mode == "vector":
        return retrieve_relevant_chunks(
            question,
            chunks,
            top_k=top_k,
            embedding_mode=embedding_mode,
        )

    if retrieval_mode == "bm25":
        return retrieve_relevant_chunks_bm25(
            question,
            chunks,
            top_k=top_k,
        )

    if retrieval_mode == "rrf":
        return retrieve_relevant_chunks_rrf(
            question,
            chunks,
            top_k=top_k,
            embedding_mode=embedding_mode,
        )

    if retrieval_mode == "rerank":
        candidates = retrieve_relevant_chunks_rrf(
            question,
            chunks,
            top_k=min(len(chunks), max(top_k * 2, top_k)),
            embedding_mode=embedding_mode,
        )
        return rerank_chunks(question, candidates, top_k=top_k)

    raise ValueError("unsupported retrieval mode")


def format_best_score(retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return "none"

    best_chunk = retrieved_chunks[0]
    for key in ("rerank_score", "distance", "score", "rrf_score"):
        if key in best_chunk and best_chunk[key] is not None:
            return f"{best_chunk[key]:.4f}"

    return "none"


def calculate_hit_rate(evaluation_rows: list[dict]) -> float:
    if not evaluation_rows:
        return 0.0

    hit_count = sum(1 for row in evaluation_rows if row["hit"])
    return hit_count / len(evaluation_rows)


def calculate_top_k_hit_rate(evaluation_rows: list[dict]) -> float:
    if not evaluation_rows:
        return 0.0

    hit_count = sum(1 for row in evaluation_rows if row["top_k_hit"])
    return hit_count / len(evaluation_rows)
