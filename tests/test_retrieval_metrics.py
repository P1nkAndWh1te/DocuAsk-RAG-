import importlib.util
from pathlib import Path

from conftest import FAQ_PATH, ONCALL_RUNBOOK_PATH
from backend.services.evaluation import (
    ONCALL_EVALUATION_CASES,
    calculate_hit_rate,
    calculate_top_k_hit_rate,
    evaluate_retrieval,
    select_evaluation_cases,
)
from backend.services.bm25 import retrieve_relevant_chunks_bm25
from backend.services.rrf import retrieve_relevant_chunks_rrf


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_rag_app_module():
    spec = importlib.util.spec_from_file_location("rag_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load app module from {APP_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_teaching_keyword_retrieval_metrics_stay_stable():
    rag_app = load_rag_app_module()
    text = FAQ_PATH.read_text(encoding="utf-8")
    chunks = rag_app.split_text_into_chunks(
        text,
        chunk_size=rag_app.DEFAULT_CHUNK_SIZE,
        chunk_overlap=rag_app.DEFAULT_CHUNK_OVERLAP,
    )

    rows = rag_app.evaluate_retrieval(
        rag_app.EVALUATION_CASES,
        chunks,
        top_k=rag_app.TOP_K,
        embedding_mode=rag_app.KEYWORD_EMBEDDING_MODE,
    )

    assert len(chunks) == 6
    assert rag_app.calculate_hit_rate(rows) == 0.7333333333333333
    assert rag_app.calculate_top_k_hit_rate(rows) == 1.0


def test_bm25_retrieves_rag_flow_chunk():
    rag_app = load_rag_app_module()
    text = FAQ_PATH.read_text(encoding="utf-8")
    chunks = rag_app.split_text_into_chunks(
        text,
        chunk_size=rag_app.DEFAULT_CHUNK_SIZE,
        chunk_overlap=rag_app.DEFAULT_CHUNK_OVERLAP,
    )

    results = retrieve_relevant_chunks_bm25(
        "RAG 的基本流程是什么？",
        chunks,
        top_k=3,
    )

    assert results[0]["chunk_index"] == 6
    assert results[0]["score"] > 0


def test_rrf_retrieves_rag_flow_chunk():
    rag_app = load_rag_app_module()
    text = FAQ_PATH.read_text(encoding="utf-8")
    chunks = rag_app.split_text_into_chunks(
        text,
        chunk_size=rag_app.DEFAULT_CHUNK_SIZE,
        chunk_overlap=rag_app.DEFAULT_CHUNK_OVERLAP,
    )

    results = retrieve_relevant_chunks_rrf(
        "RAG 的基本流程是什么？",
        chunks,
        top_k=3,
        embedding_mode=rag_app.KEYWORD_EMBEDDING_MODE,
    )

    assert results[0]["chunk_index"] == 6
    assert results[0]["rrf_score"] > 0
    assert len(results[0]["sources"]) >= 1


def test_oncall_runbook_evaluation_cases_are_selected_from_document():
    text = ONCALL_RUNBOOK_PATH.read_text(encoding="utf-8")

    assert select_evaluation_cases(text, "Auto") == ONCALL_EVALUATION_CASES


def test_oncall_runbook_retrieval_metrics_stay_stable():
    rag_app = load_rag_app_module()
    text = ONCALL_RUNBOOK_PATH.read_text(encoding="utf-8")
    chunks = rag_app.split_text_into_chunks(
        text,
        chunk_size=rag_app.DEFAULT_CHUNK_SIZE,
        chunk_overlap=rag_app.DEFAULT_CHUNK_OVERLAP,
    )

    rows = evaluate_retrieval(
        ONCALL_EVALUATION_CASES,
        chunks,
        top_k=rag_app.TOP_K,
        embedding_mode=rag_app.KEYWORD_EMBEDDING_MODE,
        retrieval_mode="rerank",
    )

    assert len(chunks) == 9
    assert calculate_hit_rate(rows) == 1.0
    assert calculate_top_k_hit_rate(rows) == 1.0
