from fastapi.testclient import TestClient

from backend.app import app
from backend.services.agent import (
    COMPARE_INTENT,
    DOCUMENT_QA_INTENT,
    OUT_OF_SCOPE_INTENT,
    SUMMARY_INTENT,
    classify_intent,
    run_agentic_rag,
)
from backend.services.embeddings import KEYWORD_EMBEDDING_MODE
from conftest import FAQ_PATH


def test_classify_intent_routes_common_question_types():
    assert classify_intent("RAG 的基本流程是什么？") == DOCUMENT_QA_INTENT
    assert classify_intent("总结这份文档") == SUMMARY_INTENT
    assert classify_intent("比较 BM25 和 RRF 的区别") == COMPARE_INTENT
    assert classify_intent("今天上海天气怎么样？") == OUT_OF_SCOPE_INTENT


def test_agentic_rag_auto_uses_rerank_for_document_qa():
    text = FAQ_PATH.read_text(encoding="utf-8")
    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    result = run_agentic_rag(
        question="RAG 的基本流程是什么？",
        chunks=chunks,
        embedding_mode=KEYWORD_EMBEDDING_MODE,
        top_k=3,
        retrieval_mode="auto",
        use_llm=False,
    )

    assert result["intent"] == DOCUMENT_QA_INTENT
    assert result["retrieval_mode"] == "rerank"
    assert "rerank_chunks" in result["selected_tools"]
    assert result["retrieved_chunks"]
    assert result["trace"][0]["step"] == "classify_intent"
    assert "来源：" in result["final_answer"]


def test_agentic_rag_refuses_out_of_scope_question():
    result = run_agentic_rag(
        question="今天上海天气怎么样？",
        chunks=["RAG 会先检索资料，再根据资料生成回答。"],
        embedding_mode=KEYWORD_EMBEDDING_MODE,
        top_k=3,
        retrieval_mode="auto",
        use_llm=False,
    )

    assert result["intent"] == OUT_OF_SCOPE_INTENT
    assert result["confidence"] == "low"
    assert result["fallback_reason"] == "out_of_scope"
    assert result["retrieved_chunks"] == []
    assert result["final_answer"] == "当前文档中没有足够依据回答该问题。"


def test_agentic_rag_summary_scans_document_chunks():
    result = run_agentic_rag(
        question="总结这份文档",
        chunks=[
            "第一段介绍 API Key 和环境变量。",
            "第二段介绍 embedding 和向量数据库。",
        ],
        embedding_mode=KEYWORD_EMBEDDING_MODE,
        top_k=2,
        retrieval_mode="auto",
        use_llm=False,
    )

    assert result["intent"] == SUMMARY_INTENT
    assert result["retrieval_mode"] == "scan"
    assert "summarize_context" in result["selected_tools"]
    assert len(result["retrieved_chunks"]) == 2
    assert result["retrieved_chunks"][0]["scan_score"] == 1.0


def test_agent_ask_endpoint_returns_trace_and_tools():
    client = TestClient(app)
    text = FAQ_PATH.read_text(encoding="utf-8")

    document_response = client.post(
        "/documents",
        json={
            "text": text,
            "embedding_mode": KEYWORD_EMBEDDING_MODE,
            "chunk_size": 350,
            "chunk_overlap": 50,
        },
    )
    collection_name = document_response.json()["collection_name"]

    response = client.post(
        "/agent/ask",
        json={
            "collection_name": collection_name,
            "question": "RAG 的基本流程是什么？",
            "embedding_mode": KEYWORD_EMBEDDING_MODE,
            "top_k": 3,
            "retrieval_mode": "auto",
            "use_llm": False,
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["intent"] == DOCUMENT_QA_INTENT
    assert payload["retrieval_mode"] == "rerank"
    assert "rerank_chunks" in payload["selected_tools"]
    assert payload["trace"][0]["step"] == "classify_intent"
    assert payload["retrieved_chunks"]
    assert payload["final_answer"]


def test_agent_ask_endpoint_refuses_out_of_scope_question():
    client = TestClient(app)
    text = FAQ_PATH.read_text(encoding="utf-8")

    document_response = client.post(
        "/documents",
        json={
            "text": text,
            "embedding_mode": KEYWORD_EMBEDDING_MODE,
            "chunk_size": 350,
            "chunk_overlap": 50,
        },
    )
    collection_name = document_response.json()["collection_name"]

    response = client.post(
        "/agent/ask",
        json={
            "collection_name": collection_name,
            "question": "今天上海天气怎么样？",
            "embedding_mode": KEYWORD_EMBEDDING_MODE,
            "top_k": 3,
            "retrieval_mode": "auto",
            "use_llm": False,
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["intent"] == OUT_OF_SCOPE_INTENT
    assert payload["confidence"] == "low"
    assert payload["fallback_reason"] == "out_of_scope"
    assert payload["retrieved_chunks"] == []
