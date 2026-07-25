import sys
from pathlib import Path

import streamlit as st
from openai import OpenAIError, RateLimitError


APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.services.chunking import split_text_into_chunks
from backend.services.agent import (
    AGENT_EVALUATION_CASES,
    AGENT_RETRIEVAL_MODES,
    evaluate_agentic_rag,
    run_agentic_rag,
)
from backend.services.document_parser import parse_uploaded_document
from backend.services.embeddings import (
    BGE_EMBEDDING_MODE,
    BGE_MODEL_NAME,
    KEYWORD_EMBEDDING_MODE,
    get_matched_concepts,
)
from backend.services.evaluation import (
    EVALUATION_CASES,
    RETRIEVAL_MODES,
    calculate_hit_rate,
    calculate_top_k_hit_rate,
    evaluate_retrieval,
    retrieve_for_mode,
)
from backend.services.generation import (
    MissingApiKeyError,
    generate_answer_with_deepseek,
)
from backend.services.retrieval import format_retrieved_context


MAX_PREVIEW_CHARS = 2000
DEFAULT_CHUNK_SIZE = 350
DEFAULT_CHUNK_OVERLAP = 50
TOP_K = 3

def read_uploaded_text(uploaded_file) -> tuple[str, str]:
    raw_bytes = uploaded_file.getvalue()
    text, parser_name = parse_uploaded_document(raw_bytes, uploaded_file.name)
    return text, parser_name


def main() -> None:
    st.set_page_config(page_title="RAG QA System", page_icon="RAG", layout="wide")
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    st.title("DocuAsk Agentic RAG")
    st.caption("Local document QA with intent routing, tool selection, evaluation, and source citations")

    embedding_mode = st.sidebar.radio(
        "Embedding mode",
        [KEYWORD_EMBEDDING_MODE, BGE_EMBEDDING_MODE],
        help=(
            "Teaching mode is fast and explainable. BGE mode uses a real Chinese "
            "embedding model and may take longer on first load."
        ),
    )

    st.sidebar.caption(f"Current mode: {embedding_mode}")
    if embedding_mode == BGE_EMBEDDING_MODE:
        st.sidebar.caption(f"Model: {BGE_MODEL_NAME}")
        st.sidebar.info(
            "BGE mode loads a local Chinese embedding model. The first evaluation "
            "or question may take longer."
        )

    retrieval_mode = st.sidebar.selectbox(
        "Retrieval mode",
        sorted(RETRIEVAL_MODES),
        index=sorted(RETRIEVAL_MODES).index("rerank"),
        help="Rerank retrieves candidates first and then reorders them with a local reranker.",
    )
    st.sidebar.caption(f"Current retrieval: {retrieval_mode}")
    agentic_mode = st.sidebar.checkbox(
        "Agentic RAG mode",
        value=True,
        help="Use intent routing, tool selection, trace, and fallback logic before answering.",
    )
    agent_retrieval_mode = st.sidebar.selectbox(
        "Agent retrieval",
        sorted(AGENT_RETRIEVAL_MODES),
        index=sorted(AGENT_RETRIEVAL_MODES).index("auto"),
        help="Auto lets the agent choose a retrieval strategy based on the question intent.",
    )

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "md", "pdf", "docx"],
        help="Supports TXT, Markdown, PDF, and Word documents.",
    )

    document_text = ""
    chunks = []

    if uploaded_file is not None:
        try:
            document_text, encoding = read_uploaded_text(uploaded_file)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error("Could not parse the uploaded document.")
            st.exception(exc)
            st.stop()

        file_size = len(uploaded_file.getvalue())
        chunks = split_text_into_chunks(
            document_text,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )

        st.success(f"Uploaded file: {uploaded_file.name}")
        st.write(f"Size: {file_size} bytes")
        st.write(f"Characters: {len(document_text)}")
        st.write(f"Detected encoding: {encoding}")
        st.write(f"Chunks: {len(chunks)}")

        with st.expander("Document preview", expanded=True):
            st.code(document_text[:MAX_PREVIEW_CHARS], language="markdown")

        st.subheader("Chunks")
        st.caption(
            f"Chunk size: {DEFAULT_CHUNK_SIZE} characters, "
            f"overlap: {DEFAULT_CHUNK_OVERLAP} characters"
        )

        for index, chunk in enumerate(chunks, start=1):
            with st.expander(f"Chunk {index} | {len(chunk)} characters"):
                st.code(chunk, language="markdown")

        st.subheader("Retrieval evaluation")
        st.caption(
            "Run fixed questions to inspect which chunks are retrieved before LLM generation."
        )
        evaluation_rows = evaluate_retrieval(
            EVALUATION_CASES,
            chunks,
            top_k=TOP_K,
            embedding_mode=embedding_mode,
            retrieval_mode=retrieval_mode,
        )
        hit_rate = calculate_hit_rate(evaluation_rows)
        top_k_hit_rate = calculate_top_k_hit_rate(evaluation_rows)
        metric_columns = st.columns(2)
        metric_columns[0].metric("Top-1 hit", f"{hit_rate:.0%}")
        metric_columns[1].metric("Top-k recall", f"{top_k_hit_rate:.0%}")
        st.dataframe(evaluation_rows, use_container_width=True, hide_index=True)

        st.subheader("Agent evaluation")
        st.caption(
            "Run fixed agent cases to inspect intent routing, tool selection, refusal, and source citation."
        )
        agent_evaluation = evaluate_agentic_rag(
            AGENT_EVALUATION_CASES,
            chunks,
            embedding_mode=embedding_mode,
            top_k=TOP_K,
        )
        agent_metric_columns = st.columns(4)
        agent_metric_columns[0].metric("Intent accuracy", f"{agent_evaluation['intent_accuracy']:.0%}")
        agent_metric_columns[1].metric("Tool accuracy", f"{agent_evaluation['tool_selection_accuracy']:.0%}")
        agent_metric_columns[2].metric("Refusal accuracy", f"{agent_evaluation['refusal_accuracy']:.0%}")
        agent_metric_columns[3].metric("Source rate", f"{agent_evaluation['source_citation_rate']:.0%}")
        st.dataframe(agent_evaluation["rows"], use_container_width=True, hide_index=True)

    question = st.text_area(
        "Question",
        placeholder="Ask a question about the uploaded document.",
        height=120,
    )

    submitted = st.button("Ask", type="primary")

    if submitted:
        if uploaded_file is None:
            st.warning("Please upload a document first.")
            return

        if not document_text.strip():
            st.warning("The uploaded document is empty.")
            return

        if not question.strip():
            st.warning("Please enter a question.")
            return

        if agentic_mode:
            st.subheader("Agentic answer")
            try:
                agent_result = run_agentic_rag(
                    question=question,
                    chunks=chunks,
                    top_k=TOP_K,
                    embedding_mode=embedding_mode,
                    retrieval_mode=agent_retrieval_mode,
                    use_llm=True,
                    conversation_history=st.session_state.conversation_history,
                )
            except RateLimitError as exc:
                st.error("DeepSeek request reached the server, but quota or rate limit failed.")
                st.exception(exc)
                return
            except OpenAIError as exc:
                st.error("DeepSeek request failed.")
                st.exception(exc)
                return

            st.write(agent_result["final_answer"])
            if agent_result["effective_question"] != agent_result["question"]:
                st.caption(f"Effective question: {agent_result['effective_question']}")

            metric_columns = st.columns(3)
            metric_columns[0].metric("Intent", agent_result["intent"])
            metric_columns[1].metric("Retrieval", agent_result["retrieval_mode"])
            metric_columns[2].metric("Confidence", agent_result["confidence"])

            if agent_result["fallback_reason"]:
                st.warning(f"Fallback: {agent_result['fallback_reason']}")

            st.write("Selected tools:", ", ".join(agent_result["selected_tools"]) or "none")
            st.write("Sources:", agent_result["sources"] or "none")

            st.subheader("Agent trace")
            st.dataframe(agent_result["trace"], use_container_width=True, hide_index=True)

            st.session_state.conversation_history.append(
                {
                    "question": question,
                    "answer": agent_result["final_answer"],
                    "intent": agent_result["intent"],
                    "sources": agent_result["sources"],
                }
            )
            st.session_state.conversation_history = st.session_state.conversation_history[-3:]

            if st.session_state.conversation_history:
                with st.expander("Conversation history"):
                    st.dataframe(
                        st.session_state.conversation_history,
                        use_container_width=True,
                        hide_index=True,
                    )

            st.subheader("Retrieved chunks")
            for rank, item in enumerate(agent_result["retrieved_chunks"], start=1):
                score_label = format_score_label(item)
                with st.expander(
                    f"Rank {rank} | Chunk {item['chunk_index']} | {score_label}",
                    expanded=rank == 1,
                ):
                    st.code(item["text"], language="markdown")

            return

        st.subheader("Answer")
        retrieved_chunks = retrieve_for_mode(
            question,
            chunks,
            top_k=TOP_K,
            embedding_mode=embedding_mode,
            retrieval_mode=retrieval_mode,
        )
        matched_concepts = get_matched_concepts(question)

        if not retrieved_chunks:
            st.warning(
                "No relevant chunks found by the selected embedding mode. Try a question "
                "with keywords such as Python, API, RAG, SQL, embedding, or Chroma."
            )
            return

        st.write(
            f"The app has embedded chunks with {embedding_mode} and retrieved "
            f"the most relevant chunks with {retrieval_mode}. DeepSeek will generate the final "
            "answer from these chunks only."
        )

        st.write("Matched concepts:", ", ".join(matched_concepts))
        st.caption("Scores depend on retrieval mode. For Chroma distance, lower means more similar.")

        st.subheader("Context for later LLM")
        st.code(format_retrieved_context(retrieved_chunks), language="markdown")

        st.subheader("Sources")
        st.write(
            ", ".join(
                f"Chunk {item['chunk_index']}" for item in retrieved_chunks
            )
        )

        try:
            with st.spinner("Generating answer with DeepSeek..."):
                final_answer = generate_answer_with_deepseek(question, retrieved_chunks)
        except MissingApiKeyError:
            st.warning(
                "DEEPSEEK_API_KEY is not set in the current environment. "
                "Set it before running the app to generate the final answer."
            )
            final_answer = ""
        except RateLimitError as exc:
            st.error("DeepSeek request reached the server, but quota or rate limit failed.")
            st.exception(exc)
            final_answer = ""
        except OpenAIError as exc:
            st.error("DeepSeek request failed.")
            st.exception(exc)
            final_answer = ""

        if final_answer:
            st.subheader("Final answer")
            st.write(final_answer)

        st.subheader("Retrieved chunks")
        for rank, item in enumerate(retrieved_chunks, start=1):
            score_label = format_score_label(item)
            with st.expander(
                f"Rank {rank} | Chunk {item['chunk_index']} | {score_label}",
                expanded=rank == 1,
            ):
                st.code(item["text"], language="markdown")


def format_score_label(item: dict) -> str:
    for key, label in (
        ("rerank_score", "Rerank score"),
        ("rrf_score", "RRF score"),
        ("score", "BM25 score"),
        ("scan_score", "Scan score"),
        ("distance", "Distance"),
    ):
        if key in item and item[key] is not None:
            return f"{label} {item[key]:.4f}"

    return "Score n/a"


if __name__ == "__main__":
    main()
