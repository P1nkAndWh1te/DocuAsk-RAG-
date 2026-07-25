from fastapi import FastAPI, File, Form, UploadFile
from openai import OpenAIError, RateLimitError
from pydantic import BaseModel, Field

from backend.services.agent import AGENT_RETRIEVAL_MODES, run_agentic_rag
from backend.services.bm25 import retrieve_relevant_chunks_bm25
from backend.services.chunking import split_text_into_chunks
from backend.services.document_parser import (
    SUPPORTED_UPLOAD_EXTENSIONS,
    get_file_extension,
    parse_uploaded_document,
)
from backend.services.embeddings import COLLECTION_NAMES, KEYWORD_EMBEDDING_MODE
from backend.services.errors import ErrorCode, raise_http_error
from backend.services.evaluation import (
    EVALUATION_CASES,
    RETRIEVAL_MODES,
    calculate_hit_rate,
    calculate_top_k_hit_rate,
    evaluate_retrieval,
    get_failure_cases,
)
from backend.services.generation import (
    MissingApiKeyError,
    format_sources,
    generate_answer_with_deepseek,
)
from backend.services.logging_config import get_logger
from backend.services.rerank import rerank_chunks
from backend.services.retrieval import (
    build_chunk_collection,
    format_retrieved_context,
    get_collection_name,
    get_chunks_from_collection,
    retrieve_relevant_chunks_from_collection,
)
from backend.services.rrf import retrieve_relevant_chunks_rrf


DEFAULT_CHUNK_SIZE = 350
DEFAULT_CHUNK_OVERLAP = 50
logger = get_logger(__name__)


class DocumentRequest(BaseModel):
    text: str = Field(..., min_length=1)
    embedding_mode: str = KEYWORD_EMBEDDING_MODE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


class DocumentResponse(BaseModel):
    document_id: str
    embedding_mode: str
    collection_name: str
    chunk_count: int
    stored_chunk_count: int


class RetrievedChunk(BaseModel):
    chunk_index: int
    text: str
    distance: float | None = None
    score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    scan_score: float | None = None


class QaRequest(BaseModel):
    collection_name: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    embedding_mode: str = KEYWORD_EMBEDDING_MODE
    top_k: int = Field(default=3, ge=1, le=10)
    retrieval_mode: str = "vector"


class QaResponse(BaseModel):
    question: str
    embedding_mode: str
    collection_name: str
    retrieval_mode: str
    top_k: int
    retrieved_chunks: list[RetrievedChunk]
    context: str


class AnswerRequest(QaRequest):
    pass


class AnswerResponse(QaResponse):
    answer: str
    sources: str


class AgentRequest(BaseModel):
    collection_name: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    embedding_mode: str = KEYWORD_EMBEDDING_MODE
    top_k: int = Field(default=3, ge=1, le=10)
    retrieval_mode: str = "auto"
    use_llm: bool = True


class AgentTraceStep(BaseModel):
    step: str
    tool: str = ""
    result: str
    details: str = ""


class AgentResponse(BaseModel):
    question: str
    intent: str
    embedding_mode: str
    collection_name: str
    retrieval_mode: str
    selected_tools: list[str]
    top_k: int
    retrieved_chunks: list[RetrievedChunk]
    context: str
    sources: str
    final_answer: str
    trace: list[AgentTraceStep]
    confidence: str
    fallback_reason: str


class EvaluationCase(BaseModel):
    question: str = Field(..., min_length=1)
    expected_top_chunk: int = Field(..., ge=1)


class EvaluationRequest(BaseModel):
    text: str = Field(..., min_length=1)
    embedding_mode: str = KEYWORD_EMBEDDING_MODE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = Field(default=3, ge=1, le=10)
    retrieval_mode: str = "vector"
    evaluation_cases: list[EvaluationCase] | None = None


class EvaluationRow(BaseModel):
    question: str
    embedding_mode: str
    retrieval_mode: str
    expected_top_chunk: str
    matched_concepts: str
    top_chunks: str
    best_score: str
    hit: bool
    top_k_hit: bool
    failure_reason: str


class EvaluationResponse(BaseModel):
    embedding_mode: str
    retrieval_mode: str
    chunk_count: int
    case_count: int
    top_1_hit_rate: float
    top_k_recall: float
    rows: list[EvaluationRow]
    failure_cases: list[EvaluationRow]


app = FastAPI(
    title="DocuAsk API",
    description="Backend service for the local document RAG QA system.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "docuask-api",
        "version": "0.1.0",
    }


@app.post("/documents", response_model=DocumentResponse)
def create_document(request: DocumentRequest) -> DocumentResponse:
    return create_document_from_text(
        text=request.text,
        embedding_mode=request.embedding_mode,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )


@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    embedding_mode: str = Form(KEYWORD_EMBEDDING_MODE),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
) -> DocumentResponse:
    filename = file.filename or ""
    extension = get_file_extension(filename)

    if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise_http_error(
            400,
            ErrorCode.UNSUPPORTED_FILE_TYPE,
            "unsupported file type",
        )

    raw_bytes = await file.read()
    try:
        text, parser_name = parse_uploaded_document(raw_bytes, filename)
    except ValueError as exc:
        raise_http_error(400, ErrorCode.UNSUPPORTED_FILE_TYPE, str(exc))
    except Exception as exc:
        logger.exception("document parse failed filename=%s", filename)
        raise_http_error(
            400,
            ErrorCode.DOCUMENT_PARSE_FAILED,
            "document parse failed",
        )

    logger.info("uploaded document parsed filename=%s parser=%s", filename, parser_name)
    return create_document_from_text(
        text=text,
        embedding_mode=embedding_mode,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def decode_uploaded_text(raw_bytes: bytes) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw_bytes.decode("utf-8", errors="replace")


def create_document_from_text(
    text: str,
    embedding_mode: str,
    chunk_size: int,
    chunk_overlap: int,
) -> DocumentResponse:
    if embedding_mode not in COLLECTION_NAMES:
        raise_http_error(
            400,
            ErrorCode.UNSUPPORTED_EMBEDDING_MODE,
            "unsupported embedding mode",
        )

    try:
        chunks = split_text_into_chunks(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as exc:
        raise_http_error(400, ErrorCode.INVALID_CHUNKING_CONFIG, str(exc))

    if not chunks:
        raise_http_error(400, ErrorCode.EMPTY_DOCUMENT, "document text is empty")

    collection = build_chunk_collection(chunks, embedding_mode)
    if collection is None:
        raise_http_error(
            400,
            ErrorCode.NO_EMBEDDABLE_CHUNKS,
            "document has no embeddable chunks",
        )

    collection_name = get_collection_name(chunks, embedding_mode)
    document_id = collection_name.rsplit("_", maxsplit=1)[-1]

    return DocumentResponse(
        document_id=document_id,
        embedding_mode=embedding_mode,
        collection_name=collection_name,
        chunk_count=len(chunks),
        stored_chunk_count=collection.count(),
    )


@app.post("/qa", response_model=QaResponse)
def query_document(request: QaRequest) -> QaResponse:
    if request.embedding_mode not in COLLECTION_NAMES:
        raise_http_error(400, ErrorCode.UNSUPPORTED_EMBEDDING_MODE, "unsupported embedding mode")

    if request.retrieval_mode not in RETRIEVAL_MODES:
        raise_http_error(400, ErrorCode.UNSUPPORTED_RETRIEVAL_MODE, "unsupported retrieval mode")

    retrieved_chunks = retrieve_chunks_for_request(request)

    if retrieved_chunks is None:
        raise_http_error(404, ErrorCode.COLLECTION_NOT_FOUND, "collection not found")

    return build_qa_response(request, retrieved_chunks)


@app.post("/answer", response_model=AnswerResponse)
def answer_document(request: AnswerRequest) -> AnswerResponse:
    if request.embedding_mode not in COLLECTION_NAMES:
        raise_http_error(400, ErrorCode.UNSUPPORTED_EMBEDDING_MODE, "unsupported embedding mode")

    if request.retrieval_mode not in RETRIEVAL_MODES:
        raise_http_error(400, ErrorCode.UNSUPPORTED_RETRIEVAL_MODE, "unsupported retrieval mode")

    retrieved_chunks = retrieve_chunks_for_request(request)
    if retrieved_chunks is None:
        raise_http_error(404, ErrorCode.COLLECTION_NOT_FOUND, "collection not found")

    try:
        answer = generate_answer_with_deepseek(request.question, retrieved_chunks)
    except MissingApiKeyError as exc:
        raise_http_error(503, ErrorCode.MISSING_API_KEY, str(exc))
    except RateLimitError as exc:
        raise_http_error(429, ErrorCode.LLM_RATE_LIMIT, "LLM quota or rate limit failed")
    except OpenAIError as exc:
        raise_http_error(502, ErrorCode.LLM_REQUEST_FAILED, "LLM request failed")

    qa_response = build_qa_response(request, retrieved_chunks)
    return AnswerResponse(
        **qa_response.model_dump(),
        answer=answer,
        sources=format_sources(retrieved_chunks),
    )


@app.post("/agent/ask", response_model=AgentResponse)
def ask_agent(request: AgentRequest) -> AgentResponse:
    if request.embedding_mode not in COLLECTION_NAMES:
        raise_http_error(400, ErrorCode.UNSUPPORTED_EMBEDDING_MODE, "unsupported embedding mode")

    if request.retrieval_mode not in AGENT_RETRIEVAL_MODES:
        raise_http_error(400, ErrorCode.UNSUPPORTED_RETRIEVAL_MODE, "unsupported retrieval mode")

    chunks = get_chunks_from_collection(request.collection_name)
    if chunks is None:
        raise_http_error(404, ErrorCode.COLLECTION_NOT_FOUND, "collection not found")

    try:
        result = run_agentic_rag(
            question=request.question,
            chunks=chunks,
            embedding_mode=request.embedding_mode,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            use_llm=request.use_llm,
        )
    except RateLimitError as exc:
        raise_http_error(429, ErrorCode.LLM_RATE_LIMIT, "LLM quota or rate limit failed")
    except OpenAIError as exc:
        raise_http_error(502, ErrorCode.LLM_REQUEST_FAILED, "LLM request failed")

    return AgentResponse(
        question=result["question"],
        intent=result["intent"],
        embedding_mode=request.embedding_mode,
        collection_name=request.collection_name,
        retrieval_mode=result["retrieval_mode"],
        selected_tools=result["selected_tools"],
        top_k=request.top_k,
        retrieved_chunks=[
            RetrievedChunk(
                chunk_index=item["chunk_index"],
                text=item["text"],
                distance=item.get("distance"),
                score=item.get("score"),
                rrf_score=item.get("rrf_score"),
                rerank_score=item.get("rerank_score"),
                scan_score=item.get("scan_score"),
            )
            for item in result["retrieved_chunks"]
        ],
        context=result["context"],
        sources=result["sources"],
        final_answer=result["final_answer"],
        trace=[AgentTraceStep(**item) for item in result["trace"]],
        confidence=result["confidence"],
        fallback_reason=result["fallback_reason"],
    )


def retrieve_chunks_for_request(request: QaRequest) -> list[dict] | None:
    if request.retrieval_mode == "vector":
        return retrieve_relevant_chunks_from_collection(
            request.collection_name,
            request.question,
            top_k=request.top_k,
            embedding_mode=request.embedding_mode,
        )

    chunks = get_chunks_from_collection(request.collection_name)
    if chunks is None:
        return None

    if request.retrieval_mode == "bm25":
        return retrieve_relevant_chunks_bm25(
            request.question,
            chunks,
            top_k=request.top_k,
        )

    rrf_chunks = retrieve_relevant_chunks_rrf(
        request.question,
        chunks,
        top_k=min(len(chunks), max(request.top_k * 2, request.top_k)),
        embedding_mode=request.embedding_mode,
    )
    if request.retrieval_mode == "rerank":
        return rerank_chunks(request.question, rrf_chunks, top_k=request.top_k)

    return rrf_chunks[:request.top_k]


def build_qa_response(request: QaRequest, retrieved_chunks: list[dict]) -> QaResponse:
    return QaResponse(
        question=request.question,
        embedding_mode=request.embedding_mode,
        collection_name=request.collection_name,
        retrieval_mode=request.retrieval_mode,
        top_k=request.top_k,
        retrieved_chunks=[
            RetrievedChunk(
                chunk_index=item["chunk_index"],
                text=item["text"],
                distance=item.get("distance"),
                score=item.get("score"),
                rrf_score=item.get("rrf_score"),
                rerank_score=item.get("rerank_score"),
                scan_score=item.get("scan_score"),
            )
            for item in retrieved_chunks
        ],
        context=format_retrieved_context(retrieved_chunks),
    )


@app.post("/evaluation", response_model=EvaluationResponse)
def evaluate_document(request: EvaluationRequest) -> EvaluationResponse:
    if request.embedding_mode not in COLLECTION_NAMES:
        raise_http_error(400, ErrorCode.UNSUPPORTED_EMBEDDING_MODE, "unsupported embedding mode")

    if request.retrieval_mode not in RETRIEVAL_MODES:
        raise_http_error(400, ErrorCode.UNSUPPORTED_RETRIEVAL_MODE, "unsupported retrieval mode")

    try:
        chunks = split_text_into_chunks(
            request.text,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except ValueError as exc:
        raise_http_error(400, ErrorCode.INVALID_CHUNKING_CONFIG, str(exc))

    if not chunks:
        raise_http_error(400, ErrorCode.EMPTY_DOCUMENT, "document text is empty")

    evaluation_cases = (
        [case.model_dump() for case in request.evaluation_cases]
        if request.evaluation_cases is not None
        else EVALUATION_CASES
    )

    rows = evaluate_retrieval(
        evaluation_cases,
        chunks,
        top_k=request.top_k,
        embedding_mode=request.embedding_mode,
        retrieval_mode=request.retrieval_mode,
    )

    return EvaluationResponse(
        embedding_mode=request.embedding_mode,
        retrieval_mode=request.retrieval_mode,
        chunk_count=len(chunks),
        case_count=len(rows),
        top_1_hit_rate=calculate_hit_rate(rows),
        top_k_recall=calculate_top_k_hit_rate(rows),
        rows=[EvaluationRow(**row) for row in rows],
        failure_cases=[EvaluationRow(**row) for row in get_failure_cases(rows)],
    )
