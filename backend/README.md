# DocuAsk FastAPI Backend

This backend provides the reusable API boundary for DocuAsk's RAG and Agentic RAG flows.

Current scope:

```text
GET /health
POST /documents
POST /documents/upload
POST /qa
POST /answer
POST /agent/ask
POST /evaluation
services/chunking.py
services/agent.py
services/bm25.py
services/embeddings.py
services/evaluation.py
services/generation.py
services/retrieval.py
services/rrf.py
services/rerank.py
services/document_parser.py
services/errors.py
services/logging_config.py
storage/chroma_db*/ (local, ignored)
```

It does not replace the Streamlit app yet. The goal is to introduce a reusable
backend service boundary before moving RAG logic out of `app.py`.

## Run

From the project root:

```powershell
python -m uvicorn backend.app:app --app-dir "." --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "docuask-api",
  "version": "0.1.0"
}
```

## Index a Document

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents" -Method Post -ContentType "application/json" -Body '{
  "text": "# Python 学习 FAQ\n\n## RAG 的基本流程是什么？\nRAG 会先检索资料，再让 LLM 根据资料回答。",
  "embedding_mode": "Teaching keyword embedding",
  "chunk_size": 350,
  "chunk_overlap": 50
}'
```

Expected fields:

```text
document_id
embedding_mode
collection_name
chunk_count
stored_chunk_count
```

## Upload a Document File

```powershell
$form = @{
  embedding_mode = "Teaching keyword embedding"
  chunk_size = "350"
  chunk_overlap = "50"
  file = Get-Item "examples/rag_faq.md"
}

Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents/upload" -Method Post -Form $form
```

Current file support:

```text
.txt
.md
.pdf
.docx
```

## Query a Document

Use the `collection_name` returned by `POST /documents`.

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/qa" -Method Post -ContentType "application/json" -Body '{
  "collection_name": "uploaded_document_chunks_keyword_xxxxxxxxxxxx",
  "question": "RAG 的基本流程是什么？",
  "embedding_mode": "Teaching keyword embedding",
  "top_k": 3,
  "retrieval_mode": "vector"
}'
```

Expected fields:

```text
question
embedding_mode
collection_name
top_k
retrieval_mode
retrieved_chunks
context
```

## Generate an Answer

Use the `collection_name` returned by `POST /documents`.

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/answer" -Method Post -ContentType "application/json" -Body '{
  "collection_name": "uploaded_document_chunks_keyword_xxxxxxxxxxxx",
  "question": "RAG 的基本流程是什么？",
  "embedding_mode": "Teaching keyword embedding",
  "top_k": 3,
  "retrieval_mode": "rrf"
}'
```

Expected fields:

```text
question
retrieved_chunks
context
answer
sources
```

If `DEEPSEEK_API_KEY` is not set, the endpoint returns 503.

## Ask with Agentic RAG

Use the `collection_name` returned by `POST /documents`.

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/agent/ask" -Method Post -ContentType "application/json" -Body '{
  "collection_name": "uploaded_document_chunks_keyword_xxxxxxxxxxxx",
  "question": "RAG 的基本流程是什么？",
  "embedding_mode": "Teaching keyword embedding",
  "top_k": 3,
  "retrieval_mode": "auto",
  "use_llm": false
}'
```

Expected fields:

```text
intent
selected_tools
retrieval_mode
retrieved_chunks
final_answer
trace
confidence
fallback_reason
```

`use_llm=false` is useful for local tests and demos without an API key. In `auto` mode, the agent classifies the question, chooses tools, retrieves or scans chunks, and may refuse out-of-scope questions.

## Test

From the project root:

```powershell
python -m pytest -q
```

Current automated coverage:

```text
POST /documents
POST /documents/upload markdown/docx upload, unsupported file type, and invalid PDF
POST /qa with vector, bm25, rrf, and rerank retrieval modes
POST /answer missing API key and unknown retrieval mode
POST /agent/ask intent routing, tool selection, trace, refusal, and local no-key answer
POST /evaluation with vector, bm25, rrf, and rerank retrieval modes
POST /evaluation with custom evaluation cases
failure case recording
404 for missing collection
400 for unknown retrieval mode
Teaching keyword retrieval metrics
```

## Evaluate Retrieval

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/evaluation" -Method Post -ContentType "application/json" -Body '{
  "text": "# Python 学习 FAQ\n\n## RAG 的基本流程是什么？\nRAG 会先检索资料，再让 LLM 根据资料回答。",
  "embedding_mode": "Teaching keyword embedding",
  "chunk_size": 350,
  "chunk_overlap": 50,
  "top_k": 3,
  "retrieval_mode": "vector"
}'
```

Expected fields:

```text
chunk_count
case_count
retrieval_mode
top_1_hit_rate
top_k_recall
rows
failure_cases
```
