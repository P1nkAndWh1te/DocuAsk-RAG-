# Delivery Checklist

## Project Scope

- Local document QA with Agentic RAG.
- Supported files: `.txt`, `.md`, `.pdf`, `.docx`.
- Retrieval modes: `vector`, `bm25`, `rrf`, `rerank`.
- Agent capabilities: intent routing, tool selection, trace, fallback/refusal, lightweight conversation history.
- Evaluation: retrieval metrics and Agent reliability metrics.
- Deployment: local Python and Docker Compose.

## Verified Commands

```powershell
python -m pytest -q
```

Expected current result:

```text
32 passed, 1 warning
```

```powershell
docker compose up --build -d
```

Expected current result:

```text
project002python-docuask-1 Up
http://localhost:8501 -> 200
```

## Current Metrics

Retrieval metrics on `examples/rag_faq.md`:

| Mode | Top-1 hit | Top-k recall |
|---|---:|---:|
| vector | 73.3% | 100% |
| bm25 | 86.7% | 93.3% |
| rrf | 80% | 93.3% |
| rerank | 86.7% | 100% |

Agent reliability metrics on default Agent evaluation cases:

| Metric | Value |
|---|---:|
| intent_accuracy | 100% |
| tool_selection_accuracy | 100% |
| refusal_accuracy | 100% |
| source_citation_rate | 100% |

## Public Repository Hygiene

- Learning process directories are ignored by Git.
- Chroma runtime storage is ignored by Git.
- Chroma collections are rebuilt on indexing to avoid stale local HNSW segment corruption.
- API keys are not stored in code.
- Docker uses `requirements-docker.txt` for a lighter demo path.
- BGE mode requires full `requirements.txt` because it depends on `sentence-transformers`.

## Known Limits

- No OCR for scanned PDFs.
- No multi-user auth or tenant isolation.
- No external cross-encoder reranker.
- No complex multi-agent runtime.
- No large-scale production benchmark.
