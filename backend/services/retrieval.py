import hashlib
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from backend.services.embeddings import (
    COLLECTION_NAMES,
    KEYWORD_EMBEDDING_MODE,
    embed_for_mode,
)


CHROMA_DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "chroma_db_v2"
COLLECTION_SCHEMA_VERSION = "v3"


def get_chroma_client():
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_PATH))


def get_collection_name(chunks: list[str], embedding_mode: str) -> str:
    source_text = "\n\n".join(chunks)
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:12]
    return f"{COLLECTION_NAMES[embedding_mode]}_{COLLECTION_SCHEMA_VERSION}_{digest}"


def get_chunk_collection(client, collection_name: str):
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def get_existing_collection(client, collection_name: str):
    try:
        return client.get_collection(name=collection_name)
    except NotFoundError:
        return None


def get_chunks_from_collection(collection_name: str) -> list[str] | None:
    client = get_chroma_client()
    collection = get_existing_collection(client, collection_name)
    if collection is None:
        return None

    result = collection.get(include=["documents", "metadatas"])
    chunk_items = []
    for document, metadata in zip(result["documents"], result["metadatas"]):
        chunk_items.append(
            {
                "text": document,
                "chunk_index": metadata["chunk_index"],
            }
        )

    chunk_items.sort(key=lambda item: item["chunk_index"])
    return [item["text"] for item in chunk_items]


def build_chunk_collection(chunks: list[str], embedding_mode: str):
    client = get_chroma_client()
    collection_name = get_collection_name(chunks, embedding_mode)

    embedded_chunks = []
    for index, chunk in enumerate(chunks, start=1):
        embedding = embed_for_mode(chunk, embedding_mode)
        if any(embedding):
            embedded_chunks.append((index, chunk, embedding))

    if not embedded_chunks:
        return None

    try:
        client.delete_collection(collection_name)
    except NotFoundError:
        pass

    collection = get_chunk_collection(client, collection_name)

    ids = [f"chunk-{index}" for index, _, _ in embedded_chunks]
    documents = [chunk for _, chunk, _ in embedded_chunks]
    embeddings = [embedding for _, _, embedding in embedded_chunks]
    metadatas = [{"chunk_index": index} for index, _, _ in embedded_chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection


def retrieve_relevant_chunks(
    question: str,
    chunks: list[str],
    top_k: int,
    embedding_mode: str = KEYWORD_EMBEDDING_MODE,
) -> list[dict]:
    if not chunks:
        return []

    query_embedding = embed_for_mode(question, embedding_mode)
    if not any(query_embedding):
        return []

    collection = build_chunk_collection(chunks, embedding_mode)
    if collection is None:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks = []
    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        retrieved_chunks.append(
            {
                "text": document,
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
            }
        )

    return retrieved_chunks


def retrieve_relevant_chunks_from_collection(
    collection_name: str,
    question: str,
    top_k: int,
    embedding_mode: str = KEYWORD_EMBEDDING_MODE,
) -> list[dict] | None:
    query_embedding = embed_for_mode(question, embedding_mode)
    if not any(query_embedding):
        return []

    client = get_chroma_client()
    collection = get_existing_collection(client, collection_name)
    if collection is None:
        return None

    collection_count = collection.count()
    if collection_count == 0:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection_count),
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks = []
    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        retrieved_chunks.append(
            {
                "text": document,
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
            }
        )

    return retrieved_chunks


def format_retrieved_context(retrieved_chunks: list[dict]) -> str:
    context_blocks = []

    for item in retrieved_chunks:
        context_blocks.append(
            f"[Chunk {item['chunk_index']}]\n{item['text']}"
        )

    return "\n\n---\n\n".join(context_blocks)
