from functools import lru_cache


BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
KEYWORD_EMBEDDING_MODE = "Teaching keyword embedding"
BGE_EMBEDDING_MODE = "BGE Chinese embedding"
COLLECTION_NAMES = {
    KEYWORD_EMBEDDING_MODE: "uploaded_document_chunks_keyword",
    BGE_EMBEDDING_MODE: "uploaded_document_chunks_bge",
}

CONCEPTS = {
    "python": ["python", "脚本", "程序"],
    "llm_api": ["llm api", "deepseek api", "openai-compatible"],
    "api": [
        "api",
        "api key",
        "接口",
        "调用",
        "请求",
        "密钥",
        "环境变量",
        "base_url",
        "sdk",
        "额度",
        "plus",
        "互通",
    ],
    "git": ["git", "提交", "版本", "仓库"],
    "rag": ["rag", "检索增强", "来源", "引用", "基本流程", "文档切分", "检索"],
    "data": ["数据", "csv", "表格", "pandas"],
    "sql": ["sql", "查询", "表格", "结构化"],
    "embedding": ["embedding", "向量", "数字向量", "相似度", "转换"],
    "chroma": ["chroma", "向量数据库", "存储", "查找", "相似"],
    "llamaindex": ["llamaindex", "索引"],
    "deepseek": ["deepseek", "deepseek_api_key", "大模型", "llm", "openai sdk"],
}


def embed_text(text: str) -> list[float]:
    lowered_text = text.lower()
    vector = []

    for keywords in CONCEPTS.values():
        score = 0
        for keyword in keywords:
            score += lowered_text.count(keyword.lower())
        vector.append(float(score))

    return vector


@lru_cache(maxsize=1)
def load_bge_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "BGE embedding requires sentence-transformers. "
            "Install the full requirements.txt to use BGE mode."
        ) from exc

    return SentenceTransformer(BGE_MODEL_NAME)


def embed_text_with_bge(text: str) -> list[float]:
    model = load_bge_model()
    embedding = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return embedding.tolist()


def get_matched_concepts(text: str) -> list[str]:
    vector = embed_text(text)
    concepts = list(CONCEPTS.keys())
    return [concept for concept, value in zip(concepts, vector) if value > 0]


def embed_for_mode(text: str, embedding_mode: str) -> list[float]:
    if embedding_mode == BGE_EMBEDDING_MODE:
        return embed_text_with_bge(text)

    return embed_text(text)
