# src/vectorstore.py
"""
Minimal Qdrant + OpenAI embedding helper.
Uses settings from config.py
"""

import time
from typing import List
import openai
from qdrant_client import QdrantClient

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    EMBED_BATCH,
)

# -------------------------------
# OpenAI Init
# -------------------------------
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY


# -------------------------------
# Qdrant client
# -------------------------------
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# -------------------------------
# Embedding functions
# -------------------------------
def _embed_batch(batch: List[str]) -> List[List[float]]:
    """Embed a batch of text using the modern OpenAI API."""
    resp = openai.Embedding.create(
        model=OPENAI_EMBED_MODEL,
        input=batch,
    )
    return [item["embedding"] for item in resp["data"]]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed many texts in batches."""
    if not texts:
        return []

    vectors: List[List[float]] = []

    for i in range(0, len(texts), EMBED_BATCH):
        chunk = texts[i : i + EMBED_BATCH]
        vectors.extend(_embed_batch(chunk))
        time.sleep(0.05)  # mild rate limiting

    return vectors


# -------------------------------
# Qdrant collection management
# -------------------------------
def ensure_collection(name: str, vector_size: int):
    """Create (or recreate) Qdrant collection if it doesn't already exist."""
    cols = client.get_collections()
    collections = getattr(cols, "collections", [])
    existing = {c.name for c in collections}

    if name not in existing:
        client.recreate_collection(
            collection_name=name,
            vectors_config={"size": vector_size, "distance": "Cosine"},
        )
