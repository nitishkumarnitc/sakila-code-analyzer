# src/vectorstore.py
import os
import time
from typing import List
from qdrant_client import QdrantClient
import openai

# ---------------------
# Configuration via ENV
# ---------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Embedding batch (keeps memory low)
BATCH = int(os.environ.get("QDRANT_EMBED_BATCH", "4"))

if OPENAI_API_KEY:
    # For openai>=1.x the library uses openai.api_key for some helpers; safe to set.
    openai.api_key = OPENAI_API_KEY

# ---------------------
# Qdrant client init
# ---------------------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ---------------------
# Embeddings (batched, compatible)
# ---------------------
def _openai_embed_batch(batch: List[str]) -> List[List[float]]:
    """
    Create embeddings for a single batch. Handles new and old openai SDK shapes.
    Returns list of lists (vectors).
    """
    if not batch:
        return []

    # New SDK: openai.embeddings.create
    if hasattr(openai, "embeddings"):
        resp = openai.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
        # resp.data is list of items with .embedding
        vectors = []
        for item in resp.data:
            vec = getattr(item, "embedding", None)
            if vec is None and isinstance(item, dict):
                vec = item.get("embedding")
            vectors.append(vec)
        return vectors

    # Fallback (older SDK)
    resp = openai.Embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
    return [item["embedding"] for item in resp["data"]]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Compute embeddings for `texts` in small batches to limit memory usage.
    """
    if not texts:
        return []

    all_vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        vectors = _openai_embed_batch(batch)
        all_vectors.extend(vectors)
        # small sleep to be gentle with API / memory
        time.sleep(0.1)
    return all_vectors


# ---------------------
# Collection management
# ---------------------
def ensure_collection(collection_name: str, vector_size: int):
    """
    Create collection if it doesn't exist. If it exists, do nothing.
    """
    try:
        cols = client.get_collections().collections
        existing = [c.name for c in cols]
    except Exception:
        # Be resilient to client shapes
        meta = client.get_collections()
        if isinstance(meta, dict) and "collections" in meta:
            existing = [c["name"] for c in meta["collections"]]
        else:
            existing = []

    if collection_name not in existing:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={"size": vector_size, "distance": "Cosine"},
        )
