# src/vectorstore.py
"""
Minimal, modern Qdrant + OpenAI helper (simple & robust).

Assumes:
  - qdrant-client >= 1.16.0
  - openai >= 1.0.0 (modern OpenAI client)
Exports:
  - client
  - embed_texts(texts)
  - ensure_collection(name, vector_size)
  - search(collection_name, query_vector, limit=8, with_payload=True)
  - upsert(collection_name, points)
"""

from __future__ import annotations

import time
from typing import List, Dict, Any, Optional

# modern OpenAI client
from openai import OpenAI as OpenAIClient
# qdrant
from qdrant_client import QdrantClient

# local config - adjust names/path if your project differs
from config import QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY, OPENAI_EMBED_MODEL, EMBED_BATCH

# -----------------------
# OpenAI client (modern)
# -----------------------
_openai = OpenAIClient(api_key=OPENAI_API_KEY)

# -----------------------
# Qdrant client
# -----------------------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# -----------------------
# Embeddings
# -----------------------
def _embed_batch(batch: List[str]) -> List[List[float]]:
    """
    Embed a small batch using modern OpenAI client.
    Handles both dict-like responses and response objects (CreateEmbeddingResponse).
    """
    if not batch:
        return []

    resp = _openai.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)

    # Try object-style access first (CreateEmbeddingResponse)
    data = None
    if hasattr(resp, "data"):
        data = resp.data  # list of objects with .embedding
    elif isinstance(resp, dict):
        data = resp.get("data")

    if data is None:
        # Last resort: attempt dict-like access; if it fails raise helpful error
        try:
            data = resp["data"]
        except Exception as e:
            raise RuntimeError(f"Unexpected embedding response shape: {type(resp)}") from e

    vectors: List[List[float]] = []
    for item in data:
        # item could be an object with .embedding or a dict with ['embedding']
        if hasattr(item, "embedding"):
            vectors.append(list(item.embedding))
        elif isinstance(item, dict) and "embedding" in item:
            vectors.append(list(item["embedding"]))
        else:
            raise RuntimeError("Unexpected embedding item shape; missing 'embedding' field.")
    return vectors


def embed_texts(texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
    """Embed many texts using batching. Returns list of vectors in same order."""
    if not texts:
        return []
    b = batch_size or EMBED_BATCH or 1
    vectors: List[List[float]] = []
    for i in range(0, len(texts), b):
        chunk = texts[i : i + b]
        vectors.extend(_embed_batch(chunk))
        time.sleep(0.05)
    return vectors


# -----------------------
# Collection management
# -----------------------
def ensure_collection(name: str, vector_size: int, distance: str = "Cosine") -> None:
    """Create a collection if it doesn't exist (uses recreate_collection)."""
    cols = client.get_collections()
    collections = getattr(cols, "collections", cols) or []
    existing = {c.name for c in collections} if collections else set()
    if name not in existing:
        client.recreate_collection(
            collection_name=name,
            vectors_config={"size": vector_size, "distance": distance},
        )


# -----------------------
# Search & Upsert (modern qdrant-client)
# -----------------------
def search(collection_name: str, query_vector: List[float], limit: int = 8, with_payload: bool = True) -> List[Dict[str, Any]]:
    """
    Query Qdrant (query_points) and return a simple list of hits:
      [ {"payload": {...}, "score": <float-or-none>}, ... ]
    """
    resp = client.query_points(collection_name=collection_name, query=query_vector, limit=limit, with_payload=with_payload)
    points = getattr(resp, "points", []) or []
    result = []
    for p in points:
        if isinstance(p, dict):
            payload = p.get("payload", {}) or {}
            score = p.get("score") or p.get("distance") or None
        else:
            payload = getattr(p, "payload", {}) or {}
            score = getattr(p, "score", None) or getattr(p, "distance", None)
            # try conversion if payload is not a plain dict
            try:
                if not isinstance(payload, dict):
                    payload = dict(payload)
            except Exception:
                pass
        result.append({"payload": payload, "score": score})
    return result


def upsert(collection_name: str, points: List[Dict[str, Any]]) -> None:
    """
    Upsert list of points into Qdrant.
    Point format: {"id": str|int, "vector": [...], "payload": {...}}
    """
    client.upsert(collection_name=collection_name, points=points)
