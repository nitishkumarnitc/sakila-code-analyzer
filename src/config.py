# src/config.py
"""Simple centralized config — maps environment variables to typed constants.
Includes aliases to remain compatible with analyzer/vectorstore expectations.
"""

import os
from pathlib import Path
from typing import Optional

# small helpers
def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default

def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default

def _bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

# ---------------------
# OpenAI / LangChain
# ---------------------
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


# ---------------------
# Qdrant
# ---------------------
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None

# ---------------------
# Directories
# ---------------------
WORKSPACE_DIR: str = os.getenv("WORKSPACE_DIR", "/app/workspaces")
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/app/outputs")

# ---------------------
# Retrieval / chunking
# ---------------------
TOP_K: int = _int("TOP_K", 8)
CHUNK_SIZE: int = _int("CHUNK_SIZE", 1500)
CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 270)

# ---------------------
# Batching / parallelism
# ---------------------
UPSERT_BATCH: int = _int("UPSERT_BATCH", 128)
EMBED_BATCH: int = _int("EMBED_BATCH", 64)
EMBED_CONCURRENCY: int = _int("EMBED_CONCURRENCY", 6)
UPSERTER_WORKERS: int = _int("UPSERTER_WORKERS", 6)
CHUNKER_WORKERS: int = _int("CHUNKER_WORKERS", 6)

# Provide both EMBED_BATCH and QDRANT_EMBED_BATCH aliases used elsewhere
QDRANT_EMBED_BATCH: int = EMBED_BATCH

# ---------------------
# Embed cache
# ---------------------
EMBED_CACHE_DIR: str = os.getenv("EMBED_CACHE_DIR", "/tmp/embed_cache")

# ---------------------
# Validation / flags
# ---------------------
VALIDATE_REPORT: bool = _bool("VALIDATE_REPORT", True)
VALIDATE_AFTER: bool = VALIDATE_REPORT   # analyzer compatibility

DRY_RUN: bool = _bool("DRY_RUN", False)
SKIP_CHAT: bool = _bool("SKIP_CHAT", False)
MAX_FILES: int = _int("MAX_FILES", 0)

# ---------------------
# Retry / backoff
# ---------------------
EMBED_RETRIES: int = _int("EMBED_RETRIES", 3)
EMBED_BACKOFF_BASE: float = _float("EMBED_BACKOFF_BASE", 0.8)

# ---------------------
# Logging
# ---------------------
LOG_LEVEL: str = os.getenv("ANALYSER_LOG_LEVEL", "INFO")
HTTPX_LOG_LEVEL: str = os.getenv("HTTPX_LOG_LEVEL", "WARNING")

# ---------------------
# Ensure dirs exist
# ---------------------
Path(WORKSPACE_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(EMBED_CACHE_DIR).mkdir(parents=True, exist_ok=True)
