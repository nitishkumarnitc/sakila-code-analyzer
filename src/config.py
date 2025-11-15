# src/config.py
"""
Centralized configuration for the analyzer.
Reads from environment variables and exposes typed constants.
"""

import os
from pathlib import Path
from typing import Optional

def _as_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def _as_int(v: Optional[str], default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default

def _as_float(v: Optional[str], default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default

# --- Paths ---
WORKSPACE: str = os.environ.get("WORKSPACE_DIR", "/app/workspaces")
OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "/app/outputs")

# --- OpenAI / LangChain model names ---
OPENAI_CHAT_MODEL: str = os.environ.get("OPENAI_CHAT_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
LANGCHAIN_EMBEDDING_MODEL: str = os.environ.get("LANGCHAIN_EMBEDDING_MODEL", os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

# --- Behavior flags ---
DRY_RUN: bool = _as_bool(os.environ.get("DRY_RUN"), False)
SKIP_CHAT: bool = _as_bool(os.environ.get("SKIP_CHAT"), False)
VALIDATE_AFTER: bool = _as_bool(os.environ.get("VALIDATE_REPORT") or os.environ.get("VALIDATE_AFTER"), False)

# --- Retrieval ---
TOP_K: int = _as_int(os.environ.get("TOP_K"), 8)

# --- Chunking (character-based) ---
CHUNK_SIZE: int = _as_int(os.environ.get("CHUNK_SIZE"), 1200)
CHUNK_OVERLAP: int = _as_int(os.environ.get("CHUNK_OVERLAP"), 240)

# --- Batching / parallelism ---
UPSERT_BATCH: int = _as_int(os.environ.get("UPSERT_BATCH"), 128)
EMBED_BATCH: int = _as_int(os.environ.get("EMBED_BATCH"), 64)
EMBED_CONCURRENCY: int = _as_int(os.environ.get("EMBED_CONCURRENCY"), 4)
UPSERTER_WORKERS: int = _as_int(os.environ.get("UPSERTER_WORKERS"), EMBED_CONCURRENCY)
CHUNKER_WORKERS: int = _as_int(os.environ.get("CHUNKER_WORKERS"), max(2, (os.cpu_count() or 2)))

# --- Limits ---
MAX_FILES: int = _as_int(os.environ.get("MAX_FILES"), 0)

# --- Embedding cache ---
EMBED_CACHE_DIR: str = os.environ.get("EMBED_CACHE_DIR", "/tmp/embed_cache")

# --- Retry/backoff params ---
EMBED_RETRIES: int = _as_int(os.environ.get("EMBED_RETRIES"), 3)
EMBED_BACKOFF_BASE: float = _as_float(os.environ.get("EMBED_BACKOFF_BASE"), 0.8)

# --- Logging ---
LOG_LEVEL: str = os.environ.get("ANALYSER_LOG_LEVEL", "INFO")
HTTPX_LOG_LEVEL: str = os.environ.get("HTTPX_LOG_LEVEL", "WARNING")

# Ensure directories exist
Path(WORKSPACE).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(EMBED_CACHE_DIR).mkdir(parents=True, exist_ok=True)
