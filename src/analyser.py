# src/analyser.py
"""
Final analyser for the Sakila (Java-first) codebase with batched AI method summarization (Upgrade A).

Features / requirements implemented:
- Prioritizes controllers/services/repositories/entities/security folders.
- Token-aware chunking (tiktoken optional) + fallback chunker.
- Embedding cache with atomic writes and retry/backoff.
- Parallel chunking/upsert with safe Qdrant upsert lock.
- Batched method summarization to reduce LLM calls (Upgrade A).
  - Summaries are cached on-disk per-method-hash.
  - Summaries are requested in batches (preserves ordering).
- Outputs machine-readable JSON with `grouped_modules`.
- Removes long code snippets from final output (no `snippet` fields).
- Detects CSS files as language "css".
- Uses LangChain ChatOpenAI via `call_chat_robust`.
- Validation call passes Path objects (fixes earlier AttributeError).

Usage:
    python src/analyser.py --repo <git_url> [--name <repo_name>]

Notes:
- The summarizer will not run if DRY_RUN or SKIP_CHAT are True; heuristic one-line summaries will be used instead.
- To tune behavior, adjust config.py variables referenced in this file.
"""

from __future__ import annotations

import os
import re
import json
import uuid
import time
import math
import logging
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
from functools import lru_cache

# Local project imports (expected to exist)
from downloader import clone_or_update
from ingest import load_codebase
import vectorstore as vs  # expects vs.upsert, vs.ensure_collection, vs.search, vs.embed_texts

# LangChain (langchain-openai + langchain-core)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Config
import config as cfg

# Optional token estimator
try:
    import tiktoken  # type: ignore
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _TIKTOKEN_AVAILABLE = False

# Logging
logging.getLogger("httpx").setLevel(getattr(cfg, "HTTPX_LOG_LEVEL", "WARNING").upper() if hasattr(cfg, "HTTPX_LOG_LEVEL") else logging.WARNING)
logger = logging.getLogger("analyser")
logging.basicConfig(level=getattr(cfg, "LOG_LEVEL", "INFO").upper() if hasattr(cfg, "LOG_LEVEL") else logging.INFO)

# Config variables (defaults if missing)
WORKSPACE_DIR = getattr(cfg, "WORKSPACE_DIR", "/tmp/workspace")
TOP_K = getattr(cfg, "TOP_K", 8)
OUTPUT_DIR = getattr(cfg, "OUTPUT_DIR", "out")
DRY_RUN = getattr(cfg, "DRY_RUN", True)
SKIP_CHAT = getattr(cfg, "SKIP_CHAT", True)
CHUNK_SIZE = getattr(cfg, "CHUNK_SIZE", 2400)
CHUNK_OVERLAP = getattr(cfg, "CHUNK_OVERLAP", 200)
UPSERT_BATCH = getattr(cfg, "UPSERT_BATCH", 16)
EMBED_BATCH = getattr(cfg, "EMBED_BATCH", 16)
VALIDATE_AFTER = getattr(cfg, "VALIDATE_AFTER", False)
CHUNKER_WORKERS = getattr(cfg, "CHUNKER_WORKERS", 4)
UPSERTER_WORKERS = getattr(cfg, "UPSERTER_WORKERS", 4)
MAX_FILES = getattr(cfg, "MAX_FILES", -1)
EMBED_CACHE_DIR = getattr(cfg, "EMBED_CACHE_DIR", ".embed_cache")
EMBED_RETRIES = getattr(cfg, "EMBED_RETRIES", 3)
EMBED_BACKOFF_BASE = getattr(cfg, "EMBED_BACKOFF_BASE", 0.5)
OPENAI_EMBED_MODEL = getattr(cfg, "OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = getattr(cfg, "OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Method summary cache directory (separate from embedding cache)
METHOD_SUMMARY_CACHE_DIR = getattr(cfg, "METHOD_SUMMARY_CACHE_DIR", ".method_summary_cache")
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
os.makedirs(METHOD_SUMMARY_CACHE_DIR, exist_ok=True)

# LangChain lazy instances
_lc_embeddings: Optional[OpenAIEmbeddings] = None
_lc_chat: Optional[ChatOpenAI] = None

# Qdrant upsert lock (some clients require sequential safety)
_qdrant_lock = threading.Lock()

# Folder prioritization for Java projects
PRIORITY_FOLDERS = [
    "controller", "controllers",
    "entity", "entities", "model", "models", "domain", "pojo",
    "service", "services", "serviceimpl",
    "repository", "repositories", "repo", "dao",
    "security", "securingweb", "auth", "config",
]

# ------------------------
# LangChain helpers
# ------------------------
def _get_langchain_embeddings() -> OpenAIEmbeddings:
    global _lc_embeddings
    if _lc_embeddings is not None:
        return _lc_embeddings
    _lc_embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, chunk_size=EMBED_BATCH)
    return _lc_embeddings


def _get_langchain_chat() -> ChatOpenAI:
    global _lc_chat
    if _lc_chat is not None:
        return _lc_chat
    _lc_chat = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)
    return _lc_chat


# ------------------------
# Embedding cache helpers
# ------------------------
def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path_for_hash(h: str) -> str:
    return os.path.join(EMBED_CACHE_DIR, f"{h}.json")


def load_vector_from_cache(h: str):
    p = _cache_path_for_hash(h)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            os.remove(p)
        except Exception:
            pass
        return None


def save_vector_to_cache(h: str, vec):
    p = _cache_path_for_hash(h)
    tmp = p + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(vec, f)
        os.replace(tmp, p)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# ------------------------
# Token estimation & chunking
# ------------------------
def estimate_tokens(text: str, model: str = OPENAI_CHAT_MODEL) -> int:
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return max(1, math.ceil(len(text) / 4))


def chunk_text_token_aware(text: str, token_target: int = 1500, model: str = OPENAI_CHAT_MODEL) -> List[str]:
    if not text:
        return []
    if estimate_tokens(text, model) <= token_target:
        return [text]

    boundary_re = re.compile(r"(^\s*(public|private|protected|static|final|def|class|function)\b.*$)|(\n\s*\n)", flags=re.MULTILINE)
    indices = [0]
    for m in boundary_re.finditer(text):
        idx = m.start()
        if idx > indices[-1]:
            indices.append(idx)
    indices.append(len(text))

    parts = []
    for a, b in zip(indices, indices[1:]):
        part = text[a:b].strip()
        if part:
            parts.append(part)

    final = []
    for p in parts:
        if estimate_tokens(p, model) <= token_target:
            final.append(p)
            continue
        paras = re.split(r"\n\s*\n", p)
        buf = ""
        for para in paras:
            if not para.strip():
                continue
            if not buf:
                buf = para
            elif estimate_tokens(buf + "\n\n" + para, model) <= token_target:
                buf = buf + "\n\n" + para
            else:
                final.append(buf)
                buf = para
        if buf:
            if estimate_tokens(buf, model) <= token_target:
                final.append(buf)
            else:
                chars_per_chunk = max(1000, int(len(buf) * token_target / max(1, estimate_tokens(buf, model))))
                for i in range(0, len(buf), chars_per_chunk):
                    final.append(buf[i:i+chars_per_chunk])
    if not final:
        chars_per = max(1000, token_target * 4)
        for i in range(0, len(text), chars_per):
            final.append(text[i:i+chars_per])
    return final


def chunk_text_simple(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    text = text.strip()
    L = len(text)
    if L <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end == L:
            break
        next_start = end - overlap
        start = end if next_start <= start else next_start
    return chunks


# ------------------------
# Static analysis helpers
# ------------------------
FUNC_REGEX = {
    "java": re.compile(r'(?m)^\s*(public|private|protected|static|\s)*\s*[\w\<\>\[\]]+\s+([a-zA-Z0-9_]+)\s*\(([^\)]*)\)\s*\{'),
    "python": re.compile(r'(?m)^\s*def\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)\s*:'),
    "js": re.compile(r'(?m)^\s*(?:function\s+)?([A-Za-z0-9_]+)\s*\(([^\)]*)\)\s*\{'),
    # Note: do NOT add an empty regex for css. CSS doesn't have functions to extract.
}



def detect_language(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".java", ".kt"}:
        return "java"
    if ext == ".py":
        return "python"
    if ext in {".js", ".ts"}:
        return "js"
    if ext in {".html", ".htm"}:
        return "html"
    if ext == ".css":
        return "css"
    return "other"


def line_of_index(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def estimate_cyclomatic(snippet: str) -> int:
    tokens = [" if ", " for ", " while ", " case ", "&&", "||", "elif ", "else if", "catch "]
    s = snippet.lower()
    count = 1
    for t in tokens:
        count += s.count(t)
    return max(1, count)


# ------------------------
# Method summarization (Batch + cache)
# ------------------------
def _method_summary_cache_path(h: str) -> str:
    return os.path.join(METHOD_SUMMARY_CACHE_DIR, f"{h}.txt")


def _load_method_summary_from_disk(h: str) -> Optional[str]:
    p = _method_summary_cache_path(h)
    if not os.path.exists(p):
        return None
    try:
        return Path(p).read_text(encoding="utf-8")
    except Exception:
        try:
            os.remove(p)
        except Exception:
            pass
        return None


def _save_method_summary_to_disk(h: str, summary: str):
    p = _method_summary_cache_path(h)
    tmp = p + ".tmp"
    try:
        Path(tmp).write_text(summary, encoding="utf-8")
        os.replace(tmp, p)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _shorten_signature(raw_sig: str) -> str:
    return re.sub(r"\s*\{\s*$", "", raw_sig.strip())


def _hash_for_method_context(signature: str, body_short: str) -> str:
    payload = (signature + "\n" + body_short).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize_methods_batch(method_bodies: List[str], method_signatures: List[str]) -> List[str]:
    """
    Summarize many methods in one LLM call. Returns descriptions in the same order.
    Uses disk cache per method hash to avoid repeated LLM calls.
    If DRY_RUN or SKIP_CHAT is True, returns heuristic summaries.
    """

    results: List[str] = []
    # prepare hashes & check cache
    to_request_indices: List[int] = []
    request_items: List[Dict[str, str]] = []
    for i, (sig, body) in enumerate(zip(method_signatures, method_bodies)):
        h = _hash_for_method_context(sig, body)
        cached = _load_method_summary_from_disk(h)
        if cached is not None:
            results.append(cached)
        else:
            results.append(None)  # placeholder
            to_request_indices.append(i)
            request_items.append({"idx": i, "signature": sig, "body": body, "hash": h})

    if not to_request_indices:
        return results  # all cached

    # If in dry-run/skip-chat, generate heuristic descriptions
    if DRY_RUN or SKIP_CHAT:
        for item in request_items:
            body_line = (item["body"] or "").strip().splitlines()
            first = body_line[0].strip() if body_line else ""
            heuristic = f"Performs: {first[:140]}" if first else "Performs an operation (heuristic summary)."
            idx = item["idx"]
            results[idx] = heuristic
            _save_method_summary_to_disk(item["hash"], heuristic)
        return results

    # Build a single prompt enumerating method signatures + short bodies.
    # Ask the model to return a JSON array of objects [{"hash":"...","summary":"..."}...]
    # The hash allows us to map back and persist each summary.
    chunks = []
    # limit how many methods we send in one call to avoid running over context (safety)
    MAX_METHODS_PER_CALL = 40
    for batch_start in range(0, len(request_items), MAX_METHODS_PER_CALL):
        batch = request_items[batch_start: batch_start + MAX_METHODS_PER_CALL]
        enumerated = []
        for it in batch:
            # include signature and a short body (max ~600 chars)
            body_short = it["body"][:800].replace("```", "'``'")
            enumerated.append({
                "hash": it["hash"],
                "signature": it["signature"],
                "body": body_short
            })
        # create prompt for this batch
        system = SystemMessage(content="You are a senior Java engineer. Produce one concise English sentence describing what each method does (business logic). Be precise and avoid implementation details like 'calls X' unless important. Output must be valid JSON: an array of objects [{\"hash\":\"...\",\"summary\":\"one sentence\"}, ...].")
        items_text = json.dumps(enumerated, ensure_ascii=False)
        user = HumanMessage(content=f"Summarize the following methods. Return ONLY valid JSON as described.\n\nMETHODS:\n{items_text}\n")
        try:
            raw = call_chat_robust(system, user)
            # try to parse JSON substring
            s = raw.find("[")
            e = raw.rfind("]")
            if s == -1 or e == -1:
                raise RuntimeError("LLM did not return a JSON array.")
            parsed = json.loads(raw[s:e+1])
            # parsed is list of {"hash":..., "summary":...}
            for obj in parsed:
                h = obj.get("hash")
                summ = obj.get("summary") or obj.get("description") or obj.get("summary_text") or ""
                summ = summ.strip().replace("\n", " ")
                if not summ:
                    summ = "<summary unavailable>"
                # save to disk
                _save_method_summary_to_disk(h, summ)
                # map into results
                # find indices with this hash
                for it in request_items:
                    if it["hash"] == h:
                        results[it["idx"]] = summ
            # safeguard: fill any remaining None with fallback
            for it in request_items:
                if results[it["idx"]] is None:
                    fallback = "<summary unavailable>"
                    results[it["idx"]] = fallback
                    _save_method_summary_to_disk(it["hash"], fallback)
        except Exception as e:
            logger.exception("Batch summarization failed: %s", e)
            # fallback to heuristics for this batch
            for it in batch:
                idx = it["idx"]
                first_line = (it["body"] or "").strip().splitlines()
                first = first_line[0].strip() if first_line else ""
                heuristic = f"Performs: {first[:140]}" if first else "Performs an operation (heuristic summary)."
                results[idx] = heuristic
                _save_method_summary_to_disk(it["hash"], heuristic)

    return results


# ------------------------
# Function extraction (no long snippets in output)
# ------------------------
def find_functions(text: str, lang: str) -> List[Dict[str, Any]]:
    """
    Extract functions/methods robustly. Handles missing capture groups gracefully
    and returns an empty list for languages with no function-like constructs (e.g. css).
    Each returned function contains:
      - name (or '<unknown>')
      - signature (cleaned)
      - start_line, loc, cyclomatic
      - body_short (kept temporarily for later summarization)
    """
    # Early-return for languages that don't have function constructs we extract
    if lang == "css":
        return []

    patt = FUNC_REGEX.get(lang)
    if not patt:
        return []

    funcs: List[Dict[str, Any]] = []
    for m in patt.finditer(text):
        # Determine method name robustly depending on available groups
        name = "<unknown>"
        # m.lastindex is the index of the last captured group (or None)
        last = getattr(m, "lastindex", None)
        if lang == "java":
            # Java pattern is expected to have group(2) as name
            if last and last >= 2:
                try:
                    name = m.group(2)
                except Exception:
                    name = "<unknown>"
        else:
            # python/js: group(1) is usually the name
            if last and last >= 1:
                try:
                    name = m.group(1)
                except Exception:
                    name = "<unknown>"

        raw_sig = m.group(0).strip() if m.lastindex is not None or m.groups() is not None else ""
        signature = _shorten_signature(raw_sig) if raw_sig else ""
        start_idx = m.start()
        start_line = line_of_index(text, start_idx)
        # take a short method body for summarization context (up to ~1200 chars)
        body_short = text[m.start(): m.start() + 1200]
        loc = body_short.count("\n") + 1
        cc = estimate_cyclomatic(body_short)
        funcs.append({
            "name": name,
            "signature": signature,
            "start_line": start_line,
            "loc": loc,
            "cyclomatic": cc,
            "body_short": body_short  # will be removed later before final output
        })
    return funcs

def analyze_code_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    modules: List[Dict[str, Any]] = []
    for doc in docs:
        path = doc.get("path", "<unknown>")
        content = doc.get("content", "") or ""
        if not content.strip():
            continue
        lang = detect_language(path)
        total_lines = content.count("\n") + 1
        funcs = find_functions(content, lang)
        # We'll keep top functions for quick inspection (by cyclomatic/loc)
        top_funcs = sorted(funcs, key=lambda f: (-f["cyclomatic"], -f["loc"]))[:12]
        modules.append({
            "module_name": Path(path).name,
            "path": path,
            "language": lang,
            "total_lines": total_lines,
            "num_functions": len(funcs),
            "top_functions": top_funcs,
            "confidence": 0.95
        })
    return modules


# ------------------------
# Chunking + embedding + upsert
# ------------------------
def process_file_to_chunks(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = doc.get("path", "<unknown>")
    text = doc.get("content", "") or ""
    try:
        chks = chunk_text_token_aware(text, token_target=1500)
        if not chks:
            chks = chunk_text_simple(text)
    except Exception:
        chks = chunk_text_simple(text)
    return [{"path": path, "chunk_index": i, "text": c} for i, c in enumerate(chks)]


def _embed_texts_with_retry(texts: List[str]) -> List[List[float]]:
    emb = _get_langchain_embeddings()
    vectors = [None] * len(texts)
    to_fetch = []
    indices = []
    for i, t in enumerate(texts):
        h = _chunk_hash(t)
        cached = load_vector_from_cache(h)
        if cached is not None:
            vectors[i] = cached
        else:
            to_fetch.append(t)
            indices.append(i)

    if not to_fetch:
        return vectors

    attempt = 0
    last_exc = None
    while attempt <= EMBED_RETRIES:
        try:
            fetched = emb.embed_documents(to_fetch)
            for idx_local, vec in enumerate(fetched):
                idx_global = indices[idx_local]
                vectors[idx_global] = vec
                save_vector_to_cache(_chunk_hash(texts[idx_global]), vec)
            return vectors
        except Exception as e:
            last_exc = e
            sleep_t = EMBED_BACKOFF_BASE * (2 ** attempt)
            logger.warning("Embedding failed (attempt %d/%d): %s — sleeping %.2fs", attempt + 1, EMBED_RETRIES + 1, e, sleep_t)
            time.sleep(sleep_t)
            attempt += 1

    raise RuntimeError("Embedding failed after retries") from last_exc


def embed_and_upsert_batch(collection: str, batch: List[Dict[str, Any]], dry_run: bool = DRY_RUN) -> int:
    if not batch:
        return 0
    texts = [d["text"] for d in batch]
    if dry_run:
        vectors = [[0.0] for _ in texts]
    else:
        vectors = _embed_texts_with_retry(texts)

    points = []
    for j, d in enumerate(batch):
        payload = {"path": d["path"], "chunk_index": d["chunk_index"], "text": d["text"]}
        points.append({"id": str(uuid.uuid4()), "vector": vectors[j], "payload": payload})

    with _qdrant_lock:
        vs.upsert(collection_name=collection, points=points)
    return len(points)


# ------------------------
# Retrieval & chat helpers
# ------------------------
def retrieve_top_k_from_qdrant(collection: str, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    if DRY_RUN:
        return [{"path": "<dry-run>", "chunk_index": 0, "document": "dry-run context snippet"}]
    q_emb = vs.embed_texts([query])[0]
    raw = vs.search(collection_name=collection, query_vector=q_emb, limit=k, with_payload=True)
    contexts = []
    for h in raw:
        payload = h.get("payload", {}) or {}
        contexts.append({
            "path": payload.get("path", ""),
            "chunk_index": payload.get("chunk_index"),
            "document": payload.get("text", "")
        })
    return contexts


def call_chat_robust(system: SystemMessage, user: HumanMessage) -> str:
    chat = _get_langchain_chat()
    # Try several call styles for compatibility
    try:
        result = chat.generate(messages=[[system, user]])
        gen = result.generations[0][0]
        if hasattr(gen, "message") and getattr(gen.message, "content", None) is not None:
            return gen.message.content
        if getattr(gen, "text", None) is not None:
            return gen.text
        return str(gen)
    except Exception:
        logger.debug("chat.generate failed (fallthrough)")

    for method in ("predict_messages", "predict", "__call__"):
        try:
            func = getattr(chat, method, None)
            if not func:
                continue
            if method == "__call__":
                resp = chat([system, user])
            else:
                resp = func([system, user])
            if isinstance(resp, list) and resp and hasattr(resp[0], "content"):
                return resp[0].content
            if hasattr(resp, "content"):
                return resp.content
            if isinstance(resp, str):
                return resp
            return str(resp)
        except Exception:
            continue

    raise RuntimeError("Unable to call ChatOpenAI with known methods. Check versions.")


# ------------------------
# Prompt building
# ------------------------
ENHANCED_SCHEMA = {
    "project_name": "string",
    "repo": {"url": "string", "branch": "string", "commit": "string"},
    "project_overview": "string",
    "primary_languages": ["string"],
    "architecture_summary": "string",
    "dependencies": [{"name": "string", "version": "string", "file": "string"}],
    "key_modules": [{"module_name": "string", "paths": ["string"], "description": "string", "lines": {"start":1,"end":10},
                     "example_snippet": "string", "key_methods": [{"name":"string","signature":"string","loc":10,"cyclomatic_complexity":1.0,"confidence":0.0,"source_ref":{"path":"string","start":1,"end":10}}], "complexity_notes":"string", "recommendations":["string"], "confidence":0.0}],
    "tests_and_ci": {"has_tests": True, "test_paths": ["string"], "ci_present": True},
    "global_complexity_notes": "string",
    "recommendations": ["string"],
    "generated_at": "ISO8601",
    "assumptions": ["string"]
}


def build_prompt(repo_name: str, modules: List[Dict[str, Any]], contexts: List[Dict[str, Any]]) -> str:
    evidence = []
    for m in modules[:40]:
        funcs = m.get("top_functions", [])[:6]
        funcs_s = "\n".join([f"- {f['name']} (loc={f['loc']}, cc={f['cyclomatic']}, start={f['start_line']})" for f in funcs])
        evidence.append(f"[file: {m['path']}]\nlang: {m['language']}\nlines: {m['total_lines']}\nfunctions:\n{funcs_s}")
    evidence_blob = "\n\n===MODULE_EVIDENCE===\n\n".join(evidence)
    contexts_blob = "\n\n---\n\n".join([f"[file: {c.get('path')}#{c.get('chunk_index')}]\n{(c.get('document') or '')[:1500]}" for c in contexts])
    prompt = f"""
You are a senior engineering lead. Using the evidence and code contexts below, produce a single JSON document exactly matching the schema described and populated with high-confidence findings.

Schema:
{json.dumps(ENHANCED_SCHEMA, indent=2)}

Requirements:
- Each key module must include source_refs (file path and approx start_line) and 1-2 short sentences as evidence.
- For each key method include signature, loc, cyclomatic_complexity (estimate), confidence [0..1], and a one-sentence description (human readable).
- Provide an architecture summary, top dependencies (name/version/file), tests/CI summary, top 5 risks, and prioritized recommendations (quick wins first).
- Keep project_overview to 2-4 sentences.
- Return ONLY valid JSON (no extra commentary). Include generated_at ISO8601.

MODULE EVIDENCE:
{evidence_blob}

RETRIEVED CONTEXTS:
{contexts_blob}
"""
    return prompt


# ------------------------
# Helpers: priority & grouping
# ------------------------
def priority_key(m: Dict[str, Any]) -> tuple:
    p = (m.get("path") or "").lower()
    if any(x in p for x in ["controller", "controllers"]):
        return (0, p)
    if any(x in p for x in ["service", "services", "serviceimpl"]):
        return (1, p)
    if any(x in p for x in ["repository", "repositories", "respository", "repo", "/dao/"]):
        return (2, p)
    if any(x in p for x in ["entity", "entities", "model", "models", "domain", "pojo"]):
        return (3, p)
    if any(x in p for x in ["security", "securingweb", "auth", "config"]):
        return (4, p)
    return (5, p)


def group_modules_by_role(modules: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = {"controllers": [], "services": [], "repositories": [], "entities": [], "security": [], "other": []}
    for m in modules:
        p = (m.get("path") or "").lower()
        if any(x in p for x in ["controller", "controllers"]):
            groups["controllers"].append(m)
        elif any(x in p for x in ["service", "services", "serviceimpl"]):
            groups["services"].append(m)
        elif any(x in p for x in ["repository", "repositories", "respository", "repo", "/dao/"]):
            groups["repositories"].append(m)
        elif any(x in p for x in ["entity", "entities", "model", "models", "domain", "pojo"]):
            groups["entities"].append(m)
        elif any(x in p for x in ["security", "securingweb", "auth", "config"]):
            groups["security"].append(m)
        else:
            groups["other"].append(m)
    return groups


# ------------------------
# Orchestration
# ------------------------
def analyze_repo(git_url: str, repo_name: Optional[str] = None) -> str:
    repo_path = clone_or_update(git_url, WORKSPACE_DIR, repo_name)
    repo_basename = Path(repo_path).name
    collection = f"repo_{repo_basename.lower()}"
    logger.info("Loaded repo: %s -> collection %s", repo_path, collection)

    code_docs = load_codebase(repo_path)
    if not isinstance(code_docs, list):
        raise RuntimeError("load_codebase must return a list of docs with 'path' and 'content' keys")

    # Prioritize files by folder relevance
    code_docs = sorted(code_docs, key=priority_key)
    if MAX_FILES > 0:
        code_docs = code_docs[:MAX_FILES]

    logger.info("Discovered %d code documents (prioritized)", len(code_docs))

    # Static analysis: extract functions and module metadata
    modules = analyze_code_docs(code_docs)

    # Keep modules sorted by priority
    modules = sorted(modules, key=priority_key)
    grouped = group_modules_by_role(modules)

    # ---------------------
    # Prepare batched method summaries
    # ---------------------
    # Collect per-method short bodies and signatures (for top functions only to limit volume)
    method_signatures: List[str] = []
    method_bodies: List[str] = []
    method_locations: List[Dict[str, Any]] = []  # for mapping back

    for m in modules:
        for f in m.get("top_functions", []):
            sig = f.get("signature", "")
            body = f.get("body_short", "")
            # only include if there is some short body or signature
            if not sig and not body:
                continue
            method_signatures.append(sig)
            method_bodies.append(body)
            method_locations.append({"module_path": m["path"], "func_name": f["name"], "start_line": f["start_line"]})

    logger.info("Collected %d methods for batched summarization", len(method_signatures))

    # Request batched summaries and populate descriptions
    descriptions: List[str] = summarize_methods_batch(method_bodies, method_signatures)

    # Map descriptions back into modules.top_functions; remove body_short and add description + confidence
    desc_idx = 0
    for m in modules:
        new_top_funcs = []
        for f in m.get("top_functions", []):
            if desc_idx < len(descriptions):
                desc = descriptions[desc_idx]
                desc_idx += 1
            else:
                desc = "<summary unavailable>"
            # populate cleaned function entry (no long snippets)
            func_entry = {
                "name": f.get("name"),
                "signature": f.get("signature"),
                "start_line": f.get("start_line"),
                "loc": f.get("loc"),
                "cyclomatic": f.get("cyclomatic"),
                "description": desc,
                "confidence": 0.9
            }
            new_top_funcs.append(func_entry)
        m["top_functions"] = new_top_funcs

    # ---------------------
    # Chunking, embedding & upsert
    # ---------------------
    logger.info("Chunking files in parallel (%d workers)...", CHUNKER_WORKERS)
    file_docs_all: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=CHUNKER_WORKERS) as tpe:
        futures = {tpe.submit(process_file_to_chunks, doc): doc.get("path", "<unknown>") for doc in code_docs}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                file_docs_all.extend(fut.result())
            except Exception as e:
                logger.exception("Chunking failed for %s: %s", path, e)

    total_chunks = len(file_docs_all)
    logger.info("Produced %d chunks across %d files", total_chunks, len(code_docs))

    # Ensure collection once
    if total_chunks > 0:
        if DRY_RUN:
            vector_size = 1
        else:
            sample_text = file_docs_all[0]["text"][:128]
            try:
                vector_size = len(_get_langchain_embeddings().embed_documents([sample_text])[0])
            except Exception:
                vector_size = 1536
        vs.ensure_collection(collection, vector_size)

    # Parallel embed + upsert
    logger.info("Embedding & upserting in parallel (%d workers) with batch size %d...", UPSERTER_WORKERS, UPSERT_BATCH)
    batches = [file_docs_all[i: i + UPSERT_BATCH] for i in range(0, len(file_docs_all), UPSERT_BATCH)]
    upserted_total = 0
    with ThreadPoolExecutor(max_workers=UPSERTER_WORKERS) as tpe:
        fut_map = {tpe.submit(embed_and_upsert_batch, collection, batch, DRY_RUN): idx for idx, batch in enumerate(batches)}
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                upserted = fut.result()
                upserted_total += upserted
                logger.info("[upsert] Batch %d/%d upserted %d points", idx + 1, len(batches), upserted)
            except Exception as e:
                logger.exception("Upsert batch %d failed: %s", idx + 1, e)

    logger.info("Created and upserted %d points", upserted_total)

    # ---------------------
    # Retrieval + LLM (project-level synthesis)
    # ---------------------
    logger.info("Retrieving top-k contexts")
    contexts = retrieve_top_k_from_qdrant(collection, "Provide high-level summary, architecture, key methods, and risks", k=TOP_K)

    logger.info("Building prompt and calling chat model...")
    prompt = build_prompt(repo_basename, modules, contexts)

    if SKIP_CHAT or DRY_RUN:
        logger.info("Skipping chat (SKIP_CHAT/DRY_RUN). Writing minimal JSON output.")
        out = {
            "project_name": repo_basename,
            "repo": {"url": git_url, "branch": "unknown", "commit": "unknown"},
            "project_overview": "DRY_RUN",
            "primary_languages": sorted(list({m["language"] for m in modules})),
            "architecture_summary": "",
            "dependencies": [],
            "key_modules": modules[:20],
            "grouped_modules": grouped,
            "tests_and_ci": {"has_tests": False, "test_paths": [], "ci_present": False},
            "global_complexity_notes": "",
            "recommendations": [],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "assumptions": ["dry-run"]
        }
    else:
        system = SystemMessage(content="You are a senior engineering lead. Output only JSON and follow instructions exactly.")
        user = HumanMessage(content=prompt)
        raw = call_chat_robust(system, user)
        try:
            out = json.loads(raw)
        except Exception:
            s = raw.find("{")
            e = raw.rfind("}")
            if s == -1 or e == -1:
                logger.exception("LLM response did not contain JSON. Raw output: %s", raw)
                raise RuntimeError("LLM response did not contain JSON.")
            out = json.loads(raw[s:e+1])
        # attach grouped modules as additional metadata
        out.setdefault("grouped_modules", grouped)

    # ensure generated_at and key_modules
    out.setdefault("generated_at", datetime.utcnow().isoformat() + "Z")
    out.setdefault("key_modules", out.get("key_modules", []))

    # Write result
    out_dir = Path(OUTPUT_DIR) / repo_basename
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "extracted_knowledge.json"
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved report to %s", str(out_file))

    # Optional validation (pass Path objects)
    if VALIDATE_AFTER:
        try:
            from validate_report import validate_report  # type: ignore
            repo_root = Path(repo_path)
            validated_out = out_dir / "extracted_knowledge_validated.json"
            validate_report(report_path=out_file, repo_root=repo_root, out_path=validated_out)
            logger.info("Validation completed, validated report written to %s", str(validated_out))
        except Exception as e:
            logger.exception("Validation step failed: %s", e)

    return str(out_file)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Git URL or local path to repository")
    p.add_argument("--name", required=False, help="Optional name for the repo/collection")
    args = p.parse_args()
    print(analyze_repo(args.repo, repo_name=args.name))
