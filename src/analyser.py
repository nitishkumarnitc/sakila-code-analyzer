# src/analyser.py
"""
Analyzer for Sakila - prioritizes controllers, services, repositories, entities, securingweb.
Produces grouped_modules in the final JSON for easier review.

Usage:
    python src/analyser.py --repo <git_url> [--name <repo_name>]
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

from downloader import clone_or_update
from ingest import load_codebase
import vectorstore as vs

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import config as cfg

# Optional tiktoken
try:
    import tiktoken  # type: ignore
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _TIKTOKEN_AVAILABLE = False

logger = logging.getLogger("analyser")
logging.basicConfig(level=getattr(cfg, "LOG_LEVEL", "INFO").upper())

# Config
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

os.makedirs(EMBED_CACHE_DIR, exist_ok=True)

_lc_embeddings: Optional[OpenAIEmbeddings] = None
_lc_chat: Optional[ChatOpenAI] = None
_qdrant_lock = threading.Lock()

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

# ---------- embedding cache ----------
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

# ---------- token estimation & chunking ----------
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
        p = text[a:b].strip()
        if p:
            parts.append(p)
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

# ---------- static analysis ----------
FUNC_REGEX = {
    "java": re.compile(r'(?m)^\s*(public|private|protected|static|\s)*\s*[\w\<\>\[\]]+\s+([a-zA-Z0-9_]+)\s*\(([^\)]*)\)\s*\{'),
    "python": re.compile(r'(?m)^\s*def\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)\s*:'),
    "js": re.compile(r'(?m)^\s*(?:function\s+)?([A-Za-z0-9_]+)\s*\(([^\)]*)\)\s*\{'),
}

def detect_language(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".java", ".kt"}:
        return "java"
    if ext == ".py":
        return "python"
    if ext in {".js", ".ts"}:
        return "js"
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

def find_functions(text: str, lang: str, keep_long_snippet: bool = False) -> List[Dict[str, Any]]:
    """
    Extract functions/methods with:
      - cleaned signature (no trailing '{')
      - start_line, loc, cyclomatic
      - description (short 1-line human-readable)
      - example_snippet (very short, <= 200 chars)
      - long_snippet only if keep_long_snippet True
    """
    patt = FUNC_REGEX.get(lang)
    if not patt:
        return []
    funcs: List[Dict[str, Any]] = []
    for m in patt.finditer(text):
        # extract method name
        if lang == "java":
            name = m.group(2)
        else:
            name = m.group(1)
        raw_sig = m.group(0).strip()
        # Clean signature: remove trailing '{' and any trailing spaces
        clean_sig = re.sub(r"\s*\{\s*$", "", raw_sig).strip()

        start_idx = m.start()
        start_line = line_of_index(text, start_idx)

        # Try to capture a method body region (naive but practical)
        # We'll take up to 3000 chars starting from match start (to capture body lines)
        raw_body = text[m.start(): m.start() + 3000]
        loc = raw_body.count("\n") + 1
        cc = estimate_cyclomatic(raw_body)

        # Build a short description:
        # Approach: take first non-empty code line(s) inside method body, remove leading braces and indentation,
        # and convert into a one-line summary (truncate).
        # This is heuristic — it gives a useful quick hint without dumping the full method.
        body_inner = raw_body.lstrip()
        # remove the opening brace if present
        if body_inner.startswith("{"):
            body_inner = body_inner[1:].lstrip()
        # get first 1-3 meaningful lines
        lines = [ln.strip() for ln in body_inner.splitlines() if ln.strip()]
        first_lines = lines[:3]
        example_snip = " ".join(first_lines)[:200] if first_lines else ""
        # Create description from first line (or example snippet) by normalizing spacing
        if first_lines:
            desc = first_lines[0]
            # remove trailing semicolon or closing brace fragments for readability
            desc = re.sub(r"[;{}]\s*$", "", desc)
        else:
            desc = "<no immediate body summary available>"

        # Optionally keep long snippet (disabled by default to avoid token bloat)
        long_snip = raw_body[:4000] if keep_long_snippet else None

        func_obj = {
            "name": name,
            "signature": clean_sig,
            "start_line": start_line,
            "loc": loc,
            "cyclomatic": cc,
            "description": desc,
            "example_snippet": example_snip
        }
        if keep_long_snippet:
            func_obj["snippet"] = long_snip
        funcs.append(func_obj)
    return funcs

def analyze_code_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    modules = []
    for doc in docs:
        path = doc.get("path", "<unknown>")
        content = doc.get("content", "") or ""
        if not content.strip():
            continue
        lang = detect_language(path)
        total_lines = content.count("\n") + 1
        funcs = find_functions(content, lang)
        top_funcs = sorted(funcs, key=lambda f: (-f["cyclomatic"], -f["loc"]))[:6]
        modules.append({
            "module_name": Path(path).name,
            "path": path,
            "language": lang,
            "total_lines": total_lines,
            "num_functions": len(funcs),
            "top_functions": top_funcs,
            "sample_snippet": content[:1200],
            "confidence": 0.95
        })
    return modules

# ---------- chunking & embedding ----------
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
            logger.warning("Embedding failed (attempt %d): %s — sleeping %.2fs", attempt + 1, e, sleep_t)
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

# ---------- retrieval & chat ----------
def retrieve_top_k_from_qdrant(collection: str, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    if DRY_RUN:
        return [{"path": "<dry-run>", "chunk_index": 0, "document": "dry-run context snippet"}]
    q_emb = vs.embed_texts([query])[0]
    raw = vs.search(collection_name=collection, query_vector=q_emb, limit=k, with_payload=True)
    contexts = []
    for h in raw:
        payload = h.get("payload", {}) or {}
        contexts.append({"path": payload.get("path", ""), "chunk_index": payload.get("chunk_index"), "document": payload.get("text", "")})
    return contexts

def call_chat_robust(system: SystemMessage, user: HumanMessage) -> str:
    chat = _get_langchain_chat()
    try:
        result = chat.generate(messages=[[system, user]])
        gen = result.generations[0][0]
        if hasattr(gen, "message") and getattr(gen.message, "content", None) is not None:
            return gen.message.content
        if getattr(gen, "text", None) is not None:
            return gen.text
        return str(gen)
    except Exception:
        pass
    for method in ("predict_messages", "predict", "__call__"):
        try:
            func = getattr(chat, method, None)
            if not func:
                continue
            resp = func([system, user]) if method != "__call__" else chat([system, user])
            if isinstance(resp, list) and resp and hasattr(resp[0], "content"):
                return resp[0].content
            if hasattr(resp, "content"):
                return resp.content
            if isinstance(resp, str):
                return resp
            return str(resp)
        except Exception:
            continue
    raise RuntimeError("Unable to call ChatOpenAI with known methods.")

# ---------- prompt ----------
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
        funcs = m.get("top_functions", [])[:4]
        funcs_s = "\n".join([f"- {f['name']} (loc={f['loc']}, cc={f['cyclomatic']}, start={f['start_line']})" for f in funcs])
        evidence.append(f"[file: {m['path']}]\nlang: {m['language']}\nlines: {m['total_lines']}\nfunctions:\n{funcs_s}\nsnippet:\n{m['sample_snippet'][:800]}")
    evidence_blob = "\n\n===MODULE_EVIDENCE===\n\n".join(evidence)
    contexts_blob = "\n\n---\n\n".join([f"[file: {c.get('path')}#{c.get('chunk_index')}]\n{(c.get('document') or '')[:1500]}" for c in contexts])
    prompt = f"""
You are a senior engineering lead. Using the evidence and code contexts below, produce a single JSON document exactly matching the schema described and populated with high-confidence findings.

Schema:
{json.dumps(ENHANCED_SCHEMA, indent=2)}

Requirements:
- Each key module must include source_refs (file path and approx start_line) and 1-2 short snippets as evidence.
- For each key method include signature, loc, cyclomatic_complexity (estimate), confidence [0..1], and source_ref.
- Provide an architecture summary, top dependencies (name/version/file), tests/CI summary, top 5 risks, and prioritized recommendations (quick wins first).
- Keep project_overview to 2-4 sentences.
- Return ONLY valid JSON (no extra commentary). Include generated_at ISO8601.

MODULE EVIDENCE:
{evidence_blob}

RETRIEVED CONTEXTS:
{contexts_blob}
"""
    return prompt

# ---------- priority_key & grouping ----------
def priority_key(m: Dict[str, Any]) -> tuple:
    p = (m.get("path") or "").lower()
    # controllers
    if any(x in p for x in ["controller", "controllers"]):
        return (0, p)
    # services
    if any(x in p for x in ["service", "services", "serviceimpl"]):
        return (1, p)
    # repositories / dao
    if any(x in p for x in ["repository", "repositories", "respository", "repo", "/dao/"]):
        return (2, p)
    # entities / models
    if any(x in p for x in ["entity", "entities", "model", "models", "domain", "pojo"]):
        return (3, p)
    # security / config / auth
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

# ---------- orchestration ----------
def analyze_repo(git_url: str, repo_name: Optional[str] = None) -> str:
    repo_path = clone_or_update(git_url, WORKSPACE_DIR, repo_name)
    repo_basename = Path(repo_path).name
    collection = f"repo_{repo_basename.lower()}"
    logger.info("Loaded repo: %s -> collection %s", repo_path, collection)

    code_docs = load_codebase(repo_path)
    if not isinstance(code_docs, list):
        raise RuntimeError("load_codebase must return a list of docs with 'path' and 'content'")

    # Prioritize using priority_key
    code_docs = sorted(code_docs, key=priority_key)

    if MAX_FILES > 0:
        code_docs = code_docs[:MAX_FILES]

    logger.info("Discovered %d code documents (prioritized)", len(code_docs))

    modules = analyze_code_docs(code_docs)

    # Keep modules sorted consistently by priority_key
    modules = sorted(modules, key=priority_key)

    # Group modules for output
    grouped = group_modules_by_role(modules)

    # Chunking
    logger.info("Chunking files (%d workers)...", CHUNKER_WORKERS)
    file_docs_all: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=CHUNKER_WORKERS) as tpe:
        futs = {tpe.submit(process_file_to_chunks, doc): doc.get("path", "<unknown>") for doc in code_docs}
        for fut in as_completed(futs):
            path = futs[fut]
            try:
                file_docs_all.extend(fut.result())
            except Exception as e:
                logger.exception("Chunking failed for %s: %s", path, e)

    total_chunks = len(file_docs_all)
    logger.info("Produced %d chunks across %d files", total_chunks, len(code_docs))

    # Ensure vector collection
    if total_chunks > 0:
        if DRY_RUN:
            vector_size = 1
        else:
            try:
                sample_text = file_docs_all[0]["text"][:128]
                vector_size = len(_get_langchain_embeddings().embed_documents([sample_text])[0])
            except Exception:
                vector_size = 1536
        vs.ensure_collection(collection, vector_size)

    # Embed & upsert
    logger.info("Embedding & upserting ...")
    batches = [file_docs_all[i:i+UPSERT_BATCH] for i in range(0, len(file_docs_all), UPSERT_BATCH)]
    upserted_total = 0
    with ThreadPoolExecutor(max_workers=UPSERTER_WORKERS) as tpe:
        fut_map = {tpe.submit(embed_and_upsert_batch, collection, batch, DRY_RUN): idx for idx, batch in enumerate(batches)}
        for fut in as_completed(fut_map):
            idx = fut_map[fut]
            try:
                upserted = fut.result()
                upserted_total += upserted
            except Exception as e:
                logger.exception("Upsert batch %d failed: %s", idx + 1, e)
    logger.info("Upserted %d points", upserted_total)

    # Retrieval + LLM
    contexts = retrieve_top_k_from_qdrant(collection, "Provide high-level summary, architecture, key methods, and risks", k=TOP_K)
    prompt = build_prompt(repo_basename, modules, contexts)

    if SKIP_CHAT or DRY_RUN:
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
                raise RuntimeError("LLM response did not contain JSON.")
            out = json.loads(raw[s:e+1])
        # attach grouped modules as additional metadata (non-invasive)
        out.setdefault("grouped_modules", grouped)

    # Ensure generated_at
    out.setdefault("generated_at", datetime.utcnow().isoformat() + "Z")
    out.setdefault("key_modules", out.get("key_modules", []))

    out_dir = Path(OUTPUT_DIR) / repo_basename
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "extracted_knowledge.json"
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved report to %s", str(out_file))

    # Validation (pass Path objects)
    # if VALIDATE_AFTER:
    #     try:
    #         from validate_report import validate_report  # type: ignore
    #         repo_root = Path(repo_path)
    #         validated_out = out_dir / "extracted_knowledge_validated.json"
    #         validate_report(report_path=out_file, repo_root=repo_root, out_path=validated_out)
    #         logger.info("Validation completed -> %s", str(validated_out))
    #     except Exception as e:
    #         logger.exception("Validation failed: %s", e)

    return str(out_file)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--name", required=False)
    args = p.parse_args()
    print(analyze_repo(args.repo, repo_name=args.name))
