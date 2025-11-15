# src/analyser.py
import os
import json
import uuid
import time
import re
import logging
import hashlib
import threading
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from downloader import clone_or_update
from ingest import load_codebase
import vectorstore as vs  # expects vs.client, vs.ensure_collection

# LangChain (langchain-openai + langchain-core)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Central config
import config as cfg

# Configure logging
logging.getLogger("httpx").setLevel(getattr(logging, cfg.HTTPX_LOG_LEVEL.upper(), logging.WARNING))
logger = logging.getLogger("analyser")
logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO))

# -----------------------
# Config (use cfg.*)
# -----------------------
WORKSPACE = cfg.WORKSPACE
TOP_K = cfg.TOP_K
OUTPUT_DIR = cfg.OUTPUT_DIR

DRY_RUN = cfg.DRY_RUN
SKIP_CHAT = cfg.SKIP_CHAT

CHUNK_SIZE = cfg.CHUNK_SIZE
CHUNK_OVERLAP = cfg.CHUNK_OVERLAP

UPSERT_BATCH = cfg.UPSERT_BATCH
EMBED_BATCH = cfg.EMBED_BATCH
VALIDATE_AFTER = cfg.VALIDATE_AFTER

EMBED_CONCURRENCY = cfg.EMBED_CONCURRENCY
CHUNKER_WORKERS = cfg.CHUNKER_WORKERS
UPSERTER_WORKERS = cfg.UPSERTER_WORKERS

MAX_FILES = cfg.MAX_FILES
CACHE_DIR = cfg.EMBED_CACHE_DIR

EMBED_RETRIES = cfg.EMBED_RETRIES
EMBED_BACKOFF_BASE = cfg.EMBED_BACKOFF_BASE

# LangChain lazy instances
_lc_embeddings: Optional[OpenAIEmbeddings] = None
_lc_chat: Optional[ChatOpenAI] = None

# Qdrant upsert lock (some clients require safety)
_qdrant_lock = threading.Lock()


# -----------------------
# LangChain helpers
# -----------------------
def _get_langchain_embeddings() -> OpenAIEmbeddings:
    global _lc_embeddings
    if _lc_embeddings is not None:
        return _lc_embeddings
    model_name = cfg.LANGCHAIN_EMBEDDING_MODEL
    _lc_embeddings = OpenAIEmbeddings(model=model_name, chunk_size=EMBED_BATCH)
    return _lc_embeddings


def _get_langchain_chat() -> ChatOpenAI:
    global _lc_chat
    if _lc_chat is not None:
        return _lc_chat
    model = cfg.OPENAI_CHAT_MODEL
    _lc_chat = ChatOpenAI(model_name=model, temperature=0)
    return _lc_chat


# -----------------------
# Embedding cache helpers
# -----------------------
def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path_for_hash(h: str) -> str:
    return os.path.join(CACHE_DIR, f"{h}.json")


def load_vector_from_cache(h: str):
    p = _cache_path_for_hash(h)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Cache read failed for %s: %s — removing corrupted cache", p, e)
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
    except Exception as e:
        logger.debug("Failed to persist cache %s: %s", p, e)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# -----------------------
# Static analysis helpers (lightweight)
# -----------------------
FUNC_REGEX = {
    "python": re.compile(r'(?m)^\s*def\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)\s*:'),
    "java": re.compile(r'(?m)^\s*(public|private|protected|static|\s)*\s*[\w\<\>\[\]]+\s+([a-zA-Z0-9_]+)\s*\(([^\)]*)\)\s*\{'),
    "js": re.compile(r'(?m)^\s*(?:function\s+)?([A-Za-z0-9_]+)\s*\(([^\)]*)\)\s*\{'),
}


def detect_language(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".py":
        return "python"
    if ext in {".java", ".kt"}:
        return "java"
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


def find_functions(text: str, lang: str) -> List[Dict]:
    patt = FUNC_REGEX.get(lang)
    if not patt:
        return []
    funcs = []
    for m in patt.finditer(text):
        name = m.group(2) if lang == "java" else m.group(1)
        sig = m.group(0).strip()
        start_idx = m.start()
        start_line = line_of_index(text, start_idx)
        snippet = text[m.start(): m.start() + 3000]
        loc = snippet.count("\n") + 1
        cc = estimate_cyclomatic(snippet)
        funcs.append({
            "name": name,
            "signature": sig,
            "start_line": start_line,
            "loc": loc,
            "cyclomatic": cc,
            "snippet": snippet[:1500]
        })
    return funcs


def analyze_code_docs(docs: List[Dict]) -> List[Dict]:
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


# -----------------------
# Chunking + embedding + upsert helpers (parallel)
# -----------------------
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


def process_file_to_chunks(doc: Dict) -> List[Dict]:
    path = doc.get("path", "<unknown>")
    text = doc.get("content", "") or ""
    chks = chunk_text_simple(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    file_docs = [{"path": path, "chunk_index": i, "text": c} for i, c in enumerate(chks)]
    return file_docs


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


def embed_and_upsert_batch(collection: str, batch: List[Dict], dry_run: bool = DRY_RUN) -> int:
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
        vs.client.upsert(collection_name=collection, points=points)
    return len(points)


# -----------------------
# Retrieval & Chat helpers
# -----------------------
def retrieve_top_k_from_qdrant(collection: str, query: str, k: int = TOP_K) -> List[Dict]:
    if DRY_RUN:
        return [{"path": "<dry-run>", "chunk_index": 0, "document": "dry-run context snippet"}]
    q_emb = _get_langchain_embeddings().embed_query(query)
    hits = vs.client.search(collection_name=collection, query_vector=q_emb, limit=k)
    contexts = []
    for h in hits:
        payload = getattr(h, "payload", None) or (h.get("payload") if isinstance(h, dict) else {})
        text = payload.get("text") or ""
        path = payload.get("path") or ""
        chunk_index = payload.get("chunk_index", None)
        contexts.append({"path": path, "chunk_index": chunk_index, "document": text})
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
    except Exception as e:
        logger.debug("chat.generate failed: %s", e)

    try:
        resp = chat.predict_messages([system, user])
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e:
        logger.debug("chat.predict_messages failed: %s", e)

    try:
        resp = chat.predict([system, user])
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e:
        logger.debug("chat.predict failed: %s", e)

    try:
        resp = chat([system, user])
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e:
        logger.debug("chat call-style failed: %s", e)

    raise RuntimeError("Unable to call ChatOpenAI with known methods. Check langchain-openai / langchain-core versions.")


# -----------------------
# Prompt building (lead-level)
# -----------------------
ENHANCED_SCHEMA = {
    "project_name": "string",
    "repo": {"url": "string", "branch": "string", "commit": "string"},
    "project_overview": "string",
    "primary_languages": ["string"],
    "architecture_summary": "string",
    "dependencies": [{"name": "string", "version": "string", "file": "string"}],
    "key_modules": [{"module_name": "string", "paths": ["string"], "description": "string", "lines": {"start": 1, "end": 10},
                     "example_snippet": "string", "key_methods": [{"name":"string","signature":"string","loc":10,"cyclomatic_complexity":1.0,"confidence":0.0,"source_ref":{"path":"string","start":1,"end":10}}], "complexity_notes":"string", "recommendations":["string"], "confidence":0.0}],
    "tests_and_ci": {"has_tests": True, "test_paths": ["string"], "ci_present": True},
    "global_complexity_notes": "string",
    "recommendations": ["string"],
    "generated_at": "ISO8601",
    "assumptions": ["string"]
}


def build_prompt(repo_name: str, modules: List[Dict], contexts: List[Dict]) -> str:
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


# -----------------------
# Orchestration
# -----------------------
def analyze_repo(git_url: str, repo_name: Optional[str] = None) -> str:
    repo_path = clone_or_update(git_url, WORKSPACE, repo_name)
    repo_basename = Path(repo_path).name
    collection = f"repo_{repo_basename.lower()}"

    logger.info("Loaded repo: %s -> collection %s", repo_path, collection)

    code_docs = load_codebase(repo_path)
    if MAX_FILES > 0:
        code_docs = code_docs[:MAX_FILES]
    logger.info("Discovered %d code documents", len(code_docs))

    logger.info("Running static analysis...")
    modules = analyze_code_docs(code_docs)

    # ---------------------
    # Parallel chunking
    # ---------------------
    logger.info("Chunking files in parallel (%d workers)...", CHUNKER_WORKERS)
    file_docs_all: List[Dict] = []
    with ThreadPoolExecutor(max_workers=CHUNKER_WORKERS) as tpe:
        futures = {tpe.submit(process_file_to_chunks, doc): doc.get("path", "<unknown>") for doc in code_docs}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                file_docs = fut.result()
                file_docs_all.extend(file_docs)
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

    # ---------------------
    # Parallel embed + upsert
    # ---------------------
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
    # Retrieval + LLM
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
            "primary_languages": list({m["language"] for m in modules}),
            "architecture_summary": "",
            "dependencies": [],
            "key_modules": modules[:8],
            "tests_and_ci": {"has_tests": False, "test_paths": [], "ci_present": False},
            "global_complexity_notes": "",
            "recommendations": [],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "assumptions": ["dry-run"]
        }
    else:
        system = SystemMessage(content="You are a senior engineering lead. Output only JSON and follow instructions.")
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

    if "generated_at" not in out:
        out["generated_at"] = datetime.utcnow().isoformat() + "Z"
    out.setdefault("key_modules", [])

    out_dir = Path(OUTPUT_DIR) / repo_basename
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "extracted_knowledge.json"
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved report to %s", str(out_file))

    # --- new: run validation if requested ---
    if VALIDATE_AFTER:
        try:
            # import local validator module (assumes src/validate_report.py exists and defines validate_report)
            from validate_report import validate_report  # relative import if validate_report.py is in same folder
            repo_root = Path(repo_path)
            # write validated output beside original (optional: pass out path)
            validated_out = out_dir / "extracted_knowledge_validated.json"
            validate_report(report_path=out_file, repo_root=repo_root, out_path=validated_out)
            logger.info("Validation completed, validated report written to %s", str(validated_out))
        except Exception as e:
            logger.exception("Validation step failed: %s", e)

    return str(out_file)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--name", required=False)
    args = p.parse_args()
    analyze_repo(args.repo, repo_name=args.name)
