# src/analyser.py
import os
import json
import uuid
import time
import re
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

from downloader import clone_or_update
from ingest import load_codebase
import vectorstore as vs  # expects client, embed_texts, ensure_collection

# LangChain imports (langchain-openai + langchain-core)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("analyser")
logging.basicConfig(level=logging.INFO)

# -----------------------
# Config (env-driven)
# -----------------------
WORKSPACE = os.environ.get("WORKSPACE_DIR", "/app/workspaces")
TOP_K = int(os.environ.get("TOP_K", "8"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
SKIP_CHAT = os.environ.get("SKIP_CHAT", "0") == "1"

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

UPSERT_BATCH = int(os.environ.get("UPSERT_BATCH", "64"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "4"))

MAX_FILES = int(os.environ.get("MAX_FILES", "0"))  # 0 => no limit

# LangChain lazy instances
_lc_embeddings: Optional[OpenAIEmbeddings] = None
_lc_chat: Optional[ChatOpenAI] = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _lc_embeddings
    if _lc_embeddings:
        return _lc_embeddings
    model = os.environ.get("LANGCHAIN_EMBEDDING_MODEL", "text-embedding-3-small")
    _lc_embeddings = OpenAIEmbeddings(model=model, chunk_size=EMBED_BATCH)
    return _lc_embeddings


def _get_chat() -> ChatOpenAI:
    global _lc_chat
    if _lc_chat:
        return _lc_chat
    model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini" if os.environ.get("OPENAI_CHAT_MODEL") is None else os.environ.get("OPENAI_CHAT_MODEL"))
    _lc_chat = ChatOpenAI(model_name=model, temperature=0)
    return _lc_chat


# -----------------------
# Static analysis helpers
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
        # heuristic snippet capture: from start to next 2000 chars to provide evidence
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
# Dependency & CI detection
# -----------------------
def detect_dependencies(repo_root: str) -> List[Dict]:
    root = Path(repo_root)
    deps = []
    # Maven (pom.xml)
    p = root / "pom.xml"
    if p.exists():
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            # crude: find <artifactId>... and <version>...
            arts = re.findall(r"<artifactId>([^<]+)</artifactId>.*?<version>([^<]+)</version>", txt, re.S)
            for a, v in arts[:30]:
                deps.append({"name": a.strip(), "version": v.strip(), "file": "pom.xml"})
        except Exception:
            pass
    # Gradle (build.gradle) simple parse
    for g in ["build.gradle", "build.gradle.kts"]:
        p = root / g
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            libs = re.findall(r"(implementation|compile|api)\(['\"]([^:'\" ]+):([^:'\" ]+):?([^'\" ]*)['\"]\)", txt)
            for _t, group, name, ver in libs[:30]:
                dep_name = f"{group}:{name}"
                deps.append({"name": dep_name, "version": ver or "unknown", "file": g})
    # package.json
    pj = root / "package.json"
    if pj.exists():
        try:
            pjobj = json.loads(pj.read_text(encoding="utf-8", errors="ignore"))
            for k, v in (pjobj.get("dependencies") or {}).items():
                deps.append({"name": k, "version": str(v), "file": "package.json"})
        except Exception:
            pass
    return deps


def detect_tests_and_ci(repo_root: str) -> Dict:
    root = Path(repo_root)
    has_tests = any(root.rglob("test") )  # simple existence check
    ci_files = list(root.glob(".github/workflows/*.yml")) + list(root.glob(".github/workflows/*.yaml"))
    return {"has_tests": has_tests, "test_paths": ["src/test", "tests"] if has_tests else [], "ci_present": len(ci_files) > 0}


# -----------------------
# Chunking & Qdrant helpers
# -----------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
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


def upsert_chunks(collection: str, docs: List[Dict], dry_run: bool = DRY_RUN):
    if not docs:
        return
    if dry_run:
        vector_size = 1
    else:
        emb = _get_embeddings()
        sample = docs[0]["text"][:128] if docs else ""
        try:
            vec = emb.embed_documents([sample])[0]
            vector_size = len(vec)
        except Exception:
            vector_size = 1536
    vs.ensure_collection(collection, vector_size)
    for i in range(0, len(docs), UPSERT_BATCH):
        batch = docs[i: i + UPSERT_BATCH]
        texts = [d["text"] for d in batch]
        if dry_run:
            vectors = [[0.0] for _ in texts]
        else:
            vectors = _get_embeddings().embed_documents(texts)
        points = []
        for j, d in enumerate(batch):
            payload = {"path": d["path"], "chunk_index": d["chunk_index"], "text": d["text"]}
            points.append({"id": str(uuid.uuid4()), "vector": vectors[j], "payload": payload})
        vs.client.upsert(collection_name=collection, points=points)
        time.sleep(0.03)


def retrieve_top_k(collection: str, query: str, k: int = TOP_K) -> List[Dict]:
    if DRY_RUN:
        return [{"path": "<dry-run>", "chunk_index": 0, "document": "dry-run"}]
    q_emb = _get_embeddings().embed_query(query)
    hits = vs.client.search(collection_name=collection, query_vector=q_emb, limit=k)
    contexts = []
    for h in hits:
        payload = getattr(h, "payload", None) or (h.get("payload") if isinstance(h, dict) else {})
        contexts.append({"path": payload.get("path"), "chunk_index": payload.get("chunk_index"), "document": payload.get("text") or ""})
    return contexts


# -----------------------
# Robust chat invocation
# -----------------------
def call_chat_robust(system: SystemMessage, user: HumanMessage) -> str:
    """
    Try multiple calling conventions across langchain versions and return assistant content (string).
    """
    chat = _get_chat()
    # 1) try generate
    try:
        result = chat.generate(messages=[[system, user]])
        # generations -> list[list[Generation]]
        gen = result.generations[0][0]
        # Generation.message.content or Generation.text
        if hasattr(gen, "message") and getattr(gen.message, "content", None) is not None:
            return gen.message.content
        if getattr(gen, "text", None) is not None:
            return gen.text
        return str(gen)
    except Exception as e_gen:
        logger.debug("generate failed: %s", e_gen)

    # 2) try predict_messages
    try:
        resp = chat.predict_messages([system, user])
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e_pred_msg:
        logger.debug("predict_messages failed: %s", e_pred_msg)

    # 3) try predict
    try:
        resp = chat.predict([system, user])
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e_pred:
        logger.debug("predict failed: %s", e_pred)

    # 4) try __call__ style
    try:
        resp = chat([system, user])
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e_call:
        logger.debug("__call__ failed: %s", e_call)

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
    for m in modules[:40]:  # cap modules
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

    # static analysis
    logger.info("Running static analysis...")
    modules = analyze_code_docs(code_docs)

    # detect deps/ci
    deps = detect_dependencies(repo_path)
    tests_ci = detect_tests_and_ci(repo_path)

    # chunk + upsert
    logger.info("Chunking and upserting to vectorstore (Qdrant)...")
    total_chunks = 0
    for doc in code_docs:
        path = doc.get("path")
        chks = chunk_text(doc.get("content", ""))
        total_chunks += len(chks)
        file_docs = [{"path": path, "chunk_index": i, "text": c} for i, c in enumerate(chks)]
        for i in range(0, len(file_docs), UPSERT_BATCH):
            upsert_chunks(collection, file_docs[i: i + UPSERT_BATCH], dry_run=DRY_RUN)
    logger.info("Upserted %d chunks", total_chunks)

    # retrieve contexts
    logger.info("Retrieving top-k contexts")
    contexts = retrieve_top_k(collection, "Provide high-level summary, architecture, key methods, and risks", k=TOP_K)

    # build prompt and call LLM
    prompt = build_prompt(repo_basename, modules, contexts)
    logger.info("Calling chat model...")
    if SKIP_CHAT or DRY_RUN:
        logger.info("Skipping LLM (DRY_RUN/SKIP_CHAT). Writing minimal JSON.")
        out = {
            "project_name": repo_basename,
            "repo": {"url": git_url, "branch": "unknown", "commit": "unknown"},
            "project_overview": "DRY_RUN; LLM skipped",
            "primary_languages": list({m["language"] for m in modules}),
            "architecture_summary": "",
            "dependencies": deps,
            "key_modules": modules[:6],
            "tests_and_ci": tests_ci,
            "global_complexity_notes": "",
            "recommendations": [],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "assumptions": ["dry-run"]
        }
    else:
        system = SystemMessage(content="You are a senior engineering lead. Output only JSON, follow instructions exactly.")
        user = HumanMessage(content=prompt)
        raw = call_chat_robust(system, user)
        # parse JSON defensively
        try:
            out = json.loads(raw)
        except Exception:
            s = raw.find("{")
            e = raw.rfind("}")
            if s == -1 or e == -1:
                raise RuntimeError("LLM response did not contain JSON")
            out = json.loads(raw[s:e+1])

    # enrich and validate some fields
    if "generated_at" not in out:
        out["generated_at"] = datetime.utcnow().isoformat() + "Z"
    if "repo" not in out:
        out["repo"] = {"url": git_url, "branch": "unknown", "commit": "unknown"}
    # minimal post-check: ensure key_modules exists
    out.setdefault("key_modules", [])

    # save
    out_dir = Path(OUTPUT_DIR) / repo_basename
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "extracted_knowledge_lead.json"
    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved report to %s", str(out_file))
    return str(out_file)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--name", required=False)
    args = p.parse_args()
    analyze_repo(args.repo, repo_name=args.name)
