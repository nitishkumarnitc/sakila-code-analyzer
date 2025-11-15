import os
import json
import uuid
import time
from typing import List, Dict
from datetime import datetime

# local modules (must exist in src/)
from downloader import clone_or_update
from ingest import load_codebase
import vectorstore as vs  # expects client, embed_texts, ensure_collection

# LangChain imports — required
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------
# Config (env-driven)
# ---------------------

WORKSPACE = os.environ.get("WORKSPACE_DIR", "/app/workspaces")
TOP_K = int(os.environ.get("TOP_K", "8"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")

# Dry-run and controls
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
SKIP_CHAT = os.environ.get("SKIP_CHAT", "0") == "1"

# Chunking configuration
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# Upsert/embedding batching
UPSERT_BATCH = int(os.environ.get("UPSERT_BATCH", "64"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "4"))

# Optional debug/dev limit
MAX_FILES = int(os.environ.get("MAX_FILES", "0"))  # 0 = no limit

# LangChain instances (lazy init)
_lc_embeddings: OpenAIEmbeddings = None
_lc_chat: ChatOpenAI = None


def _get_langchain_embeddings() -> OpenAIEmbeddings:
    global _lc_embeddings
    if _lc_embeddings is not None:
        return _lc_embeddings

    model_name = os.environ.get("LANGCHAIN_EMBEDDING_MODEL", "text-embedding-3-small")

    # chunk_size controls internal batching in LangChain OpenAIEmbeddings
    _lc_embeddings = OpenAIEmbeddings(model=model_name, chunk_size=EMBED_BATCH)
    return _lc_embeddings


def _get_langchain_chat() -> ChatOpenAI:
    global _lc_chat
    if _lc_chat is not None:
        return _lc_chat

    model_name = os.environ.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
    _lc_chat = ChatOpenAI(model_name=model_name, temperature=0)
    return _lc_chat


# ---------------------
# Helpers
# ---------------------

def chunk_text_simple(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        print("[chunk_text_simple] empty text")
        return []

    text = text.strip()
    L = len(text)

    if L <= chunk_size:
        print("[chunk_text_simple] single chunk")
        return [text]

    chunks: List[str] = []
    start = 0

    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])

        if end == L:
            break

        next_start = end - overlap
        if next_start <= start:
            start = end
        else:
            start = next_start

    return chunks


def upsert_chunks_to_qdrant(collection: str, docs: List[Dict], dry_run: bool = DRY_RUN):
    """
    docs: list of {'path','chunk_index','text'}
    Upserts vectors into Qdrant in batches using LangChain embeddings.
    """
    if not docs:
        return

    if dry_run:
        vector_size = 1
    else:
        lc_emb = _get_langchain_embeddings()
        sample = docs[0]["text"][:64] if docs else ""
        try:
            vec = lc_emb.embed_documents([sample])[0]
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
            lc_emb = _get_langchain_embeddings()
            vectors = lc_emb.embed_documents(texts)

        points = []
        for j, d in enumerate(batch):
            payload = {
                "path": d["path"],
                "chunk_index": d["chunk_index"],
                "text": d["text"]
            }
            points.append({
                "id": str(uuid.uuid4()),
                "vector": vectors[j],
                "payload": payload
            })

        vs.client.upsert(collection_name=collection, points=points)
        print(f"[upsert] Upserted batch {(i // UPSERT_BATCH) + 1} ({len(points)} points)")
        time.sleep(0.05)


def retrieve_top_k_from_qdrant(collection: str, query: str, k: int = TOP_K) -> List[Dict]:
    """
    Embed query and search Qdrant; return list of contexts with path & text.
    """
    if DRY_RUN:
        return [{"path": "<dry-run>", "chunk_index": 0, "document": "dry-run context snippet"}]

    lc_emb = _get_langchain_embeddings()
    q_emb = lc_emb.embed_query(query)
    hits = vs.client.search(collection_name=collection, query_vector=q_emb, limit=k)

    contexts = []
    for h in hits:
        payload = getattr(h, "payload", None) or (
            h.get("payload") if isinstance(h, dict) else {}
        )
        text = payload.get("text") or ""
        path = payload.get("path") or ""
        chunk_index = payload.get("chunk_index", None)

        contexts.append({
            "path": path,
            "chunk_index": chunk_index,
            "document": text
        })

    return contexts


def build_prompt(project_name: str, top_k_contexts: List[Dict]) -> str:
    parts = []

    for c in top_k_contexts:
        p = c.get("path", "unknown")
        txt = c.get("document", "")[:2500]
        parts.append(f"[file: {p}]\n{txt}")

    context_combined = "\n\n---\n\n".join(parts)

    schema = {
        "project_name": "string",
        "project_overview": "string",
        "primary_languages": ["string"],
        "key_modules": [
            {
                "module_name": "string",
                "path": "string",
                "description": "string",
                "key_methods": [
                    {
                        "name": "string",
                        "signature": "string",
                        "description": "string",
                        "complexity_notes": "string (optional)"
                    }
                ]
            }
        ],
        "global_complexity_notes": "string",
        "assumptions": ["string"],
        "generated_at": "ISO8601 timestamp"
    }

    prompt = f"""
You are a code analyst. Based on the provided snippets (from project '{project_name}'),
return a single valid JSON matching this schema:

{json.dumps(schema, indent=2)}

Provide project overview (2-4 sentences), 2-4 key modules, and 2-4 key methods
per module if possible. If uncertain, add short entries in the 'assumptions' array.

Return ONLY JSON (no extra text).

Context (code snippets):
{context_combined}
"""
    return prompt


def call_chat_model(prompt: str) -> str:
    """Call ChatOpenAI via LangChain and return the assistant content as string."""

    if DRY_RUN or SKIP_CHAT:
        print("[chat] SKIP_CHAT/DRY_RUN enabled — returning placeholder JSON.")
        stub = {
            "project_name": "DRY_RUN placeholder",
            "project_overview": "Dry-run mode; actual LLM call skipped.",
            "primary_languages": ["unknown"],
            "key_modules": [],
            "global_complexity_notes": "",
            "assumptions": ["dry-run: no LLM call executed"],
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        return json.dumps(stub)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing in environment for real runs.")

    chat = _get_langchain_chat()
    system = SystemMessage(
        content="You are a strict JSON-outputting code analyst. Output only JSON."
    )
    user = HumanMessage(content=prompt)

    result = chat.generate(messages=[[system, user]])

    try:
        gen = result.generations[0][0]
        content = gen.message.content
    except Exception:
        resp = chat([system, user])
        if hasattr(resp, "content"):
            content = resp.content
        else:
            content = str(resp)

    return content


# ---------------------
# Main orchestration
# ---------------------

def analyze_repo(git_url: str, repo_name: str = None):
    repo_path = clone_or_update(git_url, WORKSPACE, repo_name)
    repo_basename = os.path.basename(repo_path.rstrip("/"))
    collection = f"repo_{repo_basename.lower()}"

    print(f"Loaded repo at: {repo_path} -> collection: {collection}")

    code_docs = load_codebase(repo_path)
    print(f"Found {len(code_docs)} files.")

    # optional MAX_FILES limit
    if MAX_FILES > 0:
        code_docs = code_docs[:MAX_FILES]
        print(f"Limiting to first {MAX_FILES} files for this run.")

    print("Adding chunks to Qdrant...")

    total_chunks = 0
    for idx, doc in enumerate(code_docs):
        path = doc.get("path", "<unknown>")
        print(f"Processing file [{idx + 1}/{len(code_docs)}]: {path}")

        chks = chunk_text_simple(
            doc.get("content", ""),
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )
        print(f" → {len(chks)} chunk(s)")

        total_chunks += len(chks)

        file_docs = [
            {"path": path, "chunk_index": i, "text": c}
            for i, c in enumerate(chks)
        ]

        for j in range(0, len(file_docs), UPSERT_BATCH):
            batch = file_docs[j: j + UPSERT_BATCH]
            upsert_chunks_to_qdrant(collection, batch, dry_run=DRY_RUN)

    print(f"Created and upserted {total_chunks} chunks (streaming).")

    print("Retrieving top-k contexts...")
    top_k = retrieve_top_k_from_qdrant(
        collection,
        "Provide high-level summary and key methods",
        k=TOP_K
    )
    print(f"Retrieved {len(top_k)} contexts.")

    prompt = build_prompt(repo_basename, top_k)
    print("Building prompt and calling chat model...")

    raw = call_chat_model(prompt)

    try:
        payload = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise RuntimeError("LLM response did not contain JSON.")
        payload = json.loads(raw[start: end + 1])

    out_dir = os.path.join(OUTPUT_DIR, repo_basename)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "extracted_knowledge.json")

    if "generated_at" not in payload:
        payload["generated_at"] = datetime.utcnow().isoformat() + "Z"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved extracted JSON to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Git URL or local path")
    parser.add_argument("--name", required=False, help="Local repo folder name (optional)")
    args = parser.parse_args()

    analyze_repo(args.repo, repo_name=args.name)
