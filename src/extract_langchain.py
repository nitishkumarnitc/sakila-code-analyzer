# src/extract_langchain.py
import os, openai, json
from vectorstore_qdrant import client as qclient
from typing import List, Dict

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
TOP_K = int(os.environ.get("TOP_K", 8))

def retrieve_top_k(collection_name: str, query: str, k: int = TOP_K):
    # embed query (compat wrapper)
    q_emb = openai.embeddings.create(model=os.environ["OPENAI_EMBED_MODEL"], input=[query])["data"][0]["embedding"]
    res = qclient.search(collection_name=collection_name, query_vector=q_emb, limit=k)
    # res items contain .payload and .vector
    out = []
    for hit in res:
        out.append({"path": hit.payload.get("path"), "chunk_index": hit.payload.get("chunk_index"), "text": hit.payload.get("text") or hit.payload.get("document") or ""})
    return out

PROMPT_TEMPLATE = """You are a precise code analyst...
Return only JSON with this schema: { ... } 
Context: {context}
"""

def ask_llm(project_name: str, contexts: List[Dict]):
    context_combined = "\n\n---\n\n".join([f"[{c['path']}#{c['chunk_index']}]\n{c['text'][:3000]}" for c in contexts])
    prompt = PROMPT_TEMPLATE.replace("{context}", context_combined).replace("{project}", project_name)
    # call OpenAI Chat Completion (compat)
    resp = openai.ChatCompletion.create(model=OPENAI_MODEL, messages=[
        {"role":"system","content":"You are a code summarizer. Output only JSON."},
        {"role":"user","content":prompt}
    ], temperature=0, max_tokens=1200)
    return resp["choices"][0]["message"]["content"]
