# src/extract.py
import os
import json
import logging
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from vectorstore import client as qclient  # your existing Qdrant wrapper

logger = logging.getLogger("extract")
logging.basicConfig(level=logging.INFO)

OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
TOP_K = int(os.environ.get("TOP_K", "8"))

_lc_embeddings: Optional[OpenAIEmbeddings] = None
_lc_chat: Optional[ChatOpenAI] = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _lc_embeddings
    if _lc_embeddings is None:
        _lc_embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    return _lc_embeddings


def _get_chat() -> ChatOpenAI:
    global _lc_chat
    if _lc_chat is None:
        _lc_chat = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)
    return _lc_chat


def retrieve_top_k(collection_name: str, query: str, k: int = TOP_K) -> List[Dict]:
    if not query:
        return []
    emb = _get_embeddings()
    q_emb = emb.embed_query(query)
    res = qclient.search(collection_name=collection_name, query_vector=q_emb, limit=k)
    out = []
    for hit in res:
        payload = getattr(hit, "payload", None) or (hit.get("payload") if isinstance(hit, dict) else {})
        out.append({"path": payload.get("path"), "chunk_index": payload.get("chunk_index"), "text": payload.get("text") or payload.get("document") or ""})
    return out


PROMPT_TEMPLATE = """
You are a senior engineering lead and code reviewer. Return only JSON strictly following the schema given below. Use the provided contexts as evidence. Prioritize clarity, risks, and actionable recommendations.

Schema:
{{
  "project_name": "string",
  "project_overview": "string",
  "primary_languages": ["string"],
  "key_modules": [...],
  "tests_and_ci": {{ "has_tests": bool, "test_paths": ["string"], "ci_present": bool }},
  "dependencies": [...],
  "global_complexity_notes": "string",
  "recommendations": ["string"],
  "generated_at": "ISO8601",
  "assumptions": ["string"]
}}

Contexts:
{context}
"""


def ask_llm(project_name: str, contexts: List[Dict]) -> str:
    ctx = "\n\n---\n\n".join([f"[{c.get('path')}#{c.get('chunk_index')}]\n{(c.get('text') or '')[:3000]}" for c in contexts])
    prompt = PROMPT_TEMPLATE.format(context=ctx)

    chat = _get_chat()
    system = SystemMessage(content="You are a senior engineering lead. Output only JSON and follow instructions exactly.")
    user = HumanMessage(content=prompt)

    # Try multiple invocation styles
    # 1) generate
    try:
        res = chat.generate(messages=[[system, user]])
        gen = res.generations[0][0]
        if hasattr(gen, "message") and getattr(gen.message, "content", None) is not None:
            return gen.message.content
        if getattr(gen, "text", None) is not None:
            return gen.text
    except Exception:
        logger.debug("generate failed, trying predict_messages")

    try:
        resp = chat.predict_messages([system, user])
        if hasattr(resp, "content"):
            return resp.content
    except Exception:
        logger.debug("predict_messages failed, trying predict")

    try:
        resp = chat.predict([system, user])
        if hasattr(resp, "content"):
            return resp.content
    except Exception:
        logger.debug("predict failed, trying call style")

    try:
        resp = chat([system, user])
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e:
        raise RuntimeError("Unable to call ChatOpenAI. Check langchain versions.") from e


def ask_and_parse(project_name: str, contexts: List[Dict]) -> Dict:
    raw = ask_llm(project_name, contexts)
    try:
        return json.loads(raw)
    except Exception:
        s = raw.find("{")
        e = raw.rfind("}")
        if s == -1 or e == -1:
            raise RuntimeError("LLM output did not contain JSON")
        return json.loads(raw[s:e+1])
