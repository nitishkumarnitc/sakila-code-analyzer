# src/extract.py
import json
import logging
from typing import List, Dict, Optional

# langchain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# qdrant client wrapper
from vectorstore import client as qclient

# centralized config
import config as cfg

# logging
logging.getLogger("httpx").setLevel(getattr(logging, cfg.HTTPX_LOG_LEVEL.upper(), logging.WARNING))
logger = logging.getLogger("extract")
logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO))

# config-driven constants
TOP_K = cfg.TOP_K
EMBED_MODEL = cfg.LANGCHAIN_EMBEDDING_MODEL
CHAT_MODEL = cfg.OPENAI_CHAT_MODEL

# lazy langchain instances
_lc_embeddings: Optional[OpenAIEmbeddings] = None
_lc_chat: Optional[ChatOpenAI] = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _lc_embeddings
    if _lc_embeddings is None:
        _lc_embeddings = OpenAIEmbeddings(model=EMBED_MODEL, chunk_size=cfg.EMBED_BATCH)
    return _lc_embeddings


def _get_chat() -> ChatOpenAI:
    global _lc_chat
    if _lc_chat is None:
        _lc_chat = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
    return _lc_chat


def retrieve_top_k(collection_name: str, query: str, k: int = TOP_K) -> List[Dict]:
    """
    Embed the query using LangChain embeddings and search qdrant via qclient.
    Returns list of {path, chunk_index, text}
    """
    if not query:
        return []

    emb = _get_embeddings()
    try:
        q_emb = emb.embed_query(query)
    except Exception as e:
        logger.exception("Embedding query failed: %s", e)
        raise

    res = qclient.search(collection_name=collection_name, query_vector=q_emb, limit=k)
    out = []
    for hit in res:
        payload = getattr(hit, "payload", None) or (hit.get("payload") if isinstance(hit, dict) else {})
        out.append({
            "path": payload.get("path"),
            "chunk_index": payload.get("chunk_index"),
            "text": payload.get("text") or payload.get("document") or ""
        })
    return out


PROMPT_TEMPLATE = """
You are a senior engineering lead and code reviewer. Return only JSON strictly following the schema given below.
Use the provided contexts as evidence. Prioritize clarity, risks, and actionable recommendations.

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


def _call_chat_robust(system: SystemMessage, user: HumanMessage) -> str:
    """
    Try multiple invocation styles of ChatOpenAI for compatibility across langchain versions.
    """
    chat = _get_chat()

    # 1) generate (structured)
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

    # 2) predict_messages
    try:
        resp = chat.predict_messages([system, user])
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e:
        logger.debug("chat.predict_messages failed: %s", e)

    # 3) predict
    try:
        resp = chat.predict([system, user])
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e:
        logger.debug("chat.predict failed: %s", e)

    # 4) call-style
    try:
        resp = chat([system, user])
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e:
        logger.debug("chat call-style failed: %s", e)

    raise RuntimeError("Unable to call ChatOpenAI with known methods. Check langchain-openai / langchain-core versions.")


def ask_llm(project_name: str, contexts: List[Dict]) -> str:
    """
    Build prompt from contexts and call the chat model. Returns raw assistant string.
    """
    ctx = "\n\n---\n\n".join([f"[{c.get('path')}#{c.get('chunk_index')}]\n{(c.get('text') or '')[:3000]}" for c in contexts])
    prompt = PROMPT_TEMPLATE.format(context=ctx)

    system = SystemMessage(content="You are a senior engineering lead. Output only JSON and follow instructions exactly.")
    user = HumanMessage(content=prompt)

    return _call_chat_robust(system, user)


def ask_and_parse(project_name: str, contexts: List[Dict]) -> Dict:
    raw = ask_llm(project_name, contexts)
    try:
        return json.loads(raw)
    except Exception:
        s = raw.find("{")
        e = raw.rfind("}")
        if s == -1 or e == -1:
            logger.error("LLM output did not contain JSON: %s", raw[:500])
            raise RuntimeError("LLM output did not contain JSON")
        return json.loads(raw[s:e+1])
