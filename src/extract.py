import os
from typing import List, Dict

# Your existing qdrant client wrapper
from vectorstore import client as qclient

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# -----------------------------
# Config
# -----------------------------

OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
TOP_K = int(os.environ.get("TOP_K", "8"))

# Lazy-loaded LangChain instances
_lc_embeddings: OpenAIEmbeddings | None = None
_lc_chat: ChatOpenAI | None = None


# -----------------------------
# Embeddings + Chat Helpers
# -----------------------------

def _get_embeddings() -> OpenAIEmbeddings:
    """Lazy-init OpenAI Embeddings."""
    global _lc_embeddings
    if _lc_embeddings is None:
        _lc_embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    return _lc_embeddings


def _get_chat() -> ChatOpenAI:
    """Lazy-init ChatOpenAI."""
    global _lc_chat
    if _lc_chat is None:
        _lc_chat = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)
    return _lc_chat


# -----------------------------
# Qdrant Retrieval
# -----------------------------

def retrieve_top_k(collection_name: str, query: str, k: int = TOP_K) -> List[Dict]:
    """
    Use LangChain OpenAIEmbeddings to embed the query and query Qdrant via qclient.
    Returns list of dicts: {path, chunk_index, text}
    """
    if not query:
        return []

    lc_emb = _get_embeddings()
    q_emb = lc_emb.embed_query(query)

    res = qclient.search(
        collection_name=collection_name,
        query_vector=q_emb,
        limit=k
    )

    output = []
    for hit in res:
        payload = getattr(hit, "payload", None) or (
            hit.get("payload") if isinstance(hit, dict) else {}
        )

        output.append({
            "path": payload.get("path"),
            "chunk_index": payload.get("chunk_index"),
            "text": payload.get("text")
                or payload.get("document")
                or ""
        })

    return output


# -----------------------------
# Prompt Template
# -----------------------------

PROMPT_TEMPLATE = """
You are a precise code analyst. Return only JSON with this schema:

{
  "project_name": "string",
  "project_overview": "string",
  "primary_languages": ["string"],
  "key_modules": [...],
  "global_complexity_notes": "string",
  "assumptions": ["string"],
  "generated_at": "ISO8601 timestamp"
}

Context:
{context}
"""


# -----------------------------
# LLM Call
# -----------------------------

def ask_llm(project_name: str, contexts: List[Dict]) -> str:
    """
    Build prompt from contexts and call ChatOpenAI.
    Returns the raw assistant response as a string.
    """

    context_combined = "\n\n---\n\n".join(
        [
            f"[{c.get('path')}#{c.get('chunk_index')}]\n"
            f"{(c.get('text') or '')[:3000]}"
            for c in contexts
        ]
    )

    prompt = (
        PROMPT_TEMPLATE
        .replace("{context}", context_combined)
        .replace("{project}", project_name)
    )

    chat = _get_chat()
    system_msg = SystemMessage(content="You are a code summarizer. Output only JSON.")
    user_msg = HumanMessage(content=prompt)

    # Most langchain-openai versions allow chat([messages...])
    response = chat([system_msg, user_msg])

    if hasattr(response, "content"):
        return response.content

    return str(response)
