# 📘 **Sakila Codebase Analyzer (LLM-Powered, Production-Grade)**

A fully containerized, scalable system that analyzes **any GitHub codebase** using:

* **Parallel file chunking**
* **OpenAI embeddings with caching**
* **Qdrant vector database**
* **Semantic retrieval**
* **Robust LangChain-based LLM summarization**
* **Structured JSON output with optional validation**

This system was built for a **technical coding assignment for PeerIslands**, but the framework is general-purpose and can analyze any repository at scale.

---

# 🚀 High-Level Architecture

```
GitHub Repo → Clone/Update → File Loader → Parallel Chunker
 → Embedding Cache → Batched Embedding → Qdrant Upsert (Thread-Safe)
 → Semantic Retrieval → LLM Summary → Validated JSON Output
```

---

# 🧩 Components Overview

| Component            | Responsibility                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `config.py`          | Loads & validates all environment-based configuration in one place                                                  |
| `downloader.py`      | Clones/pulls the target GitHub repository                                                                           |
| `ingest.py`          | Walks the repo file tree and loads allowed text-based files                                                         |
| `analyzer.py`        | Main orchestration: static analysis, parallel chunking, embedding, upsert, retrieval, LLM summarization, validation |
| `vectorstore.py`     | Qdrant client wrapper (collection management + safe upserts)                                                        |
| `extract.py`         | Smaller utility for producing structured JSON summaries                                                             |
| `validate_report.py` | Optional validation layer to annotate/verify final output                                                           |
| `cli.py`             | Command-line interface runner                                                                                       |
| `docker-compose.yml` | App + Qdrant as isolated microservices                                                                              |
| `Dockerfile`         | Application runtime with required libraries                                                                         |

---

# ⚙️ Key Features

## 🔹 **1. Centralized Configuration**

All environment variables (OpenAI models, chunk sizes, workers, cache dirs) are imported via `config.py`.

Supports `.env`, `docker-compose`, or environment export.

---

## 🔹 **2. Parallel Chunking Engine**

Each file is split into overlapping chunks using configurable settings:

```
CHUNK_SIZE=1200
CHUNK_OVERLAP=240
CHUNKER_WORKERS=4+
```

Parallelism drastically speeds up large codebase processing.

---

## 🔹 **3. Embedding Cache (SHA256-based)**

To minimize OpenAI usage and speed up re-runs:

* Each chunk is hashed
* Embeddings are cached under `EMBED_CACHE_DIR`
* Atomic writes protect cache from corruption
* Cache is reused on future runs

---

## 🔹 **4. Thread-Safe Embedding + Qdrant Upsert**

Embedding and upserting happen in parallel:

* `UPSERTER_WORKERS` controls concurrency
* `UPSERT_BATCH` controls batch size
* `_qdrant_lock` ensures thread-safe Qdrant upserts

---

## 🔹 **5. Robust LangChain Chat Invocation**

LangChain changes APIs often.

This project includes a **multi-strategy failover**:

```python
chat.generate → chat.predict_messages → chat.predict → chat() call
```

This prevents version-related crashes like:

```
TypeError: 'ChatOpenAI' is not callable
```

---

## 🔹 **6. Static Analysis (Lightweight but Useful)**

The analyzer extracts:

* Approx function signatures
* Cyclomatic complexity estimates
* LOC metrics
* Small contextual snippets

This helps the LLM produce **lead-level architectural analysis**.

---

## 🔹 **7. Top-K Semantic Retrieval via Qdrant**

Retrieval prompt example:

```
Provide high-level summary, architecture, key modules, and risks
```

Default:

```
TOP_K = 8
```

---

## 🔹 **8. Lead-Level Structured JSON Output**

The LLM is forced (via schema prompt) to produce:

* Project overview
* Architecture summary
* Key modules + methods (LOC, cyclomatic complexity, source_ref)
* Dependencies
* Tests & CI presence
* Global risks & recommendations
* ISO timestamp

Crypto-stable JSON is saved:

```
outputs/<repo>/extracted_knowledge.json
```

---

## 🔹 **9. Optional Validation Step**

If enabled (`VALIDATE_AFTER=1`), a validated version is also created:

```
outputs/<repo>/extracted_knowledge_validated.json
```

---

# 🐳 Dockerized Execution

## ✔ Start Qdrant

```bash
docker compose up -d qdrant
```

---

## ✔ Full Production Run

```bash
docker compose run --rm \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  app python -u src/cli.py \
    --repo https://github.com/janjakovacevic/SakilaProject.git \
    --name SakilaProject
```

This performs:

* Chunking
* Embedding (cached where possible)
* Upsert
* Retrieval
* LLM summarization
* Final JSON generation
* Optional validation

---

## ✔ Development / Fast Mode (No API Calls)

```bash
docker compose run --rm \
  -e DRY_RUN=1 \
  -e SKIP_CHAT=1 \
  -e MAX_FILES=20 \
  app python -u src/cli.py \
    --repo https://github.com/janjakovacevic/SakilaProject.git \
    --name TestRun
```

Produces a valid JSON stub instantly.

---

# 📁 Output Example (Schema)

```json
{
  "project_name": "SakilaProject",
  "project_overview": "...",
  "architecture_summary": "...",
  "primary_languages": ["Java"],
  "dependencies": [...],
  "key_modules": [...],
  "tests_and_ci": {...},
  "global_complexity_notes": "...",
  "recommendations": [...],
  "generated_at": "2025-01-31T10:22:33Z",
  "assumptions": [...]
}
```

---

# 📁  Example (.env file)
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini
LANGCHAIN_EMBEDDING_MODEL=text-embedding-3-small

# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=

# Directories
WORKSPACE_DIR=/app/workspaces
OUTPUT_DIR=/app/outputs

# Retrieval
TOP_K=8

# Chunking
CHUNK_SIZE=1500
CHUNK_OVERLAP=270

# Batching & Parallelism
UPSERT_BATCH=128
EMBED_BATCH=64
EMBED_CONCURRENCY=6
UPSERTER_WORKERS=6
CHUNKER_WORKERS=6

# Embed cache
EMBED_CACHE_DIR=/tmp/embed_cache

# Post-run validation
VALIDATE_REPORT=1

# 🧪 Upcoming Enhancements

* Token-aware chunking (`tiktoken`-based)
* Java/Python/TS AST-based parsing for deeper static analysis
* Incremental re-analysis (only changed files)
* A small UI dashboard for insights
* Model streaming for low-latency summarization

---

# 🙌 Credits

Developed as part of a engineering challenge.
Uses:

* **OpenAI embeddings + chat models**
* **Qdrant vector database**
* **LangChain for LLM orchestration**
* **Docker for reproducibility**

---



