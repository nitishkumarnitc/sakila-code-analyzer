# 📘 **Sakila Codebase Analyzer (LLM-Powered)**

A fully containerized, end-to-end solution to **analyze any GitHub codebase** using **Large Language Models (LLMs)**, **vector embeddings**, and **Qdrant vector database**.
This project was built as part of a coding assignment for **PeerIslands**, focusing on:

* Automated codebase ingestion
* Efficient chunking of large source trees
* Embedding with OpenAI
* Semantic search using Qdrant
* LLM-based summarization
* Structured JSON output

---

## ⭐ **High-Level Architecture**

```
GitHub Repo → Clone/Update → File Loader → Chunker 
 → Embeddings → Qdrant → Semantic Retrieval → LLM Summary 
 → structured JSON output
```

### **Key Components**

| Component               | Responsibility                                                             |
| ----------------------- | -------------------------------------------------------------------------- |
| `downloader.py`         | Git clone / pull logic                                                     |
| `ingest.py`             | Recursively scan project & load text-based files                           |
| `analyzer.py`           | Main orchestration: chunking, embedding, upserting, retrieval, LLM summary |
| `vectorstore.py` | Qdrant client + collection management + embedding batch logic              |
| `cli.py`                | Command-line interface                                                     |
| `extract.py`            | JSON schema + final structured output formatting                           |
| `Dockerfile`            | Python runtime with OpenAI + Qdrant client                                 |
| `docker-compose.yml`    | Multi-container setup (App + Qdrant)                                       |

---

# 🧠 **Approach and Methodology**

## 1️⃣ Repository Acquisition

The system accepts a GitHub URL:

```
--repo https://github.com/janjakovacevic/SakilaProject.git
```

`downloader.py` handles:

* First-time clone
* Incremental updates (git pull)
* Workspace isolation (`/app/workspaces/...`)

---

## 2️⃣ Codebase Processing

`ingest.py` scans for meaningful text-based files:

* `.java`, `.md`, `.py`, `.sql`, `.xml`, `.json`, `.yaml`, `.html`, etc.

It stores each file as:

```json
{
  "path": "...",
  "content": "string"
}
```

---

## 3️⃣ Chunking Logic (Token-Limit Friendly)

Large files are split into overlapping chunks using a **robust, infinite-loop–safe** algorithm:

```text
chunk_size = 1000 chars  
overlap     = 200 chars
```

This ensures:

* No chunk exceeds model token limits
* Semantic continuity between chunks
* No infinite loop on end boundary

---

## 4️⃣ Vector Embeddings (OpenAI)

Each chunk is embedded using:

```
text-embedding-3-large (1536-dim)
```

Embedding batching parameters (tunable):

* `QDRANT_EMBED_BATCH=4`
* `UPSERT_BATCH=32`

---

## 5️⃣ Vector Storage (Qdrant)

Qdrant stores embedded chunks with metadata:

```json
{
  "path": "file.java",
  "chunk_index": 12,
  "text": "actual code text..."
}
```

Collections are versioned per repository:

```
repo_sakilaproject
repo_mynewrepo
```

---

## 6️⃣ Semantic Retrieval

The system queries Qdrant using a natural-language prompt:

```
"Provide high-level summary and key methods"
```

Top-K most relevant chunks are retrieved.

Example default:

```
TOP_K = 8
```

---

## 7️⃣ LLM Summarization (OpenAI GPT-4o/5)

The retrieved chunks are given to an LLM to generate structured insights:

* Project overview
* Technologies used
* Key classes
* Important methods
* Method signatures
* Complexity notes
* Assumptions

---

## 8️⃣ Structured JSON Output

Results are written to:

```
outputs/<RepoName>/extracted_knowledge.json
```

Example structure:

```json
{
  "project_name": "...",
  "project_overview": "...",
  "key_modules": [...],
  "generated_at": "timestamp"
}
```

---

# 🐳 **Docker Setup**

This project is fully containerized:

## **Services**

### **1. App Container**

* Python 3.10
* OpenAI client
* Qdrant client
* LangChain-lite integration
* All repo code mounted inside `/app/src`

### **2. Qdrant Vector Database**

Vector similarity search engine.

Expose:

```
localhost:6333
```

---

# ▶️ **How to Run**

## 0. Prerequisite: Set OpenAI API Key

You may place it inside `.env`:

```
OPENAI_API_KEY=your_key_here
```

Or export it:

```bash
export OPENAI_API_KEY="your_key_here"
```

---

## 1. Start Qdrant (if not already running)

```bash
docker compose up -d qdrant
```

---

## 2. Run the Analyzer (Full Production Run)

```bash
docker compose run --rm \
  -e DRY_RUN=0 \
  -e SKIP_CHAT=0 \
  -e MAX_FILES=0 \
  -e CHUNK_SIZE=1000 \
  -e CHUNK_OVERLAP=200 \
  -e QDRANT_EMBED_BATCH=4 \
  -e UPSERT_BATCH=32 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  app python -u src/cli.py \
    --repo https://github.com/janjakovacevic/SakilaProject.git \
    --name SakilaProject
```

---

## 3. Output Location

After successful run:

```
outputs/SakilaProject/extracted_knowledge.json
```

Open it:

```bash
cat outputs/SakilaProject/extracted_knowledge.json | jq .
```

---

# 🛠️ **Development Mode (Dry Run)**

Useful during debugging:

```bash
docker compose run --rm --no-deps \
  -e DRY_RUN=1 \
  -e SKIP_CHAT=1 \
  -e MAX_FILES=10 \
  -e CHUNK_SIZE=200 \
  -e CHUNK_OVERLAP=50 \
  app python -u src/cli.py \
    --repo https://github.com/janjakovacevic/SakilaProject.git \
    --name TestRun
```

No OpenAI calls.
No real embeddings.
No LLM summary.
Fast & memory-safe.

---

# 📚 **Libraries Used**

| Library                      | Purpose                             |
| ---------------------------- | ----------------------------------- |
| **OpenAI Python SDK**        | Embeddings + chat LLM summarization |
| **Qdrant Client**            | Vector storage and semantic search  |
| **GitPython / subprocess**   | Repo cloning & updating             |
| **Pydantic**                 | Output schema validation            |
| **tqdm**                     | Progress loops                      |
| **LangChain-lite utilities** | Text handling & similarity helpers  |
| **Docker + Docker Compose**  | Full reproducible environment       |

---

# 🚀 **Why This Approach?**

### ✔ Handles Large Codebases

Chunking + embeddings + vector storage allow processing >1000 files.

### ✔ Scalable

Qdrant + OpenAI = fast & distributed.

### ✔ Deterministic JSON Output

Stable schema for automation or dashboards.

### ✔ Containerized & Repeatable

Everything works identically on any system.

---

# 🎯 **Future Improvements**

* Token-level chunking (better for transformer models)
* Language-specific parsers (Java AST extraction)
* UI dashboard for visualizing code insights
* Model streaming for faster summarization
* Multi-repo batch analysis

---

# 🙌 **Credits**
Developed as part of a coding challenge.
Includes Sakila Sample Project (open source).
---

