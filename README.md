# 🤖 Advanced Local RAG System

A fully local, production-oriented **Retrieval-Augmented Generation (RAG)** system designed to answer technical questions from heterogeneous documents with high retrieval accuracy, citation traceability, hallucination detection, and intelligent query routing.

The system combines **hybrid retrieval**, **dense vector search**, **BM25 keyword matching**, **cross-encoder reranking**, **dynamic LLM routing**, and **Natural Language Inference (NLI)** verification.

Everything runs locally, ensuring privacy and eliminating dependency on external LLM APIs.

---

# 🏗️ Architecture Overview

The system follows a modular Retrieval-Augmented Generation pipeline:

```text
                    ┌───────────────────┐
                    │   User Query      │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   Smart Router    │
                    │ Simple / Complex  │
                    └─────────┬─────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌───────────────┐           ┌───────────────┐
        │ Dense Search  │           │ BM25 Search   │
        │    Qdrant     │           │ Sparse Index  │
        └───────┬───────┘           └───────┬───────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    ┌───────────────────┐
                    │ Hybrid Retrieval  │
                    │       RRF         │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Cross-Encoder     │
                    │    Reranking      │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Local LLM         │
                    │ Qwen 2.5 Models   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ NLI Faithfulness  │
                    │   Verification    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Answer + Citations│
                    └───────────────────┘
```

---

# 🧠 Technology Stack

| Component         | Technology / Model             | Configuration / Note                                                 |
| ----------------- | ------------------------------ | -------------------------------------------------------------------- |
| **Backend API**   | FastAPI, Python                | REST API, fully containerized with Docker Compose                    |
| **Vector Store**  | Qdrant                         | On-disk persistence with HNSW indexing disabled during bulk uploads  |
| **Sparse Index**  | BM25 (`rank_bm25`)             | Persisted to disk for exact keyword matching                         |
| **Embeddings**    | FastEmbed (ONNX)               | `BAAI/bge-small-en-v1.5` — 384 dimensions, CPU optimized, no PyTorch |
| **LLM (Simple)**  | `qwen2.5:1.5b-instruct-q4_k_m` | Fast path for definitions and factual lookups                        |
| **LLM (Complex)** | `qwen2.5:3b-instruct-q4_k_m`   | Deep reasoning for multi-document comparisons                        |
| **LLM Server**    | Ollama                         | Executes quantized models locally on CPU                             |
| **Faithfulness**  | `nli-deberta-v3-small`         | Dynamic NLI verification to reduce hallucinations                    |
| **Reranker**      | `ms-marco-MiniLM-L-6-v2`       | Enabled for Medium and Complex queries via Smart Router              |
| **Caching**       | Redis 7-alpine                 | Sub-10ms responses for repeated queries                              |
| **Frontend**      | Gradio                         | Connected to the FastAPI `/query` endpoint                           |

---

# 📊 Performance Metrics & Evaluation

The system was evaluated using a custom **100% local evaluation framework** based on:

* ROUGE-L
* Faithfulness
* Citation Accuracy
* Retrieval Recall
* Response Latency

| Metric                         |            Result |
| ------------------------------ | ----------------: |
| **Overall Accuracy**           |         **87.2%** |
| **Target Accuracy**            |               75% |
| **Simple Query Accuracy**      |          **100%** |
| **Retrieval Context Recall@5** |        **98.46%** |
| **Faithfulness Rate**          |         **94.9%** |
| **Citation Accuracy**          |          **100%** |
| **Simple Query Latency**       | **~7–15 seconds** |
| **Complex Query Latency**      |   **~35 seconds** |
| **Cached Response Latency**    |         **<10ms** |

> ⚡ Performance measurements were obtained on an Intel i5 CPU using fully local inference.

---

# ⚙️ Key Features

## 🔍 Hybrid Retrieval

The retrieval system combines:

* Dense semantic search using **Qdrant**
* Sparse keyword search using **BM25**
* Reciprocal Rank Fusion (**RRF**) for result fusion

This architecture allows the system to retrieve both:

* Semantically relevant information
* Exact technical keywords
* Identifiers
* Function names
* Configuration values
* Domain-specific codes

---

## 🧠 Intelligent Query Routing

A Smart Router analyzes query complexity and dynamically selects the appropriate processing pipeline.

### Simple Queries

Examples:

* Definitions
* Direct factual questions
* Single-document lookups

Pipeline:

```text
Query
  ↓
Fast Retrieval
  ↓
Qwen 1.5B
  ↓
NLI Verification
  ↓
Answer
```

Average response time:

```text
~7–15 seconds
```

### Medium and Complex Queries

Examples:

* Multi-document comparisons
* Technical reasoning
* Complex explanations

Pipeline:

```text
Query
  ↓
Hybrid Retrieval
  ↓
Cross-Encoder Reranking
  ↓
Qwen 3B
  ↓
NLI Verification
  ↓
Answer with Citations
```

Average response time:

```text
~35 seconds
```

---

# 🛡️ Hallucination Prevention

The system integrates an **NLI-based faithfulness verification layer** using:

```text
nli-deberta-v3-small
```

The generated answer is dynamically evaluated against the retrieved context.

This allows the system to:

* Detect unsupported claims
* Reduce hallucinated information
* Validate answer consistency
* Trigger fallback mechanisms when necessary

The measured faithfulness rate reached:

```text
94.9%
```

---

# 🏭 Production-Ready Engineering Features

This project goes beyond a basic RAG prototype by implementing several software engineering and production-oriented patterns.

## 📄 Dynamic Document Ingestion

The ingestion pipeline uses `python-magic` for MIME-type detection.

Supported formats include:

* PDF
* DOCX
* PPTX
* HTML
* CSV
* JSON
* YAML
* Excel
* Markdown
* Source code files
* Additional structured document formats

---

## 💻 Code-Aware Chunking

The system integrates Abstract Syntax Tree (**AST**) parsing for:

* Python
* Java
* JavaScript

Functions and classes are preserved as coherent units whenever possible.

This improves retrieval quality for technical documentation and source code.

---

## 🧩 Memory-Efficient Streaming

Documents are processed using streaming batches of:

```text
32 chunks
```

This approach:

* Keeps RAM usage stable
* Reduces memory pressure
* Prevents Out-Of-Memory errors
* Enables ingestion of larger document collections on CPU hardware

---

## 🔁 Idempotent Ingestion

Each chunk receives a deterministic SHA-1-based identifier.

```text
chunk_id = SHA1(content + metadata)
```

This prevents duplicate vectors during:

* Re-ingestion
* Pipeline restarts
* Document updates

---

## ⚡ CPU Thread Optimization

The system configures:

```bash
OMP_NUM_THREADS=4
```

This is tuned to match the available physical CPU resources and helps:

* Prevent thread oversubscription
* Reduce thread thrashing
* Improve CPU inference efficiency

---

## 🔄 Graceful Degradation

The system includes fallback mechanisms for critical components.

If a primary service fails, the pipeline can fall back to:

* Secondary LLM models
* Alternative verification mechanisms
* Prompt-based safety strategies

This improves system resilience and availability.

---

# 🚀 Getting Started

## 1. Prerequisites

Make sure the following software is installed:

* Docker
* Docker Compose
* Ollama

---

## 2. Pull Local Models

Pull the quantized models locally:

```bash
ollama pull qwen2.5:3b-instruct-q4_k_m
ollama pull qwen2.5:1.5b-instruct-q4_k_m
```

---

## 3. Clone the Repository

```bash
git clone https://github.com/your-username/excellia-rag.git
cd excellia-rag
```

---

## 4. Configure Environment Variables

Create your environment file:

```bash
cp .env.example .env
```

Update the configuration if necessary.

Example configuration:

```env
QDRANT_HOST=qdrant
QDRANT_PORT=6333

REDIS_HOST=redis
REDIS_PORT=6379

OLLAMA_HOST=http://host.docker.internal:11434

OMP_NUM_THREADS=4
```

---

## 5. Start the Backend

Launch the containers:

```bash
docker-compose up -d
```

Check the running services:

```bash
docker-compose ps
```

---

# 📥 Document Ingestion

Place your documents inside the:

```text
documents/
```

directory.

Then trigger ingestion:

```bash
curl -X POST http://localhost:8000/ingest \
-H "Content-Type: application/json" \
-d '{"path": null, "recreate": false}'
```

The ingestion pipeline will:

1. Detect the document MIME type
2. Extract the content
3. Apply code-aware or semantic chunking
4. Generate embeddings
5. Index vectors in Qdrant
6. Update the BM25 sparse index
7. Persist metadata and indexes

---

# 🖥️ Launch the User Interface

Install the frontend dependencies:

```bash
pip install gradio requests
```

Start the Gradio interface:

```bash
python ui.py
```

Open:

```text
http://localhost:8001
```

The Gradio frontend communicates with the FastAPI backend through:

```text
POST /query
```

---

# 📂 Project Structure

```text
excellia-rag/
│
├── app/
│   ├── api/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── verification/
│   └── services/
│
├── documents/
│
├── data/
│   ├── qdrant/
│   └── bm25/
│
├── evaluation/
│
├── ui.py
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 🔮 Future Evolution: Multimodal RAG

The current system focuses primarily on:

* Text
* Structured documents
* Tables
* Source code

The next evolution of the project is a transition toward a **multimodal RAG architecture** capable of understanding:

* Technical diagrams
* Images
* Schematics
* Complex tables
* Multi-format documents

## 🎯 Target Architecture

The planned technology stack includes:

* RAG-Anything
* LightRAG
* MinerU
* Docling

The objective is to build a multimodal knowledge architecture capable of linking:

```text
Text
   ↕
Images
   ↕
Tables
   ↕
Knowledge Graph
```

This will enable more advanced multi-document reasoning and richer contextual understanding.

---

# 📌 Project Highlights

* Fully local RAG architecture
* No external LLM API dependency
* Hybrid Dense + Sparse Retrieval
* Reciprocal Rank Fusion
* Cross-Encoder Reranking
* Dynamic LLM Routing
* NLI-based Hallucination Detection
* Citation Traceability
* Redis Caching
* Code-Aware Chunking
* Streaming Memory Management
* Idempotent Ingestion
* CPU-Optimized Local Inference
* Dockerized Architecture
* Multimodal RAG roadmap

---

# 📈 Results

The system achieved an overall evaluation accuracy of:

# **87.2%**

while exceeding the initial target of **75%**.

It achieved:

* **100% simple query accuracy**
* **98.46% retrieval recall@5**
* **94.9% faithfulness**
* **100% citation accuracy**

The project demonstrates that a high-quality, explainable, and resilient RAG system can be deployed entirely locally on consumer-grade CPU hardware.

---

# 👩‍💻 Author

**Emna Gammoudi**

Data Science & Artificial Intelligence Engineering Student

---

⭐ If you find this project useful, consider giving the repository a star!

