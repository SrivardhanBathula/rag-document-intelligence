# 📄 RAG-Powered Document Intelligence App

> **Production-ready Retrieval-Augmented Generation pipeline for PDF/document Q&A, semantic search, and multi-document summarization — built with LangChain, OpenAI, FAISS, and FastAPI.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-purple?logo=openai)](https://openai.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS%20%7C%20Pinecone-orange)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

A fully production-ready RAG system that ingests PDFs, Word docs, and web pages, indexes them into a vector store, and answers natural language questions with cited, grounded responses. Supports multi-document Q&A, semantic search, and abstractive summarization.

### 🏆 Key Capabilities
| Feature | Details |
|---|---|
| Document Ingestion | PDF, DOCX, TXT, HTML, web scraping |
| Chunking Strategy | Recursive + semantic chunking with overlap |
| Embedding Models | OpenAI `text-embedding-3-large`, HuggingFace fallback |
| Vector Stores | FAISS (local), Pinecone (cloud), ChromaDB |
| LLM Backend | GPT-4o, GPT-3.5-turbo, Claude 3, local Llama |
| Answer Grounding | Source citations with page numbers |
| Hallucination Guard | Faithfulness scoring on every response |
| API | FastAPI with streaming responses |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                      │
│   PDF/DOCX/HTML → Text Extraction → Chunking → Embed    │
│   → Vector Store (FAISS / Pinecone / Chroma)            │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                   RETRIEVAL LAYER                         │
│   Query → Embedding → ANN Search → Re-ranking            │
│   Hybrid Search (dense + BM25 sparse)                    │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                  GENERATION LAYER                         │
│   Retrieved Chunks + Query → Prompt Assembly             │
│   → LLM (GPT-4o) → Grounded Answer + Citations          │
│   → Faithfulness Check → Response                        │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                   SERVING LAYER                           │
│   FastAPI  │  Streaming SSE  │  Async  │  Docker         │
└──────────────────────────────────────────────────────────┘
```

---

## 🧠 Core Components

### 1. Document Ingestion (`src/ingestion/`)
- Multi-format loaders: PDF (PyMuPDF), DOCX, HTML, web scraping
- Recursive + semantic chunking with configurable overlap
- Metadata extraction: source, page number, section heading
- Async batch ingestion for large document sets

### 2. Retrieval Engine (`src/retrieval/`)
- Dense retrieval via OpenAI embeddings + FAISS / Pinecone ANN search
- Sparse retrieval via BM25 for keyword-heavy queries
- **Hybrid search** with Reciprocal Rank Fusion (RRF)
- Cross-encoder re-ranking for top-k precision

### 3. Generation Pipeline (`src/generation/`)
- Dynamic prompt assembly with retrieved context + chat history
- Streaming responses via Server-Sent Events (SSE)
- Source citation with document name and page number
- Faithfulness scoring to catch hallucinations before serving

### 4. FastAPI Application (`src/api/`)
- `/ingest` — upload and index documents
- `/query` — question answering with citations
- `/search` — semantic search over document corpus
- `/summarize` — multi-document abstractive summarization
- WebSocket endpoint for real-time streaming chat

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **LLM & Orchestration** | LangChain, LlamaIndex, OpenAI GPT-4o |
| **Embeddings** | OpenAI text-embedding-3-large, HuggingFace BGE |
| **Vector Stores** | FAISS, Pinecone, ChromaDB |
| **Document Parsing** | PyMuPDF, python-docx, BeautifulSoup4, Unstructured |
| **Re-ranking** | CrossEncoder (sentence-transformers), Cohere Rerank |
| **API** | FastAPI, WebSockets, SSE streaming |
| **Evaluation** | RAGAS, DeepEval faithfulness scoring |

---

## 📁 Project Structure

```
rag-document-intelligence/
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py      # Multi-format document loading
│   │   ├── chunker.py              # Recursive + semantic chunking
│   │   └── embedder.py             # Embedding generation + indexing
│   ├── retrieval/
│   │   ├── vector_store.py         # FAISS / Pinecone vector store
│   │   ├── hybrid_retriever.py     # Dense + sparse hybrid search
│   │   └── reranker.py             # Cross-encoder re-ranking
│   ├── generation/
│   │   ├── rag_chain.py            # Full RAG chain with citations
│   │   ├── prompt_templates.py     # Prompt engineering templates
│   │   └── faithfulness_guard.py   # Hallucination detection guard
│   ├── api/
│   │   ├── main.py                 # FastAPI application
│   │   ├── routers/                # Route handlers
│   │   └── schemas.py              # Request/response models
│   └── utils/
│       └── text_utils.py           # Text preprocessing utilities
├── tests/
├── configs/
│   └── config.yaml
├── notebooks/
│   ├── 01_RAG_Pipeline_Demo.ipynb
│   └── 02_Evaluation_RAGAS.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/SrivardhanBathula/rag-document-intelligence.git
cd rag-document-intelligence

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Set your API keys
export OPENAI_API_KEY=your_key_here
export PINECONE_API_KEY=your_key_here  # optional

# Run the API
uvicorn src.api.main:app --reload --port 8000

# Ingest a document
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_document.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "top_k": 5}'
```

---

## 📊 Evaluation Results (RAGAS)

| Metric | Score |
|---|---|
| Faithfulness | 0.94 |
| Answer Relevancy | 0.91 |
| Context Precision | 0.89 |
| Context Recall | 0.87 |

---

## 👤 Author

**Srivardhan Bathula** — AI/ML Engineer
📧 Srivardhan.Bathula1@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/srivardhan-bathula) | [GitHub](https://github.com/SrivardhanBathula)
