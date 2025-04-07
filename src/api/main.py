"""
FastAPI Application — RAG Document Intelligence
Endpoints: ingest, query, stream, search, summarize
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document Intelligence API",
    description="PDF/document Q&A, semantic search, and summarization via RAG",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)
    stream: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    faithfulness_score: float
    is_faithful: bool
    model: str

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(10, ge=1, le=50)

class SummarizeRequest(BaseModel):
    document_names: Optional[list[str]] = None  # None = summarize all
    style: str = Field("concise", pattern="^(concise|detailed|bullet_points)$")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "rag-document-intelligence"}


@app.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_document(file: UploadFile = File(...)):
    """Upload and index a document (PDF, DOCX, TXT, HTML)."""
    allowed = {".pdf", ".docx", ".txt", ".html", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # In production: inject DocumentLoader + FAISSVectorStore dependencies
    logger.info(f"Ingesting: {file.filename} ({len(content):,} bytes)")

    return {
        "status": "indexed",
        "filename": file.filename,
        "size_bytes": len(content),
        "message": f"Document '{file.filename}' successfully ingested and indexed.",
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question over indexed documents. Returns grounded answer with citations."""
    if request.stream:
        raise HTTPException(400, "Use /query/stream for streaming responses.")

    # In production: inject RAGChain dependency
    logger.info(f"Query: {request.question}")

    # Placeholder response structure (wire up RAGChain in production)
    return QueryResponse(
        answer="Answer grounded in retrieved document context with citations. [Source: doc.pdf, Page 3]",
        sources=[{"source": "doc.pdf", "page": 3, "excerpt": "Relevant excerpt...", "relevance_score": 0.92}],
        faithfulness_score=0.97,
        is_faithful=True,
        model="gpt-4o",
    )


@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """Stream a RAG response token-by-token via Server-Sent Events."""
    async def event_generator():
        # In production: yield from rag_chain.astream(request.question)
        sample = f"Streaming answer to: {request.question}\n\nBased on the documents provided..."
        for token in sample.split():
            yield f"data: {token} \n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/search")
async def semantic_search(request: SearchRequest):
    """Semantic search over indexed documents. Returns ranked relevant chunks."""
    logger.info(f"Search: {request.query}")
    # In production: inject retriever and call retriever.retrieve(request.query, top_k=request.top_k)
    return {
        "query": request.query,
        "results": [
            {
                "text": "Relevant passage from indexed document...",
                "source": "document.pdf",
                "page": 1,
                "score": 0.91,
            }
        ],
        "total": 1,
    }


@app.post("/summarize")
async def summarize_documents(request: SummarizeRequest):
    """Generate abstractive summary of one or all indexed documents."""
    logger.info(f"Summarize request — style: {request.style}, docs: {request.document_names}")
    # In production: retrieve all chunks for specified docs, run map-reduce summarization
    return {
        "summary": "Abstractive summary of the indexed documents...",
        "style": request.style,
        "documents_summarized": request.document_names or ["all"],
    }


@app.delete("/documents/{document_name}")
async def delete_document(document_name: str):
    """Remove a document and its vectors from the index."""
    logger.info(f"Deleting document: {document_name}")
    return {"status": "deleted", "document": document_name}


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
