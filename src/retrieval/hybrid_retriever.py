"""
Hybrid Retriever — Dense + Sparse Search with Cross-Encoder Re-ranking
Combines OpenAI embeddings (dense) + BM25 (sparse) via Reciprocal Rank Fusion.
Cross-encoder re-ranking for top-k precision before generation.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    retrieval_method: str = "hybrid"  # dense | sparse | hybrid


class FAISSVectorStore:
    """FAISS-backed dense vector store with OpenAI embeddings."""

    def __init__(self, embedding_model: str = "text-embedding-3-large", dimension: int = 3072):
        import faiss
        from openai import OpenAI

        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine on normalized vectors)
        self.chunks: list[dict] = []  # Stores text + metadata

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings for a batch of texts."""
        response = self.client.embeddings.create(model=self.embedding_model, input=texts)
        embeddings = np.array([e.embedding for e in response.data], dtype=np.float32)
        # L2 normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-9)

    def add(self, chunks: list[dict]):
        """Add chunks (text + metadata) to the FAISS index."""
        texts = [c["text"] for c in chunks]
        embeddings = self.embed(texts)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"FAISS index: {self.index.ntotal} total vectors")

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Dense ANN search for top-k similar chunks."""
        q_emb = self.embed([query])
        scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievedChunk(
                text=chunk["text"],
                score=float(score),
                metadata=chunk.get("metadata", {}),
                retrieval_method="dense",
            ))
        return results

    def save(self, path: str):
        import faiss, pickle
        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str):
        import faiss, pickle
        self.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)


class BM25Retriever:
    """Sparse BM25 retriever for keyword-based complementary retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[dict] = []
        self._index = None

    def fit(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self._index = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        logger.info(f"BM25 index built with {len(chunks)} documents")

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        if self._index is None:
            raise RuntimeError("BM25 index not built. Call fit() first.")
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievedChunk(
                text=self.chunks[i]["text"],
                score=float(scores[i]),
                metadata=self.chunks[i].get("metadata", {}),
                retrieval_method="sparse",
            )
            for i in top_indices if scores[i] > 0
        ]


class CrossEncoderReranker:
    """Cross-encoder re-ranker for precision-focused top-k selection."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        logger.info(f"Cross-encoder loaded: {model_name}")

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        reranked = []
        for score, chunk in ranked[:top_k]:
            chunk.score = float(score)
            chunk.retrieval_method = "reranked"
            reranked.append(chunk)
        return reranked


class HybridRetriever:
    """
    Hybrid retriever combining dense (FAISS) + sparse (BM25) via
    Reciprocal Rank Fusion, followed by cross-encoder re-ranking.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25: BM25Retriever,
        reranker: Optional[CrossEncoderReranker] = None,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5, fetch_k: int = 20) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline:
        1. Dense search (FAISS)
        2. Sparse search (BM25)
        3. Reciprocal Rank Fusion
        4. Cross-encoder re-ranking
        """
        dense_results = self.vector_store.search(query, top_k=fetch_k)
        sparse_results = self.bm25.search(query, top_k=fetch_k)

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        if self.reranker and len(fused) > top_k:
            return self.reranker.rerank(query, fused[:fetch_k], top_k=top_k)

        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Merge two ranked lists via Reciprocal Rank Fusion (RRF)."""
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense):
            key = chunk.text[:100]
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_map[key] = chunk

        for rank, chunk in enumerate(sparse):
            key = chunk.text[:100]
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = chunk

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        result = []
        for key in sorted_keys:
            c = chunk_map[key]
            c.score = scores[key]
            c.retrieval_method = "hybrid"
            result.append(c)

        return result
