from typing import List, Tuple, Dict
from langchain_core.documents import Document
from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines FAISS dense + BM25 sparse retrieval using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever,
                 rrf_k: int = 60, dense_weight: float = 0.6):
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight

    def _rrf_score(self, rank: int, weight: float) -> float:
        return weight / (self.rrf_k + rank + 1)

    def retrieve(self, query: str, k: int = 8) -> List[Document]:
        dense_results = self.dense.retrieve(query, k=k*2)
        sparse_results = self.sparse.retrieve(query, k=k*2)
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, (doc, _) in enumerate(dense_results):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + self._rrf_score(rank, self.dense_weight)
            doc_map[key] = doc

        for rank, (doc, _) in enumerate(sparse_results):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + self._rrf_score(rank, self.sparse_weight)
            doc_map[key] = doc

        sorted_keys = sorted(scores, key=scores.get, reverse=True)[:k]
        logger.info(f"Hybrid retrieval: {len(dense_results)} dense + {len(sparse_results)} sparse -> {k} merged")
        return [doc_map[k] for k in sorted_keys]
