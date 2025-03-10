from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker using ms-marco for improved retrieval precision."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 batch_size: int = 32):
        self.model = CrossEncoder(model_name, max_length=512)
        self.batch_size = batch_size

    def rerank(self, query: str, documents: List[Document],
               top_k: int = 5) -> List[Tuple[Document, float]]:
        if not documents:
            return []
        pairs = [(query, doc.page_content[:512]) for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size,
                                   show_progress_bar=False)
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Reranked {len(documents)} docs, returning top {top_k}")
        return scored[:top_k]
