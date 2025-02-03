from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from typing import List, Tuple
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)


class SparseRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 2]

    def index(self, documents: List[Document]):
        self.documents = documents
        tokenized = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        logger.info(f"BM25 index built: {len(documents)} documents")

    def retrieve(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        if not self.bm25:
            raise ValueError("Index documents first")
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [(self.documents[i], float(scores[i])) for i in top_k_idx if scores[i] > 0]
