"""
Document Chunker — Recursive + Semantic Chunking
Splits documents into optimal chunks for RAG retrieval with configurable overlap.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    chunk_id: str
    metadata: dict = field(default_factory=dict)
    token_count: int = 0


class RecursiveChunker:
    """
    Recursive character-based chunker with overlap.
    Respects natural boundaries: paragraphs → sentences → words.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[list[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        """Split text into overlapping chunks respecting natural boundaries."""
        metadata = metadata or {}
        raw_chunks = self._split_recursive(text, self.separators)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            source = metadata.get("source", "unknown")
            page = metadata.get("page", 0)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{source}_p{page}_c{i}",
                metadata={**metadata, "chunk_index": i, "chunk_total": len(raw_chunks)},
                token_count=len(chunk_text.split()),
            ))

        logger.debug(f"Chunked into {len(chunks)} pieces (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split on separators until all chunks are within size."""
        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        separator = separators[0]
        splits = text.split(separator) if separator else list(text)
        splits = [s for s in splits if s.strip()]

        chunks, current = [], ""
        for split in splits:
            candidate = current + (separator if current else "") + split
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    # If current chunk itself is too big, recurse
                    if len(current) > self.chunk_size:
                        chunks.extend(self._split_recursive(current, separators[1:]))
                    else:
                        chunks.append(current)
                    # Overlap: carry forward last portion of current
                    overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current = overlap_text + (separator if overlap_text else "") + split
                else:
                    current = split

        if current.strip():
            if len(current) > self.chunk_size:
                chunks.extend(self._split_recursive(current, separators[1:]))
            else:
                chunks.append(current)

        return chunks


class SemanticChunker:
    """
    Semantic chunker that groups sentences by embedding similarity.
    Produces more coherent chunks than fixed-size splitting.
    Requires an embedding model.
    """

    def __init__(self, embedding_fn, similarity_threshold: float = 0.85, max_chunk_size: int = 800):
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        """Group semantically similar sentences into coherent chunks."""
        import numpy as np
        metadata = metadata or {}

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        embeddings = self.embedding_fn(sentences)
        groups = self._group_by_similarity(sentences, embeddings)

        chunks = []
        for i, group in enumerate(groups):
            chunk_text = " ".join(group).strip()
            if not chunk_text:
                continue
            source = metadata.get("source", "unknown")
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{source}_sem{i}",
                metadata={**metadata, "chunk_index": i, "chunking": "semantic"},
                token_count=len(chunk_text.split()),
            ))

        logger.debug(f"Semantic chunking produced {len(chunks)} chunks.")
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _group_by_similarity(self, sentences: list[str], embeddings) -> list[list[str]]:
        import numpy as np
        groups, current_group = [], [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            projected_len = current_len + len(sentences[i])

            if sim >= self.similarity_threshold and projected_len <= self.max_chunk_size:
                current_group.append(sentences[i])
                current_len += len(sentences[i])
            else:
                groups.append(current_group)
                current_group = [sentences[i]]
                current_len = len(sentences[i])

        if current_group:
            groups.append(current_group)

        return groups

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        import numpy as np
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0
