"""
Embedding Generation Pipeline
Generates dense vector embeddings for document chunks using OpenAI
or HuggingFace models, then indexes them into FAISS or Pinecone.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for document chunks using OpenAI or HuggingFace.
    Supports batching, caching, and multiple backend models.
    """

    SUPPORTED_MODELS = {
        "openai-large": "text-embedding-3-large",   # 3072 dims
        "openai-small": "text-embedding-3-small",   # 1536 dims
        "bge-large": "BAAI/bge-large-en-v1.5",      # 1024 dims, local
        "bge-base": "BAAI/bge-base-en-v1.5",        # 768 dims, local
    }

    def __init__(
        self,
        model: str = "openai-large",
        batch_size: int = 64,
        cache_dir: Optional[str] = None,
    ):
        self.model_key = model
        self.model_name = self.SUPPORTED_MODELS.get(model, model)
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._client = None
        self._hf_model = None
        self.dimension = self._get_dimension()

    def _get_dimension(self) -> int:
        dims = {
            "openai-large": 3072,
            "openai-small": 1536,
            "bge-large": 1024,
            "bge-base": 768,
        }
        return dims.get(self.model_key, 1536)

    def _get_openai_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _get_hf_model(self):
        if self._hf_model is None:
            from sentence_transformers import SentenceTransformer
            self._hf_model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded HuggingFace model: {self.model_name}")
        return self._hf_model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""
        if not texts:
            return np.array([])

        if self.model_key.startswith("openai"):
            return self._embed_openai(texts)
        else:
            return self._embed_hf(texts)

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        """Batch embed using OpenAI API."""
        client = self._get_openai_client()
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            # Clean texts
            batch = [t.replace("\n", " ").strip() for t in batch]
            response = client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.debug(f"Embedded batch {i // self.batch_size + 1}, {len(batch)} texts")

        embeddings = np.array(all_embeddings, dtype=np.float32)
        return self._normalize(embeddings)

    def _embed_hf(self, texts: list[str]) -> np.ndarray:
        """Batch embed using HuggingFace SentenceTransformer."""
        model = self._get_hf_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings for cosine similarity via dot product."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-9)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed([query])[0]

    def save_cache(self, embeddings: np.ndarray, key: str):
        """Cache embeddings to disk to avoid recomputation."""
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"{key}.pkl"
            with open(path, "wb") as f:
                pickle.dump(embeddings, f)
            logger.info(f"Cached {len(embeddings)} embeddings to {path}")

    def load_cache(self, key: str) -> Optional[np.ndarray]:
        """Load cached embeddings from disk."""
        if self.cache_dir:
            path = self.cache_dir / f"{key}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    embeddings = pickle.load(f)
                logger.info(f"Loaded {len(embeddings)} cached embeddings from {path}")
                return embeddings
        return None


class IndexBuilder:
    """
    Builds and persists FAISS or Pinecone vector index from document chunks.
    """

    def __init__(self, embedder: EmbeddingGenerator, store: str = "faiss"):
        self.embedder = embedder
        self.store = store

    def build_faiss_index(self, chunks: list[dict], save_path: str) -> "FAISSIndex":
        """
        Embed all chunks and build a FAISS flat index.
        chunks: list of {"text": ..., "metadata": {...}}
        """
        import faiss

        logger.info(f"Building FAISS index for {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        Path(save_path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, f"{save_path}/index.faiss")
        with open(f"{save_path}/chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

        logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}. Saved to {save_path}")
        return index

    def build_pinecone_index(
        self,
        chunks: list[dict],
        index_name: str,
        api_key: str,
        namespace: str = "default",
    ):
        """Embed chunks and upsert into Pinecone index."""
        import pinecone

        pc = pinecone.Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        logger.info(f"Upserting {len(chunks)} chunks to Pinecone index: {index_name}")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)

        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": chunk.get("chunk_id", f"chunk_{i}"),
                "values": emb.tolist(),
                "metadata": {
                    "text": chunk["text"][:1000],
                    **chunk.get("metadata", {}),
                },
            })

        # Upsert in batches of 100
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i:i+100], namespace=namespace)
            logger.debug(f"Upserted batch {i // 100 + 1}")

        logger.info(f"Pinecone upsert complete: {len(vectors)} vectors in namespace '{namespace}'")
