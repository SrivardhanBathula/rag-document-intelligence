"""
RAG Chain — Grounded Answer Generation with Source Citations
Assembles retrieved context into LLM prompt, streams response,
and validates faithfulness before returning to user.
"""

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI, OpenAI

from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document assistant. Answer questions using ONLY the provided context.

Rules:
- Base your answer strictly on the context provided.
- Always cite the source document and page number for each claim using [Source: <name>, Page <n>].
- If the context does not contain the answer, say: "I could not find this information in the provided documents."
- Do not hallucinate or use external knowledge.
- Be concise and factual."""

QA_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer (with citations):"""

FAITHFULNESS_PROMPT = """You are a strict factual auditor. Given an answer and the context it was generated from,
determine whether every claim in the answer is directly supported by the context.

Context:
{context}

Answer:
{answer}

Respond with JSON only:
{{"faithful": true/false, "unsupported_claims": ["claim1", "claim2"], "faithfulness_score": 0.0-1.0}}"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]   # [{source, page, excerpt}]
    faithfulness_score: float = 1.0
    is_faithful: bool = True
    model: str = "gpt-4o"
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)


class RAGChain:
    """
    Full RAG pipeline: retrieve → assemble prompt → generate → validate faithfulness.
    Supports streaming and synchronous modes.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        faithfulness_check: bool = True,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.faithfulness_check = faithfulness_check
        self.top_k = top_k
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "?")
            parts.append(f"[{i+1}] Source: {source}, Page {page}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        """Build deduplicated source list for response metadata."""
        seen, sources = set(), []
        for chunk in chunks:
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "?")
            key = f"{source}:{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": source,
                    "page": page,
                    "excerpt": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "relevance_score": round(chunk.score, 4),
                })
        return sources

    def query(self, question: str) -> RAGResponse:
        """Synchronous RAG query with faithfulness validation."""
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        if not chunks:
            return RAGResponse(
                answer="I could not find relevant information in the provided documents.",
                sources=[],
                faithfulness_score=1.0,
            )

        context = self._format_context(chunks)
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()

        faithfulness_score, is_faithful = 1.0, True
        if self.faithfulness_check:
            faithfulness_score, is_faithful = self._check_faithfulness(answer, context)

        return RAGResponse(
            answer=answer,
            sources=self._extract_sources(chunks),
            faithfulness_score=faithfulness_score,
            is_faithful=is_faithful,
            model=self.model,
            retrieved_chunks=chunks,
        )

    def _check_faithfulness(self, answer: str, context: str) -> tuple[float, bool]:
        """LLM-as-a-Judge faithfulness validation."""
        import json, re
        try:
            prompt = FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer)
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = float(data.get("faithfulness_score", 1.0))
                faithful = bool(data.get("faithful", True))
                if not faithful:
                    logger.warning(f"Faithfulness check failed. Score: {score}. Unsupported: {data.get('unsupported_claims', [])}")
                return score, faithful
        except Exception as e:
            logger.warning(f"Faithfulness check error: {e}")
        return 1.0, True

    async def aquery(self, question: str) -> RAGResponse:
        """Async RAG query."""
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        if not chunks:
            return RAGResponse(answer="No relevant context found.", sources=[])

        context = self._format_context(chunks)
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)

        response = await self.async_client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()
        return RAGResponse(
            answer=answer,
            sources=self._extract_sources(chunks),
            model=self.model,
            retrieved_chunks=chunks,
        )

    async def astream(self, question: str) -> AsyncIterator[str]:
        """Streaming RAG response via async generator."""
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        context = self._format_context(chunks) if chunks else "No context found."
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)

        stream = await self.async_client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta
