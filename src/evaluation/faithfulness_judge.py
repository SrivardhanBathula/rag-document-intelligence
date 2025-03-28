from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """You are an expert evaluator assessing RAG system quality.
Given a question, retrieved context, and generated answer, evaluate:
1. Faithfulness (0-1): Is the answer grounded in the provided context?
2. Relevance (0-1): Does the answer address the question?
3. Completeness (0-1): Does the answer cover all relevant context?

Return ONLY valid JSON: {"faithfulness": 0.0, "relevance": 0.0, "completeness": 0.0, "reasoning": ""}"""


class FaithfulnessJudge:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def evaluate(self, question: str, context_docs: List[Document],
                answer: str) -> Dict[str, Any]:
        context = "\n\n".join([d.page_content for d in context_docs[:5]])
        messages = [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=f"Question: {question}\n\nContext:\n{context}\n\nAnswer: {answer}")
        ]
        response = self.llm.invoke(messages)
        try:
            result = json.loads(response.content)
            result["passed"] = result.get("faithfulness", 0) >= 0.7
            return result
        except json.JSONDecodeError:
            return {"faithfulness": 0.0, "relevance": 0.0, "completeness": 0.0,
                   "reasoning": "Parse error", "passed": False}

    def batch_evaluate(self, samples: List[Dict]) -> List[Dict]:
        return [self.evaluate(s["question"], s["contexts"], s["answer"]) for s in samples]
