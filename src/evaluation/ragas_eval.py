from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from typing import List, Dict
import pandas as pd
import mlflow
import logging

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    def __init__(self, metrics=None):
        self.metrics = metrics or [faithfulness, answer_relevancy,
                                   context_precision, context_recall]

    def evaluate(self, samples: List[Dict]) -> pd.DataFrame:
        dataset = Dataset.from_list([{
            "question": s["question"],
            "answer": s["answer"],
            "contexts": s["contexts"],
            "ground_truth": s.get("ground_truth", "")
        } for s in samples])

        with mlflow.start_run(run_name="ragas_evaluation"):
            result = evaluate(dataset, metrics=self.metrics)
            df = result.to_pandas()
            mean_scores = df.mean(numeric_only=True).to_dict()
            mlflow.log_metrics(mean_scores)
            logger.info(f"RAGAS evaluation complete: {mean_scores}")
            return df

    def run_benchmark(self, pipeline, test_questions: List[str]) -> Dict:
        samples = []
        for q in test_questions:
            docs = pipeline.retrieve(q)
            answer = pipeline.generate(q, docs)
            samples.append({"question": q, "answer": answer,
                           "contexts": [d.page_content for d in docs]})
        df = self.evaluate(samples)
        return {"faithfulness": df["faithfulness"].mean(),
                "answer_relevancy": df["answer_relevancy"].mean(),
                "overall": df.mean(numeric_only=True).mean()}
