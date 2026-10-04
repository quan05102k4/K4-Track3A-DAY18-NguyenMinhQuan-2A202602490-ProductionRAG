from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Cây chẩn đoán: chỉ số thấp nhất → (nguyên nhân, hướng xử lý)
DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM tự bịa câu trả lời ngoài tài liệu",
                     "Thắt chặt system prompt, giảm temperature về 0"),
    "context_recall": ("Hệ thống tìm kiếm bỏ sót đoạn văn đúng",
                       "Cải thiện bước cắt đoạn hoặc bổ sung từ khóa BM25"),
    "context_precision": ("Đoạn văn không liên quan bị xếp lên đầu",
                          "Bổ sung tầng Cross-Encoder reranking hoặc lọc theo metadata"),
    "answer_relevancy": ("Câu trả lời bị lệch trọng tâm câu hỏi",
                         "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn"),
}


def _zeros(n: int = 0) -> dict:
    return {**{m: 0.0 for m in METRICS}, "per_question": []}


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation (LLM + embedding qua Gemini endpoint tương thích OpenAI)."""
    try:
        import math
        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

        llm = ChatOpenAI(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL, temperature=0)
        embeddings = OpenAIEmbeddings(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall],
                          llm=llm, embeddings=embeddings,
                          run_config=RunConfig(max_workers=8, max_retries=6, timeout=120))
        df = result.to_pandas()

        def _num(v) -> float:
            v = float(v)
            return 0.0 if math.isnan(v) else v

        per_question = [EvalResult(question=row["question"], answer=row["answer"],
                                   contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                                   faithfulness=_num(row["faithfulness"]),
                                   answer_relevancy=_num(row["answer_relevancy"]),
                                   context_precision=_num(row["context_precision"]),
                                   context_recall=_num(row["context_recall"]))
                        for _, row in df.iterrows()]
        aggregate = {m: (sum(getattr(r, m) for r in per_question) / len(per_question)
                         if per_question else 0.0) for m in METRICS}
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return _zeros()


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    scored = []
    for r in eval_results:
        scores = {m: getattr(r, m) for m in METRICS}
        avg = sum(scores.values()) / len(METRICS)
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[worst_metric]
        scored.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "worst_metric": worst_metric,
            "score": avg,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    scored.sort(key=lambda x: x["score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
