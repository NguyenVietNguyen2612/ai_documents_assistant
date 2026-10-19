import argparse
import asyncio
import json
import os
import sys
import uuid
from typing import Dict, Any, List
from dotenv import load_dotenv
from langsmith import evaluate
from langsmith.schemas import Run, Example
from pydantic import BaseModel, Field

# Đảm bảo Python có thể import được thư mục 'app' ở thư mục cha
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.api.routes.chat import agent_service
from app.services.llm_service import LLMService

load_dotenv()

# Khởi tạo LLM Service để làm Giám khảo (LLM-as-a-Judge)
llm_judge = LLMService()


class EvaluationScore(BaseModel):
    score: int = Field(
        description="Binary score: 1 if the evaluation criterion is satisfied, 0 otherwise."
    )
    reasoning: str = Field(
        description="Concise explanation justifying why the score (1 or 0) was assigned."
    )


def _run_async(coro):
    """Chạy coroutine an toàn từ hàm đồng bộ (hỗ trợ cả khi đã có hoặc chưa có event loop)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def predict(inputs: dict) -> dict:
    """
    Target function (Hàm mục tiêu) để LangSmith hoặc Local Runner gọi cho mỗi câu hỏi trong dataset.
    Đã sửa lỗi gọi đồng bộ vào hàm async `agent_service.ask()` và trích xuất đầy đủ `context` thực tế từ ToolMessage.
    """
    question = inputs["question"]
    eval_session_id = f"eval_{uuid.uuid4().hex[:8]}"
    print(f"[Eval Predict] Đang đánh giá câu hỏi ({eval_session_id}): {question}")

    try:
        res = _run_async(agent_service.ask(question, session_id=eval_session_id))
        answer = res.get("answer", "")
        context = res.get("context", "")
        latency_ms = res.get("latency_ms", 0)
        tool_calls_count = res.get("tool_calls_count", 0)
    except Exception as e:
        answer = f"Error: {e}"
        context = ""
        latency_ms = 0
        tool_calls_count = 0

    return {
        "answer": answer,
        "context": context,
        "latency_ms": latency_ms,
        "tool_calls_count": tool_calls_count,
    }


# =====================================================================
# RAG TRIAD + CORRECTNESS EVALUATORS (LLM-as-a-Judge)
# =====================================================================

def context_relevance_evaluator(run: Run, example: Example) -> dict:
    """
    [RAG Triad #1 - Context Relevance]
    Đánh giá xem ngữ cảnh (Context) truy xuất được từ Hybrid Search + Reranker
    có chứa thông tin liên quan để trả lời câu hỏi hay không.
    """
    question = example.inputs.get("question", "")
    expected_answer = example.outputs.get("answer", "") if example.outputs else ""
    context = (run.outputs or {}).get("context", "")

    if not context or not context.strip():
        # Nếu câu hỏi thuộc loại out_of_context nhưng hệ thống vẫn phải truy xuất để kiểm chứng
        return {
            "key": "context_relevance",
            "score": 0,
            "comment": "No retrieved context was recorded from retrieval tools.",
        }

    prompt = f"""You are an impartial RAG Evaluator assessing CONTEXT RELEVANCE.
Determine whether the RETRIEVED CONTEXT contains relevant domain passages or evidence needed to address the QUESTION (or to verify whether the requested detail exists in the document).
Score 1 if the retrieved context is relevant to the topic of the question, otherwise score 0.

Question: {question}
Reference Expected Answer: {expected_answer}
Retrieved Context:
{context[:4000]}"""

    try:
        result = llm_judge.generate_structured(prompt, EvaluationScore)
        return {
            "key": "context_relevance",
            "score": result.score,
            "comment": result.reasoning,
        }
    except Exception as e:
        return {
            "key": "context_relevance",
            "score": 0,
            "comment": f"Eval Failed: {e}",
        }


def groundedness_evaluator(run: Run, example: Example) -> dict:
    """
    [RAG Triad #2 - Groundedness / Faithfulness]
    Đánh giá xem câu trả lời của AI có trung thực tuyệt đối với Context truy xuất được hay bị bịa đặt (Hallucination).
    """
    context = (run.outputs or {}).get("context", "")
    actual_answer = (run.outputs or {}).get("answer", "")

    if not actual_answer or actual_answer.startswith("Error:"):
        return {
            "key": "groundedness",
            "score": 0,
            "comment": "Empty or error answer.",
        }

    prompt = f"""You are an impartial RAG Evaluator assessing GROUNDEDNESS (Faithfulness).
Is the ACTUAL ANSWER faithfully grounded in and supported by the RETRIEVED CONTEXT?
Note: If the ACTUAL ANSWER states that the provided documents do not contain enough information to answer the question, and the RETRIEVED CONTEXT indeed lacks that information, score 1 (no hallucination).
Score 1 if the answer contains no fabricated facts outside the context, score 0 if it hallucinates unsupported claims.

Retrieved Context:
{context[:4000]}

Actual Answer:
{actual_answer}"""

    try:
        result = llm_judge.generate_structured(prompt, EvaluationScore)
        return {
            "key": "groundedness",
            "score": result.score,
            "comment": result.reasoning,
        }
    except Exception as e:
        return {
            "key": "groundedness",
            "score": 0,
            "comment": f"Eval Failed: {e}",
        }


def answer_relevance_evaluator(run: Run, example: Example) -> dict:
    """
    [RAG Triad #3 - Answer Relevance]
    Đánh giá xem câu trả lời có đi thẳng vào trọng tâm câu hỏi của người dùng hay không.
    """
    question = example.inputs.get("question", "")
    actual_answer = (run.outputs or {}).get("answer", "")

    if not actual_answer or actual_answer.startswith("Error:"):
        return {
            "key": "answer_relevance",
            "score": 0,
            "comment": "Empty or error answer.",
        }

    prompt = f"""You are an impartial RAG Evaluator assessing ANSWER RELEVANCE.
Does the ACTUAL ANSWER directly and appropriately address the user's QUESTION (including appropriately abstaining when the requested fact is not in the document)?
Score 1 if the answer is relevant and directly addresses the question, otherwise score 0.

Question: {question}
Actual Answer: {actual_answer}"""

    try:
        result = llm_judge.generate_structured(prompt, EvaluationScore)
        return {
            "key": "answer_relevance",
            "score": result.score,
            "comment": result.reasoning,
        }
    except Exception as e:
        return {
            "key": "answer_relevance",
            "score": 0,
            "comment": f"Eval Failed: {e}",
        }


def correctness_evaluator(run: Run, example: Example) -> dict:
    """
    [End-to-End Fact Correctness]
    Đối chiếu câu trả lời thực tế với đáp án chuẩn (Ground Truth Expected Answer).
    """
    question = example.inputs.get("question", "")
    expected_answer = example.outputs.get("answer", "") if example.outputs else ""
    actual_answer = (run.outputs or {}).get("answer", "")

    if not actual_answer or actual_answer.startswith("Error:"):
        return {
            "key": "correctness",
            "score": 0,
            "comment": "Empty or error answer.",
        }

    prompt = f"""You are an expert evaluator assessing FACTUAL CORRECTNESS.
Compare the ACTUAL ANSWER with the EXPECTED ANSWER for the given QUESTION.
Score 1 if the ACTUAL ANSWER is factually consistent with the EXPECTED ANSWER (even if phrased in Vietnamese or with additional cited details), otherwise score 0.

Question: {question}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}"""

    try:
        result = llm_judge.generate_structured(prompt, EvaluationScore)
        return {
            "key": "correctness",
            "score": result.score,
            "comment": result.reasoning,
        }
    except Exception as e:
        return {
            "key": "correctness",
            "score": 0,
            "comment": f"Eval Failed: {e}",
        }


# Giữ alias tương thích ngược
relevance_evaluator = answer_relevance_evaluator


# =====================================================================
# LOCAL EVALUATION RUNNER (Chạy trực tiếp trên dataset.json không cần LangSmith Cloud)
# =====================================================================

class _LocalDummyObj:
    def __init__(self, inputs: dict = None, outputs: dict = None):
        self.inputs = inputs or {}
        self.outputs = outputs or {}


def run_local_evaluation(limit: int = None) -> List[Dict[str, Any]]:
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if limit and limit > 0:
        dataset = dataset[:limit]

    print(f"=== BẮT ĐẦU ĐÁNH GIÁ RAG TRIAD LOCAL ({len(dataset)} câu hỏi) ===")
    results = []

    for item in dataset:
        q_id = item.get("id", "unknown")
        question = item["question"]
        expected = item["expected_answer"]

        pred = predict({"question": question})
        run_obj = _LocalDummyObj(inputs={"question": question}, outputs=pred)
        ex_obj = _LocalDummyObj(
            inputs={"question": question},
            outputs={"answer": expected},
        )

        c_rel = context_relevance_evaluator(run_obj, ex_obj)
        ground = groundedness_evaluator(run_obj, ex_obj)
        a_rel = answer_relevance_evaluator(run_obj, ex_obj)
        corr = correctness_evaluator(run_obj, ex_obj)

        row = {
            "id": q_id,
            "question": question,
            "latency_ms": pred.get("latency_ms", 0),
            "tool_calls": pred.get("tool_calls_count", 0),
            "context_relevance": c_rel["score"],
            "groundedness": ground["score"],
            "answer_relevance": a_rel["score"],
            "correctness": corr["score"],
        }
        results.append(row)
        print(
            f"  -> [{q_id}] ContextRel={row['context_relevance']} | "
            f"Groundedness={row['groundedness']} | AnswerRel={row['answer_relevance']} | "
            f"Correctness={row['correctness']} | Latency={row['latency_ms']}ms"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá hiệu suất RAG Pipeline với bộ chỉ số RAG Triad + Correctness."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Chạy đánh giá trực tiếp trên file evaluation/dataset.json thay vì gọi LangSmith Cloud.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số câu hỏi khi chạy --local để kiểm thử nhanh.",
    )
    args = parser.parse_args()

    if args.local or not os.getenv("LANGCHAIN_API_KEY"):
        if not args.local:
            print("[Info] Không tìm thấy LANGCHAIN_API_KEY, tự động chuyển sang chế độ đánh giá Local (--local).")
        run_local_evaluation(limit=args.limit)
        return

    dataset_name = "CS338-RAG-Evaluation"
    print(f"Bắt đầu chạy đánh giá LangSmith trên dataset: {dataset_name}...")

    custom_evaluators = [
        context_relevance_evaluator,
        groundedness_evaluator,
        answer_relevance_evaluator,
        correctness_evaluator,
    ]

    evaluate(
        predict,
        data=dataset_name,
        evaluators=custom_evaluators,
        experiment_prefix="RAG-Triad-Evaluation",
        description="Đánh giá hiệu suất RAG Pipeline bằng bộ chỉ số RAG Triad (Context Relevance, Groundedness, Answer Relevance) & Correctness.",
        max_concurrency=1,
    )

    print("Đánh giá hoàn tất! Hãy kiểm tra kết quả trên giao diện LangSmith.")


if __name__ == "__main__":
    main()
