import os
import sys
from dotenv import load_dotenv
from langsmith import evaluate
from langsmith.schemas import Run, Example
from pydantic import BaseModel, Field

# Đảm bảo Python có thể import được thư mục 'app' ở thư mục cha
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.api.routes.chat import agent_service
from app.services.llm_service import LLMService

load_dotenv()

# Khởi tạo LLM Service để làm Giám khảo (Judge)
llm_judge = LLMService()

class EvaluationScore(BaseModel):
    score: int = Field(description="The binary score: 1 if the condition is met, 0 otherwise.")
    reasoning: str = Field(description="Short reasoning for the given score.")

def predict(inputs: dict) -> dict:
    """
    Target function (Hàm mục tiêu) để LangSmith gọi cho mỗi câu hỏi trong dataset.
    """
    question = inputs["question"]
    print(f"Đang đánh giá câu hỏi: {question}")
    
    try:
        # Gọi RAG pipeline thông qua agent_service
        res = agent_service.ask(question)
        answer = res.get("answer", "")
        context = res.get("context", "")
    except Exception as e:
        answer = f"Error: {e}"
        context = ""
        
    return {"answer": answer, "context": context}

# --- EVALUATORS ---

def correctness_evaluator(run: Run, example: Example) -> dict:
    question = example.inputs["question"]
    expected_answer = example.outputs["answer"]
    actual_answer = run.outputs["answer"]
    
    prompt = f"""You are an expert evaluator. 
Compare the ACTUAL ANSWER with the EXPECTED ANSWER for the given QUESTION.
Score 1 if the ACTUAL ANSWER is factually correct based on the EXPECTED ANSWER, otherwise score 0.

Question: {question}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}"""

    try:
        result = llm_judge.generate_structured(prompt, EvaluationScore)
        return {"key": "correctness", "score": result.score, "comment": result.reasoning}
    except Exception as e:
        return {"key": "correctness", "score": 0, "comment": f"Eval Failed: {e}"}

def relevance_evaluator(run: Run, example: Example) -> dict:
    question = example.inputs["question"]
    actual_answer = run.outputs["answer"]
    
    prompt = f"""You are an expert evaluator. 
Does the ACTUAL ANSWER directly address the QUESTION?
Score 1 if the answer is relevant and addresses the question without extreme hallucination, otherwise score 0.

Question: {question}
Actual Answer: {actual_answer}"""

    try:
        result = llm_judge.generate_structured(prompt, EvaluationScore)
        return {"key": "relevance", "score": result.score, "comment": result.reasoning}
    except Exception as e:
        return {"key": "relevance", "score": 0, "comment": f"Eval Failed: {e}"}

def groundedness_evaluator(run: Run, example: Example) -> dict:
    context = run.outputs.get("context", "")
    actual_answer = run.outputs.get("answer", "")
    
    prompt = f"""You are an expert evaluator. 
Is the ACTUAL ANSWER fully grounded and supported by the CONTEXT?
Score 1 if the answer is completely supported by the context, score 0 if it contains hallucinated information not present in the context.

Context:
{context}

Actual Answer:
{actual_answer}"""

    try:
        result = llm_judge.generate_structured(prompt, EvaluationScore)
        return {"key": "groundedness", "score": result.score, "comment": result.reasoning}
    except Exception as e:
        return {"key": "groundedness", "score": 0, "comment": f"Eval Failed: {e}"}

# --- MAIN ---

def main():
    dataset_name = "CS338-RAG-Evaluation"

    print(f"Bắt đầu chạy đánh giá trên dataset: {dataset_name}...")
    
    custom_evaluators = [
        correctness_evaluator,
        relevance_evaluator,
        groundedness_evaluator
    ]

    # Chạy quy trình đánh giá
    experiment_results = evaluate(
        predict,
        data=dataset_name,
        evaluators=custom_evaluators,
        experiment_prefix="RAG-Evaluation",
        description="Đánh giá hiệu suất của RAG Pipeline bằng LangSmith (Correctness, Relevance, Groundedness).",
        max_concurrency=1
    )
    
    print("Đánh giá hoàn tất! Hãy kiểm tra kết quả trên giao diện LangSmith.")

if __name__ == "__main__":
    main()
