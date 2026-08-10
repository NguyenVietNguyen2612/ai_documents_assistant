from typing import TypedDict, Any


class RAGState(TypedDict, total=False):
    question: str
    retrieved_chunks: Any
    context: str
    retrieval_score: float
    is_relevant: bool
    retry_count: int
    answer: str
