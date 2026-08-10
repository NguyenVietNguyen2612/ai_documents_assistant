from typing import TypedDict, Any


class RAGState(TypedDict, total=False):
    question: str
    retrieved_chunks: Any
    context: str
    answer: str

