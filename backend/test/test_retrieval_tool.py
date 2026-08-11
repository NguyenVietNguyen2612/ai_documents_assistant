import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever

from app.tools.retrieval_tool import (
    create_retrieval_tool,
)


def main():

    embedding_service = EmbeddingService()

    vector_store = VectorStore()

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retrieval_tool = create_retrieval_tool(
        retriever
    )

    print("=" * 60)
    print("TOOL INFORMATION")
    print("=" * 60)

    print(
        "Name:",
        retrieval_tool.name,
    )

    print(
        "Description:",
        retrieval_tool.description,
    )

    print("\n")

    # ==========================================
    # Invoke tool
    # ==========================================

    query = "What is RAG?"

    result = retrieval_tool.invoke(
        {
            "query": query
        }
    )

    print("=" * 60)
    print("TOOL RESULT")
    print("=" * 60)

    print(result)


if __name__ == "__main__":
    main()