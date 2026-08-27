import sys
import os

# Thêm thư mục 'backend' vào sys.path để Python có thể tìm thấy package 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever


def main():

    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    query = "What is RAG?"

    results = retriever.retrieve(
        query=query,
        top_k=3,
    )

    for result in results[0]:

        print(
            result["entity"]["document_name"]
        )

        print(
            result["entity"]["page"]
        )

        print(
            result["entity"]["text"]
        )

        print("-" * 50)


if __name__ == "__main__":
    main()