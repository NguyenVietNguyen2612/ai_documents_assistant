import sys
import os

# Thêm thư mục 'backend' vào sys.path để Python có thể tìm thấy package 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever

from app.graph.nodes import make_retrieve_node


def main():

    embedding_service = EmbeddingService()

    vector_store = VectorStore()

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retrieve_node = make_retrieve_node(
        retriever
    )

    state = {
        "question": "What is RAG?",
        "retry_count": 0,
    }

    result = retrieve_node(state)

    print("\nResult:")
    print(result)


if __name__ == "__main__":
    main()