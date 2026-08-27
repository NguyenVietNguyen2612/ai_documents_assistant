import sys
import os

# Thêm thư mục 'backend' vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.context_builder import ContextBuilder


def main():

    # Fake data mô phỏng kết quả từ Milvus
    fake_results = [
        [
            {
                "id": "test_001",
                "distance": 0.91,
                "entity": {
                    "document_name": "rag.pdf",
                    "page": 2,
                    "text": (
                        "RAG combines retrieval "
                        "with generation."
                    ),
                },
            },
            {
                "id": "test_002",
                "distance": 0.87,
                "entity": {
                    "document_name": "rag.pdf",
                    "page": 3,
                    "text": (
                        "The retrieval component "
                        "searches external documents."
                    ),
                },
            },
        ]
    ]

    context_builder = ContextBuilder()

    context = context_builder.build(
        fake_results
    )

    print("===== GENERATED CONTEXT =====")
    print(context)


if __name__ == "__main__":
    main()