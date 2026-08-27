import sys
import os

# Thêm thư mục 'backend' vào sys.path để Python có thể tìm thấy package 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.state import RAGState


def main():

    state: RAGState = {
        "question": "What is RAG?",
        "retry_count": 0,
    }

    print(state)


if __name__ == "__main__":
    main()