import sys
import os

# Thêm thư mục 'backend' vào sys.path để Python có thể tìm thấy package 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.llm_service import LLMService


def main():
    llm = LLMService()

    answer = llm.generate(
        "Explain what RAG is in one sentence."
    )

    print(answer)


if __name__ == "__main__":
    main()