import os
import sys

# Thêm thư mục 'backend' vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.prompt_builder import PromptBuilder


def main():

    context = """
[Source 1]
Document: rag.pdf
Page: 2

RAG combines retrieval with generation.

[Source 2]
Document: rag.pdf
Page: 3

The retrieval component searches external documents.
"""

    question = "What is RAG?"

    prompt_builder = PromptBuilder()

    prompt = prompt_builder.build(
        question=question,
        context=context,
    )

    print("===== GENERATED PROMPT =====")
    print(prompt)


if __name__ == "__main__":
    main()