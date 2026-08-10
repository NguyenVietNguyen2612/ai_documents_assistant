import sys
import os

# Thêm thư mục 'backend' vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever
from app.services.context_builder import ContextBuilder
from app.services.prompt_builder import PromptBuilder
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


def main():

    # -------------------------
    # Services
    # -------------------------

    embedding_service = EmbeddingService()

    vector_store = VectorStore()

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    context_builder = ContextBuilder()

    prompt_builder = PromptBuilder()

    llm_service = LLMService(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    rag = RAGService(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
    )

    # -------------------------
    # Question
    # -------------------------

    question = "How does retrieving external information help an LLM?"

    result = rag.answer(
        question=question,
        top_k=3,
    )

    # -------------------------
    # Output
    # -------------------------

    # 1. Print question
    print("\n===== QUESTION =====")
    print(question)

    # 2. Print retrieved chunks
    print("\n===== RETRIEVED CHUNKS =====")

    for item in result["results"][0]:

        entity = item["entity"]

        print(f"\nDocument: {entity['document_name']}")
        print(f"Page: {entity['page']}")
        print(f"Score: {item['distance']}")
        print(f"Text: {entity['text']}")


    # 5. Print answer
    print("\n===== ANSWER =====")
    print(result["answer"])

    print("\n===== ANSWER =====")

    print(
        result["answer"]
    )

    print("\n===== SOURCES =====")

    for item in result["results"][0]:

        entity = item["entity"]

        print(
            f"- {entity['document_name']} "
            f"(page {entity['page']})"
        )


if __name__ == "__main__":
    main()