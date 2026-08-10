import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever
from app.services.context_builder import ContextBuilder
from app.services.prompt_builder import PromptBuilder
from app.services.llm_service import LLMService

from app.graph.rag_graph import create_rag_graph


def main():

    # ==================================================
    # 1. Initialize services
    # ==================================================

    print("=" * 60)
    print("INITIALIZING SERVICES")
    print("=" * 60)

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

    # ==================================================
    # 2. Create LangGraph
    # ==================================================

    print("\n")
    print("=" * 60)
    print("CREATING LANGGRAPH")
    print("=" * 60)

    graph = create_rag_graph(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
    )

    print(
        graph.get_graph().draw_ascii()
    )   

    print("Graph compiled successfully.")

    # ==================================================
    # 3. User question
    # ==================================================

    question = "What is RAG?"

    print("\n")
    print("=" * 60)
    print("STARTING GRAPH")
    print("=" * 60)

    print(f"\nQuestion: {question}")

    # ==================================================
    # 4. Invoke graph
    # ==================================================

    result = graph.invoke(
        {
            "question": question
        }
    )

    # ==================================================
    # 5. Print final state
    # ==================================================

    print("\n")
    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)

    print("\nQuestion:")
    print(result["question"])

    print("\nContext:")
    print(result["context"])

    print("\nAnswer:")
    print(result["answer"])

    # ==================================================
    # 6. Print retrieved sources
    # ==================================================

    print("\n")
    print("=" * 60)
    print("SOURCES")
    print("=" * 60)

    results = result["retrieved_chunks"]

    for i, item in enumerate(
        results[0],
        start=1,
    ):

        entity = item["entity"]

        print(f"\nSource {i}")

        print(
            f"Document: "
            f"{entity['document_name']}"
        )

        print(
            f"Page: "
            f"{entity['page']}"
        )

        print(
            f"Score: "
            f"{item['distance']}"
        )

        print(
            f"Text: "
            f"{entity['text']}"
        )


if __name__ == "__main__":
    main()