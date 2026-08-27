import sys
import os

# Thêm thư mục 'backend' vào sys.path để Python có thể tìm thấy package 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever
from app.services.context_builder import ContextBuilder
from app.services.prompt_builder import PromptBuilder
from app.services.llm_service import LLMService

from app.graph.state import RAGState
from app.graph.nodes import (
    retrieve_node,
    build_context_node,
    generate_node,
)

def main():

    # ==================================================
    # 1. Initialize services
    # ==================================================

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
    # 2. Initial State
    # ==================================================

    state: RAGState = {
        "question": "What is RAG?"
    }

    print("\n")
    print("=" * 60)
    print("INITIAL STATE")
    print("=" * 60)

    print(state)

    # ==================================================
    # 3. Test retrieve_node
    # ==================================================

    print("\n")
    print("=" * 60)
    print("TEST: retrieve_node")
    print("=" * 60)

    retrieve_result = retrieve_node(
        state,
        retriever,
    )

    print("\nReturned data:")

    print(
        retrieve_result
    )

    # --------------------------------------------------
    # Update state
    # --------------------------------------------------

    state.update(
        retrieve_result
    )

    print("\nUpdated state:")

    print(state)

    # ==================================================
    # 4. Validate retrieve_node
    # ==================================================

    assert "retrieved_chunks" in state

    assert state["retrieved_chunks"] is not None

    print("\nretrieve_node: PASS")

    # ==================================================
    # 5. Test build_context_node
    # ==================================================

    print("\n")
    print("=" * 60)
    print("TEST: build_context_node")
    print("=" * 60)

    context_result = build_context_node(
        state,
        context_builder,
    )

    print("\nReturned data:")

    print(
        context_result
    )

    # --------------------------------------------------
    # Update state
    # --------------------------------------------------

    state.update(
        context_result
    )

    print("\nContext:")

    print(
        state["context"]
    )

    # ==================================================
    # 6. Validate build_context_node
    # ==================================================

    assert "context" in state

    assert state["context"]

    print("\nbuild_context_node: PASS")

    # ==================================================
    # 7. Test generate_node
    # ==================================================

    print("\n")
    print("=" * 60)
    print("TEST: generate_node")
    print("=" * 60)

    generate_result = generate_node(
        state,
        prompt_builder,
        llm_service,
    )

    print("\nReturned data:")

    print(
        generate_result
    )

    # --------------------------------------------------
    # Update state
    # --------------------------------------------------

    state.update(
        generate_result
    )

    print("\nAnswer:")

    print(
        state["answer"]
    )

    # ==================================================
    # 8. Validate generate_node
    # ==================================================

    assert "answer" in state

    assert state["answer"]

    print("\ngenerate_node: PASS")

    # ==================================================
    # 9. Final State
    # ==================================================

    print("\n")
    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)

    print(
        state
    )

    print("\n")
    print("=" * 60)
    print("ALL NODE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()