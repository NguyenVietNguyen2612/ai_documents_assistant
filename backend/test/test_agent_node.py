import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.llm_service import LLMService
from app.graph.nodes import make_agent_node


def main():

    llm_service = LLMService()

    agent_node = make_agent_node(
        llm_service
    )

    # ==========================================
    # Test 1
    # ==========================================

    state = {
        "question": "What is RAG?"
    }

    result = agent_node(state)

    print("\n=== Test 1 ===")
    print(result)

    assert result["action"] in {
        "retrieve",
        "answer",
    }

    # ==========================================
    # Test 2
    # ==========================================

    state = {
        "question": "Hello"
    }

    result = agent_node(state)

    print("\n=== Test 2 ===")
    print(result)

    assert result["action"] in {
        "retrieve",
        "answer",
    }


if __name__ == "__main__":
    main()