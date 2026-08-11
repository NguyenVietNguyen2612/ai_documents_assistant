import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.rag_graph import route_after_agent


def test_retrieve():

    state = {
        "action": "retrieve"
    }

    result = route_after_agent(state)

    assert result == "retrieve"


def test_answer():

    state = {
        "action": "answer"
    }

    result = route_after_agent(state)

    assert result == "answer"


if __name__ == "__main__":

    test_retrieve()

    test_answer()

    print(
        "\nAll agent router tests passed."
    )