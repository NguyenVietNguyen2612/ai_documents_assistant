import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.rag_graph import route_after_evaluate


def test_relevant():

    state = {
        "is_relevant": True,
        "retry_count": 1,
    }

    result = route_after_evaluate(state)

    print(
        "Relevant:",
        result
    )

    assert result == "build_context"


def test_not_relevant_retry():

    state = {
        "is_relevant": False,
        "retry_count": 1,
    }

    result = route_after_evaluate(state)

    print(
        "Not relevant:",
        result
    )

    assert result == "retrieve"


def test_max_retry():

    state = {
        "is_relevant": False,
        "retry_count": 2,
    }

    result = route_after_evaluate(state)

    print(
        "Max retry:",
        result
    )

    assert result == "build_context"


if __name__ == "__main__":

    test_relevant()

    test_not_relevant_retry()

    test_max_retry()

    print("\nAll router tests passed.")