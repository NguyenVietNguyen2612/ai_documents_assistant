import sys
import os



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.nodes import make_evaluate_node


def test_relevant_result():

    state = {
        "retrieved_chunks": [[
            {
                "distance": 0.90,
                "entity": {
                    "text": "RAG combines retrieval and generation."
                }
            }
        ]]
    }

    result = make_evaluate_node()(state)

    print("\n=== Relevant Test ===")
    print(result)

    assert result["retrieval_score"] == 0.90
    assert result["is_relevant"] is True


def test_irrelevant_result():

    state = {
        "retrieved_chunks": [[
            {
                "distance": 0.30,
                "entity": {
                    "text": "Something unrelated."
                }
            }
        ]]
    }

    result = make_evaluate_node()(state)

    print("\n=== Irrelevant Test ===")
    print(result)

    assert result["retrieval_score"] == 0.30
    assert result["is_relevant"] is False


if __name__ == "__main__":

    test_relevant_result()

    test_irrelevant_result()

    print("\nAll tests passed.")