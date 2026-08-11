import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.agent_decision import AgentDecision


def test_retrieve_decision():

    decision = AgentDecision(
        action="retrieve",
        reason="The question requires information from the documents."
    )

    assert decision.action == "retrieve"

    print(decision)


def test_answer_decision():

    decision = AgentDecision(
        action="answer",
        reason="The question can be answered without document retrieval."
    )

    assert decision.action == "answer"

    print(decision)

def test_something_decision():

    decision = AgentDecision(
        action="something",
        reason="test"
    )

    assert decision.action == "something"

    print(decision)

if __name__ == "__main__":

    test_retrieve_decision()

    test_answer_decision()

    test_something_decision()

    print("\nAll AgentDecision tests passed.")