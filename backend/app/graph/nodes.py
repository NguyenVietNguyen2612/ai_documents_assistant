from app.graph.state import RAGState
from app.graph.agent_decision import AgentDecision

def make_retrieve_node(retriever):

    def retrieve_node(
        state: RAGState,
    ) -> dict:

        question = state["question"]

        retry_count = state.get(
            "retry_count",
            0,
        )

        print("\n[Node: retrieve]")
        print(f"Question: {question}")
        print(f"Attempt: {retry_count + 1}")

        results = retriever.retrieve(
            query=question,
            top_k=3,
        )

        return {
            "retrieved_chunks": results,
            "retry_count": retry_count + 1,
        }   

    return retrieve_node


def make_context_node(context_builder):

    def build_context_node(
        state: RAGState,
    ) -> dict:

        results = state["retrieved_chunks"]

        context = context_builder.build(
            results
        )

        return {
            "context": context
        }

    return build_context_node


def make_generate_node(prompt_builder, llm_service):

    def generate_node(
        state: RAGState,
    ) -> dict:

        question = state["question"]

        context = state["context"]

        prompt = prompt_builder.build(
            question=question,
            context=context,
        )

        answer = llm_service.generate(
            prompt
        )

        return {
            "answer": answer
        }
    return generate_node


def make_evaluate_node():

    def evaluate_node(state: RAGState):

        print("\n[Node: evaluate]")

        results = state["retrieved_chunks"]

        if not results or not results[0]:
            return {
                "retrieval_score": 0.0,
                "is_relevant": False,
            }

        top_result = results[0][0]

        score = float(
            top_result["distance"]
        )

        threshold = 0.70

        is_relevant = (
            score >= threshold
        )

        print(
            f"Top retrieval score: {score}"
        )

        print(
            f"Relevant: {is_relevant}"
        )

        return {
            "retrieval_score": score,
            "is_relevant": is_relevant,
        }
    
    return evaluate_node


def make_agent_node(llm_service):

    def agent_node(state):

        question = state["question"]

        prompt = f"""
You are the decision-making agent of a RAG system.

Your job is to decide whether the user's question
requires retrieving information from the uploaded documents.

Choose:

- retrieve: when information from the documents is needed.
- answer: when the question can be answered without retrieving documents.

Do not answer the user's question.
Only decide the next action.

Question:
{question}
"""

        decision: AgentDecision = (
            llm_service.generate_structured(
                prompt,
                response_model=AgentDecision,
            )
        )

        print("\n[Node: agent]")

        print(
            f"Action: {decision.action}"
        )

        print(
            f"Reason: {decision.reason}"
        )

        return {
            "action": decision.action
        }

    return agent_node