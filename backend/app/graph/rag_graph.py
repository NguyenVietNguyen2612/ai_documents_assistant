from backend.test import test_agent_node
from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from app.graph.state import RAGState
from app.graph.nodes import (
    make_retrieve_node,
    make_context_node,
    make_generate_node,
    make_evaluate_node,
    make_agent_node,
)


def route_after_evaluate(state: RAGState):
    is_relevant = state.get("is_relevant", False)
    retry_count = state.get("retry_count", 0)
    max_retries = 2

    # Relevant → build context and generate
    if is_relevant:
        return "build_context"

    # Not relevant but still allowed to retry
    if retry_count < max_retries:
        return "retrieve"

    # Retry limit reached, proceed anyway
    return "build_context"


def route_after_agent(state):

    action = state.get(
        "action",
        "retrieve",
    )

    print(
        f"[Router] Agent action: {action}"
    )

    return action


def create_rag_graph(
    retriever,
    context_builder,
    prompt_builder,
    llm_service,
):

    builder = StateGraph(RAGState)

    builder.add_node(
        "agent",
        make_agent_node(llm_service),
    )

    builder.add_node(
        "retrieve",
        make_retrieve_node(retriever),
    )
    
    builder.add_node(
        "evaluate",
        make_evaluate_node(),
    )

    builder.add_node(
        "build_context",
        make_context_node(context_builder),
    )

    builder.add_node(
        "generate",
        make_generate_node(
            prompt_builder,
            llm_service,
        ),
    )

    builder.add_edge("agent", "retrieve")
    
    builder.add_edge("retrieve", "evaluate")

    builder.add_edge(START,"agent")

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "retrieve": "retrieve",
            "answer": "generate",
        },
    )
    
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
    )

    builder.add_edge("build_context", "generate")

    builder.add_edge("generate", END)

    return builder.compile()
