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
)

def create_rag_graph(
    retriever,
    context_builder,
    prompt_builder,
    llm_service,
):

    builder = StateGraph(RAGState)

    builder.add_node(
        "retrieve",
        make_retrieve_node(
            retriever
        ),
    )

    builder.add_node(
        "build_context",
        make_context_node(
            context_builder
        ),
    )

    builder.add_node(
        "generate",
        make_generate_node(
            prompt_builder,
            llm_service,
        ),
    )

    builder.add_edge(
        START,
        "retrieve",
    )

    builder.add_edge(
        "retrieve",
        "build_context",
    )

    builder.add_edge(
        "build_context",
        "generate",
    )

    builder.add_edge(
        "generate",
        END,
    )

    return builder.compile()

    result = graph.invoke(
    {
        "question": "What is RAG?"
    }
)