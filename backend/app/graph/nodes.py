from app.graph.state import RAGState


def make_retrieve_node(retriever):

    def retrieve_node(
        state: RAGState,
    ) -> dict:

        question = state["question"]

        results = retriever.retrieve(
            query=question,
            top_k=5,
        )

        return {
            "retrieved_chunks": results
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