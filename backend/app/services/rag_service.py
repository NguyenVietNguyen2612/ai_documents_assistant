class RAGService:

    def __init__(
        self,
        retriever,
        context_builder,
        prompt_builder,
        llm_service,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_service = llm_service

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ):

        # 1. Retrieve
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        # 2. Build context
        context = self.context_builder.build(
            results
        )

        # 3. Build prompt
        prompt = self.prompt_builder.build(
            question=question,
            context=context,
        )

        # 4. Generate answer
        answer = self.llm_service.generate(
            prompt
        )

        return {
            "answer": answer,
            "results": results,
        }
