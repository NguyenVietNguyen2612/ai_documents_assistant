class AgentService:

    def __init__(self, graph):
        self.graph = graph

    def ask(self, question: str) -> str:

        initial_state = {
            "question": question,
            "retry_count": 0,
        }

        result = self.graph.invoke(
            initial_state
        )

        return result["answer"]