class AgentService:

    def __init__(self, graph):
        self.graph = graph

    def ask(self, question: str) -> dict:

        initial_state = {
            "question": question,
            "retry_count": 0,
        }

        result = self.graph.invoke(
            initial_state
        )

        return {
            "answer": result.get("answer", ""),
            "context": result.get("context", "")
        }