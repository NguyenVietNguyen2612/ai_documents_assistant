class PromptBuilder:

    def build(
        self,
        question: str,
        context: str,
    ) -> str:

        return f"""
You are an AI assistant answering questions
based on the provided documents.

Use the provided context to answer the question.

If the answer cannot be found in the context,
say that the provided documents do not contain
enough information to answer the question.

Do not invent information.

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""