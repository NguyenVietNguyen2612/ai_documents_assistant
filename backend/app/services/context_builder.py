class ContextBuilder:

    def build(self, results) -> str:

        contexts = []

        for i, result in enumerate(
            results[0],
            start=1,
        ):

            entity = result["entity"]

            context = f"""
[Source {i}]
Document: {entity["document_name"]}
Page: {entity["page"]}

{entity["text"]}
"""

            contexts.append(context)

        return "\n".join(contexts)