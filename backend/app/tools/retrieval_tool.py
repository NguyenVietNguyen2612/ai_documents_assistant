from typing import Any

from langchain_core.tools import tool


def create_retrieval_tool(retriever):

    @tool
    def retrieve_documents(
        query: str,
    ) -> Any:
        """
        Retrieve relevant document chunks
        from the vector database based on the query.
        """

        results = retriever.retrieve(
            query=query,
            top_k=3,
        )

        return results

    return retrieve_documents