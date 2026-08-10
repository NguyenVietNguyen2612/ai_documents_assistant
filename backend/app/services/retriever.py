class Retriever:

    def __init__(
        self,
        embedding_service,
        vector_store,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ):
        query_vector = self.embedding_service.embed_chunks(query)

        results = self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
        )

        return results