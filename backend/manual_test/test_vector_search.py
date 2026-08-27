from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


def main():
    # --------------------------------
    # 1. Initialize services
    # --------------------------------
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    # --------------------------------
    # 2. Test documents
    # --------------------------------
    documents = [
        {
            "id": "test_001",
            "text": "RAG combines retrieval with generation.",
            "document_id": "doc_rag",
            "document_name": "rag.pdf",
            "page": 1,
            "chunk_index": 0,
        },
        {
            "id": "test_002",
            "text": (
                "Retrieval-Augmented Generation uses external "
                "documents to provide context to a language model."
            ),
            "document_id": "doc_rag",
            "document_name": "rag.pdf",
            "page": 2,
            "chunk_index": 1,
        },
        {
            "id": "test_003",
            "text": (
                "Docker packages applications and their dependencies "
                "into portable containers."
            ),
            "document_id": "doc_docker",
            "document_name": "docker.pdf",
            "page": 1,
            "chunk_index": 0,
        },
        {
            "id": "test_004",
            "text": (
                "Kubernetes manages containerized applications "
                "across a cluster of machines."
            ),
            "document_id": "doc_kubernetes",
            "document_name": "kubernetes.pdf",
            "page": 1,
            "chunk_index": 0,
        },
    ]

    # --------------------------------
    # 3. Determine embedding dimension
    # --------------------------------
    test_vector = embedding_service.embed_chunks(
        documents[0]["text"]
    )

    dimension = len(test_vector)

    print(f"Embedding dimension: {dimension}")

    # --------------------------------
    # 4. Create collection
    # --------------------------------
    vector_store.create_collection(
        dimension=dimension
    )

    vector_store.create_index()
    vector_store.load_collection()

    # --------------------------------
    # 5. Generate embeddings
    # --------------------------------
    data = []

    for document in documents:

        vector = embedding_service.embed_chunks(
            document["text"]
        )

        data.append(
            {
                "id": document["id"],
                "vector": vector,
                "text": document["text"],
                "document_id": document["document_id"],
                "document_name": document["document_name"],
                "page": document["page"],
                "chunk_index": document["chunk_index"],
            }
        )

    # --------------------------------
    # 6. Insert into Milvus
    # --------------------------------
    vector_store.insert(data)

    print(
        f"Inserted {len(data)} documents."
    )

    # --------------------------------
    # 7. Create query
    # --------------------------------
    query = "What is RAG?"

    print("\nQuery:")
    print(query)

    # --------------------------------
    # 8. Embed query
    # --------------------------------
    query_vector = embedding_service.embed_chunks(
        query
    )

    # --------------------------------
    # 9. Vector search
    # --------------------------------
    results = vector_store.search(
        query_vector=query_vector,
        limit=3,
    )

    # --------------------------------
    # 10. Print results
    # --------------------------------
    print("\n===== SEARCH RESULTS =====")

    for i, result in enumerate(results[0], start=1):

        print(f"\nResult {i}")

        print(
            f"ID: {result['id']}"
        )

        print(
            f"Score: {result['distance']}"
        )

        print(
            f"Document: "
            f"{result['entity']['document_name']}"
        )

        print(
            f"Page: "
            f"{result['entity']['page']}"
        )

        print(
            f"Chunk Index: "
            f"{result['entity']['chunk_index']}"
        )

        print(
            f"Text: "
            f"{result['entity']['text']}"
        )


if __name__ == "__main__":
    main()