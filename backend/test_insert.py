from app.services import vector_store
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


def main():

    text = "RAG combines retrieval with generation."

    # 1. Embedding
    embedding_service = EmbeddingService()

    vector = embedding_service.embed_text(text)

    print(f"Embedding dimension: {len(vector)}")

    # 2. Vector DB
    vector_store = VectorStore()

    # 3. Load collection
    dimension = len(vector)

    vector_store.create_collection(dimension)

    vector_store.create_index()

    vector_store.load_collection()

    print("Collection loaded.")

    # 4. Insert
    data = [
        {
            "id": "test_002",
            "vector": vector,
            "text": text,
            "document_id": "test_doc_002",
            "document_name": "test.pdf",
            "page": 1,
            "chunk_index": 0,
        }
    ]

    result = vector_store.insert(data)

    print("Insert successful.")
    print(result)

    result = vector_store.get_by_id("test_002")

    print("Retrieved record:")
    print(result)


if __name__ == "__main__":
    main()