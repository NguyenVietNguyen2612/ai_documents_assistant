from app.services.embedding_service import (
    EmbeddingService
)


service = EmbeddingService()


chunks = [
    "Retrieval-Augmented Generation combines retrieval with generation.",
    "Large language models can use external knowledge.",
    "A vector database stores embeddings.",
]


embeddings = service.embed_chunks(chunks)


print("Number of chunks:")
print(len(chunks))

print("\nNumber of embeddings:")
print(len(embeddings))

print("\nEmbedding dimension:")
print(len(embeddings[0]))