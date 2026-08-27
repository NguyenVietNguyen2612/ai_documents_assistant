from pathlib import Path

from app.services.document_service import (
    DocumentService
)

path = Path("uploads/rag.pdf")

print("Exists:", path.exists())
print("Actual path:", path.resolve())
print("Filename:", path.name)


service = DocumentService()


result = service.process_document(
    Path("uploads/rag.pdf")
)


print(
    "Number of chunks:",
    len(result["chunks"])
)

print(
    "Number of embeddings:",
    len(result["embeddings"])
)

print(
    "Embedding dimension:",
    len(result["embeddings"][0])
)


print("\nFirst chunk:")
print(result["chunks"][0][:500])


print("\nFirst embedding:")
print(result["embeddings"][0][:10])