from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from app.services.document_service import DocumentService
from app.services.vector_store import VectorStore

router = APIRouter()
document_service = DocumentService()
vector_store = VectorStore()

# Initialize Milvus collection with dimension 384 for all-MiniLM-L6-v2
vector_store.create_collection(384)
vector_store.create_index()
vector_store.load_collection()


documents = []


@router.get("/documents")
def get_documents():
    return documents


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    if file.content_type != "application/pdf":
        return {
            "error": "Only PDF files are allowed"
        }

    file_path = await document_service.save_file(file)

    import uuid
    doc_id = str(uuid.uuid4())
    file_size = getattr(file, "size", 0)
    size_mb = f"{round(file_size / 1024 / 1024, 2)} MB" if file_size else "Unknown"
    
    new_doc = {
        "id": doc_id,
        "name": file.filename,
        "size": size_mb,
    }
    documents.append(new_doc)

    # Process the document to extract text, chunk, and embed
    processed = document_service.process_document(file_path)
    
    data_to_insert = []
    for i, (chunk, embedding) in enumerate(zip(processed["chunks"], processed["embeddings"])):
        data_to_insert.append({
            "id": f"{doc_id}_{i}",
            "vector": embedding,
            "text": chunk,
            "document_id": doc_id,
            "document_name": file.filename,
            "page": 1,
            "chunk_index": i,
        })
        
    if data_to_insert:
        vector_store.insert(data_to_insert)
        print(f"Inserted {len(data_to_insert)} chunks for {file.filename} into Vector DB.")

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "path": str(file_path),
    }