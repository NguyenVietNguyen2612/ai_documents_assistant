from fastapi import APIRouter, UploadFile, File
from app.services.document_service import DocumentService
from app.services.vector_store import VectorStore
from app.services.db_service import db_service
import uuid

router = APIRouter()
document_service = DocumentService()
vector_store = VectorStore()

# Initialize Milvus collection with dimension 384 for paraphrase-multilingual-MiniLM-L12-v2
vector_store.create_collection(384)
vector_store.create_index()
vector_store.load_collection()


@router.get("/documents")
def get_documents():
    # Lấy dữ liệu trực tiếp từ SQLite Database thay vì JSON
    return db_service.get_all_documents()


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    if file.content_type != "application/pdf":
        return {
            "error": "Only PDF files are allowed"
        }

    # Bỏ lưu xuống đĩa. Xử lý trực tiếp trên RAM.
    doc_id = str(uuid.uuid4())
    file_size = getattr(file, "size", 0)
    size_mb = f"{round(file_size / 1024 / 1024, 2)} MB" if file_size else "Unknown"
    
    # Lưu metadata vào SQLite
    db_service.insert_document(doc_id, file.filename, size_mb)

    # Process the document to extract text, chunk, and embed directly from RAM
    processed = document_service.process_document(file)
    
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
        # Kiểm tra và tạo lại collection nếu nó bị xóa trong lúc server đang chạy
        if not vector_store.client.has_collection(vector_store.collection_name):
            vector_store.create_collection(384)
            vector_store.create_index()
            vector_store.load_collection()
            
        vector_store.insert(data_to_insert)
        print(f"Inserted {len(data_to_insert)} chunks for {file.filename} into Vector DB.")

    return {
        "message": "File processed and uploaded successfully in-memory",
        "filename": file.filename
    }


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    # Remove from SQLite
    db_service.delete_document(document_id)
    
    # Remove from Milvus unconditionally
    try:
        vector_store.delete_by_document_id(document_id)
    except Exception as e:
        print(f"Failed to delete from Milvus: {e}")
    
    return {"message": "Document and associated vector data deleted successfully"}