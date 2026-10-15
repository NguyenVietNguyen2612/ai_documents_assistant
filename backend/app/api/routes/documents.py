from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.services.document_service import DocumentService
from app.services.vector_store import VectorStore
from app.services.db_service import db_service
from app.services.bm25_service import bm25_service
import uuid

router = APIRouter()
document_service = DocumentService()
vector_store = VectorStore()

DOCS_DIR = Path("app_data/documents")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Milvus collection with dimension 384 for paraphrase-multilingual-MiniLM-L12-v2
vector_store.create_collection(384)
vector_store.create_index()
vector_store.load_collection()


@router.get("/documents")
def get_documents():
    # Lấy dữ liệu trực tiếp từ SQLite Database thay vì JSON
    return db_service.get_all_documents()


@router.get("/documents/{document_id}/file")
def get_document_file(document_id: str):
    """Phục vụ file PDF gốc để hiển thị xem trước (preview) trên giao diện theo số trang."""
    pdf_path = DOCS_DIR / f"{document_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="File PDF không tồn tại trên hệ thống")
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"{document_id}.pdf",
    )


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    if file.content_type != "application/pdf":
        return {
            "error": "Only PDF files are allowed"
        }

    doc_id = str(uuid.uuid4())
    file_bytes = await file.read()
    await file.seek(0)

    # 1. Lưu file PDF vật lý vào app_data/documents để hỗ trợ PDF Viewer nhảy đến đúng số trang
    pdf_path = DOCS_DIR / f"{doc_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)

    file_size = len(file_bytes)
    size_mb = f"{round(file_size / 1024 / 1024, 2)} MB" if file_size else "Unknown"
    
    # 2. Lưu metadata vào SQLite
    db_service.insert_document(doc_id, file.filename, size_mb)

    # 3. Trích xuất tài liệu theo từng trang chính xác
    processed = document_service.process_document(file)
    
    data_to_insert = []
    chunk_pages = processed.get("pages", [1] * len(processed["chunks"]))
    for i, (chunk, embedding, page_num) in enumerate(zip(processed["chunks"], processed["embeddings"], chunk_pages)):
        data_to_insert.append({
            "id": f"{doc_id}_{i}",
            "vector": embedding,
            "text": chunk,
            "document_id": doc_id,
            "document_name": file.filename,
            "page": int(page_num),  # Số trang thực tế của tài liệu!
            "chunk_index": i,
        })
        
    if data_to_insert:
        # Kiểm tra và tạo lại collection nếu nó bị xóa trong lúc server đang chạy
        if not vector_store.client.has_collection(vector_store.collection_name):
            vector_store.create_collection(384)
            vector_store.create_index()
            vector_store.load_collection()
            
        vector_store.insert(data_to_insert)
        print(f"Inserted {len(data_to_insert)} chunks (with real page numbers) for {file.filename} into Vector DB.")

        # Thêm các chunk vào chỉ mục BM25
        bm25_service.add_documents(data_to_insert)

    return {
        "message": "File processed and uploaded successfully with page-aware chunking",
        "document_id": doc_id,
        "filename": file.filename,
        "total_chunks": len(data_to_insert),
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
    
    # Remove from BM25 index
    bm25_service.remove_documents_by_id(document_id)

    # Xóa file PDF vật lý nếu có
    pdf_path = DOCS_DIR / f"{document_id}.pdf"
    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except Exception as e:
            print(f"Failed to delete PDF file: {e}")

    return {"message": "Document and associated vector data deleted successfully"}