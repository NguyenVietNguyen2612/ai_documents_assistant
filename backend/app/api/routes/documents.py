from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
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


def process_document_background(doc_id: str, filename: str, saved_file_path: Path, ext: str):
    """Xử lý Ingestion ngầm (Async Background Task): OCR/Text Extraction, Structural Chunking, Embedding và Indexing."""
    try:
        print(f"[Async Ingestion] Bắt đầu xử lý ngầm cho tài liệu: '{filename}' (ID: {doc_id})...")
        file_bytes = saved_file_path.read_bytes()
        processed = document_service.process_file_data(file_bytes, filename)
        
        data_to_insert = []
        chunk_pages = processed.get("pages", [1] * len(processed["chunks"]))
        for i, (chunk, embedding, page_num) in enumerate(zip(processed["chunks"], processed["embeddings"], chunk_pages)):
            data_to_insert.append({
                "id": f"{doc_id}_{i}",
                "vector": embedding,
                "text": chunk,
                "document_id": doc_id,
                "document_name": filename,
                "page": int(page_num),  # Số trang thực tế của tài liệu
                "chunk_index": i,
            })
            
        if data_to_insert:
            if not vector_store.client.has_collection(vector_store.collection_name):
                vector_store.create_collection(384)
                vector_store.create_index()
                vector_store.load_collection()
                
            vector_store.insert(data_to_insert)
            print(f"[Async Ingestion] Đã chèn {len(data_to_insert)} chunks vào Milvus Vector DB.")

            # Thêm các chunk vào chỉ mục BM25
            bm25_service.add_documents(data_to_insert)
            print(f"[Async Ingestion] Đã cập nhật chỉ mục BM25 cho '{filename}'.")

        db_service.update_document_status(doc_id, status="ready", chunk_count=len(data_to_insert))
        print(f"[Async Ingestion] Hoàn tất xử lý ngầm tài liệu '{filename}' (ID: {doc_id}) với {len(data_to_insert)} chunks -> Trạng thái: READY ✅")
    except Exception as e:
        print(f"[Async Ingestion] Lỗi khi xử lý ngầm tài liệu '{filename}': {e} ❌")
        db_service.update_document_status(doc_id, status="error", error_message=str(e))


@router.get("/documents")
def get_documents():
    # Lấy dữ liệu trực tiếp từ SQLite Database kèm status và chunk_count
    return db_service.get_all_documents()


@router.get("/documents/{document_id}/status")
def get_document_status(document_id: str):
    """Endpoint truy vấn trạng thái tiến trình xử lý tài liệu (processing / ready / error)."""
    doc = db_service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="File tài liệu không tồn tại trên hệ thống")
    return {
        "id": doc["id"],
        "name": doc["name"],
        "status": doc.get("status", "ready"),
        "chunk_count": doc.get("chunk_count", 0),
        "error_message": doc.get("error_message"),
    }


@router.get("/documents/{document_id}/file")
def get_document_file(document_id: str, download: bool = False):
    """Phục vụ file tài liệu gốc để hiển thị xem trước (preview) trực tiếp trên giao diện."""
    matching_files = list(DOCS_DIR.glob(f"{document_id}.*"))
    if not matching_files:
        raise HTTPException(status_code=404, detail="File tài liệu không tồn tại trên hệ thống")
    
    target_file = matching_files[0]
    ext = target_file.suffix.lower()
    # Dùng text/plain cho các file văn bản/bảng để trình duyệt luôn mở inline thay vì tự tải về
    media_map = {
        ".pdf": "application/pdf",
        ".csv": "text/plain; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/plain; charset=utf-8",
    }
    media_type = media_map.get(ext, "application/octet-stream")
    disposition = "attachment" if download else "inline"

    return FileResponse(
        path=target_file,
        media_type=media_type,
        filename=target_file.name,
        content_disposition_type=disposition,
    )


@router.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    ext = Path(file.filename or "").suffix.lower()
    allowed_extensions = {".pdf", ".csv", ".txt", ".md"}
    if ext not in allowed_extensions:
        return {
            "error": f"Định dạng tệp '{ext}' không được hỗ trợ. Vui lòng tải lên file: {', '.join(allowed_extensions)}"
        }

    doc_id = str(uuid.uuid4())
    file_bytes = await file.read()
    await file.seek(0)

    # 1. Lưu file vật lý vào app_data/documents để hỗ trợ Viewer mở lại
    saved_file_path = DOCS_DIR / f"{doc_id}{ext}"
    with open(saved_file_path, "wb") as f:
        f.write(file_bytes)

    file_size = len(file_bytes)
    size_mb = f"{round(file_size / 1024 / 1024, 2)} MB" if file_size else "Unknown"
    
    # 2. Lưu metadata vào SQLite với status='processing'
    db_service.insert_document(doc_id, file.filename, size_mb, status="processing", chunk_count=0)

    # 3. Đưa vào BackgroundTasks để trích xuất OCR, chunking và embedding bất đồng bộ
    background_tasks.add_task(process_document_background, doc_id, file.filename, saved_file_path, ext)

    return {
        "message": "File đã được tải lên thành công và đang được xử lý ngầm (Async Background Ingestion)",
        "document_id": doc_id,
        "filename": file.filename,
        "status": "processing",
        "size": size_mb,
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

    # Xóa file vật lý tương ứng nếu có
    for f in DOCS_DIR.glob(f"{document_id}.*"):
        try:
            f.unlink()
        except Exception as e:
            print(f"Failed to delete document file {f}: {e}")

    return {"message": "Document and associated vector data deleted successfully"}