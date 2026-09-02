from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import health
from app.api.routes import documents
from app.api.routes import chat

from app.services.db_service import db_service
from app.services.vector_store import VectorStore

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[Sync] Bắt đầu đồng bộ hóa Database...")
    try:
        vector_store = VectorStore()
        sqlite_ids = set(db_service.get_all_document_ids())
        milvus_ids = set(vector_store.get_all_document_ids())
        
        ghost_ids = sqlite_ids - milvus_ids
        for doc_id in ghost_ids:
            print(f"[Sync] Xóa Ghost Record khỏi SQLite (Không có trong Milvus): {doc_id}")
            db_service.delete_document(doc_id)
            
        orphan_ids = milvus_ids - sqlite_ids
        for doc_id in orphan_ids:
            print(f"[Sync] Xóa Orphan Record khỏi Milvus (Không có trong SQLite): {doc_id}")
            vector_store.delete_by_document_id(doc_id)
            
        print("[Sync] Đồng bộ hóa hoàn tất!\n")
    except Exception as e:
        print(f"[Sync] Bỏ qua đồng bộ hóa (Milvus chưa sẵn sàng): {e}\n")
        
    yield

app = FastAPI(
    title="AI Document Assistant API",
    version="0.1.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)