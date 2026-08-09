from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from app.services.document_service import DocumentService


router = APIRouter()
document_service = DocumentService()


documents = [
    {
        "id": "1",
        "name": "rag.pdf",
        "size": "1.2 MB",
    },
    {
        "id": "2",
        "name": "ai_agents.pdf",
        "size": "2.4 MB",
    },
    {
        "id": "3",
        "name": "langgraph.pdf",
        "size": "1.8 MB",
    },
]


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

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "path": str(file_path),
    }