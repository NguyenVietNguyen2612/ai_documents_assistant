from pathlib import Path

from fastapi import APIRouter, UploadFile, File


router = APIRouter()


UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(
    exist_ok=True
)


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

    file_path = UPLOAD_DIR / file.filename

    content = await file.read()

    file_path.write_bytes(content)

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
    }