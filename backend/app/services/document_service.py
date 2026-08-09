from importlib import readers
from pathlib import Path
from pypdf import PdfReader
from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .embedding_service import EmbeddingService


UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(exist_ok=True)


class DocumentService:

    async def save_file(
        self,
        file: UploadFile
    ) -> Path:

        file_path = UPLOAD_DIR / file.filename

        content = await file.read()

        file_path.write_bytes(content)

        return file_path

    def extract_text(
        self,
        file_path: Path
    ) -> str:

        reader = PdfReader(file_path)

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)


    def chunk_text(
        self,
        text: str
    ) -> list[str]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

        return splitter.split_text(text)
    
    def __init__(self):
        self.embedding_service = EmbeddingService()

    def process_document(
        self,
        file_path: Path
    ):

        text = self.extract_text(
            file_path
        )

        chunks = self.chunk_text(
            text
        )

        embeddings = self.embedding_service.embed_chunks(
            chunks
        )

        return {
            "text": text,
            "chunks": chunks,
            "embeddings": embeddings,
        }