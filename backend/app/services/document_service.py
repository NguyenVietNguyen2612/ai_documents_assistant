from pypdf import PdfReader
from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .embedding_service import EmbeddingService


class DocumentService:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    def extract_text(self, file: UploadFile) -> str:
        # Đọc trực tiếp từ file trên RAM (UploadFile.file) thay vì dùng file_path
        reader = PdfReader(file.file)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    def chunk_text(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        return splitter.split_text(text)
    
    def process_document(self, file: UploadFile):
        # file.file.seek(0) đảm bảo ta đọc từ đầu file stream
        file.file.seek(0)
        text = self.extract_text(file)
        chunks = self.chunk_text(text)
        embeddings = self.embedding_service.embed_chunks(chunks)

        return {
            "text": text,
            "chunks": chunks,
            "embeddings": embeddings,
        }