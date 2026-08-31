from pypdf import PdfReader
from fastapi import UploadFile
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from .embedding_service import EmbeddingService
from .llm_service import LLMService

class DocumentService:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    def extract_text(self, file: UploadFile) -> str:
        # Khởi tạo LLM Service để dùng Gemini Vision
        llm = LLMService()
        file_bytes = file.file.read()
        file.file.seek(0)
        
        try:
            print(f"[DocumentService] Bắt đầu OCR và trích xuất Bảng bằng Gemini Vision cho file {file.filename}...")
            # Fallback mime_type nếu không có
            mime_type = file.content_type or "application/pdf"
            text = llm.extract_document_content(file_bytes=file_bytes, mime_type=mime_type)
            print("[DocumentService] Trích xuất thành công bằng Gemini!")
            return text
        except Exception as e:
            print(f"[DocumentService] Lỗi khi dùng Gemini Vision: {e}. Fallback sang pypdf...")
            file.file.seek(0)
            if "pdf" in (file.content_type or "").lower():
                reader = PdfReader(file.file)
                pages = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                return "\n".join(pages)
            return ""

    def chunk_text(self, text: str) -> list[str]:
        # Dùng MarkdownTextSplitter để ưu tiên giữ nguyên bảng biểu Markdown
        try:
            splitter = MarkdownTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
            )
            return splitter.split_text(text)
        except Exception:
            # Fallback
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