import io
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
from fastapi import UploadFile
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from .embedding_service import EmbeddingService
from .llm_service import LLMService

class DocumentService:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    def extract_pages_from_bytes(self, file_bytes: bytes, filename: str = "", content_type: str = "") -> list[dict]:
        """
        Trích xuất văn bản tài liệu theo từng trang từ raw bytes.
        Trả về danh sách: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}]
        """
        llm = LLMService()
        fname = (filename or "").lower()
        mime_type = (content_type or "").lower()
        if not mime_type or mime_type in ("application/octet-stream", "text/plain"):
            if fname.endswith(".csv"):
                mime_type = "text/csv"
            elif fname.endswith(".md"):
                mime_type = "text/markdown"
            elif fname.endswith(".pdf"):
                mime_type = "application/pdf"

        try:
            print(f"[DocumentService] Trích xuất nội dung cho file {filename} (mime_type: {mime_type})...")
            pages = llm.extract_document_pages(file_bytes=file_bytes, mime_type=mime_type)
            if pages:
                print(f"[DocumentService] Trích xuất thành công {len(pages)} trang/phần từ {filename}!")
                return pages
        except Exception as e:
            print(f"[DocumentService] Lỗi khi trích xuất: {e}. Fallback sang pypdf...")

        pages_fallback = []
        if (fname.endswith(".pdf") or "pdf" in mime_type) and PdfReader is not None:
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                for page_idx, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages_fallback.append({"page": page_idx, "text": text.strip()})
            except Exception as e:
                print(f"[DocumentService] Fallback pypdf error: {e}")

        if not pages_fallback:
            # Nếu là file text/markdown/csv thô
            try:
                decoded = file_bytes.decode("utf-8", errors="ignore")
                if decoded.strip():
                    return [{"page": 1, "text": decoded.strip()}]
            except Exception:
                pass
            return [{"page": 1, "text": ""}]
        return pages_fallback

    def extract_pages(self, file: UploadFile) -> list[dict]:
        """
        Trích xuất văn bản tài liệu theo từng trang để lưu giữ số trang chính xác.
        """
        file.file.seek(0)
        file_bytes = file.file.read()
        file.file.seek(0)
        return self.extract_pages_from_bytes(file_bytes, file.filename or "", file.content_type or "")

    def extract_text(self, file: UploadFile) -> str:
        pages = self.extract_pages(file)
        return "\n\n".join(p["text"] for p in pages if p.get("text"))

    def chunk_text(self, text: str) -> list[str]:
        try:
            # Bước 1: Chia theo Markdown Headers
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_header_splits = markdown_splitter.split_text(text)
            
            # Bước 2: Đảm bảo độ dài không vượt ngưỡng
            char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
            )
            final_splits = char_splitter.split_documents(md_header_splits)
            
            # Bước 3: Làm giàu ngữ cảnh (Inject metadata vào text)
            final_chunks = []
            for split in final_splits:
                header_path = " > ".join(split.metadata.values())
                if header_path:
                    chunk_content = f"[{header_path}]\n{split.page_content}"
                else:
                    chunk_content = split.page_content
                final_chunks.append(chunk_content)
                
            return final_chunks if final_chunks else [text]
            
        except Exception as e:
            print(f"[DocumentService] Lỗi khi dùng Structural Chunking: {e}. Fallback sang RecursiveCharacterTextSplitter...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
            )
            return splitter.split_text(text)

    def chunk_pages(self, pages: list[dict]) -> list[dict]:
        """
        Chia nhỏ văn bản từng trang mà vẫn bảo toàn số trang (page number) và cấu trúc tiêu đề.
        Trả về: [{"text": chunk_content, "page": page_number}]
        """
        all_chunks = []
        for page_item in pages:
            page_num = page_item.get("page", 1)
            page_text = page_item.get("text", "")
            if not page_text or not page_text.strip():
                continue

            page_chunks = self.chunk_text(page_text)
            for c in page_chunks:
                all_chunks.append({
                    "text": c,
                    "page": page_num,
                })

        return all_chunks
    
    def process_file_data(self, file_bytes: bytes, filename: str = "", content_type: str = ""):
        pages = self.extract_pages_from_bytes(file_bytes, filename, content_type)
        chunk_items = self.chunk_pages(pages)

        if not chunk_items:
            full_text = "\n\n".join(p["text"] for p in pages if p.get("text"))
            chunk_items = [{"text": full_text, "page": 1}]

        chunks = [item["text"] for item in chunk_items]
        pages_list = [item["page"] for item in chunk_items]
        embeddings = self.embedding_service.embed_chunks(chunks)

        return {
            "text": "\n\n".join(p["text"] for p in pages if p.get("text")),
            "chunks": chunks,
            "pages": pages_list,
            "embeddings": embeddings,
        }

    def process_document(self, file: UploadFile):
        file.file.seek(0)
        file_bytes = file.file.read()
        file.file.seek(0)
        return self.process_file_data(file_bytes, file.filename or "", file.content_type or "")