import os
import time
import io
import csv
from pathlib import Path
from functools import wraps

if os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")
    os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
    os.environ.setdefault("CURL_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")

from google import genai
from dotenv import load_dotenv

def rate_limit_retry(max_retries=5, initial_delay=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                        print(f"[LLMService] Rate limit (429) hit. Retrying in {delay} seconds... ({i+1}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        raise e
            return func(*args, **kwargs)
        return wrapper
    return decorator

from google.genai import types

class LLMService:
    def __init__(self, api_key: str = None):
        # Tải các biến môi trường từ file .env
        env_file = Path(__file__).resolve().parents[2] / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
        else:
            load_dotenv()
        
        # Lấy API key từ tham số truyền vào hoặc từ biến môi trường
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Vui lòng cung cấp API key hoặc đặt biến GEMINI_API_KEY trong file .env")
            
        # Khởi tạo client với SDK mới
        self.client = genai.Client(api_key=self.api_key)
        
        # Đặt tên model tối ưu và mới nhất
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    @rate_limit_retry()
    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    @rate_limit_retry()
    def generate_structured(
        self,
        prompt: str,
        response_model,
    ):

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": (
                    response_model.model_json_schema()
                ),
            },
        )

        return response_model.model_validate_json(
            response.text
        )

    @rate_limit_retry()
    def extract_document_pages(self, file_bytes: bytes, mime_type: str) -> list[dict]:
        """
        Trích xuất văn bản tài liệu giữ nguyên số trang chính xác:
        Trả về danh sách các trang: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}]
        """
        prompt = """
Bạn là chuyên gia trích xuất dữ liệu tài liệu.
Nhiệm vụ: Trích xuất toàn bộ văn bản từ hình ảnh/tài liệu được cung cấp.
Yêu cầu bắt buộc:
1. Giữ nguyên toàn bộ cấu trúc đoạn văn, tiêu đề. BẮT BUỘC sử dụng thẻ Markdown chuẩn cho các Tiêu đề (ví dụ: # Tiêu đề 1, ## Tiêu đề 2, ### Tiêu đề 3) để thể hiện cấu trúc phân cấp.
2. NẾU CÓ BẢNG BIỂU (TABLE): BẮT BUỘC trình bày dưới dạng chuẩn Markdown (ví dụ: | Cột 1 | Cột 2 |). KHÔNG ĐƯỢC biến bảng thành các đoạn text rời rạc. Phải giữ nguyên hàng và cột.
3. Chỉ trả về nội dung trích xuất, không giải thích gì thêm.
"""
        pages_data = []

        mime_lower = (mime_type or "").lower()

        if "csv" in mime_lower:
            try:
                text_content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text_content = file_bytes.decode("latin-1", errors="replace")

            # Parse CSV and convert to readable Markdown table chunks
            reader = list(csv.reader(io.StringIO(text_content)))
            if not reader:
                return [{"page": 1, "text": ""}]

            headers = reader[0]
            header_line = "| " + " | ".join(headers) + " |"
            separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

            data_rows = reader[1:]
            if not data_rows:
                table_md = f"{header_line}\n{separator_line}\n"
                return [{"page": 1, "text": table_md}]

            # Group rows into page-sized blocks (e.g. 35 rows per page)
            rows_per_page = 35
            pages_data = []
            for i in range(0, len(data_rows), rows_per_page):
                page_idx = (i // rows_per_page) + 1
                chunk_rows = data_rows[i:i + rows_per_page]
                lines = [f"# Bảng dữ liệu CSV (Phần {page_idx})\n", header_line, separator_line]
                for r in chunk_rows:
                    padded_r = r + [""] * (len(headers) - len(r))
                    clean_cols = [c.replace("|", "\\|").replace("\n", " ").strip() for c in padded_r[:len(headers)]]
                    lines.append("| " + " | ".join(clean_cols) + " |")
                pages_data.append({"page": page_idx, "text": "\n".join(lines)})
            return pages_data

        elif "text" in mime_lower or "markdown" in mime_lower or "plain" in mime_lower:
            try:
                text_content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text_content = file_bytes.decode("latin-1", errors="replace")
            return [{"page": 1, "text": text_content.strip()}]

        elif "pdf" in mime_lower:
            import pymupdf
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                local_text = page.get_text()
                real_page = page_num + 1

                # Logic thông minh: Chỉ dùng OCR khi thật sự cần (chứa bảng hoặc ảnh scan)
                tables = page.find_tables().tables if hasattr(page, 'find_tables') else []
                has_tables = len(tables) > 0
                is_scanned = len(local_text.strip()) == 0 and len(page.get_images()) > 0

                page_text = ""
                if has_tables or is_scanned:
                    print(f"[LLMService] Trang {real_page}/{len(doc)} chứa Bảng/Ảnh scan. Đang gọi Gemini Vision...")
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")

                    try:
                        part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=[prompt, part],
                        )
                        if response.text:
                            page_text = response.text.strip()
                    except Exception as e:
                        print(f"[LLMService] Lỗi khi trích xuất trang {real_page}: {e}. Fallback dùng text cục bộ.")
                        page_text = local_text.strip()
                else:
                    print(f"[LLMService] Trang {real_page}/{len(doc)} là text thường. Đọc siêu tốc cục bộ...")
                    page_text = local_text.strip()

                if page_text:
                    pages_data.append({"page": real_page, "text": page_text})

            return pages_data

        else:
            # File ảnh đơn lẻ
            part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, part],
            )
            text = (response.text or "").strip()
            return [{"page": 1, "text": text}] if text else []

    @rate_limit_retry()
    def extract_document_content(self, file_bytes: bytes, mime_type: str) -> str:
        pages = self.extract_document_pages(file_bytes=file_bytes, mime_type=mime_type)
        return "\n\n".join(p["text"] for p in pages if p.get("text"))