import os
import io
import asyncio
from fastapi import UploadFile
from app.services.document_service import DocumentService

async def main():
    print("=== TEST OCR & BẢNG BẰNG GEMINI VISION ===")
    
    file_path = input("Nhập đường dẫn tuyệt đối đến file PDF hoặc Ảnh cần test: ").strip()
    
    # Loại bỏ dấu ngoặc kép nếu user kéo thả file vào terminal
    file_path = file_path.strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return

    # Lấy mime_type cơ bản
    mime_type = "application/pdf"
    if file_path.lower().endswith(".png"):
        mime_type = "image/png"
    elif file_path.lower().endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"

    print(f"\nĐang đọc file: {os.path.basename(file_path)}")
    print(f"Định dạng nhận diện: {mime_type}")

    # Giả lập đối tượng UploadFile của FastAPI
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    
    file_stream = io.BytesIO(file_bytes)
    
    # Tùy phiên bản FastAPI mà cách khởi tạo UploadFile hơi khác nhau.
    upload_file = UploadFile(
        filename=os.path.basename(file_path), 
        file=file_stream,
        headers={"content-type": mime_type}
    )
    doc_service = DocumentService()
    
    print("\n[+] Đang gọi Gemini Vision để trích xuất nội dung (Có thể mất 10-30 giây)...")
    try:
        # Bước 1: Trích xuất Text
        text = doc_service.extract_text(upload_file)
        
        print("\n" + "="*50)
        print("VĂN BẢN & BẢNG ĐÃ TRÍCH XUẤT (MARKDOWN):")
        print("="*50)
        print(text)
        print("="*50)

        # Bước 2: Chia Chunk
        print("\n[+] Đang test quá trình chia Chunk (MarkdownTextSplitter)...")
        chunks = doc_service.chunk_text(text)
        
        print(f"-> Đã chia thành {len(chunks)} chunks.")
        print("Dưới đây là nội dung của 2 chunk đầu tiên (để kiểm tra xem bảng có bị cắt nát không):")
        
        for i in range(min(2, len(chunks))):
            print(f"\n--- Chunk {i+1} ({len(chunks[i])} ký tự) ---")
            print(chunks[i])
            
    except Exception as e:
        print(f"\n[!] Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    # Nạp biến môi trường từ .env
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(main())
