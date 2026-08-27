from unittest.mock import patch
from app.services.document_service import DocumentService

# =================================================================
# ĐÂY LÀ UNIT TEST (Kiểm thử mức đơn vị)
# Mục tiêu: Chỉ test 1 hàm duy nhất (chunk_text), cô lập nó khỏi Database và AI Models.
# =================================================================

# Dùng @patch để "làm giả" (mock) EmbeddingService. 
# Việc này giúp khi DocumentService khởi tạo sẽ không load Model AI thật vào RAM -> Test chạy chớp nhoáng (0.01 giây).
@patch("app.services.document_service.EmbeddingService")
def test_chunk_text_splits_correctly(mock_embedding_service):
    # 1. Arrange (Chuẩn bị dữ liệu đầu vào)
    service = DocumentService()
    # Tạo một đoạn văn bản dài 1500 ký tự
    text = "A" * 1500
    
    # 2. Act (Thực thi hàm cần test)
    chunks = service.chunk_text(text)
    
    # 3. Assert (Kiểm chứng kết quả)
    # Vì chunk_size cấu hình là 1000, văn bản 1500 ký tự phải bị cắt thành ít nhất 2 đoạn
    assert len(chunks) >= 2
    
    # Mỗi đoạn không được vượt quá kích thước 1000 ký tự
    for chunk in chunks:
        assert len(chunk) <= 1000
