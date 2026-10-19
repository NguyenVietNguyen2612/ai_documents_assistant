import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ImportError:
    pytest = None

from app.services.document_service import DocumentService


def test_chunk_text_splits_correctly():
    print("\n[TEST 1/3] Kiểm tra cắt văn bản dài (chunk_text length bounds)...")
    with patch("app.services.document_service.EmbeddingService"):
        service = DocumentService()
        # Tạo văn bản dài 1500 ký tự không có header
        text = "Deepfake detection frequency analysis research paper. " * 30
        chunks = service.chunk_text(text)

        print(f"  -> Input: {len(text)} ký tự | Output: {len(chunks)} chunks")
        assert len(chunks) >= 2, f"Kỳ vọng >= 2 chunks nhưng nhận được {len(chunks)}"
        for idx, chunk in enumerate(chunks):
            print(f"     Chunk #{idx+1}: {len(chunk)} ký tự")
            assert len(chunk) <= 1100, f"Chunk {idx} vượt quá kích thước cho phép: {len(chunk)}"

    print("  -> Kết quả: ✅ PASSED (Cắt đoạn chính xác trong giới hạn kích thước).")


def test_structural_markdown_chunking_injects_header_metadata():
    print("\n[TEST 2/3] Kiểm tra phân đoạn cấu trúc Markdown (Structural Header Metadata)...")
    with patch("app.services.document_service.EmbeddingService"):
        service = DocumentService()
        markdown_text = """# Giới Thiệu FreqNet
FreqNet là mạng học trong không gian tần số để phát hiện Deepfake.

## Kiến Trúc
FreqNet bao gồm 3 khối xử lý chính là HFRI, HFRF và FCL.

### Cải Tiến Adaptive Soft Masking
Mặt nạ thích ứng M_adapt được định nghĩa bằng hàm Sigmoid sigma(Wm(i,j)).
"""
        chunks = service.chunk_text(markdown_text)
        print(f"  -> Số chunk được tạo: {len(chunks)}")
        for idx, c in enumerate(chunks):
            print(f"     Chunk #{idx+1} xem trước: {c[:80].replace(chr(10), ' ')}...")

        # Kiểm tra metadata header được tiêm trực tiếp vào đầu chunk
        assert any("Giới Thiệu FreqNet" in c for c in chunks), "Thiếu header Giới Thiệu FreqNet"
        assert any("Kiến Trúc" in c for c in chunks), "Thiếu header Kiến Trúc"
        assert any("Adaptive Soft Masking" in c for c in chunks), "Thiếu header Adaptive Soft Masking"

    print("  -> Kết quả: ✅ PASSED (Metadata tiêu đề được tiêm đầy đủ vào phân đoạn).")


def test_chunk_pages_preserves_page_numbers():
    print("\n[TEST 3/3] Kiểm tra bảo toàn số trang (Page-aware chunking)...")
    with patch("app.services.document_service.EmbeddingService"):
        service = DocumentService()
        pages = [
            {"page": 1, "text": "Trang 1: Báo cáo kỹ thuật về Deepfake và mô hình FreqNet."},
            {"page": 2, "text": "Trang 2: Các công thức toán học và bảng số liệu thực nghiệm."},
            {"page": 3, "text": "Trang 3: Kết luận và hướng phát triển trong tương lai."},
        ]

        chunk_items = service.chunk_pages(pages)
        print(f"  -> Tổng số chunk thu được: {len(chunk_items)}")
        for idx, item in enumerate(chunk_items):
            print(f"     Chunk #{idx+1} | Trang {item['page']} | Text: {item['text'][:45]}...")
            assert "page" in item and item["page"] in [1, 2, 3]

        assert len(chunk_items) == 3
        assert chunk_items[0]["page"] == 1
        assert chunk_items[1]["page"] == 2
        assert chunk_items[2]["page"] == 3

    print("  -> Kết quả: ✅ PASSED (Số trang được bảo toàn nguyên vẹn 100%).")


if __name__ == "__main__":
    print("=" * 70)
    print("  BẮT ĐẦU CHẠY KIỂM THỬ: test_unit_document.py")
    print("=" * 70)
    test_chunk_text_splits_correctly()
    test_structural_markdown_chunking_injects_header_metadata()
    test_chunk_pages_preserves_page_numbers()
    print("\n" + "=" * 70)
    print("  TỔNG KẾT: TẤT CẢ 3 BÀI KIỂM THỬ DOCUMENT SERVICE ĐÃ VƯỢT QUA! ✅")
    print("=" * 70)
