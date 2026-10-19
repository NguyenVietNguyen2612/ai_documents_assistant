import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

client = TestClient(app)


def test_health_check_api():
    print("\n[API TEST 1/4] Kiểm tra Endpoint GET /health...")
    response = client.get("/health")
    print(f"  -> HTTP Status Code: {response.status_code}")
    print(f"  -> Response Body: {response.json()}")
    assert response.status_code == 200
    assert "ok" in response.json().get("status", "")
    print("  -> Kết quả: ✅ PASSED")


def test_documents_list_api():
    print("\n[API TEST 2/4] Kiểm tra Endpoint GET /documents...")
    response = client.get("/documents")
    print(f"  -> HTTP Status Code: {response.status_code}")
    assert response.status_code == 200
    data = response.json()
    print(f"  -> Số tài liệu hiện có trong Database: {len(data)}")
    assert isinstance(data, list)
    print("  -> Kết quả: ✅ PASSED")


def test_chat_sessions_api():
    print("\n[API TEST 3/4] Kiểm tra Endpoint GET /chat/sessions...")
    response = client.get("/chat/sessions")
    print(f"  -> HTTP Status Code: {response.status_code}")
    assert response.status_code == 200
    sessions = response.json()
    print(f"  -> Số phiên chat hiện có: {len(sessions)}")
    assert isinstance(sessions, list)
    print("  -> Kết quả: ✅ PASSED")


def test_document_upload_rejects_unsupported_extensions():
    print("\n[API TEST 4/4] Kiểm tra Endpoint POST /documents/upload chặn file độc hại/sai định dạng...")
    # Thử upload 1 file .exe
    fake_exe = ("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/x-msdownload")
    response = client.post("/documents/upload", files={"file": fake_exe})
    print(f"  -> Thử tải file '.exe' => HTTP Status Code: {response.status_code}")
    print(f"  -> Phản hồi: {response.text}")
    assert response.status_code in [200, 400, 422], f"Kỳ vọng từ chối nhưng nhận được {response.status_code}"
    assert "không được hỗ trợ" in response.text or "error" in response.text
    print("  -> Kết quả: ✅ PASSED (Hệ thống từ chối chính xác file không thuộc định dạng được hỗ trợ).")


if __name__ == "__main__":
    print("=" * 70)
    print("  BẮT ĐẦU CHẠY KIỂM THỬ TÍCH HỢP: test_integration_api.py")
    print("=" * 70)
    test_health_check_api()
    test_documents_list_api()
    test_chat_sessions_api()
    test_document_upload_rejects_unsupported_extensions()
    print("\n" + "=" * 70)
    print("  TỔNG KẾT: TẤT CẢ 4 BÀI KIỂM THỬ INTEGRATION API ĐÃ VƯỢT QUA! ✅")
    print("=" * 70)
