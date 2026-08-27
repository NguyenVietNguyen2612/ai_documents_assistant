from fastapi.testclient import TestClient
from app.main import app

# =================================================================
# ĐÂY LÀ INTEGRATION TEST (Kiểm thử tích hợp)
# Mục tiêu: Test sự kết nối giữa Router (API) và ứng dụng FastAPI.
# =================================================================

# TestClient mô phỏng một trình duyệt/Postman gửi request HTTP ảo vào app
client = TestClient(app)

def test_health_check_api():
    # 1. Act: Gửi 1 request GET đến endpoint /health
    response = client.get("/health")
    
    # 2. Assert: Kiểm tra HTTP Status Code trả về phải là 200 (Thành công)
    assert response.status_code == 200
    
    # (Tùy chọn) Kiểm tra nội dung JSON trả về nếu có
    # assert response.json() == {"status": "ok"}
