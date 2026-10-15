#!/usr/bin/env bash
# ========================================================
# AI Documents Assistant Launcher for Linux / Ubuntu / macOS
# Tương thích 100% với start.bat trên Windows
# ========================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Starting AI Documents Assistant (Linux)..."
echo "=========================================="

# Tự động nạp chứng chỉ CA của hệ thống Linux nếu có (hỗ trợ mạng nội bộ / proxy)
if [ -f "/etc/ssl/certs/ca-certificates.crt" ]; then
    export REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
    export SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
    export CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
fi

# 1. Xác định lệnh Docker Compose
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif sg docker -c "docker compose version" >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="sg docker -c 'docker compose'"
elif which docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    echo "❌ Lỗi: Không tìm thấy Docker hoặc Docker Compose."
    echo "Vui lòng thêm tài khoản của bạn vào nhóm docker:"
    echo "  sudo usermod -aG docker \$USER"
    echo "Sau đó đăng xuất và đăng nhập lại vào Ubuntu."
    exit 1
fi

echo ""
echo "[1/3] Starting Milvus Vector Database & MinIO..."
if docker ps >/dev/null 2>&1; then
    $DOCKER_COMPOSE_CMD up -d etcd minio standalone
else
    echo "ℹ️  Kích hoạt quyền nhóm docker cho phiên làm việc..."
    sg docker -c "docker compose up -d etcd minio standalone"
fi

# Chờ Milvus sẵn sàng
echo "Đang chờ Milvus khởi động hoàn tất..."
for i in {1..30}; do
    if curl -s http://localhost:9091/healthz 2>&1 | grep -q "OK"; then
        echo "✅ Milvus Vector DB đã sẵn sàng (port 19530)!"
        break
    fi
    sleep 1
done

# 2. Kiểm tra môi trường ảo Python backend
echo ""
echo "[2/3] Checking Backend Python Environment..."
if [ ! -f "backend/.venv/bin/uvicorn" ]; then
    echo "⚙️ Đang cài đặt thư viện cho backend..."
    python3 -m venv backend/.venv
    backend/.venv/bin/pip install --upgrade pip setuptools wheel
    backend/.venv/bin/pip install -r backend/requirements.txt
fi
echo "✅ Backend environment sẵn sàng!"

# 3. Kiểm tra dependencies của Frontend
echo ""
echo "[3/3] Checking Frontend Dependencies..."
if [ ! -f "frontend/node_modules/vite/bin/vite.js" ]; then
    echo "⚙️ Đang cài đặt thư viện frontend..."
    (cd frontend && npm install)
fi
echo "✅ Frontend dependencies sẵn sàng!"

echo ""
echo "=========================================="
echo "🚀 Khởi chạy hệ thống:"
echo "   Backend:  http://localhost:8001"
echo "   Frontend: http://localhost:3000"
echo "=========================================="
echo "Nhấn Ctrl+C để dừng tất cả dịch vụ."
echo ""

# Khởi chạy Backend và Frontend trong terminal hiện tại, hiển thị log trực tiếp
(cd backend && .venv/bin/uvicorn app.main:app --reload --port 8001) &
BACKEND_PID=$!

(cd frontend && npm run dev) &
FRONTEND_PID=$!

cleanup() {
    echo ""
    echo "Đang tắt tất cả dịch vụ..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    echo "Đã dừng."
    exit 0
}

trap cleanup INT TERM EXIT

wait
