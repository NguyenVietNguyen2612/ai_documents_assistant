# Hướng dẫn Khởi động Nhanh K8s (Hybrid Approach)

Mỗi khi bạn bật máy tính lên để học tập hoặc làm việc tiếp, bạn chỉ cần gõ đúng 2 lệnh sau theo thứ tự để gọi toàn bộ hệ thống (Database, Backend, Frontend) thức dậy:

### 1. Lệnh Khởi động
**Bước 1: Bật cụm Database (bằng Docker Compose để né lỗi Windows)**
```bash
docker compose up -d etcd minio standalone
```

**Bước 2: Bật Frontend và Backend (bằng Kubernetes)**
```bash
kubectl apply -f k8s/
```
*(Mẹo: Bằng cách trỏ thẳng vào thư mục `k8s/`, Kubernetes sẽ tự động quét và chạy TẤT CẢ các file cấu hình có trong đó cùng một lúc!)*

---

### 2. Lệnh Dọn dẹp (Tắt máy)
Khi bạn không muốn làm việc nữa và cần giải phóng RAM để máy tính chạy nhanh hơn, hãy gõ 2 lệnh này:

**Tắt ứng dụng trong K8s:**
```bash
kubectl delete deployment frontend-deployment backend-deployment
```

**Tắt Database:**
```bash
docker compose down
```
