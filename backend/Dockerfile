# Sử dụng image Python mỏng nhẹ để tối ưu kích thước
FROM python:3.12-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Khắc phục lỗi thiếu thư viện hệ thống khi build các gói Python (ví dụ pypdf, hnswlib...)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt vào trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt bản Torch CPU-only trước (nhẹ hơn 4GB, tránh lỗi sập Docker vì hết RAM/Ổ cứng)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# Cài đặt các thư viện Python còn lại
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn backend vào container
COPY . .

# Expose port 8000 cho FastAPI
EXPOSE 8000

# Lệnh khởi chạy ứng dụng (giống với lệnh bạn dùng trong môi trường dev)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
