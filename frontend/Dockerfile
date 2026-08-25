# Sử dụng image Node.js gọn nhẹ
FROM node:20-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy package.json và package-lock.json trước để tận dụng cache
COPY package*.json ./

# Cài đặt thư viện Node.js
RUN npm install

# Copy toàn bộ mã nguồn frontend vào container
COPY . .

# Mở cổng 5173 (Cổng mặc định của Vite)
EXPOSE 5173

# Lệnh khởi chạy frontend ở chế độ dev
# Lưu ý: Cần thêm --host để Vite cho phép truy cập từ bên ngoài mạng container
CMD ["npm", "run", "dev", "--", "--host"]
