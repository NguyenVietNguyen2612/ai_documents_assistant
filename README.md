# AI Documents Assistant

Dự án thực hành và tìm hiểu các công cụ phổ biến trong quy trình phát triển phần mềm và xây dựng hệ thống AI Agent (FastAPI, React, LangGraph, Milvus, Docker, Kubernetes).

---

## Mục lục

1. [Giới thiệu dự án](#1-giới-thiệu-dự-án)
2. [Các chức năng chính](#2-các-chức-năng-chính)
3. [Cấu trúc thư mục dự án](#3-cấu-trúc-thư-mục-dự-án)
4. [Cấu trúc hệ thống Agent](#4-cấu-trúc-hệ-thống-agent)
   - [4.1 Sơ đồ kiến trúc Agent](#41-sơ-đồ-kiến-trúc-agent)
   - [4.2 Quản lý trạng thái (AgentState)](#42-quản-lý-trạng-thái-agentstate)
   - [4.3 Vòng lặp ReAct và kiểm soát số lần gọi Tool](#43-vòng-lặp-react-và-kiểm-soát-số-lần-gọi-tool)
   - [4.4 Hiển thị quá trình suy luận (Chain-of-Thought)](#44-hiển-thị-quá-trình-suy-luận-chain-of-thought)
   - [4.5 Công cụ truy vấn tài liệu](#45-công-cụ-truy-vấn-tài-liệu)
   - [4.6 Lưu trữ Checkpoint](#46-lưu-trữ-checkpoint)
5. [Quy trình trích xuất và RAG](#5-quy-trình-trích-xuất-và-rag)
6. [Công nghệ sử dụng](#6-công-nghệ-sử-dụng)
7. [Danh sách API](#7-danh-sách-api)
8. [Hướng dẫn cài đặt và khởi chạy](#8-hướng-dẫn-cài-đặt-và-khởi-chạy)
   - [8.1 Yêu cầu môi trường](#81-yêu-cầu-môi-trường)
   - [8.2 Cấu hình biến môi trường](#82-cấu-hình-biến-môi-trường)
   - [8.3 Khởi chạy nhanh bằng script](#83-khởi-chạy-nhanh-bằng-script)
   - [8.4 Khởi chạy thủ công](#84-khởi-chạy-thủ-công)
   - [8.5 Chạy với Docker Compose và Kubernetes](#85-chạy-với-docker-compose-và-kubernetes)

---

## 1. Giới thiệu dự án

Đây là dự án cá nhân được xây dựng với mục đích học tập và thực hành các công nghệ phổ biến trong phát triển phần mềm và ứng dụng trí tuệ nhân tạo:
- Áp dụng mô hình AI Agent đơn giản với LangGraph thay vì pipeline RAG cố định.
- Tìm hiểu cách thức hoạt động của cơ sở dữ liệu vector (Milvus) trong bài toán tìm kiếm ngữ nghĩa.
- Làm quen với việc kết nối luồng streaming dữ liệu thời gian thực (Server-Sent Events) giữa Backend (FastAPI) và Frontend (React).
- Thực hành đóng gói ứng dụng với Docker Compose và viết cấu hình triển khai cơ bản trên Kubernetes.

---

## 2. Các chức năng chính

- Hỏi đáp dựa trên tài liệu: Người dùng tải lên tài liệu (PDF) và đặt câu hỏi, hệ thống sẽ tìm kiếm nội dung liên quan để trả lời.
- Cơ chế quyết định của Agent: LLM tự xác định khi nào cần tra cứu tài liệu và khi nào có thể trả lời trực tiếp các câu hỏi thông thường.
- Hiển thị các bước suy nghĩ: Phân tách khối suy luận `<think>...</think>` để người dùng có thể theo dõi cách mô hình xử lý yêu cầu.
- Tìm kiếm linh hoạt: Hỗ trợ tìm kiếm theo tên tài liệu không phân biệt chữ hoa, chữ thường và phần mở rộng của tệp.
- Quản lý phiên trò chuyện: Lưu lịch sử hội thoại và hỗ trợ tạo phiên mới hoặc xóa phiên cũ.

---

## 3. Cấu trúc thư mục dự án

```
ai_documents_assistant/
├── backend/
│   ├── app/
│   │   ├── api/routes/          # Các endpoint: chat, documents, health
│   │   ├── graph/               # Định nghĩa đồ thị LangGraph (state, rag_graph, nodes)
│   │   ├── schemas/             # Pydantic models xác thực dữ liệu
│   │   ├── services/            # Xử lý logic: agent, db, document, retriever, vector_store...
│   │   ├── tools/               # Công cụ của agent (retrieval_tool)
│   │   └── main.py              # Điểm khởi chạy ứng dụng FastAPI
│   ├── app_data/                # File cơ sở dữ liệu SQLite cục bộ (lịch sử chat, checkpoint)
│   ├── Dockerfile               # Cấu hình container cho backend
│   ├── requirements.txt         # Danh sách thư viện Python
│   └── .env                     # File cấu hình biến môi trường
│
├── frontend/
│   ├── src/
│   │   ├── components/          # Các component React: ChatBox, DocumentList, DocumentUpload, Sidebar
│   │   ├── services/            # Xử lý gọi API và nhận SSE stream
│   │   ├── App.jsx              # Giao diện chính
│   │   └── App.css              # Định dạng giao diện
│   ├── package.json             # Danh sách dependencies của frontend
│   └── vite.config.js           # Cấu hình Vite
│
├── k8s/                         # Các file cấu hình Kubernetes cơ bản
├── docker-compose.yml           # Khởi động Milvus, etcd, MinIO, Backend và Frontend
├── start.bat                    # Script khởi chạy nhanh cho môi trường Windows
└── ...
```

---

## 4. Cấu trúc hệ thống Agent

Hệ thống sử dụng thư viện LangGraph để quản lý luồng hội thoại dưới dạng một đồ thị trạng thái (StateGraph).

### 4.1 Sơ đồ kiến trúc Agent

```
               [ START ]
                   |
                   v
       +-----------------------+
       |      agent_node       | <-------------------+
       |         (LLM)         |                     |
       +-----------------------+                     |
                   |                                 |
                   v                                 |
         { tools_condition }                         |
         /                 \                         |
(Có gọi tool)          (Không gọi tool               |
       /                hoặc đạt tối đa 3 lần)       |
      v                       \                      |
+-------------------+          v                     |
|     tool_node     |       [ END ]                  |
| (retrieve_docs)   |                                |
+-------------------+                                |
      |                                              |
      +----------------------------------------------+
```

Mô tả các thành phần và cạnh (edge) trong đồ thị:
- `START -> agent_node`: Điểm bắt đầu, đưa câu hỏi và lịch sử tin nhắn vào node Agent.
- `agent_node (LLM)`: Mô hình ngôn ngữ nhận danh sách tin nhắn, phân tích và đưa ra quyết định: gọi công cụ hay xuất câu trả lời.
- `tools_condition`: Cạnh điều kiện kiểm tra phản hồi từ LLM:
  - Nếu LLM tạo yêu cầu gọi tool và chưa vượt quá 3 lần: Chuyển tiếp sang `tool_node`.
  - Nếu không cần gọi tool hoặc đã đạt giới hạn 3 lần tra cứu: Chuyển sang `END`.
- `tool_node -> agent_node`: Sau khi thực thi việc tìm kiếm tài liệu, kết quả (`ToolMessage`) được gửi ngược lại về `agent_node` để LLM tiếp tục tổng hợp câu trả lời.

### 4.2 Quản lý trạng thái (AgentState)

Trạng thái của Agent được định nghĩa thông qua kiểu dữ liệu `AgentState`:
- Sử dụng danh sách tin nhắn có reducer `add_messages` để tự động nối tiếp các tin nhắn mới (`HumanMessage`, `AIMessage`, `ToolMessage`, `SystemMessage`) vào ngữ cảnh chung của cuộc trò chuyện.

### 4.3 Vòng lặp ReAct và kiểm soát số lần gọi Tool

- Khi người dùng hỏi thông tin cần đối chiếu, Agent sẽ phát lệnh gọi công cụ `retrieve_documents`.
- Để tránh việc mô hình rơi vào vòng lặp gọi công cụ liên tục khi không tìm thấy thông tin, hệ thống đặt giới hạn tối đa 3 lần gọi tool. Khi đạt đến ngưỡng này, hệ thống sẽ ép buộc mô hình dừng tìm kiếm và phản hồi bằng nội dung hiện có.

### 4.4 Hiển thị quá trình suy luận (Chain-of-Thought)

- Trong System Prompt, mô hình được hướng dẫn tạo phần suy nghĩ bên trong cặp thẻ `<think>...</think>` trước khi đưa ra câu trả lời chính thức.
- Tầng backend xử lý luồng stream (SSE) sẽ bóc tách các token trong thẻ `<think>` để gửi riêng sang client dưới dạng sự kiện suy nghĩ, giúp giao diện hiển thị phần phân tích tách biệt với câu trả lời.

### 4.5 Công cụ truy vấn tài liệu

- Tool `retrieve_documents`: Nhận vào từ khóa tìm kiếm và tên tài liệu (tùy chọn).
- Hệ thống hỗ trợ nới lỏng điều kiện lọc: Bỏ qua phân biệt hoa thường, bỏ đuôi `.pdf` và so khớp chuỗi con. Nếu lọc theo tên không có kết quả, hệ thống sẽ tự động tìm trên toàn bộ tài liệu hiện có.

### 4.6 Lưu trữ Checkpoint

- Sử dụng `AsyncSqliteSaver` để lưu trữ trạng thái của từng bước thực thi vào file SQLite (`app_data/checkpoints.db`).
- Hỗ trợ khôi phục phiên làm việc và quay lui (rollback) trạng thái hội thoại khi cần.

---

## 5. Quy trình trích xuất và RAG

Quy trình nạp tài liệu và tìm kiếm thông tin được thực hiện tuần tự qua các bước sau:

1. Tải lên tệp: Người dùng gửi tệp PDF thông qua giao diện upload.
2. Trích xuất nội dung:
   - Hệ thống ưu tiên sử dụng Gemini Vision để nhận diện nội dung văn bản và cấu trúc bảng biểu.
   - Nếu xảy ra lỗi hoặc không dùng được Vision, hệ thống chuyển sang dùng thư viện `pypdf` để đọc văn bản thông thường.
3. Chia nhỏ văn bản (Chunking):
   - Nội dung được phân đoạn theo các tiêu đề Markdown (`#`, `##`, `###`) bằng `MarkdownHeaderTextSplitter`.
   - Các đoạn sau đó tiếp tục được kiểm soát kích thước với `RecursiveCharacterTextSplitter` (khoảng 1000 ký tự mỗi đoạn, độ chồng lặp 150 ký tự).
4. Tạo Vector nhúng: Mỗi đoạn văn bản được chuyển đổi thành vector qua mô hình `text-embedding-004`.
5. Lưu trữ:
   - Lưu vector, đoạn trích và metadata (tên file, document_id) vào Milvus.
   - Lưu thông tin tệp (tên, kích thước, ngày tạo) vào SQLite.
6. Đồng bộ dữ liệu: Khi ứng dụng khởi động, hệ thống tự động kiểm tra và xóa các bản ghi lệch giữa SQLite và Milvus để đảm bảo tính nhất quán.

---

## 6. Công nghệ sử dụng

- Backend: Python, FastAPI, Uvicorn
- AI & Luồng Agent: LangGraph, LangChain, Google Gemini API
- Cơ sở dữ liệu: Milvus Standalone (kèm MinIO và etcd), SQLite
- Frontend: React, Vite, CSS thuần
- Triển khai: Docker, Docker Compose, Kubernetes

---

## 7. Danh sách API

| Phương thức | Đường dẫn | Chức năng |
| :--- | :--- | :--- |
| GET | /health | Kiểm tra tình trạng backend |
| GET | /documents | Lấy danh sách tài liệu |
| POST | /documents/upload | Tải lên tài liệu mới |
| DELETE | /documents/{id} | Xóa tài liệu khỏi database và vector store |
| POST | /chat/stream | Gửi câu hỏi và nhận stream phản hồi qua SSE |
| POST | /chat | Gửi câu hỏi (chế độ phản hồi thông thường) |
| GET | /chat/sessions | Lấy danh sách các phiên chat |
| POST | /chat/sessions | Tạo phiên chat mới |
| GET | /chat/sessions/{id}/messages | Lấy lịch sử tin nhắn của phiên chat |
| DELETE | /chat/sessions/{id} | Xóa một phiên chat |
| POST | /chat/rollback | Quay lui trạng thái hội thoại về một tin nhắn cụ thể |

---

## 8. Hướng dẫn cài đặt và khởi chạy

### 8.1 Yêu cầu môi trường
- Python phiên bản 3.10 trở lên.
- Node.js phiên bản 18 trở lên và npm.
- Docker Desktop (dùng để chạy Milvus).
- API Key của Google Gemini.

### 8.2 Cấu hình biến môi trường
Tạo file `backend/.env` với các giá trị cấu hình cơ bản:

```env
GEMINI_API_KEY=dien_api_key_tai_day
GEMINI_MODEL=gemini-2.5-flash
MILVUS_URI=http://localhost:19530
PORT=8001
HOST=0.0.0.0
```

### 8.3 Khởi chạy nhanh bằng script
Nếu dùng Windows, có thể chạy tệp `start.bat` ở thư mục gốc để khởi động tự động:
1. Chạy các container phụ trợ (Milvus, etcd, MinIO).
2. Chạy Backend tại `http://localhost:8001`.
3. Chạy Frontend tại `http://localhost:3000`.

### 8.4 Khởi chạy thủ công

Bước 1: Khởi động cơ sở dữ liệu Milvus
```bash
docker-compose up -d etcd minio standalone
```

Bước 2: Cài đặt và khởi chạy Backend
```bash
cd backend
python -m venv .venv

# Kích hoạt venv trên Windows:
.\.venv\Scripts\activate

# Cài thư viện:
pip install -r requirements.txt

# Chạy server:
uvicorn app.main:app --reload --port 8001
```

Bước 3: Cài đặt và khởi chạy Frontend
```bash
cd frontend
npm install
npm run dev
```

### 8.5 Chạy với Docker Compose và Kubernetes

Chạy toàn bộ dịch vụ qua Docker Compose:
```bash
docker-compose up --build
```

Triển khai lên cụm Kubernetes (tham khảo thêm tại file `K8S_QUICKSTART.md`):
```bash
kubectl apply -f k8s/backend-config.yaml
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
```
