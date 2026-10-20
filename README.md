# AI Documents Assistant

[![CI/CD Pipeline](https://github.com/NguyenVietNguyen2612/ai_documents_assistant/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/NguyenVietNguyen2612/ai_documents_assistant/actions/workflows/ci-cd.yml)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6F00?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-00A4E4?logo=linuxcontainers&logoColor=white)](https://milvus.io/)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Automated Tests](https://img.shields.io/badge/Tests-32%2F32%20PASS-success)](backend/auto_tests/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Dự án cá nhân** tìm hiểu và thực hành xây dựng hệ thống trợ lý hỏi đáp tài liệu thông minh (RAG) sử dụng **FastAPI**, **LangGraph ReAct Agent**, **Milvus Vector Database** và giao diện **React**. Dự án tập trung vào việc thử nghiệm các kỹ thuật cải thiện chất lượng tìm kiếm (Hybrid Search, Reranking, Window Expansion) và tối ưu trải nghiệm tương tác với tài liệu.

---

## Mục lục

1. [Giới thiệu & Các tính năng chính](#1-giới-thiệu--các-tính-năng-chính)
2. [Kiến trúc luồng xử lý (Two-System Architecture)](#2-kiến-trúc-luồng-xử-lý-two-system-architecture)
3. [Quy trình nạp tài liệu và tìm kiếm kết hợp (2-Stage Hybrid Retrieval)](#3-quy-trình-nạp-tài-liệu-và-tìm-kiếm-kết-hợp-2-stage-hybrid-retrieval)
4. [Thử nghiệm đánh giá hiệu quả (Ablation Study)](#4-thử-nghiệm-đánh-giá-hiệu-quả-ablation-study)
5. [Các lựa chọn thiết kế kỹ thuật](#5-các-lựa-chọn-thiết-kế-kỹ-thuật)
6. [Các công cụ của Agent (Multi-Tool Ecosystem)](#6-các-công-cụ-của-agent-multi-tool-ecosystem)
7. [Kết quả kiểm thử tự động (32/32 tests)](#7-kết-quả-kiểm-thử-tự-động-3232-tests)
8. [Danh mục REST API](#8-danh-mục-rest-api)
9. [Hướng dẫn cài đặt & khởi chạy](#9-hướng-dẫn-cài-đặt--khởi-chạy)
10. [Triển khai với Docker & Kubernetes](#10-triển-khai-với-docker--kubernetes)

---

## 1. Giới thiệu & Các tính năng chính

Dự án được xây dựng nhằm mục đích học tập các khái niệm hiện đại trong xây dựng ứng dụng AI với mô hình Agentic Workflow (LangGraph) thay vì chuỗi RAG cố định, đồng thời áp dụng các giải pháp thực tế để xử lý các vấn đề thường gặp khi làm việc với tài liệu:

- **Hỏi đáp thông minh theo ngữ cảnh tài liệu**:
  - Hỗ trợ tải lên và xử lý nhiều định dạng: `.pdf`, `.csv`, `.md`, `.txt`.
  - Phân tích văn bản kết hợp OCR hình ảnh/bảng biểu (Gemini Vision) và trích xuất cấu trúc đề mục Markdown.
  - Lưu trữ số trang thực tế của từng đoạn văn bản để người dùng có thể nhấp trực tiếp vào nguồn trích dẫn và mở đúng trang PDF tương ứng.
- **Định tuyến 2 cấp (Two-System Routing)**:
  - **Lớp định tuyến nhanh (System 1)**: Nhận diện các câu chào hỏi hoặc câu hỏi thông thường để phản hồi trực tiếp mà không cần khởi tạo luồng ReAct phức tạp.
  - **Lớp suy luận đa bước (System 2)**: Sử dụng LangGraph StateGraph cho các câu hỏi cần tra cứu tài liệu, tính toán hoặc tổng hợp thông tin qua nhiều bước.
- **Tìm kiếm hỗn hợp 2 giai đoạn (2-Stage Hybrid Search)**:
  - Kết hợp Dense Vector Search (Milvus Cosine) với Lexical Search (BM25) thông qua thuật toán Reciprocal Rank Fusion (RRF).
  - Tái chấm điểm và xếp hạng lại (Reranking) top kết quả bằng mô hình Cross-Encoder (`BAAI/bge-reranker-base`) để chọn ra các đoạn văn liên quan nhất.
- **Mở rộng cửa sổ ngữ cảnh (Small-to-Big Window Expansion)**:
  - Tự động ghép nối các đoạn văn lân cận ($i-1, i, i+1$) xung quanh đoạn trích có điểm số cao nhất nhằm tránh mất mát ngữ cảnh ở ranh giới cắt chunk.
- **Cơ chế gợi ý viết lại truy vấn (Corrective RAG)**:
  - Khi điểm tương quan của tài liệu tìm được ở mức thấp, hệ thống kích hoạt cảnh báo để Agent thực hiện diễn đạt lại câu hỏi (Query Rewriting) hoặc dịch thuật ngữ sang tiếng Anh học thuật để thử tra cứu lại.
- **Bộ nhớ đệm ngữ nghĩa (Semantic Cache)**:
  - Lưu trữ các cặp câu hỏi - câu trả lời trên Milvus; khi người dùng đặt câu hỏi có ý nghĩa tương đương ($\ge 0.95$), hệ thống có thể phản hồi nhanh từ cache mà không cần gọi lại LLM.
- **Xử lý tài liệu ngầm (Async Background Ingestion)**:
  - Quá trình đọc file, OCR, chia đoạn và tạo vector được đưa vào `BackgroundTasks` của FastAPI.
  - Phản hồi HTTP ngay lập tức và hỗ trợ giao diện polling trạng thái xử lý theo thời gian thực (2 giây/lần).
- **Trải nghiệm giao diện (UI/UX)**:
  - Hỗ trợ gắn thẻ `@tài_liệu` để giới hạn phạm vi tìm kiếm vào một tài liệu cụ thể.
  - Tách luồng suy luận `<think>...</think>` (Chain-of-Thought) hiển thị riêng biệt qua Server-Sent Events (SSE).
  - Hiển thị thanh thông số kỹ thuật (Telemetry) dưới mỗi câu trả lời: thời gian xử lý, điểm rerank, số bước gọi công cụ và nguồn trích dẫn.

---

## 2. Kiến trúc luồng xử lý (Two-System Architecture)

Luồng điều phối hội thoại kết hợp giữa lớp phân loại nhanh và đồ thị trạng thái LangGraph:

```mermaid
flowchart TD
    User["👤 Người dùng / Giao diện React"] --> API["🌐 FastAPI Server (/chat/stream)"]

    subgraph System1 ["⚡ LỚP ĐỊNH TUYẾN NHANH (System 1)"]
        API --> S1_Router{"Bộ phân loại Laya / Heuristic"}
        S1_Router -- "Câu chào hỏi / Xã giao" --> S1_Direct["💬 Trả lời trực tiếp"]
        S1_Router -- "Yêu cầu tra cứu / Phân tích" --> S1_Cache{"Kiểm tra Semantic Cache (Milvus)"}
        S1_Cache -- "Khớp ngữ nghĩa (≥ 0.95)" --> S1_CacheHit["⚡ Trả kết quả từ Cache"]
    end

    subgraph System2 ["🧠 ĐỒ THỊ SUY LUẬN REACT (System 2)"]
        S1_Cache -- "Chưa có trong Cache" --> AgentNode["🤖 agent_node (Mô hình LLM)"]
        
        AgentNode --> ToolCond{"tools_condition (Tối đa 3 lượt)"}
        
        ToolCond -- "Cần gọi công cụ" --> ToolNode["🛠️ tool_node"]
        
        subgraph ToolsCatalog ["Danh mục công cụ (backend/app/tools/)"]
            ToolNode --> T1["🔍 retrieve_documents (Hybrid Search + Mở rộng cửa sổ)"]
            ToolNode --> T2["📋 list_uploaded_documents"]
            ToolNode --> T3["📄 get_document_overview"]
            ToolNode --> T4["🧮 calculate_expression (Bộ phân tích cú pháp AST)"]
        end

        ToolsCatalog --> CRAG_Eval{"Đánh giá độ tin cậy"}
        CRAG_Eval -- "Điểm thấp / Không tìm thấy" --> QueryRewrite["🔄 Gợi ý viết lại truy vấn"]
        QueryRewrite --> AgentNode
        CRAG_Eval -- "Điểm tin cậy cao" --> ToolMsg["📝 Trả về ToolMessage"]
        ToolMsg --> AgentNode
        
        ToolCond -- "Hoàn thành câu trả lời" --> Pruner["✂️ Cắt tỉa ngữ cảnh cũ & Sliding Window"]
        Pruner --> FinalStream["📡 Stream SSE: Nội dung + Suy nghĩ + Telemetry"]
    end

    FinalStream --> User
    S1_Direct --> User
    S1_CacheHit --> User

    style System1 fill:#f0fdf4,stroke:#16a34a,stroke-width:1.5px,color:#14532d
    style System2 fill:#eff6ff,stroke:#2563eb,stroke-width:1.5px,color:#1e3a8a
```

### Các thành phần chính trong đồ thị:
1. **`agent_node`**: Tiếp nhận lịch sử tin nhắn và yêu cầu người dùng, hướng dẫn mô hình suy luận trong thẻ `<think>...</think>` trước khi đưa ra hành động.
2. **`tools_condition`**: Kiểm tra phản hồi của LLM để quyết định chuyển sang `tool_node` hay kết thúc đồ thị (giới hạn tối đa 3 lần gọi công cụ để tránh lặp vô tận).
3. **`tool_node`**: Điều hướng thực thi công cụ tương ứng và gửi kết quả về lại cho Agent.
4. **`Cắt tỉa ngữ cảnh & Sliding Window`**: Nén ngắn các `ToolMessage` từ những lượt hội thoại trước về dạng tóm tắt ngắn ($\le 150$ ký tự) và giữ tối đa 16 tin nhắn gần nhất để kiểm soát số lượng token gửi vào LLM.
5. **`Lưu trữ Checkpoint`**: Sử dụng `AsyncSqliteSaver` để lưu snapshot từng bước vào SQLite, cho phép khôi phục phiên chat và quay lui trạng thái (Rollback).

---

## 3. Quy trình nạp tài liệu và tìm kiếm kết hợp (2-Stage Hybrid Retrieval)

```mermaid
flowchart LR
    subgraph Ingestion ["📥 TIẾP NHẬN & XỬ LÝ TÀI LIỆU"]
        File["📄 Tệp tải lên (.pdf, .csv, .md, .txt)"] --> DocRoute["/documents/upload"]
        DocRoute --> BG["⚙️ FastAPI BackgroundTasks"]
        BG --> Extract["Trích xuất: Gemini Vision OCR + pypdf"]
        Extract --> Chunking["Phân đoạn theo tiêu đề Markdown + Số trang"]
        Chunking --> Embed["Tạo Vector: text-embedding-004"]
        Chunking --> BM25_Idx["Lập chỉ mục từ khóa: BM25 (SQLite)"]
        Embed --> Milvus_Idx["Lưu trữ Vector: Milvus DB (HNSW Index)"]
    end

    subgraph Retrieval ["🔎 TRUY VẤN 2 GIAI ĐOẠN"]
        Query["❓ Câu hỏi"] --> Dense["Dense Search (Milvus Cosine)"]
        Query --> Sparse["Sparse Search (BM25 Lexical)"]
        Dense --> RRF["Hợp nhất thứ hạng RRF (k=60)"]
        Sparse --> RRF
        RRF --> Candidates["Top-10 ứng viên ban đầu"]
        Candidates --> Rerank["Tái xếp hạng: BGE Cross-Encoder"]
        Rerank --> TopK["Top-3 đoạn trích phù hợp nhất"]
        TopK --> WinExp["Mở rộng cửa sổ lân cận (Chunks i-1, i, i+1)"]
        WinExp --> FinalContext["Ngữ cảnh hoàn chỉnh đưa vào LLM"]
    end

    style Ingestion fill:#fefce8,stroke:#ca8a04,stroke-width:1.5px,color:#713f12
    style Retrieval fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,color:#4c1d95
```

- **Quy trình nạp tài liệu**:
  1. Khi người dùng tải tệp lên, hệ thống phản hồi ngay mã tài liệu và trạng thái `processing`.
  2. Tác vụ ngầm thực hiện OCR nhận diện văn bản, cấu trúc bảng biểu và trích xuất số trang.
  3. Nội dung được chia đoạn theo phân cấp tiêu đề Markdown (`#`, `##`, `###`) để giữ ngữ cảnh phân mục.
  4. Lưu đồng thời vector vào Milvus và từ khóa vào chỉ mục BM25 cục bộ.
- **Hợp nhất kết quả bằng RRF**:
  Sử dụng công thức Reciprocal Rank Fusion với hằng số $k = 60$ để kết hợp thứ hạng từ Dense Vector và BM25:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + \text{rank}_m(d)}$$
- **Reranker với Cross-Encoder**:
  Mô hình `BAAI/bge-reranker-base` chấm điểm trực tiếp cặp `(Câu hỏi, Đoạn trích)` để chọn ra top kết quả có độ tương quan ngữ nghĩa cao nhất.
- **Mở rộng cửa sổ (Window Expansion)**:
  Lấy thêm các đoạn liền kề trước và sau đoạn trích tốt nhất giúp mô hình nắm được câu văn hoàn chỉnh, hạn chế việc bị cắt đứt giữa câu hoặc mất phần giải thích công thức.

---

## 4. Thử nghiệm đánh giá hiệu quả (Ablation Study)

Để kiểm tra xem việc kết hợp BM25 và Cross-Encoder có thực sự cải thiện khả năng tìm kiếm so với chỉ dùng Vector đơn thuần hay không, dự án thực hiện đo lường trên bộ câu hỏi thử nghiệm ([`backend/evaluation/dataset.json`](backend/evaluation/dataset.json)) với các dạng truy vấn khác nhau (từ khóa đặc thù, tên riêng, câu hỏi ngữ nghĩa):

| Phương pháp truy xuất | Hit Rate@1 | Hit Rate@3 | MRR@3 | Độ phủ từ khóa | Thời gian trung bình |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Chỉ dùng Dense Vector (Milvus)** | 90.0% | 90.0% | 0.900 | 96.0% | **~38 ms** |
| **Tìm kiếm kết hợp (Milvus + BM25 RRF)** | 90.0% | **100.0%** | 0.950 | **100.0%** | ~47 ms |
| **Kết hợp + Tái xếp hạng (Hybrid + BGE Reranker)** | **100.0%** | **100.0%** | **1.000** | **100.0%** | ~142 ms |

> [!NOTE]
> **Ghi nhận từ kết quả thử nghiệm:**
> - Phương pháp chỉ dùng Vector gặp khó khăn ở các câu hỏi chứa mã định danh học thuật chính xác (ví dụ các tên viết tắt kỹ thuật) do vector embedding có xu hướng dàn đều ngữ nghĩa.
> - Khi bổ sung BM25 thông qua RRF, toàn bộ tài liệu mục tiêu đều xuất hiện trong Top-3 kết quả tìm kiếm.
> - Bước tái xếp hạng bằng BGE Cross-Encoder giúp đưa đoạn trích chính xác nhất lên vị trí Top-1 trong các ca thử nghiệm, tuy nhiên thời gian phản hồi tăng thêm khoảng 95ms do mô hình Cross-Encoder cần xử lý qua nhiều lớp attention.

---

## 5. Các lựa chọn thiết kế kỹ thuật

Trong quá trình phát triển, một số giải pháp kỹ thuật đã được cân nhắc để cân đối giữa chi phí, thời gian xử lý và độ chính xác:

1. **Lớp định tuyến nhanh (System 1)**:
   - Các câu chào hỏi hoặc giao tiếp ngắn không cần thiết phải chạy qua chuỗi suy luận ReAct. Việc phân loại sớm giúp phản hồi ngay cho người dùng và giảm bớt số lần gọi API bên ngoài.
2. **Sử dụng thuật toán RRF thay vì chuẩn hóa điểm**:
   - Điểm số Cosine Similarity và điểm BM25 có phân phối và biên độ khác nhau. Thay vì cố gắng co giãn điểm số vốn dễ bị ảnh hưởng bởi giá trị ngoại lai, RRF dựa trên thứ tự xếp hạng giúp kết quả ổn định hơn.
3. **Mở rộng cửa sổ ngữ cảnh (Window Expansion)**:
   - Cắt đoạn văn bản ngắn (khoảng 800 - 1000 ký tự) giúp vector nhạy hơn trong việc nắm bắt ý chính. Sau khi tìm được vị trí phù hợp, việc kéo thêm đoạn trước và đoạn sau vào prompt giúp câu trả lời của mô hình có đủ ngữ cảnh mà không làm giảm độ chính xác lúc tìm kiếm.
4. **Bộ nhớ đệm ngữ nghĩa (Semantic Cache)**:
   - Đối với các câu hỏi tương tự nhau về mặt ngữ nghĩa (ngưỡng Cosine $\ge 0.95$), hệ thống có thể trả lời trực tiếp từ cache. Cache sẽ tự động được xóa khi có tài liệu mới tải lên hoặc bị xóa để tránh trả lời bằng dữ liệu cũ.
5. **Cắt tỉa ngữ cảnh cũ (ToolMessage Pruning)**:
   - Trong các cuộc trò chuyện kéo dài, kết quả tra cứu của các lượt chat trước chiếm rất nhiều dung lượng token. Rút gọn các kết quả cũ về dạng tóm tắt ngắn giúp giữ cho cửa sổ ngữ cảnh luôn nằm trong giới hạn hợp lý.
6. **Xử lý tài liệu ngầm (Background Ingestion)**:
   - Việc OCR và tạo embedding cho các tài liệu dài có thể mất từ 15 đến 30 giây. Việc đưa công việc này vào background task giúp người dùng không phải chờ đợi ở màn hình upload và có thể tiếp tục thao tác khác.

---

## 6. Các công cụ của Agent (Multi-Tool Ecosystem)

Các công cụ được tổ chức tách biệt tại thư mục [`backend/app/tools/`](backend/app/tools/):

- **`retrieve_documents`**: Tra cứu đoạn văn bản liên quan trong tài liệu (kết hợp Milvus, BM25, Reranker và mở rộng cửa sổ).
- **`list_uploaded_documents`**: Liệt kê danh sách các tài liệu hiện có kèm số lượng trang và trạng thái.
- **`get_document_overview`**: Lấy thông tin tóm tắt và cấu trúc các đề mục chính của một tài liệu cụ thể.
- **`calculate_expression`**: Hỗ trợ tính toán các phép toán số học (`+`, `-`, `*`, `/`, `**`, `%`, `round`). Sử dụng thư viện `ast` (Abstract Syntax Tree) của Python để phân tích cú pháp an toàn, không sử dụng hàm `eval()`.

---

## 7. Kết quả kiểm thử tự động (32/32 tests)

Dự án có sẵn bộ script kiểm thử tự động để kiểm tra các thành phần từ backend API, dịch vụ xử lý tài liệu đến luồng stream SSE:

```bash
# Thực thi toàn bộ kiểm thử
python backend/auto_tests/run_all_tests.py
```

| STT | Tập tin kiểm thử | Số test | Nội dung kiểm tra | Trạng thái |
| :---: | :--- | :---: | :--- | :---: |
| 1 | [`test_unit_retriever_bm25.py`](backend/auto_tests/test_unit_retriever_bm25.py) | **7** | Tìm kiếm Milvus, BM25, RRF, Reranker, Window Expansion, Corrective RAG | ✅ PASS |
| 2 | [`test_unit_semantic_cache_and_pruning.py`](backend/auto_tests/test_unit_semantic_cache_and_pruning.py) | **4** | Semantic Cache trên Milvus, xóa cache, cắt tỉa ToolMessage và Sliding Window | ✅ PASS |
| 3 | [`test_integration_api.py`](backend/auto_tests/test_integration_api.py) | **6** | API upload ngầm, polling trạng thái, xóa file, phạm vi `@tài_liệu`, quản lý phiên | ✅ PASS |
| 4 | [`test_integration_chat_stream.py`](backend/auto_tests/test_integration_chat_stream.py) | **3** | Luồng stream SSE, định dạng chunk suy nghĩ `<think>`, sự kiện telemetry | ✅ PASS |
| 5 | [`test_tools.py`](backend/auto_tests/test_tools.py) | **5** | Khả năng kích hoạt công cụ của Agent, máy tính AST an toàn, Document Overview | ✅ PASS |
| 6 | [`test_unit_laya_router.py`](backend/auto_tests/test_unit_laya_router.py) | **4** | Phân loại câu chào hỏi, ngưỡng tự tin tối thiểu (guardrail), fallback | ✅ PASS |
| 7 | [`test_unit_db_documents.py`](backend/auto_tests/test_unit_db_documents.py) | **3** | Cập nhật cấu trúc SQLite, đồng bộ dữ liệu giữa SQLite và Milvus | ✅ PASS |
| **Tổng cộng** | **7 Modules** | **32 tests** | **Kiểm tra chức năng toàn diện** | **32/32 (100%)** |

---

## 8. Danh mục REST API

| Phương thức | Đường dẫn | Chức năng |
| :--- | :--- | :--- |
| `GET` | `/health` | Kiểm tra trạng thái hoạt động của backend |
| `GET` | `/documents` | Lấy danh sách tài liệu kèm trạng thái (`processing`, `ready`, `error`) |
| `POST` | `/documents/upload` | Tải lên tài liệu mới (.pdf, .csv, .md, .txt) và xử lý ở chế độ nền |
| `GET` | `/documents/{id}/status` | Kiểm tra tiến trình xử lý của tài liệu |
| `GET` | `/documents/{id}/view` | Xem nội dung hoặc tải file tài liệu để hiển thị |
| `DELETE` | `/documents/{id}` | Xóa tài liệu khỏi database, vector store và xóa cache liên quan |
| `POST` | `/chat/stream` | Gửi câu hỏi và nhận stream phản hồi qua SSE (hỗ trợ `@tài_liệu`, CoT, Telemetry) |
| `POST` | `/chat` | Gửi câu hỏi nhận phản hồi dạng JSON thông thường |
| `POST` | `/chat/cache/clear` | Xóa dữ liệu cache ngữ nghĩa trên Milvus |
| `GET` | `/chat/sessions` | Lấy danh sách các phiên trò chuyện |
| `POST` | `/chat/sessions` | Tạo phiên trò chuyện mới |
| `GET` | `/chat/sessions/{id}/messages` | Lấy lịch sử tin nhắn của một phiên |
| `DELETE` | `/chat/sessions/{id}` | Xóa phiên trò chuyện và các checkpoint tương ứng |
| `POST` | `/chat/rollback` | Quay lui trạng thái hội thoại về một tin nhắn cụ thể |

---

## 9. Hướng dẫn cài đặt & khởi chạy

### Yêu cầu môi trường:
- Python 3.10 trở lên (khuyến nghị 3.12).
- Node.js 18 trở lên và npm.
- Docker Desktop (dùng để chạy cơ sở dữ liệu Milvus).
- Khóa API của Google Gemini.

### Bước 1: Cấu hình biến môi trường
Tạo tệp `backend/.env` với các thông tin cơ bản:
```env
GEMINI_API_KEY=dien_api_key_cua_ban_o_day
GEMINI_MODEL=gemini-2.5-flash
MILVUS_URI=http://localhost:19530
PORT=8001
HOST=0.0.0.0
```

### Bước 2: Khởi chạy nhanh bằng script

- **Trên Linux / macOS**:
  ```bash
  chmod +x start.sh
  ./start.sh
  ```
- **Trên Windows**:
  Chạy tệp `start.bat` từ Command Prompt hoặc PowerShell:
  ```cmd
  start.bat
  ```

*Script sẽ tự động khởi động các container phụ trợ của Milvus, chuẩn bị môi trường ảo Python, cài đặt thư viện và mở Backend (cổng 8001) cùng Frontend (cổng 3000).*

### Bước 3: Khởi chạy thủ công từng phần

1. **Khởi động Milvus qua Docker**:
   ```bash
   docker compose up -d standalone
   ```

2. **Cài đặt và chạy Backend**:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate       # Trên Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload --port 8001
   ```

3. **Cài đặt và chạy Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

---

## 10. Triển khai với Docker & Kubernetes

### Chạy bằng Docker Compose:
```bash
docker compose up --build
```
Lệnh này sẽ khởi động đầy đủ các dịch vụ gồm Milvus (kèm etcd, MinIO), Backend FastAPI và Frontend React.

### Triển khai trên Kubernetes:
Các tệp cấu hình triển khai cơ bản nằm trong thư mục [`k8s/`](k8s/):
```bash
# 1. Áp dụng ConfigMap
kubectl apply -f k8s/backend-config.yaml

# 2. Triển khai Backend
kubectl apply -f k8s/backend-deployment.yaml

# 3. Triển khai Frontend
kubectl apply -f k8s/frontend-deployment.yaml
```

---

## Giấy phép (License)
Dự án được phân phối dưới giấy phép mã nguồn mở **MIT License**.
