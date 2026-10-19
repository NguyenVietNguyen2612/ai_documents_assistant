# 📊 Báo cáo Thực nghiệm Ablation Study: Kiến trúc Retrieval 2-Stage Hybrid + Reranking

> **Bối cảnh thực nghiệm:** Đánh giá định lượng trên bộ dữ liệu chuẩn **CS338 RAG Evaluation Dataset** (`10 câu hỏi chuyên sâu` trên tập Corpus `20 chunks` gồm 10 Ground-Truth chunks từ bài báo *Adaptive FreqNet* và 10 Hard-Negative Distractor chunks có độ chồng lấn ngữ nghĩa cao).

---

## 1. Bảng So sánh Tổng hợp (Quantitative Benchmark Summary)

| Kiến trúc Truy xuất (Retrieval Mode) | Hit Rate@1 (Top-1 Acc) | Hit Rate@3 (Recall@3) | MRR@3 (Mean Reciprocal Rank) | Keyword Coverage (%) | Avg Latency (ms) | P95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Dense Vector Only** (`paraphrase-multilingual-MiniLM-L12-v2`) | 90.0% | 90.0% | `0.900` | 96.0% | `212.7 ms` | `642.8 ms` |
| **2. Hybrid Search (Milvus Vector + BM25 + RRF)** | 90.0% | 100.0% | `0.950` | 100.0% | `230.0 ms` | `658.1 ms` |
| **3. 2-Stage Hybrid + BGE Cross-Encoder Reranker** (`BAAI/bge-reranker-base`) | **100.0%** | **100.0%** | **`1.000`** | **100.0%** | `2431.9 ms` | `3800.0 ms` |

---

## 2. Phân tích Chi tiết theo Từng Câu hỏi (Per-Question Rank Breakdown)

*Chú thích: `Rank #1` = Đoạn văn bản chuẩn nằm ngay vị trí đầu tiên; `Rank #2/#3` = Nằm trong Top-3; `Miss` = Không lọt vào Top-3.*

| ID | Loại câu hỏi (`test_type`) | Câu hỏi | Vector Only Rank | Hybrid RRF Rank | Hybrid + BGE Rerank | Latency (2-Stage) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| `q1` | `basic_retrieval` | What is the main problem addressed in this document? | ❌ Miss | Rank #2 | **Rank #1** | `3800.0 ms` |
| `q2` | `direct_retrieval` | What is FreqNet and what problem is it designed to solve? | **Rank #1** | **Rank #1** | **Rank #1** | `2892.6 ms` |
| `q3` | `fact_retrieval` | What are the three main processing blocks of FreqNet? | **Rank #1** | **Rank #1** | **Rank #1** | `2078.1 ms` |
| `q4` | `technical_reasoning` | What is the main technical limitation of the original FreqNet identified in the document? | **Rank #1** | **Rank #1** | **Rank #1** | `2301.2 ms` |
| `q5` | `multi_section_retrieval` | How does Adaptive Soft Masking address the limitation of hard masking in FreqNet? | **Rank #1** | **Rank #1** | **Rank #1** | `2090.9 ms` |
| `q6` | `specific_fact` | How is the adaptive frequency mask defined for low- and high-frequency regions? | **Rank #1** | **Rank #1** | **Rank #1** | `2584.0 ms` |
| `q7` | `technical_fact` | How is the filtered frequency spectrum calculated in the proposed method? | **Rank #1** | **Rank #1** | **Rank #1** | `2469.7 ms` |
| `q8` | `numerical_fact` | What dataset was used in the experiments, and how was it divided? | **Rank #1** | **Rank #1** | **Rank #1** | `2261.6 ms` |
| `q9` | `comparison` | How does the proposed method compare with the original FreqNet in terms of parameters and accuracy? | **Rank #1** | **Rank #1** | **Rank #1** | `1834.5 ms` |
| `q10` | `out_of_context` | Was weight decay used to train the model? | **Rank #1** | **Rank #1** | **Rank #1** | `2006.1 ms` |

---

## 3. Nhận xét Kỹ thuật & Lý giải Kiến trúc (Engineering Takeaways)

1. **Điểm mù của Dense Vector Search thuần túy (`vector_only`):**
   - Mô hình Bi-Encoder nén toàn bộ câu hỏi và đoạn văn thành một vector 384 chiều duy nhất. Khi gặp các câu hỏi chứa **ký hiệu toán học, tên khối viết tắt (`HFRI`, `HFRF`, `FCL`, `sigma(Wm(i,j))`) hoặc số liệu cụ thể (`152,710`, `72.16%`)**, Dense Vector dễ bị nhiễu bởi các đoạn văn Hard-Negative cùng chủ đề tần số/Deepfake.
2. **Vai trò của Hybrid Search + Reciprocal Rank Fusion (`hybrid_rrf`):**
   - Sự kết hợp giữa **BM25 Okapi** (bắt chính xác tần suất từ khóa hiếm/công thức) và **Milvus Cosine Similarity** thông qua công thức hợp nhất thứ hạng $\text{RRF}(d) = \sum \frac{1}{60 + \text{rank}(d)}$ giúp kéo các đoạn chứa thuật ngữ chuyên ngành vào tập ứng viên Top-K, cải thiện mạnh **Keyword Coverage** và **Recall@3** mà chỉ tăng thêm vài mili-giây xử lý.
3. **Đột phá từ BGE Cross-Encoder Reranker ở Giai đoạn 2 (`hybrid_rerank`):**
   - Thay vì tính tích vô hướng độc lập như Bi-Encoder, mô hình **Cross-Encoder (`BAAI/bge-reranker-base`)** đưa đồng thời cặp `(Query, Candidate Chunk)` qua toàn bộ các tầng Self-Attention để chấm điểm tương tác sâu.
   - Kết quả thực nghiệm cho thấy kiến trúc 2-Stage đẩy **MRR@3 lên `1.000`** và **Hit Rate@1 đạt `100.0%`**, đảm bảo các đoạn văn chính xác nhất luôn nằm ở vị trí đầu tiên trước khi đưa vào LLM, giảm thiểu tối đa hiện tượng *Lost-in-the-Middle* và *Hallucination*.

## 4. Hướng dẫn Tái lập Thực nghiệm (Reproducibility)

```bash
cd backend
./.venv/bin/python evaluation/ablation_study.py
```
