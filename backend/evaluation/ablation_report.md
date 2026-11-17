# 📊 Báo cáo Thực nghiệm Ablation Study: Kiến trúc Retrieval 2-Stage Hybrid + Reranking

> **Bối cảnh thực nghiệm:** Đánh giá định lượng trên bộ dữ liệu chuẩn quốc tế **BEIR `mteb/scifact`** (split `test`): `100 câu truy vấn` (`116 cặp ground-truth qrels`) trên toàn bộ kho văn bản `5183 tài liệu khoa học`. Cấu hình truy xuất: `top_k=5` (Giai đoạn 1 thu hồi `initial_k=15` ứng viên), Vector Backend: `In-Memory Cosine (NumPy Fallback)`.

---

## 1. Bảng So sánh Tổng hợp (Quantitative Benchmark Summary)

| Kiến trúc Truy xuất (Retrieval Mode) | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR@5 | nDCG@5 | Recall@5 (%) | Avg Latency (ms) | P95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Dense Vector Only** (`paraphrase-multilingual-MiniLM-L12-v2`) | 43.00% | 51.00% | 57.00% | `0.4763` | `0.4891` | 55.58% | `156.9 ms` | `427.4 ms` |
| **2. Hybrid Search (Dense Vector + BM25 + RRF)** | 52.00% | 73.00% | 81.00% | `0.6265` | `0.6560` | 78.48% | `213.0 ms` | `696.1 ms` |
| **3. 2-Stage Hybrid + BGE Cross-Encoder Reranker** (`BAAI/bge-reranker-base`) | **65.00%** | **80.00%** | **87.00%** | **`0.7333`** | **`0.7516`** | **84.03%** | `5784.4 ms` | `7841.9 ms` |

---

## 2. Phân tích Thắng / Hòa / Thua của Reranker (Rank Improvement Analysis)

| So sánh đối chiếu | Cải thiện thứ hạng (Win) | Giữ nguyên (Tie) | Tụt hạng (Loss) |
| :--- | :---: | :---: | :---: |
| **Hybrid + Reranker** vs **Dense Vector Only** | **44** | 51 | 5 |
| **Hybrid + Reranker** vs **Hybrid RRF** | **28** | 61 | 11 |

### Các ca điển hình Reranker cải thiện thứ hạng (Top Case Studies)

| Query ID | Câu truy vấn | Vector Only | Hybrid RRF | Hybrid + BGE Rerank | Latency (2-Stage) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `238` | Cells undergoing methionine restriction may activate miRNAs. | ❌ Miss | ❌ Miss | **Rank #1** | `5552.8 ms` |
| `127` | Arginine 90 in p150n is important for interaction with EB1. | ❌ Miss | Rank #5 | **Rank #1** | `5461.3 ms` |
| `249` | Chenodeosycholic acid treatment reduces whole-body energy expenditure. | ❌ Miss | Rank #5 | **Rank #1** | `5641.9 ms` |
| `295` | Crosstalk between dendritic cells (DCs) and innate lymphoid cells (ILCs) is important in the regu... | ❌ Miss | Rank #5 | **Rank #1** | `5535.7 ms` |
| `440` | Fz/PCP-dependent Pk localizes to the anterior membrane of notochord cells during zebrafish neural... | ❌ Miss | Rank #5 | **Rank #1** | `5476.0 ms` |
| `70` | Activation of PPM1D suppresses p53 function. | ❌ Miss | ❌ Miss | Rank #2 | `7969.9 ms` |
| `128` | Arterioles have a larger lumen diameter than venules. | ❌ Miss | ❌ Miss | Rank #2 | `5711.0 ms` |
| `300` | Cytosolic proteins bind to iron-responsive elements on mRNAs coding for DMT1. Cytosolic proteins ... | ❌ Miss | ❌ Miss | Rank #2 | `5482.2 ms` |
| `324` | Deleting Raptor reduces G-CSF levels. | ❌ Miss | ❌ Miss | Rank #2 | `5617.0 ms` |
| `5` | 1/2000 in UK have abnormal PrP positivity. | ❌ Miss | Rank #3 | **Rank #1** | `6991.7 ms` |

---

## 3. Nhận xét Kỹ thuật & Lý giải Kiến trúc (Engineering Takeaways)

1. **Hạn chế của Bi-Encoder Vector Search đơn lẻ (`vector_only`):**
   - Trên tập dữ liệu học thuật `mteb/scifact` với `5183` tài liệu, Bi-Encoder đa ngữ (`paraphrase-multilingual-MiniLM-L12-v2`) đạt `Hit@1 = 43.00%` và `nDCG@5 = 0.4891` do việc nén toàn bộ đoạn văn thành một vector cố định làm mờ các thuật ngữ chuyên ngành chính xác.
2. **Hiệu quả thu hồi từ Hybrid Search + RRF (`hybrid_rrf`):**
   - Kết hợp **BM25 Okapi** (khớp chính xác thuật ngữ hiếm/định danh khoa học) và Dense Vector giúp tăng `Recall@5` từ `55.58%` lên `78.48%` và `nDCG@5` lên `0.6560`.
3. **Giá trị của BGE Cross-Encoder Reranker ở Giai đoạn 2 (`hybrid_rerank`):**
   - Cross-Encoder (`BAAI/bge-reranker-base`) đọc đồng thời cặp `(Query, Document)` qua toàn bộ các tầng Self-Attention, đẩy `Hit@1` lên **`65.00%`** (tăng `+13.00` điểm % so với Hybrid RRF) và `nDCG@5` đạt **`0.7516`** (`+9.56` điểm % so với Hybrid RRF, `+26.25` điểm % so với Vector Only).

## 4. Hướng dẫn Tái lập Thực nghiệm (Reproducibility)

```bash
cd backend
./.venv/bin/python evaluation/ablation_study.py --dataset mteb/scifact --split test --top-k 5 --max-queries 100
```
