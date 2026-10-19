import os
from app.services.reranker_service import RerankerService


class ContextBuilder:
    """
    Xây dựng ngữ cảnh cho Agent từ danh sách chunks thu hồi được,
    đồng thời chuẩn hóa điểm tin cậy về thang [0.00 - 1.00] cho cơ chế Corrective RAG:
    - Cao (>= 0.70): Ngữ cảnh khớp rất tốt.
    - Trung bình (0.40 - 0.69): Ngữ cảnh có liên quan một phần.
    - Thấp (< 0.40): Ngữ cảnh độ tin cậy thấp -> Kích hoạt cảnh báo Corrective RAG (Query Rewriting).
    """

    HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_HIGH_CONFIDENCE_THRESHOLD", "0.70"))
    LOW_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.40"))

    def build(self, results) -> str:
        if not results or not results[0]:
            return ""

        contexts = []
        max_norm_score = None

        for i, result in enumerate(
            results[0],
            start=1,
        ):
            entity = result.get("entity", {})
            doc_name = entity.get("document_name", "Unknown")
            page = entity.get("page", 1)
            page_range = entity.get("page_range", str(page))
            chunk_range = entity.get("chunk_range", str(entity.get("chunk_index", 0)))
            has_expanded = entity.get("has_expanded", False)
            text = entity.get("expanded_text") or entity.get("text", "")

            raw_score = result.get(
                "normalized_score",
                result.get("rerank_score", result.get("score", result.get("distance"))),
            )
            score_str = ""
            if raw_score is not None:
                norm_score = RerankerService.normalize_score(float(raw_score))
                if norm_score >= self.HIGH_CONFIDENCE_THRESHOLD:
                    level = "Cao"
                elif norm_score >= self.LOW_CONFIDENCE_THRESHOLD:
                    level = "Trung bình"
                else:
                    level = "Thấp"
                score_str = f" | Độ tin cậy: {norm_score:.2f}/1.00 ({level})"
                if max_norm_score is None or norm_score > max_norm_score:
                    max_norm_score = norm_score

            window_tag = f" | Cửa sổ Small-to-Big: Chunks {chunk_range}" if has_expanded else ""
            context = f"""[Source {i}] [Tài liệu: {doc_name} - Trang {page_range}] ({score_str.lstrip(' | ') if score_str else ''}{window_tag})
Document: {doc_name}
Page: {page_range}
{f'Chunks Window: {chunk_range}' if has_expanded else f'Chunk Index: {chunk_range}'}

{text}
"""
            contexts.append(context)

        joined_context = "\n".join(contexts)

        # Corrective RAG: Đánh giá độ tin cậy và kích hoạt phản xạ tự sửa truy vấn (Self-Correction)
        if max_norm_score is not None and max_norm_score < self.LOW_CONFIDENCE_THRESHOLD:
            joined_context += (
                f"\n[ĐÁNH GIÁ CORRECTIVE RAG - CẢNH BÁO ĐỘ TIN CẬY THẤP]:\n"
                f"Độ tin cậy ngữ cảnh cao nhất chỉ đạt {max_norm_score:.2f}/1.00 (thấp hơn ngưỡng chuẩn {self.LOW_CONFIDENCE_THRESHOLD:.2f}).\n"
                "HÀNH ĐỘNG BẮT BUỘC CỦA AGENT:\n"
                "1. TUYỆT ĐỐI KHÔNG KẾT LUẬN VỘI VÀNG rằng tài liệu không có thông tin khi mới tra cứu 1 lần.\n"
                "2. Hãy tự động THỰC HIỆN QUERY REWRITING: dịch/viết lại truy vấn sang thuật ngữ tiếng Anh chuyên ngành tương ứng, "
                "hoặc dùng các từ đồng nghĩa kỹ thuật ngắn gọn hơn, rồi gọi lại `retrieve_documents` (lần 2/3).\n"
                "3. Chỉ kết luận tài liệu không đề cập nếu sau khi đã thử viết lại truy vấn mà vẫn không tìm thấy bằng chứng."
            )
        elif max_norm_score is not None and max_norm_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            joined_context += (
                f"\n[ĐÁNH GIÁ CORRECTIVE RAG - ĐỘ TIN CẬY CAO]:\n"
                f"Ngữ cảnh thu được đạt độ tin cậy cao {max_norm_score:.2f}/1.00. "
                "Đầy đủ thông tin chính xác để tổng hợp câu trả lời chi tiết và trích dẫn số trang tương ứng."
            )

        return joined_context

    @classmethod
    def extract_sources_metadata(cls, results) -> list[dict]:
        """Trích xuất metadata có cấu trúc của các nguồn trích dẫn để gửi về Frontend."""
        if not results or not results[0]:
            return []
        sources = []
        for i, result in enumerate(results[0], start=1):
            entity = result.get("entity", {})
            doc_name = entity.get("document_name", "Unknown")
            doc_id = entity.get("document_id", "")
            page = entity.get("page", 1)
            page_range = entity.get("page_range", str(page))
            chunk_range = entity.get("chunk_range", str(entity.get("chunk_index", 0)))
            has_expanded = entity.get("has_expanded", False)
            raw_text = entity.get("text", "")
            expanded_text = entity.get("expanded_text", "")
            snippet_text = expanded_text if (expanded_text and len(expanded_text) > 0) else raw_text

            raw_score = result.get(
                "normalized_score",
                result.get("rerank_score", result.get("score", 0.0)),
            )
            norm_score = RerankerService.normalize_score(float(raw_score)) if raw_score is not None else 0.50
            level = (
                "Cao"
                if norm_score >= cls.HIGH_CONFIDENCE_THRESHOLD
                else ("Trung bình" if norm_score >= cls.LOW_CONFIDENCE_THRESHOLD else "Thấp")
            )
            sources.append({
                "source_id": i,
                "document_name": doc_name,
                "document_id": doc_id,
                "page": int(page) if str(page).isdigit() else 1,
                "page_range": page_range,
                "chunk_range": chunk_range,
                "has_expanded": has_expanded,
                "score": norm_score,
                "level": level,
                "snippet": (snippet_text[:280] + "...") if len(snippet_text) > 280 else snippet_text,
            })
        return sources