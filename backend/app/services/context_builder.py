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
            text = entity.get("text", "")

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

            context = f"""[Source {i}] [Tài liệu: {doc_name} - Trang {page}] ({score_str.lstrip(' | ') if score_str else ''})
Document: {doc_name}
Page: {page}

{text}
"""
            contexts.append(context)

        joined_context = "\n".join(contexts)

        # Corrective RAG: Nếu điểm chuẩn hóa cao nhất thấp hơn ngưỡng 0.40 -> Cảnh báo Agent viết lại truy vấn
        if max_norm_score is not None and max_norm_score < self.LOW_CONFIDENCE_THRESHOLD:
            joined_context += (
                f"\n[ĐÁNH GIÁ CORRECTIVE RAG]: Độ tin cậy ngữ cảnh cao nhất đạt {max_norm_score:.2f}/1.00 "
                f"(thấp hơn ngưỡng chuẩn {self.LOW_CONFIDENCE_THRESHOLD:.2f}). "
                "Hãy kiểm tra kỹ xem ngữ cảnh trên có thực sự trả lời đúng câu hỏi không. "
                "Nếu chưa đủ thông tin, hãy gọi lại công cụ `retrieve_documents` với từ khóa khác (Query Rewriting - dùng từ đồng nghĩa hoặc thuật ngữ tiếng Anh tương ứng)."
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
            text = entity.get("text", "")
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
                "page": int(page),
                "score": norm_score,
                "level": level,
                "snippet": (text[:280] + "...") if len(text) > 280 else text,
            })
        return sources