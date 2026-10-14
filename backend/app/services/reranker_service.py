import os
from typing import List, Dict, Any, Optional


class RerankerService:
    """
    Service xếp hạng lại (Reranking) sử dụng mô hình Cross-Encoder BGE Base (BAAI/bge-reranker-base).
    Mô hình nhận trực tiếp cặp (query, document) để đọc hiểu sâu mối liên hệ ngữ nghĩa
    và chấm điểm mức độ liên quan (relevance score).
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-base"
        )
        self.model = None
        self._is_loaded = False

    def _ensure_model_loaded(self):
        """Lazy load mô hình để tối ưu thời gian khởi động ứng dụng."""
        if self._is_loaded:
            return

        try:
            from sentence_transformers import CrossEncoder
            print(f"[RerankerService] Loading BGE Reranker model from '{self.model_name}'...")
            self.model = CrossEncoder(self.model_name)
            self._is_loaded = True
            print(f"[RerankerService] BGE Reranker model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(
                f"[RerankerService] Warning: Failed to load BGE Reranker '{self.model_name}': {e}. "
                "Will fallback to original candidate order."
            )
            self.model = None
            self._is_loaded = True

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Nhận danh sách candidates từ giai đoạn 1 (RRF/Hybrid Search) và xếp hạng lại bằng BGE base,
        chỉ giữ lại top_k (mặc định là 3) candidates xuất sắc nhất.
        """
        if not candidates:
            return []

        self._ensure_model_loaded()

        # Fallback nếu mô hình chưa sẵn sàng hoặc gặp lỗi khi tải
        if self.model is None:
            return candidates[:top_k]

        try:
            # Tạo các cặp (query, document_text)
            pairs = []
            for c in candidates:
                entity = c.get("entity", {})
                doc_text = entity.get("text", "")
                pairs.append((query, doc_text))

            # Chấm điểm mức độ liên quan bằng Cross-Encoder
            scores = self.model.predict(pairs)

            # Gán điểm và sao chép candidate để không ảnh hưởng dữ liệu gốc
            scored_candidates = []
            for i, c in enumerate(candidates):
                item = dict(c)
                score = float(scores[i])
                item["rerank_score"] = score
                item["score"] = score
                item["distance"] = score
                scored_candidates.append(item)

            # Sắp xếp giảm dần theo điểm rerank
            scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

            selected = scored_candidates[:top_k]
            print(
                f"[RerankerService] Reranked {len(candidates)} candidates -> selected top {len(selected)} "
                f"using '{self.model_name}'."
            )
            return selected

        except Exception as e:
            print(f"[RerankerService] Error during reranking: {e}. Falling back to original top-{top_k}.")
            return candidates[:top_k]


# Singleton instance dùng chung trong hệ thống
reranker_service = RerankerService()
