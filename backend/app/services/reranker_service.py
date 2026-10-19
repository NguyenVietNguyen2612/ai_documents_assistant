import math
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

    @staticmethod
    def normalize_score(raw_score: float) -> float:
        """
        Chuẩn hóa điểm số về thang [0.0, 1.0]:
        - Nếu điểm đã nằm trong khoảng [0.0, 1.0] (xác suất hoặc Cosine/RRF), giữ nguyên.
        - Nếu điểm là raw logit nằm ngoài [0.0, 1.0] (từ Cross-Encoder), áp dụng hàm Sigmoid.
        """
        val = float(raw_score)
        if 0.0 <= val <= 1.0:
            return round(val, 4)
        # Áp dụng Sigmoid cho raw logits âm hoặc > 1.0
        clamped = max(-20.0, min(20.0, val))
        return round(1.0 / (1.0 + math.exp(-clamped)), 4)

    def _ensure_model_loaded(self):
        """Lazy load mô hình để tối ưu thời gian khởi động ứng dụng."""
        if self._is_loaded:
            return

        try:
            from sentence_transformers import CrossEncoder
            print(f"[RerankerService] Loading BGE Reranker model from '{self.model_name}'...")
            try:
                self.model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception:
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
                raw_score = float(scores[i])
                norm_score = self.normalize_score(raw_score)
                item["rerank_score"] = raw_score
                item["normalized_score"] = norm_score
                item["score"] = raw_score
                item["distance"] = raw_score
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

