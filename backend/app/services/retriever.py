import re
from typing import List, Dict, Any, Optional, Tuple
from app.services.db_service import db_service
from app.services.bm25_service import bm25_service as default_bm25_service


class Retriever:

    def __init__(
        self,
        embedding_service,
        vector_store,
        bm25_service=None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_service = bm25_service or default_bm25_service

    def _resolve_document_ids(self, document_name: str) -> Tuple[List[str], List[str]]:
        """
        Nới lỏng giới hạn tìm kiếm tên tài liệu:
        - Không phân biệt hoa / thường (case-insensitive).
        - Không phụ thuộc đuôi file (.pdf, .docx...).
        - Bỏ qua ký tự đặc biệt, khoảng trắng, gạch nối.
        - So khớp chuỗi con (substring) 2 chiều.
        """
        if not document_name:
            return [], []

        try:
            available_docs = db_service.get_all_documents()
        except Exception as e:
            print(f"[Retriever] Lỗi khi lấy danh sách documents từ SQLite: {e}")
            available_docs = []

        if not available_docs:
            return [], []

        raw_input = document_name.lower().strip()
        clean_input = re.sub(r'\.[a-z0-9]+$', '', raw_input).strip()
        compact_input = re.sub(r'[^a-z0-9\u00C0-\u024F\u1EA0-\u1EF9]', '', raw_input)

        matched_ids = []
        matched_names = []

        for doc in available_docs:
            d_id = doc.get("id")
            d_name = doc.get("name", "")
            raw_d = d_name.lower().strip()
            clean_d = re.sub(r'\.[a-z0-9]+$', '', raw_d).strip()
            compact_d = re.sub(r'[^a-z0-9\u00C0-\u024F\u1EA0-\u1EF9]', '', raw_d)

            # 1. Khớp chính xác không phân biệt hoa thường
            if raw_input == raw_d or clean_input == clean_d:
                matched_ids.append(d_id)
                matched_names.append(d_name)
                continue

            # 2. Khớp dạng rút gọn (bỏ khoảng cách, gạch nối)
            if compact_input and (compact_input == compact_d):
                matched_ids.append(d_id)
                matched_names.append(d_name)
                continue

            # 3. So khớp chuỗi con (substring)
            if clean_input and (clean_input in clean_d or clean_d in clean_input):
                matched_ids.append(d_id)
                matched_names.append(d_name)
                continue

            if compact_input and len(compact_input) >= 2 and (compact_input in compact_d or compact_d in compact_input):
                matched_ids.append(d_id)
                matched_names.append(d_name)
                continue

        unique_ids = list(set(matched_ids))
        unique_names = list(set(matched_names))
        return unique_ids, unique_names

    def retrieve_vector(
        self,
        query: str,
        top_k: int = 15,
        document_ids: Optional[List[str]] = None,
        document_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Dense Vector Search trên Milvus (Top K)."""
        query_vector = self.embedding_service.embed_chunks(query)
        results = self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
            document_ids=document_ids,
            document_name=document_name,
        )
        if results and len(results) > 0:
            return results[0]
        return []

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 15,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Lexical Search dùng BM25 (Top K)."""
        if self.bm25_service:
            return self.bm25_service.search(
                query=query,
                top_k=top_k,
                document_ids=document_ids,
            )
        return []

    @staticmethod
    def rrf_merge(
        vector_hits: List[Dict[str, Any]],
        bm25_hits: List[Dict[str, Any]],
        top_k: int = 15,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) để hợp nhất 2 danh sách ứng viên (Dense + BM25).
        Formula: score(doc) = sum( 1 / (rrf_k + rank_i) )
        """
        rrf_scores: Dict[str, float] = {}
        entities_map: Dict[str, Dict[str, Any]] = {}
        distances_map: Dict[str, float] = {}

        # 1. Tính điểm rank cho Vector Search (Dense)
        for rank, hit in enumerate(vector_hits, start=1):
            doc_id = hit.get("id")
            if not doc_id:
                continue
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))
            entities_map[doc_id] = hit.get("entity", {})
            distances_map[doc_id] = float(hit.get("distance", 0.0))

        # 2. Tính điểm rank cho BM25 Search (Sparse)
        for rank, hit in enumerate(bm25_hits, start=1):
            doc_id = hit.get("id")
            if not doc_id:
                continue
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))
            if doc_id not in entities_map:
                entities_map[doc_id] = hit.get("entity", {})
            if doc_id not in distances_map:
                distances_map[doc_id] = float(hit.get("distance", 0.0))

        # 3. Sắp xếp các ứng viên theo điểm RRF giảm dần
        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda d_id: rrf_scores[d_id],
            reverse=True,
        )

        merged_hits = []
        for doc_id in sorted_doc_ids[:top_k]:
            merged_hits.append({
                "id": doc_id,
                "distance": distances_map.get(doc_id, 0.0),
                "rrf_score": rrf_scores[doc_id],
                "score": rrf_scores[doc_id],
                "entity": entities_map.get(doc_id, {}),
            })

        return merged_hits

    def _execute_hybrid_search(
        self,
        query: str,
        top_k: int = 15,
        document_ids: Optional[List[str]] = None,
        document_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        vector_hits = self.retrieve_vector(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
            document_name=document_name,
        )
        bm25_hits = self.retrieve_bm25(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
        )

        print(f"[Retriever] Vector Search (Dense) returned {len(vector_hits)} candidates.")
        print(f"[Retriever] BM25 Search (Lexical) returned {len(bm25_hits)} candidates.")

        if vector_hits and bm25_hits:
            merged = self.rrf_merge(vector_hits, bm25_hits, top_k=top_k)
            print(f"[Retriever] RRF Fusion merged into {len(merged)} candidates (top_k={top_k}).")
            return merged
        elif vector_hits:
            return vector_hits[:top_k]
        elif bm25_hits:
            return bm25_hits[:top_k]
        return []

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        document_name: Optional[str] = None,
    ):
        """
        Tìm kiếm kết hợp Hybrid: Vector Search + BM25 với RRF Fusion,
        hỗ trợ tìm kiếm theo tên tài liệu và tự động nới lỏng nếu không có kết quả.
        """
        matched_ids = []
        matched_names = []
        if document_name:
            matched_ids, matched_names = self._resolve_document_ids(document_name)
            if matched_names:
                print(f"[Retriever] Resolved document '{document_name}' -> Matched: {matched_names}")
            else:
                print(f"[Retriever] Document '{document_name}' not matched directly -> Relaxing filter to all documents.")

        # Lần 1: Tìm kiếm Hybrid có lọc theo tài liệu đã resolve (nếu có)
        hits = self._execute_hybrid_search(
            query=query,
            top_k=top_k,
            document_ids=matched_ids if matched_ids else None,
            document_name=document_name if not matched_ids else None,
        )

        # Lần 2: NỚI LỎNG TOÀN DIỆN - Nếu có chỉ định document_name mà kết quả trống -> Tự động fallback sang tất cả tài liệu
        if document_name and not hits:
            print(f"[Retriever] Filter for '{document_name}' yielded 0 results -> Auto-fallback to ALL documents...")
            hits = self._execute_hybrid_search(
                query=query,
                top_k=top_k,
                document_ids=None,
                document_name=None,
            )

        return [hits]
