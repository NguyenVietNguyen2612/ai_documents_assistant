import re
from typing import List, Dict, Any, Optional, Tuple
from app.services.db_service import db_service
from app.services.bm25_service import bm25_service as default_bm25_service
from app.services.reranker_service import reranker_service as default_reranker_service


class Retriever:

    def __init__(
        self,
        embedding_service,
        vector_store,
        bm25_service=None,
        reranker_service=None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_service = bm25_service or default_bm25_service
        self.reranker_service = reranker_service or default_reranker_service

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
        initial_k: int = 15,
        document_ids: Optional[List[str]] = None,
        document_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Giai đoạn 1: Thu hồi diện rộng (Recall) qua Vector + BM25 + RRF."""
        vector_hits = self.retrieve_vector(
            query=query,
            top_k=initial_k,
            document_ids=document_ids,
            document_name=document_name,
        )
        bm25_hits = self.retrieve_bm25(
            query=query,
            top_k=initial_k,
            document_ids=document_ids,
        )

        print(f"[Retriever] Vector Search (Dense) returned {len(vector_hits)} candidates.")
        print(f"[Retriever] BM25 Search (Lexical) returned {len(bm25_hits)} candidates.")

        if vector_hits and bm25_hits:
            merged = self.rrf_merge(vector_hits, bm25_hits, top_k=initial_k)
            print(f"[Retriever] RRF Fusion merged into {len(merged)} candidates.")
            return merged
        elif vector_hits:
            return vector_hits[:initial_k]
        elif bm25_hits:
            return bm25_hits[:initial_k]
        return []

    def expand_window(
        self,
        hits: List[Dict[str, Any]],
        window_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Small-to-Big Retrieval Window Expansion:
        Đối với mỗi chunk được chọn lọc ở Top-K, tự động truy xuất các chunk liền kề
        (trước và sau) trong cùng một tài liệu để mở rộng ngữ cảnh, ngăn chặn tình trạng
        đoạn văn bị cụt ý ở ranh giới cắt chunk.
        """
        if not hits or window_size <= 0:
            return hits

        expanded_hits = []
        for hit in hits:
            item = dict(hit)
            entity = dict(item.get("entity", {}))
            doc_id = entity.get("document_id")
            chunk_idx = entity.get("chunk_index")

            if doc_id and chunk_idx is not None:
                # 1. Thử lấy chunk liền kề từ VectorStore (Milvus)
                neighbors = []
                if hasattr(self.vector_store, "get_neighboring_chunks"):
                    neighbors = self.vector_store.get_neighboring_chunks(
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        window_size=window_size,
                    )

                # 2. Fallback nếu VectorStore không có hoặc trả về rỗng -> lấy từ BM25Service
                if (not neighbors or len(neighbors) <= 1) and hasattr(self.bm25_service, "get_neighboring_chunks"):
                    bm25_neighbors = self.bm25_service.get_neighboring_chunks(
                        document_id=doc_id,
                        chunk_index=chunk_idx,
                        window_size=window_size,
                    )
                    if bm25_neighbors and len(bm25_neighbors) > len(neighbors):
                        neighbors = bm25_neighbors

                if neighbors and len(neighbors) > 1:
                    # Nối văn bản liền mạch không bị lặp ranh giới
                    merged_parts = []
                    for n in neighbors:
                        t = n.get("text", "").strip()
                        if t and t not in merged_parts:
                            merged_parts.append(t)

                    min_idx = min(n.get("chunk_index", chunk_idx) for n in neighbors)
                    max_idx = max(n.get("chunk_index", chunk_idx) for n in neighbors)
                    pages = sorted(list(set(n.get("page", entity.get("page", 1)) for n in neighbors if n.get("page"))))

                    entity["expanded_text"] = "\n\n".join(merged_parts)
                    entity["chunk_range"] = f"{min_idx}-{max_idx}"
                    entity["has_expanded"] = True
                    entity["page_range"] = (
                        f"{pages[0]}-{pages[-1]}" if len(pages) > 1 and pages[0] != pages[-1] else str(pages[0])
                    )
                else:
                    entity["expanded_text"] = entity.get("text", "")
                    entity["chunk_range"] = str(chunk_idx)
                    entity["has_expanded"] = False
                    entity["page_range"] = str(entity.get("page", 1))
            else:
                entity["expanded_text"] = entity.get("text", "")
                entity["has_expanded"] = False

            item["entity"] = entity
            expanded_hits.append(item)

        return expanded_hits

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        document_name: Optional[str] = None,
        mode: str = "hybrid_rerank",
        expand_window: bool = True,
        window_size: int = 1,
    ):
        """
        Quy trình Two-stage Retrieval (hỗ trợ 3 chế độ phục vụ Ablation Study):
        - mode="vector_only": Chỉ dùng Dense Vector Search trên Milvus.
        - mode="hybrid_rrf": Kết hợp Dense Vector + BM25 qua thuật toán RRF (không dùng Reranker).
        - mode="hybrid_rerank" (mặc định):
            1. Giai đoạn 1 (Recall): Vector Search + BM25 Search kết hợp RRF gom rộng các candidates (10-15 chunks).
            2. Giai đoạn 2 (Precision): Reranker (BGE base) chấm điểm sâu mối tương quan và chọn lọc ra Top K chunks tốt nhất.
        - expand_window=True: Kích hoạt Small-to-Big Window Expansion cho các chunk Top-K để không bị mất ngữ cảnh liền kề.
        """
        matched_ids = []
        matched_names = []
        if document_name:
            matched_ids, matched_names = self._resolve_document_ids(document_name)
            if matched_names:
                print(f"[Retriever] Resolved document '{document_name}' -> Matched: {matched_names}")
            else:
                print(f"[Retriever] Document '{document_name}' not matched directly -> Relaxing filter to all documents.")

        # Chế độ 1: Chỉ dùng Dense Vector Search (Baseline)
        if mode == "vector_only":
            vector_hits = self.retrieve_vector(
                query=query,
                top_k=top_k,
                document_ids=matched_ids if matched_ids else None,
                document_name=document_name if not matched_ids else None,
            )
            if document_name and not vector_hits:
                vector_hits = self.retrieve_vector(query=query, top_k=top_k)
            selected = vector_hits[:top_k]
            if expand_window:
                selected = self.expand_window(selected, window_size=window_size)
            return [selected]

        # Lấy rộng ứng viên ở Giai đoạn 1 (ít nhất 10 chunks hoặc gấp 3 lần top_k) để RRF hợp nhất hiệu quả
        initial_k = max(10, top_k * 3)

        # Lần 1: Tìm kiếm Hybrid có lọc theo tài liệu đã resolve (nếu có)
        candidates = self._execute_hybrid_search(
            query=query,
            initial_k=initial_k,
            document_ids=matched_ids if matched_ids else None,
            document_name=document_name if not matched_ids else None,
        )

        # Lần 2: NỚI LỎNG TOÀN DIỆN - Nếu có chỉ định document_name mà kết quả trống -> Tự động fallback sang tất cả tài liệu
        if document_name and not candidates:
            print(f"[Retriever] Filter for '{document_name}' yielded 0 results -> Auto-fallback to ALL documents...")
            candidates = self._execute_hybrid_search(
                query=query,
                initial_k=initial_k,
                document_ids=None,
                document_name=None,
            )

        # Chế độ 2: Hybrid Search (Vector + BM25 RRF) không qua Cross-Encoder Reranker
        if mode == "hybrid_rrf":
            selected = candidates[:top_k]
            if expand_window:
                selected = self.expand_window(selected, window_size=window_size)
            return [selected]

        # Chế độ 3 (Mặc định): Giai đoạn 2 - Rerank với BGE Cross-Encoder để chọn lọc ra đúng top_k candidates
        if self.reranker_service and candidates:
            final_hits = self.reranker_service.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        else:
            final_hits = candidates[:top_k]

        # Small-to-Big Window Expansion cho các chunks chiến thắng
        if expand_window:
            final_hits = self.expand_window(final_hits, window_size=window_size)

        print(f"[Retriever] Final selected {len(final_hits)} chunks for context (expanded_window={expand_window}).")
        return [final_hits]

