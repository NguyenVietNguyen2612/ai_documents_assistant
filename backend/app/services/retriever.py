import re
from app.services.db_service import db_service

class Retriever:

    def __init__(
        self,
        embedding_service,
        vector_store,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def _resolve_document_ids(self, document_name: str) -> tuple[list[str], list[str]]:
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

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        document_name: str = None,
    ):
        query_vector = self.embedding_service.embed_chunks(query)

        matched_ids = []
        matched_names = []
        if document_name:
            matched_ids, matched_names = self._resolve_document_ids(document_name)
            if matched_names:
                print(f"[Retriever] Resolved document '{document_name}' -> Matched: {matched_names}")
            else:
                print(f"[Retriever] Document '{document_name}' not matched directly -> Relaxing filter to all documents.")

        # Lần 1: Tìm kiếm có lọc theo tài liệu đã resolve (nếu có)
        results = self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
            document_ids=matched_ids if matched_ids else None,
            document_name=document_name if not matched_ids else None,
        )

        # Lần 2: NỚI LỎNG TOÀN DIỆN - Nếu lọc theo document_name mà không ra kết quả
        if document_name and (not results or not results[0]):
            print(f"[Retriever] Filter for '{document_name}' yielded 0 results -> Auto-fallback to ALL documents...")
            results = self.vector_store.search(
                query_vector=query_vector,
                limit=top_k,
                document_ids=None,
                document_name=None,
            )

        return results