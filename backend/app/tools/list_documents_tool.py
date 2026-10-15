from app.tools.base import tool
from app.services.retriever import Retriever
from app.services.db_service import db_service


def make_list_documents_tool(retriever: Retriever):
    @tool
    def list_uploaded_documents() -> str:
        """
        Dùng công cụ này khi người dùng hỏi danh sách các tài liệu/file đã tải lên hệ thống
        (ví dụ: 'Hiện có những tài liệu nào?', 'Tôi đã upload file gì?'), hoặc khi bạn cần biết tên các tài liệu hiện có.
        """
        print("\n[Tool: ListDocuments] Đang lấy danh sách tài liệu trong hệ thống...")
        try:
            docs = db_service.get_all_documents()
        except Exception as e:
            return f"Không thể truy xuất danh sách tài liệu: {e}"

        if not docs:
            return "Hiện tại chưa có tài liệu nào được tải lên hệ thống."

        bm25_docs = retriever.bm25_service.documents if retriever and retriever.bm25_service else []
        chunk_counts = {}
        for chunk in bm25_docs:
            d_id = chunk.get("document_id")
            if d_id:
                chunk_counts[d_id] = chunk_counts.get(d_id, 0) + 1

        lines = [f"Hiện có {len(docs)} tài liệu trong hệ thống:"]
        for idx, d in enumerate(docs, start=1):
            d_id = d.get("id", "")
            d_name = d.get("name", "Unknown")
            d_size = d.get("size", "Unknown")
            c_count = chunk_counts.get(d_id, 0)
            chunk_info = f", {c_count} phân đoạn (chunks)" if c_count > 0 else ""
            lines.append(f"- [{idx}] [Tài liệu: {d_name}] (Dung lượng: {d_size}{chunk_info})")

        return "\n".join(lines)

    return list_uploaded_documents
