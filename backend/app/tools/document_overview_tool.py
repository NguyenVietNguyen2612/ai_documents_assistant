from app.tools.base import tool
from app.services.retriever import Retriever


def make_document_overview_tool(retriever: Retriever):
    @tool
    def get_document_overview(document_name: str) -> str:
        """
        Dùng công cụ này khi người dùng yêu cầu TÓM TẮT TỔNG QUAN một tài liệu, hỏi bố cục/mục lục của tài liệu,
        hoặc cần nắm nội dung chính từ đầu đến cuối của một file cụ thể.
        
        THAM SỐ:
        - document_name: Tên tài liệu cần lấy tổng quan (ví dụ: 'CS338', 'report.pdf').
        """
        print(f"\n[Tool: DocumentOverview] Đang lấy tổng quan tài liệu: '{document_name}'...")
        matched_ids, matched_names = retriever._resolve_document_ids(document_name)
        bm25_docs = retriever.bm25_service.documents if retriever and retriever.bm25_service else []

        target_chunks = []
        if matched_ids:
            target_chunks = [c for c in bm25_docs if c.get("document_id") in matched_ids]
        elif document_name:
            q_lower = document_name.lower().strip()
            target_chunks = [
                c for c in bm25_docs
                if q_lower in str(c.get("document_name", "")).lower()
            ]

        if not target_chunks:
            return (
                f"Không tìm thấy tài liệu nào khớp với tên '{document_name}'. "
                "Hãy gọi công cụ `list_uploaded_documents` để kiểm tra tên chính xác của các file hiện có."
            )

        # Sắp xếp các chunk theo thứ tự xuất hiện trong tài liệu
        target_chunks.sort(key=lambda c: (c.get("document_name", ""), int(c.get("chunk_index", 0))))
        doc_title = matched_names[0] if matched_names else target_chunks[0].get("document_name", document_name)

        # Trích xuất danh sách các đề mục (Header paths)
        headers = []
        for c in target_chunks:
            text = c.get("text", "")
            first_line = text.splitlines()[0].strip() if text else ""
            if first_line.startswith("[") and "]" in first_line:
                header = first_line[1:first_line.index("]")]
                if header not in headers:
                    headers.append(header)

        # Lấy các đoạn mở đầu và đoạn kết luận để nắm toàn cảnh
        head_chunks = target_chunks[:2]
        tail_chunks = target_chunks[-1:] if len(target_chunks) > 2 else []
        selected = head_chunks + tail_chunks

        overview_parts = [
            f"[Tài liệu: {doc_title}]",
            f"Tổng số phân đoạn (chunks): {len(target_chunks)}",
        ]
        if headers:
            overview_parts.append("Cấu trúc đề mục chính trong tài liệu:")
            for h in headers[:15]:
                overview_parts.append(f"  + {h}")

        overview_parts.append("\nNội dung trích lược (Mở đầu & Kết luận):")
        for c in selected:
            idx = c.get("chunk_index", 0)
            overview_parts.append(f"--- [Phân đoạn #{idx}] ---\n{c.get('text', '')}")

        return "\n".join(overview_parts)

    return get_document_overview
