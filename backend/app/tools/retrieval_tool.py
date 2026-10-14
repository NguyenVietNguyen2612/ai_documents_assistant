from typing import Optional
from langchain_core.tools import tool
from app.services.retriever import Retriever
from app.services.context_builder import ContextBuilder

def make_retrieve_tool(retriever: Retriever, context_builder: ContextBuilder):
    @tool
    def retrieve_documents(query: str, document_name: Optional[str] = None) -> str:
        """
        Dùng công cụ này để tìm kiếm thông tin từ các tài liệu đã được tải lên (PDF, hình ảnh, văn bản).
        Luôn gọi công cụ này khi người dùng hỏi về nội dung tài liệu, số liệu, hoặc thông tin cụ thể.
        
        THAM SỐ:
        - query: Câu hỏi hoặc các từ khóa nội dung cần tìm kiếm trong tài liệu.
        - document_name (tùy chọn): Tên hoặc từ khóa gợi ý tên tài liệu (ví dụ: 'CS338', 'cs338', 'RAG'). Hệ thống tự động nới lỏng tìm kiếm, không phân biệt chữ hoa/thường, không phụ thuộc đuôi .pdf. Nếu người dùng không chỉ định tài liệu hoặc bạn muốn tìm rộng trên tất cả tài liệu, hãy để trống tham số này.
        """

        print(f"\n[Tool: Retrieve] Đang tìm kiếm truy vấn: '{query}' " + (f"(Gợi ý file: {document_name})" if document_name else "(Toàn bộ tài liệu)"))
        results = retriever.retrieve(query=query, top_k=15, document_name=document_name)
        
        if not results or not results[0]:
            print("[Tool: Retrieve] Không tìm thấy kết quả.")
            return "Không tìm thấy thông tin nào liên quan trong tài liệu."
        
        context = context_builder.build(results)
        print(f"[Tool: Retrieve] Đã tìm thấy ngữ cảnh ({len(context)} ký tự).")
        return context

    return retrieve_documents