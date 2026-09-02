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
        Nếu người dùng chỉ định tìm kiếm trong một tài liệu cụ thể (ví dụ: "trong file báo cáo CS338", "của tài liệu abc.pdf"), hãy truyền tên tài liệu đó vào tham số `document_name`.
        """
        print(f"\n[Tool: Retrieve] Đang tìm kiếm truy vấn: '{query}' " + (f"(Lọc theo file: {document_name})" if document_name else "(Toàn bộ tài liệu)"))
        results = retriever.retrieve(query=query, top_k=10, document_name=document_name)
        
        if not results or not results[0]:
            print("[Tool: Retrieve] Không tìm thấy kết quả.")
            return "Không tìm thấy thông tin nào liên quan trong tài liệu."
        
        context = context_builder.build(results)
        print(f"[Tool: Retrieve] Đã tìm thấy ngữ cảnh ({len(context)} ký tự).")
        return context
        
    return retrieve_documents