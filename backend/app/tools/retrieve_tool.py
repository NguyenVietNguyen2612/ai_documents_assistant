from typing import Optional
from app.tools.base import tool
from app.services.retriever import Retriever
from app.services.context_builder import ContextBuilder


def make_retrieve_tool(retriever: Retriever, context_builder: ContextBuilder):
    @tool
    def retrieve_documents(query: str, document_name: Optional[str] = None) -> str:
        """
        Dùng công cụ này để tìm kiếm thông tin chi tiết từ các tài liệu đã được tải lên (PDF, hình ảnh, văn bản).
        Luôn gọi công cụ này khi người dùng hỏi về nội dung tài liệu, số liệu, định nghĩa, hoặc thông tin cụ thể.
        Hệ thống áp dụng 2-Stage Hybrid Search (Milvus + BM25 RRF) kết hợp BGE Cross-Encoder Reranker và
        Small-to-Big Window Expansion (tự động mở rộng sang các đoạn liền kề trước/sau để không bị cụt ý).
        
        THAM SỐ:
        - query: Câu hỏi hoặc các từ khóa nội dung cần tìm kiếm trong tài liệu. 
          [QUY TẮC CORRECTIVE RAG]: Nếu kết quả trả về có cảnh báo ĐỘ TIN CẬY THẤP hoặc chưa chứa dữ liệu cần tìm, 
          bạn BẮT BUỘC KHÔNG ĐƯỢC KẾT LUẬN VỘI VÀNG. Hãy tự động viết lại truy vấn (Query Rewriting: dịch sang thuật ngữ 
          tiếng Anh học thuật, tách từ khóa cốt lõi hoặc dùng từ đồng nghĩa) và gọi lại công cụ này lần 2/3.
        - document_name (tùy chọn): Tên hoặc từ khóa gợi ý tên tài liệu (ví dụ: 'CS338', 'cs338', 'RAG'). Hệ thống tự động nới lỏng tìm kiếm, không phân biệt chữ hoa/thường, không phụ thuộc đuôi .pdf. Nếu người dùng không chỉ định tài liệu hoặc bạn muốn tìm rộng trên tất cả tài liệu, hãy để trống tham số này.
        """
        print(f"\n[Tool: Retrieve] Đang tìm kiếm truy vấn: '{query}' " + (f"(Gợi ý file: {document_name})" if document_name else "(Toàn bộ tài liệu)"))
        results = retriever.retrieve(query=query, top_k=3, document_name=document_name, expand_window=True)
        
        if not results or not results[0]:
            print("[Tool: Retrieve] Không tìm thấy kết quả.")
            return (
                "Không tìm thấy thông tin nào liên quan trong tài liệu với truy vấn hiện tại.\n"
                "[YÊU CẦU CORRECTIVE RAG]: Hãy tự động thực hiện Query Rewriting (dịch câu hỏi sang tiếng Anh hoặc rút gọn thành 2-4 từ khóa chuyên môn cốt lõi) "
                "và gọi lại `retrieve_documents` một lần nữa, hoặc gọi `list_uploaded_documents` để kiểm tra tên các tài liệu hiện có."
            )
        
        context = context_builder.build(results)
        print(f"[Tool: Retrieve] Đã tìm thấy ngữ cảnh ({len(context)} ký tự) kèm Small-to-Big Window.")
        return context

    return retrieve_documents
