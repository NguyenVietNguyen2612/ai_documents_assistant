import ast
import math
import operator
import re
from typing import Optional, List

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(fn):
        class _ToolWrapper:
            def __init__(self, func):
                self.func = func
                self.name = func.__name__
                self.__doc__ = func.__doc__

            def __call__(self, *args, **kwargs):
                return self.func(*args, **kwargs)

            def invoke(self, input_data):
                if isinstance(input_data, dict):
                    return self.func(**input_data)
                return self.func(input_data)

        return _ToolWrapper(fn)

from app.services.retriever import Retriever
from app.services.context_builder import ContextBuilder
from app.services.db_service import db_service


def make_retrieve_tool(retriever: Retriever, context_builder: ContextBuilder):
    @tool
    def retrieve_documents(query: str, document_name: Optional[str] = None) -> str:
        """
        Dùng công cụ này để tìm kiếm thông tin chi tiết từ các tài liệu đã được tải lên (PDF, hình ảnh, văn bản).
        Luôn gọi công cụ này khi người dùng hỏi về nội dung tài liệu, số liệu, định nghĩa, hoặc thông tin cụ thể.
        
        THAM SỐ:
        - query: Câu hỏi hoặc các từ khóa nội dung cần tìm kiếm trong tài liệu. Nếu lần tìm đầu chưa ra kết quả tốt, hãy viết lại query (Query Rewriting) bằng từ khóa ngắn gọn hơn hoặc thuật ngữ tiếng Anh chuyên ngành.
        - document_name (tùy chọn): Tên hoặc từ khóa gợi ý tên tài liệu (ví dụ: 'CS338', 'cs338', 'RAG'). Hệ thống tự động nới lỏng tìm kiếm, không phân biệt chữ hoa/thường, không phụ thuộc đuôi .pdf. Nếu người dùng không chỉ định tài liệu hoặc bạn muốn tìm rộng trên tất cả tài liệu, hãy để trống tham số này.
        """
        print(f"\n[Tool: Retrieve] Đang tìm kiếm truy vấn: '{query}' " + (f"(Gợi ý file: {document_name})" if document_name else "(Toàn bộ tài liệu)"))
        results = retriever.retrieve(query=query, top_k=3, document_name=document_name)
        
        if not results or not results[0]:
            print("[Tool: Retrieve] Không tìm thấy kết quả.")
            return (
                "Không tìm thấy thông tin nào liên quan trong tài liệu với truy vấn hiện tại. "
                "[GỢI Ý CORRECTIVE RAG]: Hãy thử gọi lại `retrieve_documents` với từ khóa khác (ngắn gọn hơn hoặc dịch sang thuật ngữ tiếng Anh), "
                "hoặc gọi `list_uploaded_documents` để kiểm tra danh sách tài liệu hiện có."
            )
        
        context = context_builder.build(results)
        print(f"[Tool: Retrieve] Đã tìm thấy ngữ cảnh ({len(context)} ký tự).")
        return context

    return retrieve_documents


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


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def _safe_eval_expr(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval_expr(node.left)
        right = _safe_eval_expr(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval_expr(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name in _ALLOWED_FUNCTIONS:
            args = [_safe_eval_expr(arg) for arg in node.args]
            return _ALLOWED_FUNCTIONS[func_name](*args)
    raise ValueError(f"Biểu thức không được hỗ trợ: {ast.dump(node)}")


def make_calculator_tool():
    @tool
    def calculate_metrics(expression: str) -> str:
        """
        Công cụ máy tính toán học chính xác dùng để tính toán số liệu, tỷ lệ phần trăm, độ chênh lệch
        (ví dụ: '74.03 - 72.16', '(36000 / 50000) * 100', 'round((74.03 - 72.16) / 72.16 * 100, 2)').
        Hỗ trợ các phép toán +, -, *, /, **, % và các hàm sqrt, log, log10, exp, abs, round, min, max.
        """
        print(f"\n[Tool: Calculator] Đang tính biểu thức: '{expression}'")
        try:
            cleaned = expression.strip().replace("^", "**")
            cleaned = re.sub(r"[^0-9a-zA-Z_\+\-\*\/\%\(\)\.\,\s]", "", cleaned)
            parsed = ast.parse(cleaned, mode="eval")
            result = _safe_eval_expr(parsed)
            if isinstance(result, float):
                result = round(result, 6)
            return f"Kết quả tính toán của `{expression}` = {result}"
        except Exception as e:
            return f"Không thể tính biểu thức '{expression}': {e}"

    return calculate_metrics


def make_agent_tools(retriever: Retriever, context_builder: ContextBuilder) -> List:
    """Khởi tạo bộ công cụ Multi-Tool cho Agent."""
    return [
        make_retrieve_tool(retriever, context_builder),
        make_list_documents_tool(retriever),
        make_document_overview_tool(retriever),
        make_calculator_tool(),
    ]