import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.context_builder import ContextBuilder
from app.services.retriever import Retriever
from app.tools.calculator_tool import make_calculator_tool
from app.tools.document_overview_tool import make_document_overview_tool
from app.services.bm25_service import BM25Service


def test_small_to_big_window_boundary_edge_cases():
    print("\n[TEST 1/4] Kiểm tra ranh giới Small-to-Big Window (Chunk đầu tiên index=0)...")
    mock_embedding = MagicMock()
    mock_vector_store = MagicMock()
    mock_bm25 = MagicMock()
    mock_reranker = MagicMock()

    # Khi chunk_index = 0, target_indices không được chứa số âm: [0, 1]
    mock_vector_store.get_neighboring_chunks.return_value = [
        {"id": "c0", "chunk_index": 0, "page": 1, "text": "Tiêu đề mở đầu bài báo."},
        {"id": "c1", "chunk_index": 1, "page": 1, "text": "Nội dung phần 1."},
    ]

    retriever = Retriever(
        embedding_service=mock_embedding,
        vector_store=mock_vector_store,
        bm25_service=mock_bm25,
        reranker_service=mock_reranker,
    )

    hits = [{
        "id": "c0",
        "entity": {
            "document_id": "doc_1",
            "document_name": "Paper.pdf",
            "page": 1,
            "chunk_index": 0,
            "text": "Tiêu đề mở đầu bài báo.",
        }
    }]

    expanded = retriever.expand_window(hits, window_size=1)
    ent = expanded[0]["entity"]
    print(f"  -> Cửa sổ ranh giới chunk 0: Chunks {ent['chunk_range']} | Text: '{ent['expanded_text']}'")

    assert ent["has_expanded"] is True
    assert ent["chunk_range"] == "0-1"
    assert "Tiêu đề mở đầu" in ent["expanded_text"]
    assert "Nội dung phần 1" in ent["expanded_text"]
    print("  -> Kết quả: ✅ PASSED (Xử lý an toàn ranh giới đầu tài liệu).")


def test_corrective_rag_three_confidence_tiers():
    print("\n[TEST 2/4] Kiểm tra 3 tầng phân loại độ tin cậy RAG (Cao >= 0.70, TB 0.40-0.69, Thấp < 0.40)...")
    cb = ContextBuilder()

    # Tier 1: Cao (0.85)
    res_high = [[{"normalized_score": 0.85, "entity": {"document_name": "doc.pdf", "page": 1, "text": "Khớp cao"}}]]
    ctx_high = cb.build(res_high)
    print("  -> Tier 1 (0.85): Kiểm tra nhãn 'Độ tin cậy: 0.85/1.00 (Cao)'...")
    assert "0.85/1.00 (Cao)" in ctx_high
    assert "[ĐÁNH GIÁ CORRECTIVE RAG - ĐỘ TIN CẬY CAO]" in ctx_high

    # Tier 2: Trung bình (0.55)
    res_med = [[{"normalized_score": 0.55, "entity": {"document_name": "doc.pdf", "page": 1, "text": "Khớp vừa"}}]]
    ctx_med = cb.build(res_med)
    print("  -> Tier 2 (0.55): Kiểm tra nhãn 'Độ tin cậy: 0.55/1.00 (Trung bình)'...")
    assert "0.55/1.00 (Trung bình)" in ctx_med
    assert "[ĐÁNH GIÁ CORRECTIVE RAG - CẢNH BÁO ĐỘ TIN CẬY THẤP]" not in ctx_med

    # Tier 3: Thấp (0.28)
    res_low = [[{"normalized_score": 0.28, "entity": {"document_name": "doc.pdf", "page": 1, "text": "Ít liên quan"}}]]
    ctx_low = cb.build(res_low)
    print("  -> Tier 3 (0.28): Kiểm tra nhãn 'Độ tin cậy: 0.28/1.00 (Thấp)' & Kích hoạt Self-Correction...")
    assert "0.28/1.00 (Thấp)" in ctx_low
    assert "[ĐÁNH GIÁ CORRECTIVE RAG - CẢNH BÁO ĐỘ TIN CẬY THẤP]" in ctx_low
    assert "QUERY REWRITING" in ctx_low
    print("  -> Kết quả: ✅ PASSED (Cả 3 tầng phân loại hoạt động chuẩn xác).")


def test_calculator_tool_ast_safety_and_math():
    print("\n[TEST 3/4] Kiểm tra AST Calculator (Tính toán số học chính xác & Chống Code Injection)...")
    calc = make_calculator_tool()

    # 1. Phép tính phần trăm thực nghiệm phức tạp
    expr1 = "round((74.03 - 72.16) / 72.16 * 100, 2)"
    out1 = calc.invoke({"expression": expr1})
    print(f"  -> Tính phần trăm: '{expr1}' => '{out1}'")
    assert "2.59" in out1

    # 2. Xử lý chia cho 0
    expr2 = "100 / 0"
    out2 = calc.invoke({"expression": expr2})
    print(f"  -> Chia cho 0: '{expr2}' => '{out2}'")
    assert any(w in out2.lower() for w in ["lỗi", "error", "zerodivision", "không thể", "division by zero"])

    # 3. Chống mã độc (Code injection)
    expr3 = "__import__('os').system('ls')"
    out3 = calc.invoke({"expression": expr3})
    print(f"  -> Thử mã độc: '{expr3}' => '{out3}'")
    assert any(w in out3.lower() for w in ["lỗi", "error", "không thể", "không hợp lệ", "không được hỗ trợ"])
    print("  -> Kết quả: ✅ PASSED (AST an toàn tuyệt đối, ngăn chặn thực thi mã độc).")


def test_document_overview_tool_hierarchy():
    print("\n[TEST 4/4] Kiểm tra công cụ Document Overview (Trích xuất cấu trúc đề mục tài liệu)...")
    bm25 = BM25Service()
    bm25.index_documents([
        {
            "id": "c1",
            "document_id": "doc_cs338",
            "document_name": "CS338.pdf",
            "page": 1,
            "chunk_index": 0,
            "text": "[1. Introduction > 1.1 Problem Statement]\nNghiên cứu phát hiện Deepfake.",
        },
        {
            "id": "c2",
            "document_id": "doc_cs338",
            "document_name": "CS338.pdf",
            "page": 2,
            "chunk_index": 1,
            "text": "[2. Methodology > 2.2 Adaptive Soft Masking]\nĐề xuất mặt nạ mềm thích ứng.",
        },
    ])
    mock_retriever = MagicMock()
    mock_retriever.bm25_service = bm25
    mock_retriever._resolve_document_ids.return_value = (["doc_cs338"], ["CS338.pdf"])

    overview_tool = make_document_overview_tool(mock_retriever)
    res = overview_tool.invoke({"document_name": "CS338"})
    print(f"  -> Cấu trúc trích xuất được:\n{res}")

    assert "CS338.pdf" in res
    assert "1. Introduction" in res
    assert "2. Methodology" in res
    print("  -> Kết quả: ✅ PASSED (Trích xuất mục lục và bố cục đề mục thành công).")


if __name__ == "__main__":
    print("=" * 70)
    print("  BẮT ĐẦU CHẠY KIỂM THỬ: test_unit_context_and_crag.py")
    print("=" * 70)
    test_small_to_big_window_boundary_edge_cases()
    test_corrective_rag_three_confidence_tiers()
    test_calculator_tool_ast_safety_and_math()
    test_document_overview_tool_hierarchy()
    print("\n" + "=" * 70)
    print("  TỔNG KẾT: TẤT CẢ 4 BÀI KIỂM THỬ CONTEXT & CRAG ĐÃ VƯỢT QUA! ✅")
    print("=" * 70)
