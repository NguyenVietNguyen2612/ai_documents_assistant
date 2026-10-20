import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from app.graph.rag_graph import _sanitize_messages_for_llm
from app.services.semantic_cache import SemanticCacheService


def test_semantic_cache_store_and_lookup_hit():
    print("\n[CACHE TEST 1/4] Kiểm tra Semantic Cache lưu trữ & tra cứu thành công (Cache HIT <50ms)...")
    cache = SemanticCacheService(collection_name="test_semantic_cache_suite")

    query = "Mô hình FreqNet phát hiện deepfake như thế nào?"
    answer = "FreqNet trích xuất đặc trưng miền tần số kết hợp HFRI, HFRF và FCL để phát hiện dấu vết giả mạo."
    sources = [{"document_name": "CS338.pdf", "page": 1, "snippet": "FreqNet frequency domain analysis"}]

    # 1. Lưu vào cache
    stored = cache.store(query=query, answer=answer, sources=sources)
    print(f"  -> Lưu câu hỏi gốc: '{query}' => Result: {stored}")
    assert stored is True

    # 2. Tra cứu câu hỏi tương đương ngữ nghĩa
    equivalent_query = "Cách thức mô hình FreqNet nhận diện ảnh deepfake?"
    hit = cache.lookup(equivalent_query, similarity_threshold=0.85)
    print(f"  -> Tra cứu câu tương đương: '{equivalent_query}'")
    assert hit is not None, "Kỳ vọng tìm thấy Cache HIT nhưng trả về None"
    print(f"  -> Trạng thái: HIT! (Similarity: {hit['similarity']:.4f} | Độ trễ: {hit['latency_ms']}ms)")
    print(f"  -> Câu trả lời trả về: '{hit['answer'][:60]}...'")
    assert "FreqNet" in hit["answer"]
    assert len(hit["sources"]) == 1
    print("  -> Kết quả: ✅ PASSED (Semantic Cache phản hồi siêu tốc).")

    # Dọn dẹp collection test
    cache.clear()


def test_semantic_cache_miss_for_unrelated_query():
    print("\n[CACHE TEST 2/4] Kiểm tra Semantic Cache từ chối câu hỏi không liên quan (Cache MISS)...")
    cache = SemanticCacheService(collection_name="test_semantic_cache_suite")

    cache.store(
        query="Kiến trúc Kubernetes gồm các thành phần nào?",
        answer="Gồm Control Plane (API server, etcd, scheduler) và Worker Nodes (kubelet, kube-proxy)."
    )

    unrelated_query = "Hướng dẫn làm bánh gato sinh nhật đơn giản tại nhà?"
    hit = cache.lookup(unrelated_query, similarity_threshold=0.95)
    print(f"  -> Câu hỏi không liên quan: '{unrelated_query}'")
    print(f"  -> Kết quả tra cứu: {hit}")
    assert hit is None, f"Kỳ vọng Cache MISS nhưng nhận được hit: {hit}"
    print("  -> Kết quả: ✅ PASSED (Không xảy ra False-Positive khi câu hỏi khác ngữ nghĩa).")

    cache.clear()


def test_semantic_cache_scoped_document_isolation():
    print("\n[CACHE TEST 3/4] Kiểm tra Cô lập Cache theo phạm vi tài liệu (@tài_liệu)...")
    cache = SemanticCacheService(collection_name="test_semantic_cache_suite")

    # Lưu câu hỏi có khoanh vùng tài liệu Doc_A.pdf
    cache.store(
        query="Tóm tắt chương 1 của tài liệu",
        answer="Tóm tắt chương 1 theo Doc_A: Khởi đầu công nghệ AI.",
        scope_doc="Doc_A.pdf"
    )

    # Tra cứu cùng câu hỏi nhưng ở tài liệu Doc_B.pdf
    hit_diff_scope = cache.lookup("Tóm tắt chương 1 của tài liệu", scope_doc="Doc_B.pdf")
    print(f"  -> Tra cứu cùng câu hỏi nhưng scope='Doc_B.pdf' => Kết quả: {hit_diff_scope}")
    assert hit_diff_scope is None, "Kỳ vọng không hit do khác scope tài liệu"

    # Tra cứu đúng scope Doc_A.pdf
    hit_same_scope = cache.lookup("Tóm tắt chương 1 của tài liệu", scope_doc="Doc_A.pdf")
    print(f"  -> Tra cứu đúng scope='Doc_A.pdf' => Kết quả: {hit_same_scope is not None}")
    assert hit_same_scope is not None
    assert "Doc_A" in hit_same_scope["answer"]
    print("  -> Kết quả: ✅ PASSED (Cô lập cache chính xác theo từng tài liệu).")

    cache.clear()


def test_tool_message_pruning_and_sliding_window():
    print("\n[CACHE TEST 4/4] Kiểm tra Nén ToolMessage cũ (Pruning) & Cắt tỉa Sliding Window...")

    # Giả lập lịch sử 3 lượt hỏi đáp:
    # Lượt 1: User hỏi -> Tool trả về đoạn text cực dài 600 ký tự -> AI trả lời
    # Lượt 2: User hỏi -> Tool trả về đoạn text 500 ký tự -> AI trả lời
    # Lượt 3 (hiện tại): User hỏi câu mới nhất -> Tool lượt hiện tại
    huge_context_1 = "DỮ LIỆU NGỮ CẢNH RẤT DÀI CỦA LƯỢT 1: " + ("FreqNet analysis. " * 30)
    huge_context_2 = "DỮ LIỆU NGỮ CẢNH RẤT DÀI CỦA LƯỢT 2: " + ("Deepfake masking details. " * 30)
    current_turn_context = "NGỮ CẢNH LƯỢT HIỆN TẠI (CẦN GIỮ NGUYÊN VẸN 100%): " + ("Fresh context. " * 5)

    messages = [
        SystemMessage(content="System instruction"),
        HumanMessage(content="Lượt 1: FreqNet là gì?"),
        ToolMessage(content=huge_context_1, tool_call_id="call_1"),
        AIMessage(content="<think>Suy nghĩ lượt 1</think>FreqNet là mạng tần số."),
        HumanMessage(content="Lượt 2: Cơ chế Soft Masking hoạt động thế nào?"),
        ToolMessage(content=huge_context_2, tool_call_id="call_2"),
        AIMessage(content="<think>Suy nghĩ lượt 2</think>Nó áp dụng mặt nạ mềm thích ứng."),
        HumanMessage(content="Lượt 3: Hãy so sánh kết quả 2 phương pháp trên?"),
        ToolMessage(content=current_turn_context, tool_call_id="call_3"),
    ]

    sanitized = _sanitize_messages_for_llm(messages, max_window=10)

    # 1. Kiểm tra ToolMessage lượt 1 và 2 đã bị thu gọn
    t1_pruned = sanitized[2]
    t2_pruned = sanitized[5]
    print(f"  -> Độ dài ToolMessage lượt 1: Gốc={len(huge_context_1)} ký tự => Sau khi nén={len(t1_pruned.content)} ký tự")
    print(f"  -> Nội dung ToolMessage lượt 1: '{t1_pruned.content}'")
    assert len(t1_pruned.content) < 180
    assert "Ngữ cảnh đã được tổng hợp" in t1_pruned.content

    print(f"  -> Độ dài ToolMessage lượt 2: Gốc={len(huge_context_2)} ký tự => Sau khi nén={len(t2_pruned.content)} ký tự")
    assert len(t2_pruned.content) < 180

    # 2. Kiểm tra ToolMessage lượt hiện tại (lượt 3) vẫn giữ nguyên 100%
    current_tool = sanitized[8]
    print(f"  -> ToolMessage lượt hiện tại (lượt 3): {len(current_tool.content)} ký tự (Giữ nguyên vẹn)")
    assert current_tool.content == current_turn_context

    # 3. Kiểm tra AIMessage cũ đã được lọc thẻ <think>
    ai_1 = sanitized[3]
    assert "<think>" not in ai_1.content
    assert "FreqNet là mạng tần số." in ai_1.content

    # 4. Kiểm tra Sliding Window khi vượt quá giới hạn
    long_chain = [SystemMessage(content="System root")] + [HumanMessage(content=f"Q{i}") for i in range(25)]
    windowed = _sanitize_messages_for_llm(long_chain, max_window=8)
    print(f"  -> Chuỗi dài 26 messages => Sau Sliding Window (max=8): {len(windowed)} messages")
    assert len(windowed) == 9  # 1 System + 8 Recent
    assert isinstance(windowed[0], SystemMessage)
    assert windowed[-1].content == "Q24"
    print("  -> Kết quả: ✅ PASSED (Nén ToolMessage và trượt cửa sổ hội thoại chính xác 100%).")


if __name__ == "__main__":
    print("=" * 70)
    print("  BẮT ĐẦU CHẠY KIỂM THỬ: test_unit_semantic_cache_and_pruning.py")
    print("=" * 70)
    test_semantic_cache_store_and_lookup_hit()
    test_semantic_cache_miss_for_unrelated_query()
    test_semantic_cache_scoped_document_isolation()
    test_tool_message_pruning_and_sliding_window()
    print("\n" + "=" * 70)
    print("  TỔNG KẾT: TẤT CẢ 4 BÀI KIỂM THỬ SEMANTIC CACHE & PRUNING ĐÃ VƯỢT QUA! ✅")
    print("=" * 70)
