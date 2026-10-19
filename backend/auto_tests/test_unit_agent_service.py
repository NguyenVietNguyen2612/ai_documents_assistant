import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from app.graph.rag_graph import (
    _count_current_turn_tool_calls,
    _sanitize_messages_for_llm,
    _rescue_text_tool_calls,
)


def test_count_current_turn_tool_calls_enforces_limit_3():
    print("\n[TEST 1/3] Kiểm tra đếm số lần gọi tool chỉ trong lượt hiện tại (Max 3 turns)...")

    # Giả lập lịch sử 2 lượt:
    # Lượt 1: User hỏi -> Tool gọi 2 lần -> AI trả lời
    # Lượt 2: User hỏi -> Tool gọi 1 lần
    messages = [
        HumanMessage(content="Lượt 1: Câu hỏi cũ"),
        AIMessage(content="", tool_calls=[{"name": "retrieve_documents", "args": {}, "id": "call_1"}]),
        ToolMessage(content="Kết quả tool 1", tool_call_id="call_1"),
        AIMessage(content="", tool_calls=[{"name": "retrieve_documents", "args": {}, "id": "call_2"}]),
        ToolMessage(content="Kết quả tool 2", tool_call_id="call_2"),
        AIMessage(content="<think>Phân tích</think>Câu trả lời lượt 1."),
        HumanMessage(content="Lượt 2: Câu hỏi mới hiện tại"),
        AIMessage(content="", tool_calls=[{"name": "retrieve_documents", "args": {}, "id": "call_3"}]),
        ToolMessage(content="Kết quả tool 3", tool_call_id="call_3"),
    ]

    count = _count_current_turn_tool_calls(messages)
    print(f"  -> Lịch sử gồm 3 ToolMessages tổng cộng, nhưng chỉ có 1 ToolMessage sau HumanMessage mới nhất.")
    print(f"  -> Kết quả hàm đếm: {count} (Kỳ vọng: 1)")

    assert count == 1, f"Kỳ vọng 1 nhưng đếm được {count}"

    # Bổ sung thêm 2 tool calls nữa vào lượt 2
    messages.extend([
        AIMessage(content="", tool_calls=[{"name": "retrieve_documents", "args": {}, "id": "call_4"}]),
        ToolMessage(content="Kết quả tool 4", tool_call_id="call_4"),
        AIMessage(content="", tool_calls=[{"name": "retrieve_documents", "args": {}, "id": "call_5"}]),
        ToolMessage(content="Kết quả tool 5", tool_call_id="call_5"),
    ])

    count_max = _count_current_turn_tool_calls(messages)
    print(f"  -> Sau khi thêm 2 tools vào lượt hiện tại: {count_max} (Kỳ vọng: 3 - Chạm ngưỡng tối đa)")
    assert count_max == 3
    print("  -> Kết quả: ✅ PASSED (Phân tách chính xác lượt hiện tại và lượt cũ).")


def test_sanitize_messages_removes_old_think_tags():
    print("\n[TEST 2/3] Kiểm tra lọc sạch khối <think>...</think> từ lượt cũ (Multi-turn Sanitization)...")

    raw_ai_content = "<think>Đây là suy nghĩ nội tại của mô hình cần giấu.</think>Đây là câu trả lời thực tế."
    messages = [
        SystemMessage(content="System prompt"),
        HumanMessage(content="Câu hỏi lượt 1"),
        AIMessage(content=raw_ai_content),
        HumanMessage(content="Câu hỏi lượt 2"),
    ]

    sanitized = _sanitize_messages_for_llm(messages)
    ai_msg_clean = sanitized[2]

    print(f"  -> Trước khi lọc: '{raw_ai_content}'")
    print(f"  -> Sau khi lọc:   '{ai_msg_clean.content}'")

    assert "<think>" not in ai_msg_clean.content
    assert "</think>" not in ai_msg_clean.content
    assert "Đây là câu trả lời thực tế." in ai_msg_clean.content
    print("  -> Kết quả: ✅ PASSED (Làm sạch thẻ think thành công, chống nhiễm token).")


def test_rescue_text_tool_calls_from_llm_output():
    print("\n[TEST 3/3] Kiểm tra tự động phục hồi tool_calls dạng text do LLM vô tình xuất...")

    # Khi Gemini hoặc LLM xuất text "call:default_api:retrieve_documents{...}" thay vì payload structured
    mock_ai_resp = AIMessage(
        content='Tôi sẽ tra cứu tài liệu: call:default_api:retrieve_documents{"query": "FreqNet", "document_name": "CS338.pdf"}'
    )

    valid_tools = {"retrieve_documents", "list_uploaded_documents", "calculate_metrics"}
    rescued = _rescue_text_tool_calls(mock_ai_resp, valid_tools)

    print(f"  -> Text sau khi bóc tách: '{rescued.content}'")
    print(f"  -> Tool calls phục hồi được: {len(rescued.tool_calls)}")
    for tc in rescued.tool_calls:
        print(f"     Tool: {tc['name']} | Args: {tc['args']}")

    assert len(rescued.tool_calls) == 1
    assert rescued.tool_calls[0]["name"] == "retrieve_documents"
    assert rescued.tool_calls[0]["args"]["query"] == "FreqNet"
    assert rescued.tool_calls[0]["args"]["document_name"] == "CS338.pdf"
    assert "call:" not in rescued.content
    print("  -> Kết quả: ✅ PASSED (Phục hồi cấu trúc gọi tool thành công 100%).")


if __name__ == "__main__":
    print("=" * 70)
    print("  BẮT ĐẦU CHẠY KIỂM THỬ: test_unit_agent_service.py")
    print("=" * 70)
    test_count_current_turn_tool_calls_enforces_limit_3()
    test_sanitize_messages_removes_old_think_tags()
    test_rescue_text_tool_calls_from_llm_output()
    print("\n" + "=" * 70)
    print("  TỔNG KẾT: TẤT CẢ 3 BÀI KIỂM THỬ AGENT SERVICE ĐÃ VƯỢT QUA! ✅")
    print("=" * 70)
