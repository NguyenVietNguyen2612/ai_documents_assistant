import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ImportError:
    pytest = None

from app.services.laya_service import LayaDecisionService
from app.graph.agent_decision import AgentDecision

# Vô hiệu hóa nạp lại checkpoint 800MB từ đĩa trong unit tests để tăng tốc thực thi (<0.1s)
LayaDecisionService._load_agent = lambda self: None


def test_laya_decision_service_with_agent_mock():
    print("\n[TEST 1/6] Kiểm tra Laya Decision Model dự đoán 'retrieve' (độ tự tin 0.96)...")
    service = LayaDecisionService()
    mock_agent = MagicMock()
    mock_agent.predict.return_value = {
        "answers": {
            "action": {
                "choice": "retrieve",
                "confidence": 0.96
            }
        }
    }
    service.agent = mock_agent

    query = "What is FreqNet architecture and its main processing blocks?"
    decision = service.decide(query)
    print(f"  -> Câu hỏi: '{query}'")
    print(f"  -> Quyết định: action='{decision.action}' | reason='{decision.reason}'")

    assert isinstance(decision, AgentDecision)
    assert decision.action == "retrieve"
    assert "confidence: 0.96" in decision.reason
    print("  -> Kết quả: ✅ PASSED")


def test_laya_decision_service_answer_choice():
    print("\n[TEST 2/6] Kiểm tra Laya Decision Model dự đoán 'answer' (chào hỏi xã giao)...")
    service = LayaDecisionService()
    mock_agent = MagicMock()
    mock_agent.predict.return_value = {
        "answers": {
            "action": {
                "choice": "answer",
                "confidence": 0.99
            }
        }
    }
    service.agent = mock_agent

    query = "Hello, who are you and what can you help me with?"
    decision = service.decide(query)
    print(f"  -> Câu hỏi: '{query}'")
    print(f"  -> Quyết định: action='{decision.action}' | reason='{decision.reason}'")

    assert decision.action == "answer"
    assert "confidence: 0.99" in decision.reason
    print("  -> Kết quả: ✅ PASSED")


def test_laya_fast_conversational_heuristic():
    print("\n[TEST 3/6] Kiểm tra Fast Heuristic System-1 phân loại chào hỏi tức thì (<1ms)...")
    service = LayaDecisionService()
    service.agent = None
    service.fallback_llm = None

    greeting = "Xin chào bạn!"
    g_decision = service.decide(greeting)
    print(f"  -> Câu hỏi chào: '{greeting}' => action='{g_decision.action}' (Kỳ vọng: answer)")
    assert g_decision.action == "answer"

    doc_q = "Kiến trúc của mô hình FreqNet gồm những khối nào?"
    d_decision = service.decide(doc_q)
    print(f"  -> Câu hỏi tài liệu: '{doc_q}' => action='{d_decision.action}' (Kỳ vọng: retrieve)")
    assert d_decision.action == "retrieve"
    print("  -> Kết quả: ✅ PASSED")


def test_laya_low_confidence_answer_fallback_to_retrieve():
    print("\n[TEST 4/6] Kiểm tra Ngưỡng Tự tin Tối thiểu (Confidence Guardrail): Phạt độ tự tin thấp...")
    # Khi Laya chọn 'answer' nhưng độ tự tin chỉ 0.60 < 0.75 -> Tự động ép về 'retrieve' để không bỏ sót thông tin
    service = LayaDecisionService(min_confidence=0.75)
    mock_agent = MagicMock()
    mock_agent.predict.return_value = {
        "answers": {
            "action": {
                "choice": "answer",
                "confidence": 0.60
            }
        }
    }
    service.agent = mock_agent

    query = "Cho tôi hỏi về phần mềm lọc tần số trong tài liệu?"
    decision = service.decide(query)
    print(f"  -> Laya raw choice='answer' (conf=0.60 < 0.75) => Ép sang action='{decision.action}'")
    print(f"  -> Lý do: {decision.reason}")

    assert decision.action == "retrieve"
    assert "below threshold (0.75)" in decision.reason
    print("  -> Kết quả: ✅ PASSED")


def test_laya_decision_fallback_llm():
    print("\n[TEST 5/6] Kiểm tra Fallback sang LLM khi mô hình Laya cục bộ gặp sự cố...")
    mock_llm = MagicMock()
    mock_llm.generate_structured.return_value = AgentDecision(
        action="retrieve",
        reason="Gemini fallback decision"
    )

    service = LayaDecisionService(fallback_llm=mock_llm)
    service.agent = None

    decision = service.decide("Summarize section 3.")
    print(f"  -> Fallback action='{decision.action}' | reason='{decision.reason}'")
    assert decision.action == "retrieve"
    assert decision.reason == "Gemini fallback decision"
    assert mock_llm.generate_structured.called
    print("  -> Kết quả: ✅ PASSED")


def test_multi_tools_and_corrective_rag():
    print("\n[TEST 6/6] Kiểm tra AST Calculator & Cảnh báo Corrective RAG...")
    from app.tools import make_calculator_tool
    from app.services.context_builder import ContextBuilder

    # 1. Tính toán số học bằng AST Calculator
    calc_tool = make_calculator_tool()
    expr = "round(74.03 - 72.16, 2)"
    calc_out = calc_tool.invoke({"expression": expr})
    print(f"  -> Máy tính AST: '{expr}' = {calc_out}")
    assert "1.87" in calc_out

    # 2. Cảnh báo Corrective RAG khi điểm < 0.40
    cb = ContextBuilder()
    low_prob_results = [[{
        "rerank_score": 0.25,
        "entity": {"document_name": "CS338.pdf", "page": 1, "text": "Nội dung ít liên quan"}
    }]]
    built_ctx = cb.build(low_prob_results)
    assert "[ĐÁNH GIÁ CORRECTIVE RAG" in built_ctx
    print("  -> Cảnh báo Corrective RAG kích hoạt chính xác khi điểm rerank = 0.25")

    print("  -> Kết quả: ✅ PASSED")


if __name__ == "__main__":
    print("=" * 70)
    print("  BẮT ĐẦU CHẠY KIỂM THỬ: test_unit_laya.py (Laya System 1 Router)")
    print("=" * 70)
    test_laya_decision_service_with_agent_mock()
    test_laya_decision_service_answer_choice()
    test_laya_fast_conversational_heuristic()
    test_laya_low_confidence_answer_fallback_to_retrieve()
    test_laya_decision_fallback_llm()
    test_multi_tools_and_corrective_rag()
    print("\n" + "=" * 70)
    print("  TỔNG KẾT: TẤT CẢ 6 BÀI KIỂM THỬ LAYA ROUTER ĐÃ VƯỢT QUA! ✅")
    print("=" * 70)
