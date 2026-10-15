import pytest
from unittest.mock import MagicMock, patch
from app.services.laya_service import LayaDecisionService
from app.graph.agent_decision import AgentDecision


def test_laya_decision_service_with_agent_mock():
    # 1. Arrange: Giả lập đối tượng Laya agent
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

    # 2. Act
    decision = service.decide("What is FreqNet architecture?")

    # 3. Assert
    assert isinstance(decision, AgentDecision)
    assert decision.action == "retrieve"
    assert "confidence: 0.96" in decision.reason


def test_laya_decision_service_answer_choice():
    # Test khi Laya trả về choice="answer" (câu hỏi thông thường, chào hỏi)
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

    decision = service.decide("Hello, who are you?")
    assert decision.action == "answer"
    assert "confidence: 0.99" in decision.reason


def test_laya_decision_fallback_llm():
    # Test khi agent bị lỗi hoặc chưa load, fallback sang LLMService
    mock_llm = MagicMock()
    mock_llm.generate_structured.return_value = AgentDecision(
        action="retrieve",
        reason="Gemini fallback decision"
    )

    service = LayaDecisionService(fallback_llm=mock_llm)
    service.agent = None  # Không có agent Laya

    decision = service.decide("Summarize section 3.")
    assert decision.action == "retrieve"
    assert decision.reason == "Gemini fallback decision"
    assert mock_llm.generate_structured.called


def test_laya_fast_conversational_heuristic():
    service = LayaDecisionService()
    service.agent = None
    service.fallback_llm = None

    greeting_decision = service.decide("Xin chào bạn!")
    assert greeting_decision.action == "answer"

    doc_decision = service.decide("Kiến trúc của mô hình FreqNet gồm những khối nào?")
    assert doc_decision.action == "retrieve"


def test_multi_tools_and_corrective_rag():
    # Test import từ app.tools và từ các module riêng biệt
    from app.tools import (
        make_calculator_tool,
        make_document_overview_tool,
        make_list_documents_tool,
        make_retrieve_tool,
        make_agent_tools,
    )
    from app.tools.calculator_tool import make_calculator_tool as make_calc_direct
    from app.tools.retrieve_tool import make_retrieve_tool as make_retrieve_direct
    from app.tools.retrieval_tool import make_agent_tools as make_agent_tools_compat
    from app.services.context_builder import ContextBuilder
    from app.services.bm25_service import BM25Service

    # 1. Kiểm tra công cụ calculate_metrics
    calc_tool = make_calculator_tool()
    calc_out = calc_tool.invoke({"expression": "round(74.03 - 72.16, 2)"})
    assert "1.87" in calc_out

    # 2. Kiểm tra công cụ get_document_overview
    bm25 = BM25Service()
    bm25.index_documents([
        {
            "id": "c1",
            "document_id": "doc_cs338",
            "document_name": "CS338.pdf",
            "page": 1,
            "chunk_index": 0,
            "text": "[1. Introduction > 1.1 FreqNet]\nFreqNet là kiến trúc phát hiện Deepfake trong miền tần số.",
        },
        {
            "id": "c2",
            "document_id": "doc_cs338",
            "document_name": "CS338.pdf",
            "page": 2,
            "chunk_index": 1,
            "text": "[2. Proposed Method > 2.1 Adaptive Soft Masking]\nAdaptive Soft Masking cải thiện độ chính xác lên 74.03%.",
        },
    ])
    mock_retriever = MagicMock()
    mock_retriever.bm25_service = bm25
    mock_retriever._resolve_document_ids.return_value = (["doc_cs338"], ["CS338.pdf"])

    overview_tool = make_document_overview_tool(mock_retriever)
    overview_out = overview_tool.invoke({"document_name": "CS338"})
    assert "CS338.pdf" in overview_out
    assert "1. Introduction > 1.1 FreqNet" in overview_out
    assert "Adaptive Soft Masking" in overview_out

    # 3. Kiểm tra Corrective RAG cảnh báo khi điểm chuẩn hóa < 0.40 (cả logit âm lẫn xác suất thấp)
    cb = ContextBuilder()
    low_prob_results = [[{
        "rerank_score": 0.25,
        "entity": {"document_name": "CS338.pdf", "page": 1, "text": "Nội dung ít liên quan"}
    }]]
    built_ctx = cb.build(low_prob_results)
    assert "0.25/1.00 (Thấp)" in built_ctx
    assert "[ĐÁNH GIÁ CORRECTIVE RAG]" in built_ctx

    high_prob_results = [[{
        "rerank_score": 0.88,
        "entity": {"document_name": "CS338.pdf", "page": 1, "text": "Nội dung rất khớp"}
    }]]
    high_ctx = cb.build(high_prob_results)
    assert "0.88/1.00 (Cao)" in high_ctx
    assert "[ĐÁNH GIÁ CORRECTIVE RAG]" not in high_ctx


def test_laya_low_confidence_answer_fallback_to_retrieve():
    # Khi Laya dự đoán 'answer' nhưng confidence < 0.75 -> Tự động ép về 'retrieve'
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

    decision = service.decide("Cho tôi hỏi về phần mềm lọc tần số?")
    assert decision.action == "retrieve"
    assert "below threshold (0.75)" in decision.reason



