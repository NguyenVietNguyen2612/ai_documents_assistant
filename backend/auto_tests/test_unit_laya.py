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
