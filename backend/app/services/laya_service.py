import os
from typing import Optional
from app.graph.agent_decision import AgentDecision


class LayaDecisionService:
    """
    Decision Service powered by Laya (Convai Innovations).
    Laya is a non-autoregressive 'System 1' Decision Model (Jev-like architecture),
    evaluating state and typed questions in a single forward pass without generating free-form text.
    """

    DEFAULT_MIN_CONFIDENCE = 0.75

    def __init__(
        self,
        model_name: Optional[str] = None,
        fallback_llm = None,
        min_confidence: Optional[float] = None,
    ):
        self.model_name = model_name or os.getenv(
            "LAYA_MODEL", "convaiinnovations/laya"
        )
        self.fallback_llm = fallback_llm
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else float(os.getenv("LAYA_MIN_CONFIDENCE", str(self.DEFAULT_MIN_CONFIDENCE)))
        )
        self.agent = None
        self._load_agent()

    def _load_agent(self):
        try:
            import laya
            print(f"[LayaDecisionService] Loading Laya Decision Model from '{self.model_name}'...")
            self.agent = laya.load(self.model_name)
            print("[LayaDecisionService] Laya model loaded successfully.")
        except ImportError:
            print(
                "[LayaDecisionService] Notice: 'laya' package is not installed. "
                "Will use fallback LLM if available. To install: pip install laya"
            )
        except Exception as e:
            print(
                f"[LayaDecisionService] Warning: Failed to load Laya model '{self.model_name}': {e}. "
                "Will use fallback LLM if available."
            )

    def decide(self, question: str) -> AgentDecision:
        """
        Determines the next action ('retrieve' or 'answer') for the given question.
        Uses Laya non-autoregressive decision model with typed choices and confidence threshold.
        """
        if self.agent is not None:
            try:
                state = {
                    "question": question,
                }
                questions = {
                    "action": {
                        "type": "choice",
                        "instructions": (
                            "Decide whether the user's question requires retrieving information "
                            "from uploaded documents, or can be answered directly."
                        ),
                        "criteria": {
                            "retrieve": (
                                "Information from uploaded documents, papers, files, "
                                "or domain-specific document context is needed to answer."
                            ),
                            "answer": (
                                "Greetings, conversational chit-chat, general common knowledge, "
                                "or questions that can be answered directly without document lookup."
                            ),
                        },
                    }
                }

                result = self.agent.predict(state, questions)
                choice = result["answers"]["action"]["choice"]
                confidence = float(result["answers"]["action"].get("confidence", 1.0))

                # Validate choice matches expected literal values
                if choice not in ("retrieve", "answer"):
                    choice = "retrieve"

                # Enforce minimum confidence threshold when skipping retrieval ('answer')
                if choice == "answer" and confidence < self.min_confidence:
                    return AgentDecision(
                        action="retrieve",
                        reason=(
                            f"Laya Decision Model: 'answer' confidence ({confidence:.2f}) "
                            f"below threshold ({self.min_confidence:.2f}) -> safe fallback to 'retrieve'"
                        ),
                    )

                return AgentDecision(
                    action=choice,
                    reason=f"Laya Decision Model: classified as '{choice}' (confidence: {confidence:.2f})",
                )
            except Exception as e:
                print(f"[LayaDecisionService] Error during Laya prediction: {e}. Falling back...")

        # Fallback to LLM if Laya is unavailable or encounters an error
        if self.fallback_llm is not None:
            prompt = f"""
You are the decision-making agent of a RAG system.

Your job is to decide whether the user's question
requires retrieving information from the uploaded documents.

Choose:
- retrieve: when information from the documents is needed.
- answer: when the question can be answered without retrieving documents.

Do not answer the user's question.
Only decide the next action.

Question:
{question}
"""
            try:
                if hasattr(self.fallback_llm, "generate_structured"):
                    return self.fallback_llm.generate_structured(
                        prompt,
                        response_model=AgentDecision,
                    )
                elif hasattr(self.fallback_llm, "with_structured_output"):
                    structured_llm = self.fallback_llm.with_structured_output(AgentDecision)
                    decision = structured_llm.invoke(prompt)
                    if isinstance(decision, AgentDecision):
                        return decision
            except Exception as e:
                print(f"[LayaDecisionService] Fallback LLM structured output error: {e}")

        # Fast System-1 heuristic for pure greetings / chitchat when Laya & fallback LLM are unavailable
        if self._is_pure_conversational(question):
            return AgentDecision(
                action="answer",
                reason="System-1 Fast Heuristic: classified as conversational greeting/chitchat (confidence: 0.95)",
            )

        # Default safe fallback action
        return AgentDecision(
            action="retrieve",
            reason="Default fallback: routed to document retrieval pipeline",
        )

    @staticmethod
    def _is_pure_conversational(question: str) -> bool:
        """Nhận diện nhanh các câu chào hỏi/xã giao ngắn không yêu cầu tra cứu tài liệu."""
        import re
        if not question:
            return False
        cleaned = re.sub(r"[^\w\s\u00C0-\u024F\u1EA0-\u1EF9]", " ", question.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        words = cleaned.split()
        if len(words) == 0 or len(words) > 7:
            return False

        # Nếu có từ khóa ám chỉ tài liệu/bài báo/nội dung chuyên môn thì không phải xã giao thuần túy
        doc_keywords = {
            "tài liệu", "file", "pdf", "bài báo", "paper", "document", "mô hình",
            "model", "kiến trúc", "architecture", "dataset", "kết quả", "số liệu",
            "accuracy", "freqnet", "masking", "chương", "mục", "trang", "section", "table",
        }
        if any(kw in cleaned for kw in doc_keywords):
            return False

        conversational_patterns = [
            r"^(xin )?chào(\s+bạn|\s+ad|\s+ai|\s+nhé|\s+nha|\s+ạ)*$",
            r"^(hi|hello|hey|alo)(\s+there|\s+bạn|\s+ai)*$",
            r"^(cảm ơn|cám ơn|thank|thanks|thank you)(\s+bạn|\s+nhiều|\s+nhé|\s+ạ)*$",
            r"^(tạm biệt|bye|goodbye)(\s+bạn|\s+nhé)*$",
            r"^(bạn là ai|bạn tên gì|bạn làm được gì|bạn có thể làm gì|who are you|what can you do)(\s+vậy|\s+thế|\s+ạ)*$",
        ]
        return any(re.match(pattern, cleaned) for pattern in conversational_patterns)


# Singleton instance dùng chung trong hệ thống
laya_service = LayaDecisionService()


