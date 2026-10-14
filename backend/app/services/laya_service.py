import os
from typing import Optional
from app.graph.agent_decision import AgentDecision


class LayaDecisionService:
    """
    Decision Service powered by Laya (Convai Innovations).
    Laya is a non-autoregressive 'System 1' Decision Model (Jev-like architecture),
    evaluating state and typed questions in a single forward pass without generating free-form text.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        fallback_llm = None,
    ):
        self.model_name = model_name or os.getenv(
            "LAYA_MODEL", "convaiinnovations/laya"
        )
        self.fallback_llm = fallback_llm
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
        Uses Laya non-autoregressive decision model with typed choices.
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
                confidence = result["answers"]["action"].get("confidence", 1.0)

                # Validate choice matches expected literal values
                if choice not in ("retrieve", "answer"):
                    choice = "retrieve"

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
            return self.fallback_llm.generate_structured(
                prompt,
                response_model=AgentDecision,
            )

        # Default safe fallback action
        return AgentDecision(
            action="retrieve",
            reason="Default fallback: Laya agent and fallback LLM unavailable",
        )
