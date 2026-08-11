from pydantic import BaseModel
from typing import Literal


class AgentDecision(BaseModel):

    action: Literal[
        "retrieve",
        "answer",
    ]

    reason: str