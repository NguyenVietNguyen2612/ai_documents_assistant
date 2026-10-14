from typing import Optional
from langgraph.graph import MessagesState

# Sử dụng MessagesState mặc định của LangGraph để hỗ trợ chuẩn Tool-Calling
# kết hợp với kết quả định tuyến từ Laya Decision Router (System 1)
class AgentState(MessagesState):
    route_action: Optional[str] = None
    route_reason: Optional[str] = None


RAGState = AgentState


