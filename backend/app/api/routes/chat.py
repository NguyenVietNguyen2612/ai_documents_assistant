from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)

from app.services.agent_service import (
    AgentService,
)

from app.graph.rag_graph import graph


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


agent_service = AgentService(
    graph=graph
)


@router.post(
    "",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    res = agent_service.ask(
        request.question,
        request.history
    )

    return ChatResponse(
        question=request.question,
        answer=res["answer"],
    )


@router.post(
    "/stream",
)
async def chat_stream(
    request: ChatRequest,
):
    return StreamingResponse(
        agent_service.ask_stream(
            request.question,
            request.history
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )