from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Optional

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SessionItem,
    CreateSessionRequest,
    UpdateSessionRequest,
    RollbackRequest,
    RollbackResponse,
    CheckpointInfo
)
from app.services.agent_service import agent_service
from app.services.db_service import db_service

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

# --- Chat endpoints ---

@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):
    res = await agent_service.ask(
        request.question,
        request.history,
        session_id=request.session_id or "default",
        checkpoint_id=request.checkpoint_id,
        document_filter=request.document_filter,
    )
    return ChatResponse(
        question=request.question,
        answer=res["answer"],
        session_id=res.get("session_id"),
        checkpoint_id=res.get("checkpoint_id")
    )


@router.post(
    "/stream",
)
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        agent_service.ask_stream(
            question=request.question,
            history=request.history,
            session_id=request.session_id or "default",
            checkpoint_id=request.checkpoint_id,
            rollback_from_message_id=request.rollback_from_message_id,
            document_filter=request.document_filter,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# --- Sessions endpoints ---

@router.get(
    "/sessions",
    response_model=List[SessionItem],
)
def get_sessions():
    return db_service.get_all_sessions()


@router.post(
    "/sessions",
    response_model=SessionItem,
)
def create_session(request: CreateSessionRequest):
    sess = db_service.create_session(title=request.title or "Cuộc trò chuyện mới")
    return sess


@router.delete(
    "/sessions/{session_id}",
)
def delete_session(session_id: str):
    db_service.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@router.patch(
    "/sessions/{session_id}",
    response_model=SessionItem,
)
def update_session(session_id: str, request: UpdateSessionRequest):
    db_service.update_session_title(session_id, request.title)
    sessions = db_service.get_all_sessions()
    for s in sessions:
        if s["id"] == session_id:
            return s
    raise HTTPException(status_code=404, detail="Session not found")


@router.get(
    "/sessions/{session_id}/messages",
)
def get_session_messages(session_id: str):
    messages = db_service.get_session_messages(session_id)
    return messages

# --- Checkpoints & Rollback endpoints ---

@router.get(
    "/checkpoints",
    response_model=List[CheckpointInfo],
)
async def get_checkpoints(session_id: str = Query(..., description="ID của phiên hội thoại")):
    return await agent_service.get_checkpoints(session_id)


@router.post(
    "/rollback",
    response_model=RollbackResponse,
)
async def rollback_chat(request: RollbackRequest):
    remaining_msgs = await agent_service.rollback(
        session_id=request.session_id,
        checkpoint_id=request.checkpoint_id,
        message_id=request.message_id
    )
    return RollbackResponse(
        status="success",
        session_id=request.session_id,
        checkpoint_id=request.checkpoint_id,
        messages=remaining_msgs
    )