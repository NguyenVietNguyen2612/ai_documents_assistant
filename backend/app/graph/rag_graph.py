import os
if os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")
    os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
    os.environ.setdefault("CURL_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")

import re
import json
import uuid
import aiosqlite
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.graph.state import AgentState
from app.tools import make_retrieve_tool, make_agent_tools
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever
from app.services.context_builder import ContextBuilder
from app.services.bm25_service import bm25_service
from app.services.reranker_service import reranker_service
from app.services.laya_service import laya_service, LayaDecisionService


def _count_current_turn_tool_calls(messages) -> int:
    """Đếm số lần gọi tool chỉ trong lượt hỏi hiện tại (tính từ HumanMessage mới nhất)."""
    count = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            count += 1
    return count


def _extract_latest_question(messages) -> str:
    """Lấy nội dung câu hỏi mới nhất của người dùng."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            return str(content)
    return ""


def _extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content or "")


def _sanitize_messages_for_llm(messages, max_window: int = 16) -> list:
    """
    Tối ưu hóa Context Window & Nén Bộ nhớ Hội thoại (ToolMessage Pruning & Sliding Window):
    1. Làm sạch các AIMessage cũ (loại bỏ khối <think>...</think> cũ tránh làm mô hình bắt chước).
    2. Cắt tỉa (Pruning) các ToolMessage từ các lượt hội thoại trước (idx < last_human_idx) 
       để giảm 60-70% lượng token Context Window tiêu thụ trong các phiên trao đổi dài.
    3. Áp dụng Sliding Window giữ tối đa `max_window` tin nhắn gần nhất kèm SystemMessage ở đầu.
    """
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    sanitized = []
    for idx, msg in enumerate(messages):
        if idx == 0 and isinstance(msg, SystemMessage):
            content_str = _extract_text_from_content(msg.content)
            if "TUYỆT ĐỐI KHÔNG sinh thẻ <think>" not in content_str:
                from app.services.agent_service import AgentService
                sanitized.append(SystemMessage(content=AgentService.SYSTEM_PROMPT))
            else:
                sanitized.append(msg)
        elif idx < last_human_idx and isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            raw_text = _extract_text_from_content(msg.content)
            cleaned = re.sub(r"<(?:think|thought)>.*?</(?:think|thought|quist)>", "", raw_text, flags=re.DOTALL)
            cleaned = re.sub(r"<accessibility>.*?</accessibility>", "", cleaned, flags=re.DOTALL)
            cleaned = re.sub(r"</?(?:think|thought|quist|accessibility)>", "", cleaned)
            cleaned = re.sub(r"call:(?:default_api:)?[a-zA-Z0-9_]+\s*\{.*?\}", "", cleaned, flags=re.DOTALL).strip()
            if not cleaned:
                cleaned = "(Đã phản hồi)"
            sanitized.append(AIMessage(content=cleaned))
        elif idx < last_human_idx and isinstance(msg, ToolMessage):
            # [ToolMessage Pruning]: Thu gọn ngữ cảnh ToolMessage ở các lượt cũ
            raw_content = _extract_text_from_content(msg.content)
            if len(raw_content) > 150:
                short_summary = raw_content[:100] + "... [Ngữ cảnh đã được tổng hợp ở câu trả lời trước]"
                pruned_tool = ToolMessage(
                    content=short_summary,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                )
                sanitized.append(pruned_tool)
            else:
                sanitized.append(msg)
        else:
            sanitized.append(msg)

    # [Sliding Window]: Nếu lịch sử vượt ngưỡng max_window, giữ lại SystemMessage đầu tiên + các message gần nhất
    if len(sanitized) > (max_window + 1):
        system_msg = sanitized[0]
        recent_msgs = sanitized[-max_window:]
        sanitized = [system_msg] + recent_msgs

    return sanitized


def _rescue_text_tool_calls(response: AIMessage, valid_tool_names: set) -> AIMessage:
    """
    Nếu LLM vô tình xuất lệnh gọi tool dưới dạng text `call:default_api:tool_name{...}`
    thay vì structured tool_calls, tự động bóc tách và chuyển thành tool_calls chuẩn cho LangGraph.
    """
    if getattr(response, "tool_calls", None):
        return response

    raw_text = _extract_text_from_content(response.content)
    if "call:" not in raw_text:
        return response

    pattern = r"call:(?:default_api:)?([a-zA-Z0-9_]+)\s*(\{.*?\})"
    rescued_calls = []
    for match in re.finditer(pattern, raw_text, flags=re.DOTALL):
        t_name, t_args_str = match.groups()
        if t_name in valid_tool_names:
            try:
                parsed_args = json.loads(t_args_str) if t_args_str.strip() else {}
            except Exception:
                parsed_args = {}
            rescued_calls.append({
                "name": t_name,
                "args": parsed_args if isinstance(parsed_args, dict) else {},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            })

    if rescued_calls:
        print(f"[Agent] Đã phục hồi {len(rescued_calls)} tool_call từ định dạng text: {[c['name'] for c in rescued_calls]}")
        cleaned_text = re.sub(pattern, "", raw_text, flags=re.DOTALL).strip()
        response.content = cleaned_text
        response.tool_calls = rescued_calls

    return response


def create_agent_graph(llm, tools, checkpointer=None, decision_service: LayaDecisionService = None):
    builder = StateGraph(AgentState)
    router_service = decision_service or laya_service
    valid_tool_names = {getattr(t, "name", "") for t in tools}

    # Ràng buộc bộ công cụ Multi-Tool vào LLM
    llm_with_tools = llm.bind_tools(tools)

    # 1. Định nghĩa Router Node (Laya System-1 Decision Model)
    def router_node(state: AgentState):
        messages = state.get("messages", [])
        question = _extract_latest_question(messages)
        decision = router_service.decide(question)
        print(f"\n[Node: router - Laya Decision] Action: {decision.action} | Reason: {decision.reason}")
        return {
            "route_action": decision.action,
            "route_reason": decision.reason,
        }

    # 2. Định nghĩa Agent Node (ReAct + Corrective RAG)
    def agent_node(state: AgentState):
        messages = _sanitize_messages_for_llm(state["messages"])
        route_action = state.get("route_action", "retrieve")

        # Đếm số lần công cụ đã được gọi trong lượt câu hỏi hiện tại
        tool_calls_count = _count_current_turn_tool_calls(messages)

        if tool_calls_count >= 3:
            print("[Agent] Đã đạt giới hạn 3 lần gọi công cụ trong lượt này. Ép buộc tổng hợp trả lời...")
            forced_response = llm.invoke(messages + [
                HumanMessage(content="[Hệ thống]: Bạn đã đạt giới hạn tối đa 3 lần tra cứu công cụ cho câu hỏi này. BẮT BUỘC KHÔNG GỌI CÔNG CỤ THÊM. Hãy mở đầu bằng khối <think>...</think> để phân tích tổng hợp toàn bộ dữ liệu đã thu thập được, sau đó đóng thẻ </think> và đưa ra câu trả lời chi tiết cuối cùng cho người dùng.")
            ])
            return {"messages": [forced_response]}

        # Nếu Laya Router (System 1) phân loại là câu giao tiếp/trả lời trực tiếp ở bước đầu tiên
        if route_action == "answer" and tool_calls_count == 0:
            print("[Agent] Laya Router chỉ định 'answer' -> Phản hồi trực tiếp không qua Tool.")
            direct_response = llm.invoke(messages)
            return {"messages": [direct_response]}

        # Ngược lại: Chạy vòng lặp ReAct với bộ công cụ Multi-Tool
        response = llm_with_tools.invoke(messages)
        response = _rescue_text_tool_calls(response, valid_tool_names)
        return {"messages": [response]}

    builder.add_node("router", router_node)
    builder.add_node("agent", agent_node)

    # Định nghĩa Tool Node (thực thi bộ công cụ)
    tool_node = ToolNode(tools)
    builder.add_node("tools", tool_node)

    # Cấu hình luồng (Graph): START -> router -> agent <-> tools -> END
    builder.add_edge(START, "router")
    builder.add_edge("router", "agent")

    # Cạnh điều kiện: Nếu LLM muốn gọi tool -> Sang node tools, nếu không -> END
    builder.add_conditional_edges("agent", tools_condition)

    # Sau khi thực thi tool xong, trả kết quả về lại cho Agent phân tích tiếp (Corrective RAG loop)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


_llm = None
_tools = None
_checkpointer = None
_conn = None
_graph = None


def _get_llm_and_tools():
    global _llm, _tools
    if _llm is None or _tools is None:
        embedding_service = EmbeddingService()
        vector_store = VectorStore()
        
        # Đồng bộ dữ liệu hiện có trong Milvus vào chỉ mục BM25
        bm25_service.sync_from_milvus(vector_store)

        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            bm25_service=bm25_service,
            reranker_service=reranker_service,
        )
        context_builder = ContextBuilder()
        _tools = make_agent_tools(retriever, context_builder)

        from pathlib import Path
        from dotenv import load_dotenv
        env_file = Path(__file__).resolve().parents[2] / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
        else:
            load_dotenv()

        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        _llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.environ.get("GEMINI_API_KEY", ""),
        )
    return _llm, _tools


async def get_agent_graph():
    global _graph, _checkpointer, _conn
    if _graph is None:
        os.makedirs("app_data", exist_ok=True)
        _conn = await aiosqlite.connect("app_data/checkpoints.db")
        _checkpointer = AsyncSqliteSaver(_conn)
        await _checkpointer.setup()
        llm, tools = _get_llm_and_tools()
        _graph = create_agent_graph(
            llm,
            tools,
            checkpointer=_checkpointer,
            decision_service=laya_service,
        )
    return _graph


async def close_checkpointer():
    global _conn, _graph, _checkpointer
    if _conn is not None:
        await _conn.close()
        _conn = None
        _graph = None
        _checkpointer = None

