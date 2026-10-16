import os
if os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")
    os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
    os.environ.setdefault("CURL_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")

import aiosqlite
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
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


def create_agent_graph(llm, tools, checkpointer=None, decision_service: LayaDecisionService = None):
    builder = StateGraph(AgentState)
    router_service = decision_service or laya_service

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
        messages = state["messages"]
        route_action = state.get("route_action", "retrieve")

        # Đếm số lần công cụ đã được gọi trong lượt câu hỏi hiện tại
        tool_calls_count = _count_current_turn_tool_calls(messages)

        if tool_calls_count >= 3:
            print("[Agent] Đã đạt giới hạn 3 lần gọi công cụ trong lượt này. Ép buộc tổng hợp trả lời...")
            forced_response = llm.invoke(messages + [
                SystemMessage(content="Bạn đã đạt giới hạn tối đa 3 lần tra cứu công cụ cho câu hỏi này. BẮT BUỘC KHÔNG GỌI CÔNG CỤ THÊM. Hãy mở đầu bằng khối <think>...</think> để phân tích tổng hợp toàn bộ dữ liệu đã thu thập được, sau đó đưa ra câu trả lời cuối cùng. LƯU Ý BẮT BUỘC: Không dùng các ký hiệu Markdown như **, ##. Nếu phân cấp hoặc liệt kê, chỉ dùng -, +, . hoặc số thứ tự ở đầu dòng.")
            ])
            return {"messages": [forced_response]}

        # Nếu Laya Router (System 1) phân loại là câu giao tiếp/trả lời trực tiếp ở bước đầu tiên
        if route_action == "answer" and tool_calls_count == 0:
            print("[Agent] Laya Router chỉ định 'answer' -> Phản hồi trực tiếp không qua Tool.")
            direct_response = llm.invoke(messages)
            return {"messages": [direct_response]}

        # Ngược lại: Chạy vòng lặp ReAct với bộ công cụ Multi-Tool
        response = llm_with_tools.invoke(messages)
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
            temperature=0,
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

