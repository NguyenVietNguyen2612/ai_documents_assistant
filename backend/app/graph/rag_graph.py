import os
import aiosqlite
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.graph.state import AgentState
from app.tools.retrieval_tool import make_retrieve_tool
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever
from app.services.context_builder import ContextBuilder
from app.services.bm25_service import bm25_service
from app.services.reranker_service import reranker_service


def create_agent_graph(llm, tools, checkpointer=None):
    builder = StateGraph(AgentState)
    
    # Ràng buộc công cụ vào LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Định nghĩa Agent Node
    def agent_node(state: AgentState):
        messages = state["messages"]
        
        # Đếm số lần công cụ đã được gọi (dựa trên số ToolMessage)
        from langchain_core.messages import ToolMessage, SystemMessage
        tool_calls_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        
        if tool_calls_count >= 3:
            print("[Agent] Đã đạt giới hạn 3 lần tìm kiếm. Ép buộc trả lời...")
            # Dùng LLM nguyên thủy (không bind tool) để ép nó phải trả lời bằng chữ
            forced_response = llm.invoke(messages + [
                SystemMessage(content="Bạn đã đạt giới hạn 3 lần tìm kiếm tài liệu. BẮT BUỘC KHÔNG TÌM KIẾM THÊM. Hãy mở đầu bằng khối <think>...</think> để phân tích tổng hợp tài liệu, sau đó đưa ra câu trả lời cuối cùng. LƯU Ý BẮT BUỘC: Không dùng các ký hiệu Markdown như **, ##. Nếu phân cấp hoặc liệt kê, chỉ dùng -, +, . hoặc số thứ tự ở đầu dòng.")
            ])
            return {"messages": [forced_response]}
            
        # LLM tự động đọc lịch sử chat và quyết định gọi tool hay trả lời trực tiếp
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
        
    builder.add_node("agent", agent_node)
    
    # Định nghĩa Tool Node (thực thi tool)
    tool_node = ToolNode(tools)
    builder.add_node("tools", tool_node)
    
    # Cấu hình luồng (Graph)
    builder.add_edge(START, "agent")
    
    # Cạnh điều kiện: Nếu LLM muốn gọi tool -> Sang node tools, nếu không -> END
    builder.add_conditional_edges("agent", tools_condition)
    
    # Sau khi thực thi tool xong, trả kết quả về lại cho Agent phân tích tiếp
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
        retrieve_tool = make_retrieve_tool(retriever, context_builder)
        _tools = [retrieve_tool]

        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
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
        _graph = create_agent_graph(llm, tools, checkpointer=_checkpointer)
    return _graph


async def close_checkpointer():
    global _conn, _graph, _checkpointer
    if _conn is not None:
        await _conn.close()
        _conn = None
        _graph = None
        _checkpointer = None
