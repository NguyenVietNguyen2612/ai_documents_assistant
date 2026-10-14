<<<<<<< ours
=======

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from app.graph.state import RAGState
from app.graph.nodes import (
    make_retrieve_node,
    make_context_node,
    make_generate_node,
    make_evaluate_node,
    make_agent_node,
)


def route_after_evaluate(state: RAGState):
    is_relevant = state.get("is_relevant", False)
    retry_count = state.get("retry_count", 0)
    max_retries = 2

    # Relevant → build context and generate
    if is_relevant:
        return "build_context"

    # Not relevant but still allowed to retry
    if retry_count < max_retries:
        return "retrieve"

    # Retry limit reached, proceed anyway
    return "build_context"


def route_after_agent(state):

    action = state.get(
        "action",
        "retrieve",
    )

    print(
        f"[Router] Agent action: {action}"
    )

    return action


def create_rag_graph(
    retriever,
    context_builder,
    prompt_builder,
    llm_service,
    decision_service=None,
):

    builder = StateGraph(RAGState)

    builder.add_node(
        "agent",
        make_agent_node(decision_service or llm_service),
    )

    builder.add_node(
        "retrieve",
        make_retrieve_node(retriever),
    )
    
    builder.add_node(
        "evaluate",
        make_evaluate_node(),
    )

    builder.add_node(
        "build_context",
        make_context_node(context_builder),
    )

    builder.add_node(
        "generate",
        make_generate_node(
            prompt_builder,
            llm_service,
        ),
    )


    builder.add_edge("retrieve", "evaluate")

    builder.add_edge(START,"agent")

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "retrieve": "retrieve",
            "answer": "generate",
        },
    )
    
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
    )

    builder.add_edge("build_context", "generate")

    builder.add_edge("generate", END)

    return builder.compile()

>>>>>>> theirs
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
<<<<<<< ours

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
        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
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
=======
from app.services.prompt_builder import PromptBuilder
from app.services.llm_service import LLMService
from app.services.laya_service import LayaDecisionService
from app.services.bm25_service import bm25_service

def _init_graph():
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    
    # Đồng bộ dữ liệu hiện có trong Milvus vào chỉ mục BM25
    bm25_service.sync_from_milvus(vector_store)

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_service=bm25_service,
    )
    context_builder = ContextBuilder()
    prompt_builder = PromptBuilder()
    
    # Provide a default empty string to avoid KeyError if not set during build/import time, 
    # but actual usage will require a valid API key.
    llm_service = LLMService(
        api_key=os.environ.get("GEMINI_API_KEY", "")
    )
    decision_service = LayaDecisionService(fallback_llm=llm_service)

    return create_rag_graph(
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
        decision_service=decision_service,
    )

graph = _init_graph()
>>>>>>> theirs
