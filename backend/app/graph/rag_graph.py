import os
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI

from app.graph.state import AgentState
from app.tools.retrieval_tool import make_retrieve_tool

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.retriever import Retriever
from app.services.context_builder import ContextBuilder

def create_agent_graph(llm, tools):
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
                SystemMessage(content="Bạn đã đạt giới hạn 3 lần tìm kiếm tài liệu. BẮT BUỘC KHÔNG TÌM KIẾM THÊM. Hãy tổng hợp và đưa ra câu trả lời cuối cùng dựa trên các thông tin đã tìm được.")
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
    
    return builder.compile()

def _init_graph():
    # Khởi tạo các dịch vụ cơ sở
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )
    context_builder = ContextBuilder()
    
    # 1. Khởi tạo các Công cụ (Tools)
    retrieve_tool = make_retrieve_tool(retriever, context_builder)
    
    # Tạm thời chỉ đưa vào 1 tool (Retrieval), sau này thêm Calculator/Tavily vào danh sách này
    tools = [retrieve_tool]
    
    # 2. Khởi tạo LLM cho Agent
    # Model gemini-1.5-flash hoặc pro đều hỗ trợ Tool Calling cực tốt
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.environ.get("GEMINI_API_KEY", ""),
        temperature=0,
    )
    
    # Build Graph
    return create_agent_graph(llm, tools)

graph = _init_graph()
