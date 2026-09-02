from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class AgentService:

    def __init__(self, graph):
        self.graph = graph

    def ask(self, question: str, history: list = None) -> dict:
        if history is None:
            history = []
            
        # System Prompt đóng vai trò là "Não bộ định hướng" cho ReAct Agent
        system_prompt = """Bạn là một Trợ lý AI thông minh chuyên phân tích tài liệu (AI Documents Assistant).
Nhiệm vụ của bạn là giải đáp các thắc mắc của người dùng dựa trên tài liệu đã tải lên.

QUY TẮC HOẠT ĐỘNG:
1. Luôn sử dụng công cụ `retrieve_documents` ĐẦU TIÊN để tìm kiếm ngữ cảnh nếu câu hỏi liên quan đến tài liệu, số liệu, hoặc thông tin cụ thể.
2. Tuyệt đối không tự bịa ra thông tin. Nếu công cụ trả về "Không tìm thấy", hãy thành thật nói rằng tài liệu không đề cập đến vấn đề này.
3. Khi trả lời, hãy tổng hợp thông tin từ ngữ cảnh trả về một cách rõ ràng, dễ hiểu và chuyên nghiệp.
4. Nếu câu hỏi là giao tiếp thông thường (ví dụ: "Chào bạn", "Bạn có thể làm gì?"), bạn có thể trả lời trực tiếp mà KHÔNG CẦN gọi công cụ.
"""

        # Chuyển đổi lịch sử chat
        messages = [SystemMessage(content=system_prompt)]
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
                
        # Câu hỏi hiện tại
        messages.append(HumanMessage(content=question))

        # Khởi tạo state dạng MessagesState cho ReAct loop
        initial_state = {
            "messages": messages
        }

        result = self.graph.invoke(
            initial_state
        )
        
        # Lấy tin nhắn cuối cùng (AIMessage) làm câu trả lời
        final_message = result["messages"][-1]
        answer = final_message.content
        
        # Xử lý trường hợp Langchain Google GenAI trả về list thay vì string
        if isinstance(answer, list):
            text_parts = [part["text"] for part in answer if isinstance(part, dict) and "text" in part]
            answer = "".join(text_parts) if text_parts else str(answer)

        return {
            "answer": answer,
            "context": "" # Context tạm thời rỗng do ReAct tự tổng hợp
        }