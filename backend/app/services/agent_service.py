import json
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class AgentService:

    def __init__(self, graph):
        self.graph = graph

    def _build_initial_state(self, question: str, history: list = None) -> dict:
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

QUY TẮC ĐỊNH DẠNG VĂN BẢN (CỰC KỲ QUAN TRỌNG - BẮT BUỘC TUÂN THỦ 100%):
1. TRẢ VỀ VĂN BẢN THƯỜNG (PLAIN TEXT):
   - Tuyệt đối CẤM sử dụng cặp dấu sao ** để in đậm từ ngữ (ví dụ: KHÔNG VIẾT **lọc mềm**, PHẢI VIẾT lọc mềm).
   - Tuyệt đối CẤM sử dụng dấu thăng # làm tiêu đề (ví dụ: KHÔNG VIẾT ## Tiêu đề).
   - Tuyệt đối CẤM dùng bảng Markdown hoặc gạch đầu dòng dạng dấu sao (*).
   - LƯU Ý KHI ĐỌC TÀI LIỆU: Dù nội dung tài liệu trích xuất có chứa dấu ** hay # thì khi viết câu trả lời bạn BẮT BUỘC phải bỏ hết các ký tự ** và # này đi.
2. CÁCH PHÂN CẤP VÀ LIỆT KÊ: Nếu cần phân cấp mục, tiêu đề phân đoạn hoặc liệt kê, CHỈ ĐƯỢC PHÉP dùng các ký hiệu sau ở đầu câu/đầu dòng:
   - Dấu gạch ngang (-)
   - Dấu cộng (+)
   - Dấu chấm (.)
   - Hoặc số thứ tự (ví dụ: 1., 2., 3. hoặc 1.1, 1.2)
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

        return {
            "messages": messages
        }

    def ask(self, question: str, history: list = None) -> dict:
        initial_state = self._build_initial_state(question, history)

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

    async def ask_stream(self, question: str, history: list = None) -> AsyncGenerator[str, None]:
        initial_state = self._build_initial_state(question, history)

        # 1. Bước 1: Tiếp nhận câu hỏi và phân tích ban đầu
        short_q = (question[:60] + "...") if len(question) > 60 else question
        yield f"data: {json.dumps({'type': 'thought', 'step': 'receive', 'content': f'Tiếp nhận câu hỏi: \"{short_q}\". Bắt đầu phân tích ý định người dùng...'}, ensure_ascii=False)}\n\n"

        first_token_sent = False
        iteration = 0

        try:
            async for event in self.graph.astream_events(initial_state, version="v2"):
                event_type = event.get("event")
                name = event.get("name", "")
                data = event.get("data", {})

                # Khi Agent bắt đầu vòng lặp tư duy
                if event_type == "on_chain_start" and name == "agent":
                    iteration += 1
                    if iteration == 1:
                        yield f"data: {json.dumps({'type': 'thought', 'step': 'analyze', 'content': 'Đang phân tích câu hỏi để quyết định xem có cần tra cứu tài liệu hay trả lời trực tiếp...'}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'thought', 'step': 'synthesize', 'content': 'Đang tổng hợp thông tin từ ngữ cảnh tài liệu vừa tra cứu để chuẩn bị câu trả lời...'}, ensure_ascii=False)}\n\n"

                # Khi LLM kết thúc một lượt suy luận (ra quyết định gọi tool hay trả lời)
                elif event_type == "on_chat_model_end":
                    output = data.get("output")
                    tool_calls = getattr(output, "tool_calls", None) or []
                    if tool_calls and iteration == 1:
                        tool_names = ", ".join([t.get("name", "") for t in tool_calls])
                        yield f"data: {json.dumps({'type': 'thought', 'step': 'decision', 'content': f'Xác định câu hỏi liên quan đến tài liệu. Quyết định kích hoạt công cụ: [{tool_names}].'}, ensure_ascii=False)}\n\n"
                    elif not tool_calls and iteration == 1:
                        yield f"data: {json.dumps({'type': 'thought', 'step': 'decision', 'content': 'Xác định câu hỏi mang tính chất giao tiếp / kiến thức chung. Quyết định trả lời trực tiếp mà không cần truy vấn cơ sở dữ liệu.'}, ensure_ascii=False)}\n\n"

                # Khi Agent kích hoạt Tool
                elif event_type == "on_tool_start":
                    tool_input = data.get("input", {})
                    query = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
                    doc_name = tool_input.get("document_name") if isinstance(tool_input, dict) else None

                    thought_msg = f"Đang tra cứu vector database với từ khóa: '{query}'"
                    if doc_name:
                        thought_msg += f" (lọc tài liệu: {doc_name})"

                    yield f"data: {json.dumps({'type': 'thought', 'step': 'tool_call', 'content': thought_msg}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': name, 'input': tool_input}, ensure_ascii=False)}\n\n"

                # Khi Tool hoàn thành
                elif event_type == "on_tool_end":
                    output = data.get("output", "")
                    out_content = getattr(output, "content", str(output))
                    has_found = "Không tìm thấy" not in out_content
                    if has_found:
                        # Đếm số đoạn tài liệu hoặc ước tính độ dài
                        doc_count = out_content.count("[Tài liệu:") if "[Tài liệu:" in out_content else 1
                        result_msg = f"Đã tìm thấy {doc_count} đoạn ngữ cảnh phù hợp từ tài liệu. Đang nạp vào bộ nhớ..."
                    else:
                        result_msg = "Không tìm thấy đoạn văn bản trực tiếp nào khớp trong tài liệu."

                    yield f"data: {json.dumps({'type': 'thought', 'step': 'tool_result', 'content': result_msg}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': name, 'summary': result_msg}, ensure_ascii=False)}\n\n"

                # Khi LLM stream từng token
                elif event_type == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk:
                        if getattr(chunk, "tool_call_chunks", None):
                            continue

                        additional = getattr(chunk, "additional_kwargs", {}) or {}
                        thinking_token = additional.get("thinking") or additional.get("thought")
                        if thinking_token:
                            yield f"data: {json.dumps({'type': 'thought_token', 'content': thinking_token}, ensure_ascii=False)}\n\n"

                        content = chunk.content
                        if content:
                            if not first_token_sent:
                                first_token_sent = True
                                yield f"data: {json.dumps({'type': 'thought', 'step': 'responding', 'content': 'Bắt đầu xuất câu trả lời cho người dùng...'}, ensure_ascii=False)}\n\n"

                            if isinstance(content, str):
                                cleaned = content.replace("**", "").replace("__", "")
                                if cleaned:
                                    yield f"data: {json.dumps({'type': 'token', 'content': cleaned}, ensure_ascii=False)}\n\n"
                            elif isinstance(content, list):
                                for part in content:
                                    if isinstance(part, dict) and "text" in part:
                                        cleaned = part["text"].replace("**", "").replace("__", "")
                                        if cleaned:
                                            yield f"data: {json.dumps({'type': 'token', 'content': cleaned}, ensure_ascii=False)}\n\n"
                                    elif isinstance(part, str):
                                        cleaned = part.replace("**", "").replace("__", "")
                                        if cleaned:
                                            yield f"data: {json.dumps({'type': 'token', 'content': cleaned}, ensure_ascii=False)}\n\n"

            # 2. Hoàn tất toàn bộ luồng
            yield f"data: {json.dumps({'type': 'thought', 'step': 'complete', 'content': 'Hoàn tất toàn bộ quy trình suy luận và phản hồi.'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_msg = f"Đã xảy ra lỗi khi xử lý: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"