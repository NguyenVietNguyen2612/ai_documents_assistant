import json
from typing import AsyncGenerator, Optional, List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.graph.rag_graph import get_agent_graph
from app.services.db_service import db_service

class AgentService:

    SYSTEM_PROMPT = """Bạn là một Trợ lý AI thông minh chuyên phân tích tài liệu (AI Documents Assistant).
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

    def __init__(self, graph=None):
        self._graph = graph

    async def _get_graph(self):
        if self._graph is None:
            self._graph = await get_agent_graph()
        return self._graph

    async def ask(
        self,
        question: str,
        history: list = None,
        session_id: str = "default",
        checkpoint_id: Optional[str] = None
    ) -> dict:
        graph = await self._get_graph()
        config = {"configurable": {"thread_id": session_id}}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id

        # Kiểm tra state hiện tại của thread
        state = await graph.aget_state(config)
        existing_msgs = state.values.get("messages", []) if state else []

        if not existing_msgs:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=question)
            ]
        else:
            messages = [HumanMessage(content=question)]

        result = await graph.ainvoke({"messages": messages}, config=config)
        final_message = result["messages"][-1]
        answer = final_message.content

        if isinstance(answer, list):
            text_parts = [part["text"] for part in answer if isinstance(part, dict) and "text" in part]
            answer = "".join(text_parts) if text_parts else str(answer)

        # Lấy checkpoint id sau khi hoàn thành
        new_state = await graph.aget_state(config)
        new_checkpoint_id = new_state.config["configurable"].get("checkpoint_id")

        return {
            "answer": answer,
            "session_id": session_id,
            "checkpoint_id": new_checkpoint_id,
            "context": ""
        }

    async def ask_stream(
        self,
        question: str,
        history: list = None,
        session_id: str = "default",
        checkpoint_id: Optional[str] = None,
        rollback_from_message_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        graph = await self._get_graph()

        # Đảm bảo session tồn tại trong DB
        db_service.create_session(session_id=session_id)

        # Nếu có rollback_from_message_id, cắt tỉa DB messages từ vị trí đó trở đi
        if rollback_from_message_id is not None:
            db_service.delete_messages_from_and_after(session_id, rollback_from_message_id)

        # Lưu tin nhắn User vào database
        user_msg_id = db_service.save_message(
            session_id=session_id,
            role="user",
            content=question
        )

        # Cấu hình thread và checkpoint cho LangGraph
        config = {"configurable": {"thread_id": session_id}}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id

        # Kiểm tra state của thread tại checkpoint này
        state = await graph.aget_state(config)
        existing_msgs = state.values.get("messages", []) if state else []

        if not existing_msgs:
            initial_input = {
                "messages": [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=question)
                ]
            }
        else:
            initial_input = {
                "messages": [HumanMessage(content=question)]
            }

        # Lưu trữ danh sách suy luận để lưu vào DB
        thoughts_history: List[Dict[str, Any]] = []
        full_answer_parts: List[str] = []

        # 1. Tiếp nhận câu hỏi
        short_q = (question[:60] + "...") if len(question) > 60 else question
        receive_thought = {
            "type": "thought",
            "step": "receive",
            "content": f'Tiếp nhận câu hỏi: "{short_q}". Bắt đầu phân tích ý định người dùng...'
        }
        thoughts_history.append(receive_thought)
        yield f"data: {json.dumps(receive_thought, ensure_ascii=False)}\n\n"

        first_token_sent = False
        iteration = 0

        try:
            async for event in graph.astream_events(initial_input, config=config, version="v2"):
                event_type = event.get("event")
                name = event.get("name", "")
                data = event.get("data", {})

                # Khi Agent bắt đầu vòng lặp tư duy
                if event_type == "on_chain_start" and name == "agent":
                    iteration += 1
                    if iteration == 1:
                        thought = {
                            "type": "thought",
                            "step": "analyze",
                            "content": "Đang phân tích câu hỏi để quyết định xem có cần tra cứu tài liệu hay trả lời trực tiếp..."
                        }
                    else:
                        thought = {
                            "type": "thought",
                            "step": "synthesize",
                            "content": "Đang tổng hợp thông tin từ ngữ cảnh tài liệu vừa tra cứu để chuẩn bị câu trả lời..."
                        }
                    thoughts_history.append(thought)
                    yield f"data: {json.dumps(thought, ensure_ascii=False)}\n\n"

                # Khi LLM kết thúc một lượt suy luận
                elif event_type == "on_chat_model_end":
                    output = data.get("output")
                    tool_calls = getattr(output, "tool_calls", None) or []
                    if tool_calls and iteration == 1:
                        tool_names = ", ".join([t.get("name", "") for t in tool_calls])
                        thought = {
                            "type": "thought",
                            "step": "decision",
                            "content": f"Xác định câu hỏi liên quan đến tài liệu. Quyết định kích hoạt công cụ: [{tool_names}]."
                        }
                        thoughts_history.append(thought)
                        yield f"data: {json.dumps(thought, ensure_ascii=False)}\n\n"
                    elif not tool_calls and iteration == 1:
                        thought = {
                            "type": "thought",
                            "step": "decision",
                            "content": "Xác định câu hỏi mang tính chất giao tiếp / kiến thức chung. Quyết định trả lời trực tiếp mà không cần truy vấn cơ sở dữ liệu."
                        }
                        thoughts_history.append(thought)
                        yield f"data: {json.dumps(thought, ensure_ascii=False)}\n\n"

                # Khi Agent kích hoạt Tool
                elif event_type == "on_tool_start":
                    tool_input = data.get("input", {})
                    query = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
                    doc_name = tool_input.get("document_name") if isinstance(tool_input, dict) else None

                    thought_msg = f"Đang tra cứu vector database với từ khóa: '{query}'"
                    if doc_name:
                        thought_msg += f" (lọc tài liệu: {doc_name})"

                    thought = {
                        "type": "thought",
                        "step": "tool_call",
                        "content": thought_msg
                    }
                    thoughts_history.append(thought)
                    yield f"data: {json.dumps(thought, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': name, 'input': tool_input}, ensure_ascii=False)}\n\n"

                # Khi Tool hoàn thành
                elif event_type == "on_tool_end":
                    output = data.get("output", "")
                    out_content = getattr(output, "content", str(output))
                    has_found = "Không tìm thấy" not in out_content
                    if has_found:
                        doc_count = out_content.count("[Tài liệu:") if "[Tài liệu:" in out_content else 1
                        result_msg = f"Đã tìm thấy {doc_count} đoạn ngữ cảnh phù hợp từ tài liệu. Đang nạp vào bộ nhớ..."
                    else:
                        result_msg = "Không tìm thấy đoạn văn bản trực tiếp nào khớp trong tài liệu."

                    thought = {
                        "type": "thought",
                        "step": "tool_result",
                        "content": result_msg
                    }
                    thoughts_history.append(thought)
                    yield f"data: {json.dumps(thought, ensure_ascii=False)}\n\n"
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
                                thought = {
                                    "type": "thought",
                                    "step": "responding",
                                    "content": "Bắt đầu xuất câu trả lời cho người dùng..."
                                }
                                thoughts_history.append(thought)
                                yield f"data: {json.dumps(thought, ensure_ascii=False)}\n\n"

                            if isinstance(content, str):
                                cleaned = content.replace("**", "").replace("__", "")
                                if cleaned:
                                    full_answer_parts.append(cleaned)
                                    yield f"data: {json.dumps({'type': 'token', 'content': cleaned}, ensure_ascii=False)}\n\n"
                            elif isinstance(content, list):
                                for part in content:
                                    if isinstance(part, dict) and "text" in part:
                                        cleaned = part["text"].replace("**", "").replace("__", "")
                                        if cleaned:
                                            full_answer_parts.append(cleaned)
                                            yield f"data: {json.dumps({'type': 'token', 'content': cleaned}, ensure_ascii=False)}\n\n"
                                    elif isinstance(part, str):
                                        cleaned = part.replace("**", "").replace("__", "")
                                        if cleaned:
                                            full_answer_parts.append(cleaned)
                                            yield f"data: {json.dumps({'type': 'token', 'content': cleaned}, ensure_ascii=False)}\n\n"

            # 2. Hoàn tất toàn bộ luồng
            complete_thought = {
                "type": "thought",
                "step": "complete",
                "content": "Hoàn tất toàn bộ quy trình suy luận và phản hồi."
            }
            thoughts_history.append(complete_thought)
            yield f"data: {json.dumps(complete_thought, ensure_ascii=False)}\n\n"

            # Lấy Checkpoint ID mới nhất từ LangGraph state
            latest_state = await graph.aget_state({"configurable": {"thread_id": session_id}})
            latest_checkpoint_id = latest_state.config["configurable"].get("checkpoint_id")

            # Lưu Assistant Message vào DB
            full_answer = "".join(full_answer_parts)
            asst_msg_id = db_service.save_message(
                session_id=session_id,
                role="assistant",
                content=full_answer,
                thoughts=thoughts_history,
                checkpoint_id=latest_checkpoint_id
            )

            # Cập nhật tiêu đề session nếu là câu hỏi đầu tiên
            session_msgs = db_service.get_session_messages(session_id)
            if len(session_msgs) <= 2:
                auto_title = (question[:28] + "...") if len(question) > 28 else question
                db_service.update_session_title(session_id, auto_title)

            # Phát sự kiện checkpoint thông báo cho frontend
            yield f"data: {json.dumps({'type': 'checkpoint', 'session_id': session_id, 'checkpoint_id': latest_checkpoint_id, 'user_message_id': user_msg_id, 'assistant_message_id': asst_msg_id}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_msg = f"Đã xảy ra lỗi khi xử lý: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"

    async def get_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        """Lấy danh sách các checkpoint của một phiên chat."""
        graph = await self._get_graph()
        config = {"configurable": {"thread_id": session_id}}
        
        checkpoints: List[Dict[str, Any]] = []
        step = 0
        try:
            async for state in graph.aget_state_history(config):
                cid = state.config["configurable"].get("checkpoint_id")
                parent_cid = state.parent_config["configurable"].get("checkpoint_id") if state.parent_config else None
                msgs = state.values.get("messages", [])
                
                # Chỉ lấy các checkpoint có ý nghĩa (khi có tin nhắn mới hoặc hoàn thành lượt trả lời)
                last_msg = msgs[-1] if msgs else None
                last_content = ""
                if last_msg:
                    raw = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                    last_content = (raw[:50] + "...") if len(raw) > 50 else raw
                
                step += 1
                checkpoints.append({
                    "checkpoint_id": cid,
                    "parent_checkpoint_id": parent_cid,
                    "step": step,
                    "summary": f"{type(last_msg).__name__ if last_msg else 'Init'}: {last_content}",
                    "message_count": len(msgs)
                })
        except Exception as e:
            print(f"[Checkpoints] Lỗi khi đọc checkpoint history: {e}")

        return checkpoints

    async def rollback(self, session_id: str, checkpoint_id: str, message_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Quay lui cuộc trò chuyện về một checkpoint hoặc một message_id."""
        if message_id is not None:
            # Xóa các tin nhắn sau message_id
            db_service.delete_messages_after(session_id, message_id)
        else:
            # Tìm xem checkpoint_id ứng với tin nhắn nào trong DB
            messages = db_service.get_session_messages(session_id)
            target_msg_id = None
            for msg in messages:
                if msg.get("checkpoint_id") == checkpoint_id:
                    target_msg_id = msg.get("id")
                    break
            if target_msg_id is not None:
                db_service.delete_messages_after(session_id, target_msg_id)

        # Trả về danh sách tin nhắn còn lại
        remaining_messages = db_service.get_session_messages(session_id)
        return remaining_messages

agent_service = AgentService()