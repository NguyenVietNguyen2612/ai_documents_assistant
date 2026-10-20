import json
from typing import AsyncGenerator, Optional, List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.graph.rag_graph import get_agent_graph
from app.services.db_service import db_service
from app.services.semantic_cache import semantic_cache_service

class AgentService:

    SYSTEM_PROMPT = """Bạn là một Trợ lý AI thông minh chuyên phân tích tài liệu (AI Documents Assistant).
Nhiệm vụ của bạn là giải đáp các thắc mắc của người dùng dựa trên tài liệu đã tải lên.

BỘ CÔNG CỤ CỦA BẠN (MULTI-TOOL):
1. `retrieve_documents`: Tìm kiếm chi tiết ngữ cảnh trong tài liệu (Hybrid Search: Vector + BM25 + Reranker). Luôn ưu tiên dùng khi hỏi về thông tin cụ thể, định nghĩa, kiến trúc, thực nghiệm.
2. `list_uploaded_documents`: Liệt kê tất cả các file tài liệu hiện có trong hệ thống kèm dung lượng và số lượng phân đoạn. Dùng khi người dùng hỏi về danh sách file đã tải lên hoặc khi bạn cần biết tên file.
3. `get_document_overview`: Lấy cấu trúc đề mục (mục lục) và nội dung mở đầu/kết luận của một tài liệu. Dùng khi người dùng yêu cầu tóm tắt tổng quan một tài liệu hoặc hỏi bố cục toàn bài.
4. `calculate_metrics`: Máy tính toán học chính xác. Dùng khi cần tính toán số liệu, độ chênh lệch, tỷ lệ phần trăm từ các con số trong tài liệu.

QUY TẮC HOẠT ĐỘNG & CORRECTIVE RAG (SELF-CORRECTION):
1. Luôn sử dụng công cụ phù hợp ĐẦU TIÊN nếu câu hỏi liên quan đến tài liệu hoặc số liệu.
2. CƠ CHẾ SMALL-TO-BIG WINDOW EXPANSION: Khi tra cứu tài liệu, hệ thống tự động mở rộng sang các phân đoạn liền kề (Chunks Window) để bảo toàn ngữ cảnh trọn vẹn, không bị đứt gãy ý ở ranh giới cắt đoạn. Hãy tận dụng tối đa ngữ cảnh mở rộng này.
3. CƠ CHẾ CORRECTIVE RAG (Tự đánh giá & Viết lại truy vấn): 
   - Nếu kết quả từ `retrieve_documents` có nhãn `[CẢNH BÁO ĐỘ TIN CẬY THẤP]` hoặc không tìm thấy thông tin, bạn BẮT BUỘC KHÔNG KẾT LUẬN VỘI VÀNG.
   - Hãy tự động THỰC HIỆN QUERY REWRITING:
     + Nếu người dùng hỏi bằng tiếng Việt mà tài liệu nghiên cứu là tiếng Anh: Dịch và viết lại từ khóa sang tiếng Anh học thuật (ví dụ: "vấn đề chính của FreqNet" -> "main limitation problem of FreqNet hard masking").
     + Nếu truy vấn quá dài hoặc có tên tài liệu sai: Rút ngắn thành 2-4 từ khóa chuyên môn cốt lõi và bỏ lọc tên tài liệu để tìm kiếm diện rộng.
     + Kích hoạt gọi lại công cụ `retrieve_documents` lần 2 hoặc lần 3 với truy vấn đã được tinh chỉnh.
4. Tuyệt đối không tự bịa ra thông tin. Nếu sau khi đã thử tự sửa truy vấn (Query Rewriting) mà vẫn không tìm thấy bằng chứng, hãy khẳng định rõ ràng và trung thực rằng tài liệu hiện tại không đề cập đến thông tin này.
5. Nếu câu hỏi là giao tiếp thông thường (ví dụ: "Chào bạn", "Bạn có thể làm gì?"), bạn có thể trả lời trực tiếp mà KHÔNG CẦN gọi công cụ.

QUY TẮC TRÍCH DẪN NGUỒN (CITATIONS - BẮT BUỘC):
1. Khi cung cấp câu trả lời có chứa dữ kiện hoặc số liệu từ tài liệu, bạn BẮT BUỘC phải đặt thẻ trích dẫn nguồn ngay cạnh thông tin đó theo đúng định dạng chuẩn:
   [Tài liệu: <tên_tệp> - Trang <số_trang>] (Ví dụ: [Tài liệu: CS338.pdf - Trang 2]).
2. Đối chiếu cẩn thận số trang ghi trong [Source ...] của ngữ cảnh để ghi đúng số trang, tuyệt đối không bịa số trang.

QUY TẮC SUY LUẬN (CHAIN-OF-THOUGHT - TƯƠNG TỰ DEEPSEEK):
- LƯU Ý KHI GỌI CÔNG CỤ (TOOL CALL): Khi bạn cần gọi công cụ (`list_uploaded_documents`, `retrieve_documents`, `get_document_overview`, `calculate_metrics`), HÃY KÍCH HOẠT CÔNG CỤ TRỰC TIẾP NGAY LẬP TỨC. TUYỆT ĐỐI KHÔNG sinh thẻ <think> hay viết văn bản trước khi gọi công cụ.
- KHI VIẾT CÂU TRẢ LỜI CUỐI CÙNG CHO NGƯỜI DÙNG (sau khi đã có kết quả từ công cụ hoặc khi trả lời trực tiếp không cần công cụ): bạn BẮT BUỘC mở đầu bằng cặp thẻ <think>...</think> (chú ý đóng đúng bằng thẻ </think>).
Bên trong khối <think>, hãy trình bày chi tiết quá trình suy nghĩ và phân tích:
- Phân tích câu hỏi của người dùng và các điểm cốt lõi cần giải đáp.
- Đánh giá thông tin ngữ cảnh tìm được từ tài liệu (thông tin nào có sẵn, số liệu nào được tìm thấy, có điểm nào tài liệu chưa đề cập hay không).
- Lập luận logic từng bước để đi đến kết luận và giải pháp.
- Lên kế hoạch cấu trúc câu trả lời rõ ràng, mạch lạc, dễ hiểu.
Sau khi đóng thẻ </think>, bạn mới viết nội dung câu trả lời chính thức gửi tới người dùng.

QUY TẮC ĐỊNH DẠNG VĂN BẢN (MARKDOWN CHUẨN):
- Sử dụng Markdown chuẩn để câu trả lời mạch lạc và chuyên nghiệp:
  + Dùng in đậm (**từ khóa**) để làm nổi bật các khái niệm hoặc kết luận quan trọng.
  + Dùng tiêu đề (##, ###) để phân chia các phần nếu câu trả lời dài.
  + Dùng danh sách gạch đầu dòng (-) hoặc số thứ tự (1., 2.) khi phân loại hoặc liệt kê.
  + Dùng bảng Markdown (| Cột 1 | Cột 2 |) khi so sánh số liệu hoặc trình bày kết quả thực nghiệm.
  + Dùng công thức toán học ($...$ hoặc $$...$$) khi trình bày biểu thức toán/tần số.
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
        checkpoint_id: Optional[str] = None,
        document_filter: Optional[str] = None,
    ) -> dict:
        import time
        import re
        from langchain_core.messages import ToolMessage

        t0 = time.perf_counter()
        graph = await self._get_graph()
        config = {"configurable": {"thread_id": session_id}}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id

        # Kiểm tra state hiện tại của thread
        state = await graph.aget_state(config)
        existing_msgs = state.values.get("messages", []) if state else []
        prior_len = len(existing_msgs)

        target_doc = document_filter
        if not target_doc:
            at_match = re.search(r'@([a-zA-Z0-9_\-\.\s]+?\.(?:pdf|csv|md|txt))', question, re.IGNORECASE)
            if at_match:
                target_doc = at_match.group(1).strip()

        # [Semantic Cache]: Kiểm tra phản hồi có sẵn trong cache (<50ms)
        cached_hit = semantic_cache_service.lookup(question, scope_doc=target_doc)
        if cached_hit:
            return {
                "answer": cached_hit["answer"],
                "session_id": session_id,
                "checkpoint_id": None,
                "context": "",
                "tool_calls_count": 0,
                "latency_ms": cached_hit["latency_ms"],
                "cached": True,
                "cache_similarity": cached_hit["similarity"],
            }

        effective_prompt = question
        if target_doc:
            effective_prompt += f"\n\n[LƯU Ý PHẠM VI TÀI LIỆU]: Người dùng chỉ định tập trung phân tích tài liệu '{target_doc}'. Bạn hãy ưu tiên truyền tham số document_name='{target_doc}' khi kích hoạt công cụ tra cứu."

        if not existing_msgs:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=effective_prompt)
            ]
        else:
            messages = [HumanMessage(content=effective_prompt)]

        result = await graph.ainvoke({"messages": messages}, config=config)
        all_msgs = result.get("messages", [])
        new_msgs = all_msgs[prior_len:] if prior_len < len(all_msgs) else all_msgs

        # Trích xuất toàn bộ ngữ cảnh thu được từ các ToolMessage trong lượt hỏi này
        tool_contexts = []
        for msg in new_msgs:
            if isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool":
                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content_str.strip():
                    tool_contexts.append(content_str.strip())

        final_message = all_msgs[-1]
        raw_answer = final_message.content

        if isinstance(raw_answer, list):
            text_parts = [part["text"] for part in raw_answer if isinstance(part, dict) and "text" in part]
            raw_answer = "".join(text_parts) if text_parts else str(raw_answer)

        clean_answer = re.sub(r'<(?:think|thought)>.*?</(?:think|thought|quist)>', '', raw_answer, flags=re.DOTALL)
        clean_answer = re.sub(r'<accessibility>.*?</accessibility>', '', clean_answer, flags=re.DOTALL).strip()

        # Lưu câu trả lời vào Semantic Cache nếu hợp lệ
        if clean_answer and len(clean_answer) > 10:
            semantic_cache_service.store(
                query=question,
                answer=clean_answer,
                thoughts=[],
                sources=[],
                scope_doc=target_doc,
            )

        # Lấy checkpoint id sau khi hoàn thành
        new_state = await graph.aget_state(config)
        new_checkpoint_id = new_state.config["configurable"].get("checkpoint_id")
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "answer": clean_answer,
            "session_id": session_id,
            "checkpoint_id": new_checkpoint_id,
            "context": "\n\n---\n\n".join(tool_contexts),
            "tool_calls_count": len(tool_contexts),
            "latency_ms": latency_ms,
            "cached": False,
        }


    async def ask_stream(
        self,
        question: str,
        history: list = None,
        session_id: str = "default",
        checkpoint_id: Optional[str] = None,
        rollback_from_message_id: Optional[str] = None,
        document_filter: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        import re
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

        # Nhận diện tài liệu mục tiêu (Scoped Retrieval / Mention)
        target_doc = document_filter
        if not target_doc:
            at_match = re.search(r'@([a-zA-Z0-9_\-\.\s]+?\.(?:pdf|csv|md|txt))', question, re.IGNORECASE)
            if at_match:
                target_doc = at_match.group(1).strip()

        # [Semantic Cache]: Tra cứu câu trả lời tương đương trong Semantic Cache (<50ms, 0 token LLM)
        cache_hit = semantic_cache_service.lookup(question, scope_doc=target_doc)
        if cache_hit:
            hit_similarity = cache_hit["similarity"]
            hit_latency = cache_hit["latency_ms"]
            cache_thought = {
                "type": "thought",
                "step": "router",
                "content": f"⚡ Semantic Cache Hit! Tìm thấy câu trả lời tương đồng ngữ nghĩa: {(hit_similarity * 100):.1f}% (>= 95%) trong {hit_latency}ms. Phản hồi tức thì với 0 token LLM!"
            }
            yield f"data: {json.dumps(cache_thought, ensure_ascii=False)}\n\n"

            cached_sources = cache_hit.get("sources") or []
            if cached_sources:
                yield f"data: {json.dumps({'type': 'sources', 'sources': cached_sources}, ensure_ascii=False)}\n\n"

            cached_answer = cache_hit["answer"]
            yield f"data: {json.dumps({'type': 'token', 'content': cached_answer}, ensure_ascii=False)}\n\n"

            telemetry_event = {
                "type": "telemetry",
                "step": "telemetry",
                "telemetry": {
                    "latency_ms": hit_latency,
                    "router_ms": hit_latency,
                    "tool_calls": 0,
                    "top_rerank_score": hit_similarity,
                    "cached": True,
                    "cache_similarity": hit_similarity,
                },
            }
            yield f"data: {json.dumps(telemetry_event, ensure_ascii=False)}\n\n"

            asst_msg_id = db_service.save_message(
                session_id=session_id,
                role="assistant",
                content=cached_answer,
                thoughts=[cache_thought] + ([{"type": "sources", "sources": cached_sources}] if cached_sources else []),
                checkpoint_id=None
            )

            session_msgs = db_service.get_session_messages(session_id)
            if len(session_msgs) <= 2:
                auto_title = (question[:28] + "...") if len(question) > 28 else question
                db_service.update_session_title(session_id, auto_title)

            yield f"data: {json.dumps({'type': 'checkpoint', 'session_id': session_id, 'checkpoint_id': None, 'user_message_id': user_msg_id, 'assistant_message_id': asst_msg_id}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            return

        effective_prompt = question
        if target_doc:
            effective_prompt += f"\n\n[LƯU Ý PHẠM VI TÀI LIỆU]: Người dùng chỉ định tập trung phân tích tài liệu '{target_doc}'. Bạn hãy ưu tiên truyền tham số document_name='{target_doc}' khi kích hoạt công cụ tra cứu."

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
                    HumanMessage(content=effective_prompt)
                ]
            }
        else:
            initial_input = {
                "messages": [HumanMessage(content=effective_prompt)]
            }

        # Lưu trữ danh sách suy luận để lưu vào DB
        thoughts_history: List[Dict[str, Any]] = []
        full_answer_parts: List[str] = []
        full_thinking_parts: List[str] = []

        # Trạng thái bóc tách khối suy nghĩ <think>...</think> (DeepSeek-style)
        stream_buffer = ""
        in_think_tag = False
        think_ended = False

        # 1. Tiếp nhận câu hỏi
        short_q = (question[:60] + "...") if len(question) > 60 else question
        status_suffix = f" (Tập trung tài liệu: {target_doc})" if target_doc else ""
        receive_thought = {
            "type": "thought",
            "step": "receive",
            "content": f'Tiếp nhận câu hỏi: "{short_q}"{status_suffix}. Bắt đầu phân tích ý định người dùng...'
        }
        thoughts_history.append(receive_thought)
        yield f"data: {json.dumps(receive_thought, ensure_ascii=False)}\n\n"

        first_token_sent = False
        iteration = 0
        import time
        stream_start_time = time.perf_counter()
        router_start_time = time.perf_counter()
        router_ms: Optional[float] = None
        tool_calls_count = 0
        top_rerank_score: Optional[float] = None

        try:
            async for event in graph.astream_events(initial_input, config=config, version="v2"):
                event_type = event.get("event")
                name = event.get("name", "")
                data = event.get("data", {})

                if event_type == "on_chain_start" and name == "router":
                    router_start_time = time.perf_counter()

                # Khi Laya Decision Router (System 1) hoàn tất phân loại
                elif event_type == "on_chain_end" and name == "router":
                    router_ms = round((time.perf_counter() - router_start_time) * 1000, 1)
                    output = data.get("output", {}) or {}
                    r_action = output.get("route_action", "retrieve") if isinstance(output, dict) else "retrieve"
                    r_reason = output.get("route_reason", "") if isinstance(output, dict) else ""
                    action_label = "Tra cứu & Phân tích tài liệu (ReAct)" if r_action == "retrieve" else "Phản hồi trực tiếp (Fast-path)"
                    router_thought = {
                        "type": "thought",
                        "step": "router",
                        "content": f"Laya Decision Router ({router_ms}ms): Quyết định '{r_action}' -> {action_label}. ({r_reason})"
                    }
                    thoughts_history.append(router_thought)
                    yield f"data: {json.dumps(router_thought, ensure_ascii=False)}\n\n"

                # Khi Agent bắt đầu vòng lặp tư duy mới
                elif event_type == "on_chain_start" and name == "agent":
                    # Xả buffer suy luận của vòng lặp trước (nếu có) và đặt lại trạng thái thẻ <think>
                    if stream_buffer.strip():
                        leftover = re.sub(r"</?(?:think|thought|quist|accessibility)>", "", stream_buffer)
                        leftover = re.sub(r"call:(?:default_api:)?[a-zA-Z0-9_]+\s*\{.*?\}", "", leftover, flags=re.DOTALL).strip()
                        if leftover:
                            full_thinking_parts.append(leftover + "\n")
                            yield f"data: {json.dumps({'type': 'thought_token', 'content': leftover}, ensure_ascii=False)}\n\n"
                    stream_buffer = ""
                    in_think_tag = False
                    think_ended = False

                    iteration += 1
                    if iteration == 1:
                        thought = {
                            "type": "thought",
                            "step": "analyze",
                            "content": "Đang phân tích yêu cầu và lựa chọn công cụ xử lý tối ưu..."
                        }
                    else:
                        thought = {
                            "type": "thought",
                            "step": "synthesize",
                            "content": f"Vòng lặp ReAct #{iteration}: Đang đánh giá ngữ cảnh thu được (Corrective RAG) để tổng hợp hoặc tra cứu bổ sung..."
                        }
                    thoughts_history.append(thought)
                    yield f"data: {json.dumps(thought, ensure_ascii=False)}\n\n"

                # Khi LLM kết thúc một lượt suy luận
                elif event_type == "on_chat_model_end":
                    output = data.get("output")
                    tool_calls = getattr(output, "tool_calls", None) or []
                    if tool_calls:
                        tool_names = ", ".join([t.get("name", "") for t in tool_calls])
                        step_msg = (
                            f"Quyết định kích hoạt công cụ: [{tool_names}]."
                            if iteration == 1
                            else f"Corrective RAG: Kích hoạt bổ sung công cụ [{tool_names}] để làm giàu thông tin."
                        )
                        thought = {
                            "type": "thought",
                            "step": "decision",
                            "content": step_msg
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
                    tool_calls_count += 1
                    tool_input = data.get("input", {})
                    if name == "retrieve_documents":
                        query = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)
                        doc_name = tool_input.get("document_name") if isinstance(tool_input, dict) else None
                        thought_msg = f"Đang tra cứu Hybrid Search (Milvus + BM25 + BGE Reranker) với truy vấn: '{query}'"
                        if doc_name:
                            thought_msg += f" (lọc tài liệu: {doc_name})"
                    elif name == "list_uploaded_documents":
                        thought_msg = "Đang truy xuất danh sách toàn bộ tài liệu hiện có trong hệ thống..."
                    elif name == "get_document_overview":
                        doc_name = tool_input.get("document_name", "") if isinstance(tool_input, dict) else str(tool_input)
                        thought_msg = f"Đang trích xuất mục lục và tổng quan tài liệu: '{doc_name}'..."
                    elif name == "calculate_metrics":
                        expr = tool_input.get("expression", "") if isinstance(tool_input, dict) else str(tool_input)
                        thought_msg = f"Đang tính toán biểu thức số học: `{expr}`..."
                    else:
                        thought_msg = f"Đang thực thi công cụ `{name}`..."

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
                    if name == "retrieve_documents":
                        has_found = "Không tìm thấy thông tin" not in out_content
                        if has_found:
                            doc_count = out_content.count("[Tài liệu:") if "[Tài liệu:" in out_content else 1
                            has_low_conf = "[ĐÁNH GIÁ CORRECTIVE RAG]" in out_content
                            result_msg = f"Đã tìm thấy {doc_count} đoạn ngữ cảnh phù hợp nhất sau khi Rerank."
                            if has_low_conf:
                                result_msg += " (Điểm tương quan thấp -> kích hoạt đánh giá Corrective RAG)."

                            # Trích xuất cấu trúc sources để hỗ trợ Interactive Citations trên UI
                            extracted_sources = []
                            pattern = r"\[Source (\d+)\]\s*\[Tài liệu:\s*([^\]\-]+?)(?:\s*-\s*Trang\s*(\d+))?\](?:\s*\(([^)]+)\))?"
                            docs_in_db = db_service.get_all_documents()
                            doc_map = {d["name"].lower().strip(): d["id"] for d in docs_in_db}

                            for match in re.finditer(pattern, out_content):
                                s_id, d_name, p_num, s_meta = match.groups()
                                score_match = re.search(r"([\d\.]+)/1\.00", s_meta or "")
                                lvl_match = re.search(r"\((Cao|Trung bình|Thấp)\)", s_meta or "")
                                clean_doc = d_name.strip() if d_name else ""
                                extracted_sources.append({
                                    "source_id": int(s_id),
                                    "document_name": clean_doc,
                                    "document_id": doc_map.get(clean_doc.lower(), ""),
                                    "page": int(p_num) if p_num else 1,
                                    "score": float(score_match.group(1)) if score_match else 0.5,
                                    "level": lvl_match.group(1) if lvl_match else "Trung bình",
                                })

                            if extracted_sources:
                                max_s = max(s["score"] for s in extracted_sources)
                                if top_rerank_score is None or max_s > top_rerank_score:
                                    top_rerank_score = max_s
                                sources_event = {
                                    "type": "sources",
                                    "step": "sources",
                                    "sources": extracted_sources,
                                }
                                thoughts_history.append(sources_event)
                                yield f"data: {json.dumps(sources_event, ensure_ascii=False)}\n\n"
                        else:
                            result_msg = "Không tìm thấy đoạn văn bản khớp trực tiếp -> Gợi ý Agent tự động viết lại truy vấn (Query Rewriting)."
                    elif name == "list_uploaded_documents":
                        doc_count = out_content.count("[Tài liệu:")
                        result_msg = f"Đã lấy danh sách {doc_count} tài liệu trong kho lưu trữ."
                    elif name == "get_document_overview":
                        result_msg = "Đã trích xuất cấu trúc đề mục và nội dung tổng quan của tài liệu."
                    elif name == "calculate_metrics":
                        result_msg = out_content
                    else:
                        result_msg = f"Công cụ `{name}` đã hoàn tất thực thi."

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
                            full_thinking_parts.append(thinking_token)
                            yield f"data: {json.dumps({'type': 'thought_token', 'content': thinking_token}, ensure_ascii=False)}\n\n"

                        raw_content = chunk.content
                        if not raw_content:
                            continue

                        chunk_text = ""
                        if isinstance(raw_content, str):
                            chunk_text = raw_content
                        elif isinstance(raw_content, list):
                            parts = []
                            for p in raw_content:
                                if isinstance(p, dict) and "text" in p:
                                    parts.append(p["text"])
                                elif isinstance(p, str):
                                    parts.append(p)
                            chunk_text = "".join(parts)

                        if not chunk_text:
                            continue

                        stream_buffer += chunk_text
                        # Chuẩn hóa các biến thể thẻ đóng/mở của Gemini 3.5
                        stream_buffer = re.sub(r"<accessibility>\s*</accessibility>", "", stream_buffer)
                        stream_buffer = stream_buffer.replace("<thought>", "<think>")
                        stream_buffer = stream_buffer.replace("</thought>", "</think>").replace("</quist>", "</think>")

                        # 1. Phát hiện thẻ mở <think> (kể cả ở vòng lặp thứ 2-3)
                        if not in_think_tag:
                            if "<think>" in stream_buffer:
                                before_open, after_open = stream_buffer.split("<think>", 1)
                                before_clean = re.sub(r"</?accessibility>", "", before_open)
                                if before_clean.strip():
                                    full_answer_parts.append(before_clean)
                                    yield f"data: {json.dumps({'type': 'token', 'content': before_clean}, ensure_ascii=False)}\n\n"
                                in_think_tag = True
                                think_ended = False
                                stream_buffer = after_open
                            elif not think_ended and len(stream_buffer) > 25:
                                think_ended = True

                        # 2. Đang trong thẻ <think> (Suy nghĩ chi tiết của mô hình)
                        if in_think_tag:
                            if "</think>" in stream_buffer:
                                think_part, after_close = stream_buffer.split("</think>", 1)
                                in_think_tag = False
                                think_ended = True
                                if think_part:
                                    full_thinking_parts.append(think_part)
                                    yield f"data: {json.dumps({'type': 'thought_token', 'content': think_part}, ensure_ascii=False)}\n\n"

                                complete_think_text = "".join(full_thinking_parts).strip()
                                thoughts_history = [t for t in thoughts_history if t.get("step") != "reasoning"]
                                reason_item = {
                                    "type": "thought",
                                    "step": "reasoning",
                                    "content": complete_think_text
                                }
                                thoughts_history.append(reason_item)
                                yield f"data: {json.dumps(reason_item, ensure_ascii=False)}\n\n"
                                stream_buffer = after_close
                            else:
                                safe_cut = max(0, len(stream_buffer) - 18)
                                if safe_cut > 0:
                                    to_send = stream_buffer[:safe_cut]
                                    stream_buffer = stream_buffer[safe_cut:]
                                    full_thinking_parts.append(to_send)
                                    yield f"data: {json.dumps({'type': 'thought_token', 'content': to_send}, ensure_ascii=False)}\n\n"

                        # 3. Đã kết thúc thẻ <think>, xuất token câu trả lời chính thức
                        if think_ended and not in_think_tag and stream_buffer:
                            # Giữ lại phần đuôi nếu nghi ngờ đang mở thẻ <think> hoặc call: dở dang
                            hold_back = 0
                            if "<" in stream_buffer[-16:]:
                                hold_back = len(stream_buffer) - stream_buffer.rfind("<")
                            elif "call:" in stream_buffer[-30:]:
                                hold_back = len(stream_buffer) - stream_buffer.rfind("call:")

                            emit_len = len(stream_buffer) - hold_back
                            if emit_len > 0:
                                to_emit = stream_buffer[:emit_len]
                                to_emit = re.sub(r"call:(?:default_api:)?[a-zA-Z0-9_]+\s*\{.*?\}", "", to_emit, flags=re.DOTALL)
                                stream_buffer = stream_buffer[emit_len:]
                                if to_emit:
                                    if not first_token_sent:
                                        first_token_sent = True
                                        thought = {
                                            "type": "thought",
                                            "step": "responding",
                                            "content": "Bắt đầu xuất câu trả lời cho người dùng..."
                                        }
                                        thoughts_history.append(thought)
                                        yield f"data: {json.dumps(thought, ensure_ascii=False)}\n\n"

                                    full_answer_parts.append(to_emit)
                                    yield f"data: {json.dumps({'type': 'token', 'content': to_emit}, ensure_ascii=False)}\n\n"

            # Xử lý phần còn tồn đọng trong buffer nếu kết thúc stream
            if stream_buffer:
                cleaned_rem = re.sub(r"</?(?:think|thought|quist|accessibility)>", "", stream_buffer)
                cleaned_rem = re.sub(r"call:(?:default_api:)?[a-zA-Z0-9_]+\s*\{.*?\}", "", cleaned_rem, flags=re.DOTALL)
                if in_think_tag:
                    if cleaned_rem.strip():
                        full_thinking_parts.append(cleaned_rem)
                        yield f"data: {json.dumps({'type': 'thought_token', 'content': cleaned_rem}, ensure_ascii=False)}\n\n"
                    complete_think_text = "".join(full_thinking_parts).strip()
                    thoughts_history = [t for t in thoughts_history if t.get("step") != "reasoning"]
                    reason_item = {
                        "type": "thought",
                        "step": "reasoning",
                        "content": complete_think_text
                    }
                    thoughts_history.append(reason_item)
                    yield f"data: {json.dumps(reason_item, ensure_ascii=False)}\n\n"
                elif cleaned_rem:
                    full_answer_parts.append(cleaned_rem)
                    yield f"data: {json.dumps({'type': 'token', 'content': cleaned_rem}, ensure_ascii=False)}\n\n"

            # Bảo vệ chống câu trả lời rỗng (nếu LLM quên đóng thẻ </think> ở lượt cuối)
            if not "".join(full_answer_parts).strip() and full_thinking_parts:
                fallback_text = "".join(full_thinking_parts).strip()
                fallback_text = re.sub(r"call:(?:default_api:)?[a-zA-Z0-9_]+\s*\{.*?\}", "", fallback_text, flags=re.DOTALL).strip()
                if fallback_text:
                    full_answer_parts.append(fallback_text)
                    yield f"data: {json.dumps({'type': 'token', 'content': fallback_text}, ensure_ascii=False)}\n\n"

            # 2. Hoàn tất toàn bộ luồng
            total_latency_ms = round((time.perf_counter() - stream_start_time) * 1000, 1)
            complete_thought = {
                "type": "thought",
                "step": "complete",
                "content": f"Hoàn tất toàn bộ quy trình suy luận và phản hồi ({total_latency_ms}ms)."
            }
            thoughts_history.append(complete_thought)
            yield f"data: {json.dumps(complete_thought, ensure_ascii=False)}\n\n"

            telemetry_event = {
                "type": "telemetry",
                "step": "telemetry",
                "telemetry": {
                    "latency_ms": total_latency_ms,
                    "router_ms": router_ms,
                    "tool_calls": tool_calls_count,
                    "top_rerank_score": top_rerank_score,
                },
            }
            thoughts_history.append(telemetry_event)
            yield f"data: {json.dumps(telemetry_event, ensure_ascii=False)}\n\n"

            # Lấy Checkpoint ID mới nhất từ LangGraph state
            latest_state = await graph.aget_state({"configurable": {"thread_id": session_id}})
            latest_checkpoint_id = latest_state.config["configurable"].get("checkpoint_id")

            # Lưu Assistant Message vào DB
            full_answer = "".join(full_answer_parts).strip()
            asst_msg_id = db_service.save_message(
                session_id=session_id,
                role="assistant",
                content=full_answer,
                thoughts=thoughts_history,
                checkpoint_id=latest_checkpoint_id
            )

            # [Semantic Cache]: Lưu phản hồi vào cache để tái sử dụng với 0 token LLM
            if full_answer and len(full_answer) > 10:
                saved_sources = []
                for t in thoughts_history:
                    if t.get("type") == "sources" and t.get("sources"):
                        saved_sources = t["sources"]
                        break
                semantic_cache_service.store(
                    query=question,
                    answer=full_answer,
                    thoughts=thoughts_history,
                    sources=saved_sources,
                    scope_doc=target_doc,
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
            import traceback
            traceback.print_exc()
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