import { useState, useRef, useEffect } from "react"
import { sendMessageStream, getSessionMessages } from "../services/chatAPI"
import MarkdownViewer from "./MarkdownViewer"
import PDFModal from "./PDFModal"

function formatPlainText(text) {
    if (!text) return ""
    return text
        .replace(/^[\*\•]\s+/gm, "- ")
}

const STEP_CONFIG = {
    receive: { label: "Tiếp nhận câu hỏi", bg: "#eef2ff", color: "#4338ca", border: "#c7d2fe" },
    router: { label: "Laya Router (System 1)", bg: "#fdf4ff", color: "#a21caf", border: "#f5d0fe" },
    analyze: { label: "Phân tích yêu cầu", bg: "#f5f3ff", color: "#6d28d9", border: "#ddd6fe" },
    decision: { label: "Quyết định xử lý", bg: "#f0f9ff", color: "#0369a1", border: "#bae6fd" },
    tool_call: { label: "Kích hoạt công cụ", bg: "#eff6ff", color: "#1d4ed8", border: "#bfdbfe" },
    tool_result: { label: "Kết quả công cụ", bg: "#ecfdf5", color: "#047857", border: "#a7f3d0" },
    synthesize: { label: "Corrective RAG & Tổng hợp", bg: "#fffbeb", color: "#b45309", border: "#fde68a" },
    responding: { label: "Soạn câu trả lời", bg: "#faf5ff", color: "#7e22ce", border: "#f3e8ff" },
    reasoning: { label: "Suy nghĩ của mô hình", bg: "#f8fafc", color: "#334155", border: "#cbd5e1" },
    complete: { label: "Hoàn tất suy luận", bg: "#f0fdf4", color: "#15803d", border: "#bbf7d0" },
}

function ChatBox({ sessionId = "default", sessionTitle = "Cuộc trò chuyện", onRefreshSessions }) {
    const [question, setQuestion] = useState("")
    const [messages, setMessages] = useState([])
    const [loading, setLoading] = useState(false)
    const [loadingHistory, setLoadingHistory] = useState(false)
    
    // State cho Rollback từ User Message
    const [rollbackState, setRollbackState] = useState(null)

    // State cho PDF Viewer Modal & Interactive Citations
    const [activePdfModal, setActivePdfModal] = useState(null)

    const messagesEndRef = useRef(null)
    const textareaRef = useRef(null)


    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }

    // Tải lịch sử tin nhắn khi sessionId thay đổi
    useEffect(() => {
        let isMounted = true
        setRollbackState(null)
        setLoadingHistory(true)

        const fetchHistory = async () => {
            try {
                const dbMsgs = await getSessionMessages(sessionId)
                if (!isMounted) return

                const formatted = dbMsgs.map((m) => {
                    let parsedThoughts = []
                    if (Array.isArray(m.thoughts)) {
                        parsedThoughts = m.thoughts
                    } else if (m.thoughts_json) {
                        try {
                            parsedThoughts = JSON.parse(m.thoughts_json)
                        } catch (e) {
                            console.error("Error parsing thoughts_json:", e)
                        }
                    }
                    const reasoningItem = parsedThoughts.find((t) => t.step === "reasoning")
                    const thinkingText = reasoningItem ? (reasoningItem.content || reasoningItem.text || "") : ""
                    const sourcesItem = parsedThoughts.find((t) => t.type === "sources" || t.step === "sources")
                    const loadedSources = sourcesItem?.sources || []

                    return {
                        id: m.id,
                        role: m.role,
                        content: m.content,
                        thoughts: parsedThoughts,
                        sources: loadedSources,
                        thinkingContent: thinkingText,
                        checkpoint_id: m.checkpoint_id,
                        isThinking: false,
                        showThoughts: false,
                    }
                })
                setMessages(formatted)
            } catch (err) {
                console.error("Failed to load session messages:", err)
            } finally {
                if (isMounted) setLoadingHistory(false)
            }
        }

        fetchHistory()
        return () => {
            isMounted = false
        }
    }, [sessionId])

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const handleOpenCitation = (citationInfo) => {
        if (!citationInfo) return
        setActivePdfModal({
            isOpen: true,
            documentName: citationInfo.documentName || "Tài liệu",
            documentId: citationInfo.documentId || null,
            page: citationInfo.page || 1,
            snippet: citationInfo.snippet || "",
        })
    }

    const toggleThoughts = (id) => {
        setMessages((previous) =>
            previous.map((msg) =>
                msg.id === id ? { ...msg, showThoughts: !msg.showThoughts } : msg
            )
        )
    }


    // Bắt đầu quy trình Rollback & Chỉnh sửa câu hỏi
    const handleStartRollback = (userMsg) => {
        const msgIndex = messages.findIndex((m) => m.id === userMsg.id)
        if (msgIndex === -1) return

        // Tìm checkpoint của assistant ngay trước câu hỏi này
        let priorCheckpointId = null
        for (let i = msgIndex - 1; i >= 0; i--) {
            if (messages[i].role === "assistant" && messages[i].checkpoint_id) {
                priorCheckpointId = messages[i].checkpoint_id
                break
            }
        }

        setRollbackState({
            targetMessageId: userMsg.id,
            checkpointId: priorCheckpointId,
            originalText: userMsg.content,
        })

        // Nạp nội dung câu hỏi vào textarea để người dùng cập nhật
        setQuestion(userMsg.content)
        textareaRef.current?.focus()
    }

    const cancelRollback = () => {
        setRollbackState(null)
        setQuestion("")
    }

    const handleSend = async () => {
        if (!question.trim() || loading) {
            return
        }

        const currentQuestion = question.trim()
        const userMessageId = Date.now()
        const assistantMessageId = userMessageId + 1

        const isRollingBack = Boolean(rollbackState)
        const targetMessageId = rollbackState?.targetMessageId || null
        const targetCheckpointId = rollbackState?.checkpointId || null

        // Cập nhật optimistic messages trong UI
        let updatedMessages = [...messages]
        if (isRollingBack && targetMessageId) {
            const cutIndex = updatedMessages.findIndex((m) => m.id === targetMessageId)
            if (cutIndex !== -1) {
                updatedMessages = updatedMessages.slice(0, cutIndex)
            }
        }

        const userMessage = {
            id: userMessageId,
            role: "user",
            content: currentQuestion,
        }

        const initialAssistantMessage = {
            id: assistantMessageId,
            role: "assistant",
            content: "",
            thoughts: [],
            sources: [],
            thinkingContent: "",
            isThinking: true,
            showThoughts: true,
        }

        const history = updatedMessages.map((msg) => ({
            role: msg.role,
            content: msg.content,
        }))

        setMessages([...updatedMessages, userMessage, initialAssistantMessage])
        setQuestion("")
        setRollbackState(null)
        setLoading(true)

        await sendMessageStream({
            question: currentQuestion,
            history: history,
            sessionId: sessionId,
            checkpointId: targetCheckpointId,
            rollbackFromMessageId: targetMessageId,
            onEvent: (event) => {
                const nowTime = new Date().toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                })

                if (event.type === "thought_token") {
                    setMessages((previous) =>
                        previous.map((msg) =>
                            msg.id === assistantMessageId
                                ? {
                                      ...msg,
                                      thinkingContent: (msg.thinkingContent || "") + event.content,
                                      isThinking: true,
                                      showThoughts: true,
                                  }
                                : msg
                        )
                    )
                } else if (event.type === "thought") {
                    if (event.step === "reasoning") {
                        setMessages((previous) =>
                            previous.map((msg) => {
                                if (msg.id !== assistantMessageId) return msg
                                const remaining = msg.thoughts.filter((t) => t.step !== "reasoning")
                                return {
                                    ...msg,
                                    thinkingContent: event.content,
                                    thoughts: [
                                        ...remaining,
                                        {
                                            step: "reasoning",
                                            text: event.content,
                                            time: nowTime,
                                        },
                                    ],
                                }
                            })
                        )
                    } else {
                        setMessages((previous) =>
                            previous.map((msg) =>
                                msg.id === assistantMessageId
                                    ? {
                                          ...msg,
                                          thoughts: [
                                              ...msg.thoughts,
                                              {
                                                  step: event.step || "thought",
                                                  text: event.content,
                                                  time: nowTime,
                                              },
                                          ],
                                      }
                                    : msg
                            )
                        )
                    }
                } else if (event.type === "sources") {
                    setMessages((previous) =>
                        previous.map((msg) =>
                            msg.id === assistantMessageId
                                ? {
                                      ...msg,
                                      sources: event.sources || [],
                                  }
                                : msg
                        )
                    )
                } else if (event.type === "tool_call") {
                    setMessages((previous) =>
                        previous.map((msg) => {
                            if (msg.id !== assistantMessageId) return msg
                            const thoughts = [...msg.thoughts]
                            if (thoughts.length > 0) {
                                thoughts[thoughts.length - 1] = {
                                    ...thoughts[thoughts.length - 1],
                                    toolDetails: event.input,
                                    toolName: event.tool,
                                }
                            }
                            return { ...msg, thoughts }
                        })
                    )
                } else if (event.type === "tool_result") {
                    setMessages((previous) =>
                        previous.map((msg) => {
                            if (msg.id !== assistantMessageId) return msg
                            const thoughts = [...msg.thoughts]
                            if (thoughts.length > 0) {
                                thoughts[thoughts.length - 1] = {
                                    ...thoughts[thoughts.length - 1],
                                    summary: event.summary,
                                }
                            }
                            return { ...msg, thoughts }
                        })
                    )
                } else if (event.type === "token") {
                    setMessages((previous) =>
                        previous.map((msg) =>
                            msg.id === assistantMessageId
                                ? {
                                      ...msg,
                                      content: msg.content + event.content,
                                      isThinking: false,
                                  }
                                : msg
                        )
                    )
                } else if (event.type === "checkpoint") {
                    setMessages((previous) =>
                        previous.map((msg) =>
                            msg.id === assistantMessageId
                                ? {
                                      ...msg,
                                      checkpoint_id: event.checkpoint_id,
                                      db_id: event.assistant_message_id,
                                  }
                                : msg
                        )
                    )
                    if (onRefreshSessions) {
                        onRefreshSessions()
                    }
                } else if (event.type === "error") {
                    setMessages((previous) =>
                        previous.map((msg) =>
                            msg.id === assistantMessageId
                                ? {
                                      ...msg,
                                      content:
                                          msg.content ||
                                          `Xin lỗi, đã xảy ra sự cố: ${event.content}`,
                                      isThinking: false,
                                  }
                                : msg
                        )
                    )
                } else if (event.type === "done") {
                    setMessages((previous) =>
                        previous.map((msg) =>
                            msg.id === assistantMessageId
                                ? {
                                      ...msg,
                                      isThinking: false,
                                  }
                                : msg
                        )
                    )
                }
            },
            onError: (error) => {
                console.error("Failed to stream message:", error)
                setMessages((previous) =>
                    previous.map((msg) =>
                        msg.id === assistantMessageId
                            ? {
                                  ...msg,
                                  content:
                                      msg.content ||
                                      "Rất tiếc, đã xảy ra lỗi kết nối khi xử lý câu hỏi của bạn.",
                                  isThinking: false,
                              }
                            : msg
                    )
                )
                setLoading(false)
            },
            onComplete: () => {
                setLoading(false)
            }
        })
    }

    const handleKeyDown = (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault()
            handleSend()
        }
    }

    return (
        <div className="chat-container">
            {/* Topbar điều khiển Chat & Checkpoints */}
            <header className="chat-topbar">
                <div className="chat-topbar-info">
                    <h2>{sessionTitle}</h2>
                    <span className="chat-topbar-sub">Mã phiên: {sessionId}</span>
                </div>


            </header>

            {/* Khung nội dung hội thoại */}
            <div className="messages">
                {loadingHistory ? (
                    <div className="loading-history-indicator">
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                        <span>Đang nạp hội thoại...</span>
                    </div>
                ) : messages.length === 0 ? (
                    <div className="empty-chat">
                        <div className="empty-chat-icon">✦</div>
                        <h2>Hỏi bất kỳ điều gì về tài liệu của bạn</h2>
                        <p>
                            Tải lên tài liệu và đặt câu hỏi. AI sẽ tìm kiếm
                            ngữ cảnh từ dữ liệu của bạn để đưa ra câu trả lời chính xác.
                        </p>
                    </div>
                ) : null}

                {messages.map((message) => (
                    <div
                        className={
                            message.role === "user"
                                ? "message user-message"
                                : "message assistant-message"
                        }
                        key={message.id}
                    >
                        <div
                            className={
                                message.role === "user"
                                    ? "avatar user-avatar"
                                    : "avatar assistant-avatar"
                            }
                        >
                            {message.role === "user" ? "U" : "AI"}
                        </div>

                        <div className="message-content">
                            {/* Nút Rollback & Edit hiển thị ở User Message */}
                            {message.role === "user" && (
                                <div className="user-message-header">
                                    <div className="user-message-text">
                                        {message.content}
                                    </div>
                                    <button
                                        className="rollback-msg-btn"
                                        onClick={() => handleStartRollback(message)}
                                        title="Quay về đây và sửa lại câu hỏi"
                                    >
                                        <span>Quay về & sửa</span>
                                    </button>
                                </div>
                            )}

                            {/* Khối toàn bộ lịch sử suy luận (Thinking Box) cho phản hồi của Assistant */}
                            {message.role === "assistant" &&
                                ((message.thoughts && message.thoughts.length > 0) ||
                                 message.thinkingContent ||
                                 message.isThinking) && (
                                    <div
                                        className={`thinking-box ${
                                            message.showThoughts ? "expanded" : ""
                                        }`}
                                    >
                                        <div
                                            className="thinking-header"
                                            onClick={() => toggleThoughts(message.id)}
                                        >
                                            <div className="thinking-title">
                                                <span style={{ fontWeight: 600 }}>
                                                    {message.isThinking
                                                        ? "Đang suy luận & xử lý..."
                                                        : "Lịch sử quá trình suy luận"}
                                                </span>
                                                <span className="thinking-badge-count">
                                                    {message.thoughts ? message.thoughts.length : 0} bước
                                                </span>
                                                {message.checkpoint_id && (
                                                    <span className="checkpoint-pill" title={`Checkpoint ID: ${message.checkpoint_id}`}>
                                                        {message.checkpoint_id.slice(0, 8)}...
                                                    </span>
                                                )}
                                            </div>
                                            <span
                                                className={`thinking-arrow ${
                                                    message.showThoughts ? "open" : ""
                                                }`}
                                            >
                                                ▼
                                            </span>
                                        </div>

                                        {message.showThoughts && (
                                            <div className="thinking-content">
                                                {/* Khối suy nghĩ chi tiết của mô hình (DeepSeek-style) */}
                                                {(message.thinkingContent || (message.isThinking && !message.content)) && (
                                                    <div className="deepseek-reasoning-section">
                                                        <div className="reasoning-section-header">
                                                            <span className="reasoning-section-title">Quá trình tư duy của mô hình</span>
                                                            {message.isThinking && !message.content && (
                                                                <span className="reasoning-live-badge">Đang suy nghĩ...</span>
                                                            )}
                                                        </div>
                                                        <div className="reasoning-section-body">
                                                            {message.thinkingContent ? (
                                                                <div className="reasoning-text-stream">
                                                                    {formatPlainText(message.thinkingContent)}
                                                                    {message.isThinking && !message.content && (
                                                                        <span className="reasoning-cursor" />
                                                                    )}
                                                                </div>
                                                            ) : (
                                                                <div className="thinking-typing-placeholder">
                                                                    <span>Mô hình đang phân tích và tư duy</span>
                                                                    <div className="typing-dot" />
                                                                    <div className="typing-dot" />
                                                                    <div className="typing-dot" />
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                )}

                                                {/* Danh sách các bước xử lý hệ thống */}
                                                {message.thoughts &&
                                                    message.thoughts
                                                        .filter((t) => t.step !== "reasoning")
                                                        .map((thought, idx) => {
                                                            const cfg =
                                                                STEP_CONFIG[thought.step] || {
                                                                    label: "Tư duy",
                                                                    bg: "#f1f5f9",
                                                                    color: "#475569",
                                                                    border: "#e2e8f0",
                                                                }
                                                            return (
                                                                <div key={idx} className="thinking-step">
                                                                    <div className="step-bullet-icon">
                                                                        {idx + 1}
                                                                    </div>
                                                                    <div className="step-body">
                                                                        <div className="step-meta">
                                                                            <span
                                                                                className="step-tag"
                                                                                style={{
                                                                                    background: cfg.bg,
                                                                                    color: cfg.color,
                                                                                    border: `1px solid ${cfg.border}`,
                                                                                }}
                                                                            >
                                                                                <span>{cfg.label}</span>
                                                                            </span>
                                                                            {thought.time && (
                                                                                <span className="step-timestamp">
                                                                                    {thought.time}
                                                                                </span>
                                                                            )}
                                                                        </div>
                                                                        <div className="step-description">
                                                                            {thought.text}
                                                                        </div>
                                                                        {thought.toolDetails && (
                                                                            <div className="step-details-box">
                                                                                <strong>Tham số:</strong>{" "}
                                                                                {JSON.stringify(thought.toolDetails)}
                                                                            </div>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                            )
                                                        })}
                                            </div>
                                        )}
                                    </div>
                                )}

                            {/* Nội dung câu trả lời cho Assistant */}
                            {message.role === "assistant" && (
                                message.content ? (
                                    <>
                                        <div className="assistant-text-body">
                                            <MarkdownViewer
                                                content={message.content}
                                                onCitationClick={handleOpenCitation}
                                            />
                                        </div>

                                        {/* Khối danh sách nguồn trích dẫn tương tác (Sources Grid) */}
                                        {message.sources && message.sources.length > 0 && (
                                            <div className="sources-container">
                                                <div className="sources-header">
                                                    <span className="sources-title">
                                                        📚 Nguồn trích dẫn đối chiếu ({message.sources.length})
                                                    </span>
                                                    <span className="sources-hint">
                                                        (Nhấp vào thẻ để mở trang PDF gốc)
                                                    </span>
                                                </div>
                                                <div className="sources-cards-grid">
                                                    {message.sources.map((src, sIdx) => {
                                                        const confClass =
                                                            src.level === "Cao"
                                                                ? "conf-high"
                                                                : src.level === "Trung bình"
                                                                ? "conf-medium"
                                                                : "conf-low"
                                                        return (
                                                            <div
                                                                key={sIdx}
                                                                className="source-card"
                                                                onClick={() =>
                                                                    handleOpenCitation({
                                                                        documentName: src.document_name,
                                                                        documentId: src.document_id,
                                                                        page: src.page,
                                                                        snippet: src.snippet,
                                                                    })
                                                                }
                                                                title={`Nhấp để xem trang ${src.page} của ${src.document_name}`}
                                                            >
                                                                <div className="source-card-top">
                                                                    <span className="source-card-doc">
                                                                        📄 {src.document_name}
                                                                    </span>
                                                                    <span className="source-card-page">
                                                                        Trang {src.page}
                                                                    </span>
                                                                </div>
                                                                <div className="source-card-meta">
                                                                    <span className={`source-conf-badge ${confClass}`}>
                                                                        {src.level} ({(src.score * 100).toFixed(0)}%)
                                                                    </span>
                                                                    <span className="source-open-arrow">
                                                                        Xem PDF ↗
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        )
                                                    })}
                                                </div>
                                            </div>
                                        )}
                                    </>
                                ) : message.isThinking ? (
                                    <div className="thinking-typing-placeholder">
                                        <span>AI đang soạn thảo câu trả lời</span>
                                        <div className="typing-dot" />
                                        <div className="typing-dot" />
                                        <div className="typing-dot" />
                                    </div>
                                ) : null
                            )}
                        </div>
                    </div>
                ))}

                <div ref={messagesEndRef} />
            </div>



            {/* Rollback Notification Banner */}
            {rollbackState && (
                <div className="rollback-banner">
                    <div className="rollback-banner-info">
                        <span>
                            Đang quay về câu hỏi: <strong>"{rollbackState.originalText}"</strong>. Chỉnh sửa câu hỏi bên dưới và nhấn Gửi để rẽ nhánh mới.
                        </span>
                    </div>
                    <button className="rollback-banner-cancel" onClick={cancelRollback}>
                        ✕ Hủy
                    </button>
                </div>
            )}

            {/* Ô nhập tin nhắn */}
            <div className="chat-input-wrapper">
                <div className="chat-input-box">
                    <button
                        className="attach-button"
                        title="Đính kèm tài liệu"
                    >
                        +
                    </button>

                    <textarea
                        ref={textareaRef}
                        value={question}
                        onChange={(event) => setQuestion(event.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={
                            rollbackState
                                ? "Chỉnh sửa câu hỏi để rẽ nhánh mới..."
                                : "Hỏi bất kỳ điều gì về tài liệu của bạn..."
                        }
                        rows={1}
                    />

                    <button
                        className="send-button"
                        onClick={handleSend}
                        disabled={!question.trim() || loading}
                        title={rollbackState ? "Gửi và rẽ nhánh" : "Gửi tin nhắn"}
                    >
                        {loading ? "..." : "↑"}
                    </button>
                </div>

                <p className="input-hint">
                    AI responses may contain mistakes. Check important information against the source.
                </p>
            </div>

            {/* PDF Viewer Modal */}
            <PDFModal
                isOpen={Boolean(activePdfModal?.isOpen)}
                onClose={() => setActivePdfModal(null)}
                documentName={activePdfModal?.documentName}
                documentId={activePdfModal?.documentId}
                page={activePdfModal?.page}
                snippet={activePdfModal?.snippet}
            />
        </div>
    )
}

export default ChatBox