import { useState, useRef, useEffect } from "react"
import { sendMessageStream, getSessionMessages, getCheckpoints, rollbackChat } from "../services/chatAPI"

function formatPlainText(text) {
    if (!text) return ""
    return text
        // Xóa hoàn toàn các cặp dấu ** và __
        .replace(/\*\*/g, "")
        .replace(/__/g, "")
        // Xóa các dấu # ở đầu dòng (ví dụ: ## Tiêu đề -> Tiêu đề)
        .replace(/^#{1,6}\s*/gm, "")
        // Đổi bullet dạng * hoặc • thành -
        .replace(/^[\*\•]\s+/gm, "- ")
}

const STEP_CONFIG = {
    receive: { icon: "📥", label: "Tiếp nhận câu hỏi", bg: "#eef2ff", color: "#4338ca", border: "#c7d2fe" },
    analyze: { icon: "🧠", label: "Phân tích yêu cầu", bg: "#f5f3ff", color: "#6d28d9", border: "#ddd6fe" },
    decision: { icon: "🎯", label: "Quyết định xử lý", bg: "#f0f9ff", color: "#0369a1", border: "#bae6fd" },
    tool_call: { icon: "🔍", label: "Truy vấn tài liệu", bg: "#eff6ff", color: "#1d4ed8", border: "#bfdbfe" },
    tool_result: { icon: "📑", label: "Ngữ cảnh tìm được", bg: "#ecfdf5", color: "#047857", border: "#a7f3d0" },
    synthesize: { icon: "🧩", label: "Tổng hợp thông tin", bg: "#fffbeb", color: "#b45309", border: "#fde68a" },
    responding: { icon: "✍️", label: "Soạn câu trả lời", bg: "#faf5ff", color: "#7e22ce", border: "#f3e8ff" },
    complete: { icon: "✅", label: "Hoàn tất suy luận", bg: "#f0fdf4", color: "#15803d", border: "#bbf7d0" },
}

function ChatBox({ sessionId = "default", sessionTitle = "Cuộc trò chuyện", onRefreshSessions }) {
    const [question, setQuestion] = useState("")
    const [messages, setMessages] = useState([])
    const [loading, setLoading] = useState(false)
    const [loadingHistory, setLoadingHistory] = useState(false)
    
    // State cho Rollback từ User Message
    const [rollbackState, setRollbackState] = useState(null)
    
    // State cho Checkpoint Drawer
    const [isDrawerOpen, setIsDrawerOpen] = useState(false)
    const [checkpoints, setCheckpoints] = useState([])
    const [loadingCheckpoints, setLoadingCheckpoints] = useState(false)

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
                    return {
                        id: m.id,
                        role: m.role,
                        content: m.content,
                        thoughts: parsedThoughts,
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

    // Mở / Đóng Checkpoint Drawer
    const toggleCheckpointDrawer = async () => {
        if (!isDrawerOpen) {
            setIsDrawerOpen(true)
            setLoadingCheckpoints(true)
            try {
                const cps = await getCheckpoints(sessionId)
                setCheckpoints(cps)
            } catch (err) {
                console.error("Failed to load checkpoints:", err)
            } finally {
                setLoadingCheckpoints(false)
            }
        } else {
            setIsDrawerOpen(false)
        }
    }

    // Khôi phục một checkpoint từ Drawer
    const handleRestoreCheckpoint = async (checkpointId) => {
        try {
            setLoading(true)
            const res = await rollbackChat(sessionId, checkpointId)
            const formatted = (res.messages || []).map((m) => {
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
                return {
                    id: m.id,
                    role: m.role,
                    content: m.content,
                    thoughts: parsedThoughts,
                    checkpoint_id: m.checkpoint_id,
                    isThinking: false,
                    showThoughts: false,
                }
            })
            setMessages(formatted)
            setIsDrawerOpen(false)
        } catch (err) {
            console.error("Failed to restore checkpoint:", err)
        } finally {
            setLoading(false)
        }
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

                if (event.type === "thought") {
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

                <div className="chat-topbar-actions">
                    <button
                        className={`checkpoints-toggle-btn ${isDrawerOpen ? "active" : ""}`}
                        onClick={toggleCheckpointDrawer}
                        title="Xem lịch sử Checkpoints & Khôi phục"
                    >
                        <span className="cp-btn-icon">⏱️</span>
                        <span>Checkpoints</span>
                    </button>
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
                                        <span className="rollback-btn-icon">↩️</span>
                                        <span>Quay về & sửa</span>
                                    </button>
                                </div>
                            )}

                            {/* Khối toàn bộ lịch sử suy luận (Thinking Box) cho phản hồi của Assistant */}
                            {message.role === "assistant" &&
                                message.thoughts &&
                                message.thoughts.length > 0 && (
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
                                                <span
                                                    className={`thinking-icon ${
                                                        message.isThinking ? "pulse" : ""
                                                    }`}
                                                >
                                                    {message.isThinking ? "⚙" : "🧠"}
                                                </span>
                                                <span style={{ fontWeight: 600 }}>
                                                    {message.isThinking
                                                        ? "Đang suy luận & xử lý..."
                                                        : "Lịch sử quá trình suy luận"}
                                                </span>
                                                <span className="thinking-badge-count">
                                                    {message.thoughts.length} bước
                                                </span>
                                                {message.checkpoint_id && (
                                                    <span className="checkpoint-pill" title={`Checkpoint ID: ${message.checkpoint_id}`}>
                                                        ⏱️ {message.checkpoint_id.slice(0, 8)}...
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
                                                {message.thoughts.map((thought, idx) => {
                                                    const cfg =
                                                        STEP_CONFIG[thought.step] || {
                                                            icon: "•",
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
                                                                        <span>{cfg.icon}</span>
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
                                    <div className="assistant-text-body">{formatPlainText(message.content)}</div>
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

            {/* Checkpoint Timeline Drawer / Sidebar */}
            {isDrawerOpen && (
                <div className="checkpoint-drawer-overlay" onClick={() => setIsDrawerOpen(false)}>
                    <div className="checkpoint-drawer" onClick={(e) => e.stopPropagation()}>
                        <div className="drawer-header">
                            <div className="drawer-title">
                                <span className="drawer-icon">⏱️</span>
                                <h3>Lịch sử Checkpoints</h3>
                            </div>
                            <button
                                className="drawer-close-btn"
                                onClick={() => setIsDrawerOpen(false)}
                            >
                                ✕
                            </button>
                        </div>

                        <p className="drawer-subtitle">
                            LangGraph đã lưu trạng thái hội thoại ở từng bước. Bạn có thể khôi phục để rẽ nhánh trò chuyện.
                        </p>

                        <div className="checkpoint-list">
                            {loadingCheckpoints ? (
                                <div className="drawer-loading">
                                    <div className="typing-dot" />
                                    <div className="typing-dot" />
                                    <div className="typing-dot" />
                                    <span>Đang tải checkpoints...</span>
                                </div>
                            ) : checkpoints.length === 0 ? (
                                <div className="drawer-empty">
                                    <span>Chưa có checkpoint nào cho phiên này. Hãy bắt đầu nhắn tin!</span>
                                </div>
                            ) : (
                                checkpoints.map((cp, idx) => (
                                    <div key={cp.checkpoint_id} className="checkpoint-card">
                                        <div className="cp-card-top">
                                            <span className="cp-badge">Bước #{cp.step || (checkpoints.length - idx)}</span>
                                            <span className="cp-messages-count">{cp.message_count} tin nhắn</span>
                                        </div>
                                        <div className="cp-card-id" title={cp.checkpoint_id}>
                                            ID: <code>{cp.checkpoint_id.slice(0, 18)}...</code>
                                        </div>
                                        {cp.summary && (
                                            <div className="cp-summary">{cp.summary}</div>
                                        )}
                                        <button
                                            className="checkpoint-restore-btn"
                                            onClick={() => handleRestoreCheckpoint(cp.checkpoint_id)}
                                            disabled={loading}
                                        >
                                            ↩️ Khôi phục trạng thái này
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Rollback Notification Banner */}
            {rollbackState && (
                <div className="rollback-banner">
                    <div className="rollback-banner-info">
                        <span className="rollback-banner-icon">↩️</span>
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
        </div>
    )
}

export default ChatBox