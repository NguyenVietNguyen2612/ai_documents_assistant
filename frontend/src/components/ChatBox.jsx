import { useState, useRef, useEffect } from "react"
import { sendMessageStream } from "../services/chatAPI"

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

function ChatBox() {
    const [question, setQuestion] = useState("")
    const [messages, setMessages] = useState(() => {
        try {
            const saved = localStorage.getItem("chat_messages_history")
            return saved ? JSON.parse(saved) : []
        } catch {
            return []
        }
    })
    const [loading, setLoading] = useState(false)
    const messagesEndRef = useRef(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }

    useEffect(() => {
        scrollToBottom()
        try {
            localStorage.setItem("chat_messages_history", JSON.stringify(messages))
        } catch (e) {
            console.error("Failed to save chat history:", e)
        }
    }, [messages])

    const toggleThoughts = (id) => {
        setMessages((previous) =>
            previous.map((msg) =>
                msg.id === id ? { ...msg, showThoughts: !msg.showThoughts } : msg
            )
        )
    }

    const handleSend = async () => {
        if (!question.trim() || loading) {
            return
        }

        const currentQuestion = question.trim()
        const userMessageId = Date.now()
        const assistantMessageId = userMessageId + 1

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

        const history = messages.map((msg) => ({
            role: msg.role,
            content: msg.content,
        }))

        setMessages((previous) => [
            ...previous,
            userMessage,
            initialAssistantMessage,
        ])

        setQuestion("")
        setLoading(true)

        await sendMessageStream(
            currentQuestion,
            history,
            (event) => {
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
            (error) => {
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
            () => {
                setLoading(false)
            }
        )
    }

    const handleKeyDown = (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault()
            handleSend()
        }
    }

    return (
        <div className="chat-container">
            <div className="messages">
                {messages.length === 0 && (
                    <div className="empty-chat">
                        <div className="empty-chat-icon">✦</div>
                        <h2>Ask anything about your documents</h2>
                        <p>
                            Upload your documents and ask questions. The AI will
                            search your knowledge base to find relevant
                            information.
                        </p>
                    </div>
                )}

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
                            {/* Khối toàn bộ lịch sử suy luận (Thinking Box) cho phản hồi của Assistant */}
                            {message.role === "assistant" &&
                                message.thoughts &&
                                message.thoughts.length > 0 && (
                                    <div
                                        className={`thinking-box ${
                                            message.showThoughts
                                                ? "expanded"
                                                : ""
                                        }`}
                                    >
                                        <div
                                            className="thinking-header"
                                            onClick={() =>
                                                toggleThoughts(message.id)
                                            }
                                        >
                                            <div className="thinking-title">
                                                <span
                                                    className={`thinking-icon ${
                                                        message.isThinking
                                                            ? "pulse"
                                                            : ""
                                                    }`}
                                                >
                                                    {message.isThinking
                                                        ? "⚙"
                                                        : "🧠"}
                                                </span>
                                                <span style={{ fontWeight: 600 }}>
                                                    {message.isThinking
                                                        ? "Đang suy luận & xử lý..."
                                                        : "Lịch sử quá trình suy luận"}
                                                </span>
                                                <span className="thinking-badge-count">
                                                    {message.thoughts.length} bước
                                                </span>
                                            </div>
                                            <span
                                                className={`thinking-arrow ${
                                                    message.showThoughts
                                                        ? "open"
                                                        : ""
                                                }`}
                                            >
                                                ▼
                                            </span>
                                        </div>

                                        {message.showThoughts && (
                                            <div className="thinking-content">
                                                {message.thoughts.map(
                                                    (thought, idx) => {
                                                        const cfg =
                                                            STEP_CONFIG[
                                                                thought.step
                                                            ] || {
                                                                icon: "•",
                                                                label: "Tư duy",
                                                                bg: "#f1f5f9",
                                                                color: "#475569",
                                                                border: "#e2e8f0",
                                                            }
                                                        return (
                                                            <div
                                                                key={idx}
                                                                className="thinking-step"
                                                            >
                                                                <div className="step-bullet-icon">
                                                                    {idx + 1}
                                                                </div>
                                                                <div className="step-body">
                                                                    <div className="step-meta">
                                                                        <span
                                                                            className="step-tag"
                                                                            style={{
                                                                                background:
                                                                                    cfg.bg,
                                                                                color: cfg.color,
                                                                                border: `1px solid ${cfg.border}`,
                                                                            }}
                                                                        >
                                                                            <span>
                                                                                {
                                                                                    cfg.icon
                                                                                }
                                                                            </span>
                                                                            <span>
                                                                                {
                                                                                    cfg.label
                                                                                }
                                                                            </span>
                                                                        </span>
                                                                        {thought.time && (
                                                                            <span className="step-timestamp">
                                                                                {
                                                                                    thought.time
                                                                                }
                                                                            </span>
                                                                        )}
                                                                    </div>
                                                                    <div className="step-description">
                                                                        {
                                                                            thought.text
                                                                        }
                                                                    </div>
                                                                    {thought.toolDetails && (
                                                                        <div className="step-details-box">
                                                                            <strong>
                                                                                Tham
                                                                                số:
                                                                            </strong>{" "}
                                                                            {JSON.stringify(
                                                                                thought.toolDetails
                                                                            )}
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        )
                                                    }
                                                )}
                                            </div>
                                        )}
                                    </div>
                                )}

                            {/* Nội dung câu trả lời hoặc trạng thái đang chờ token đầu tiên */}
                            {message.content ? (
                                <div>{formatPlainText(message.content)}</div>
                            ) : message.isThinking ? (
                                <div className="thinking-typing-placeholder">
                                    <span>AI đang soạn thảo câu trả lời</span>
                                    <div className="typing-dot" />
                                    <div className="typing-dot" />
                                    <div className="typing-dot" />
                                </div>
                            ) : null}
                        </div>
                    </div>
                ))}

                <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-wrapper">
                <div className="chat-input-box">
                    <button
                        className="attach-button"
                        title="Attach document"
                    >
                        +
                    </button>

                    <textarea
                        value={question}
                        onChange={(event) =>
                            setQuestion(event.target.value)
                        }
                        onKeyDown={handleKeyDown}
                        placeholder="Ask about your documents..."
                        rows={1}
                    />

                    <button
                        className="send-button"
                        onClick={handleSend}
                        disabled={!question.trim() || loading}
                    >
                        {loading ? "..." : "↑"}
                    </button>
                </div>

                <p className="input-hint">
                    AI responses may contain mistakes. Check important
                    information against the source.
                </p>
            </div>
        </div>
    )
}

export default ChatBox