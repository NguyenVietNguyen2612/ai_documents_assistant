import { useState, useRef, useEffect } from "react"
import {
    Sparkles,
    Bot,
    User,
    Send,
    Paperclip,
    RotateCcw,
    ChevronDown,
    BookOpen,
    FileText,
    ArrowUpRight,
    Clock,
    Zap,
    GitBranch,
    Copy,
    Check,
    PanelLeftOpen,
    X,
    Search,
    Layers,
    Calculator,
    ListFilter,
    Loader2,
} from "lucide-react"
import { sendMessageStream, getSessionMessages } from "../services/chatAPI"
import MarkdownViewer from "./MarkdownViewer"

function formatPlainText(text) {
    if (!text) return ""
    return text.replace(/^[\*\•]\s+/gm, "- ")
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

const STARTER_PROMPTS = [
    {
        icon: <Layers size={16} />,
        title: "Tổng quan kho tài liệu",
        prompt: "Hãy liệt kê các tài liệu hiện có trong hệ thống và tóm tắt nội dung chính của từng tài liệu.",
    },
    {
        icon: <Search size={16} />,
        title: "Truy vấn chuyên sâu & Trích dẫn",
        prompt: "Vấn đề cốt lõi và phương pháp đề xuất trong tài liệu nghiên cứu là gì? Trích dẫn rõ số trang.",
    },
    {
        icon: <ListFilter size={16} />,
        title: "Phân tích bảng biểu & Số liệu",
        prompt: "Tổng hợp các chỉ số thực nghiệm hoặc số liệu quan trọng nhất dưới dạng bảng Markdown.",
    },
    {
        icon: <Calculator size={16} />,
        title: "Tính toán & Đối chiếu",
        prompt: "So sánh hiệu năng giữa các mô hình/phương pháp trong tài liệu và tính mức chênh lệch phần trăm.",
    },
]

function ChatBox({
    sessionId = "default",
    sessionTitle = "Cuộc trò chuyện",
    onRefreshSessions,
    onOpenCitation,
    onUploadSuccess,
    sidebarOpen = true,
    onToggleSidebar,
}) {
    const [question, setQuestion] = useState("")
    const [messages, setMessages] = useState([])
    const [loading, setLoading] = useState(false)
    const [loadingHistory, setLoadingHistory] = useState(false)
    const [rollbackState, setRollbackState] = useState(null)
    const [copiedMsgId, setCopiedMsgId] = useState(null)
    const [quickUploading, setQuickUploading] = useState(false)

    const messagesEndRef = useRef(null)
    const textareaRef = useRef(null)
    const quickFileInputRef = useRef(null)

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
                    const thinkingText = reasoningItem ? reasoningItem.content || reasoningItem.text || "" : ""
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

    const handleCitationTrigger = (citationInfo) => {
        if (!citationInfo || !onOpenCitation) return
        onOpenCitation({
            documentName: citationInfo.documentName || "Tài liệu",
            documentId: citationInfo.documentId || null,
            page: citationInfo.page || 1,
            snippet: citationInfo.snippet || "",
        })
    }

    const toggleThoughts = (id) => {
        setMessages((previous) =>
            previous.map((msg) => (msg.id === id ? { ...msg, showThoughts: !msg.showThoughts } : msg))
        )
    }

    const handleCopyMessage = (msg) => {
        if (!msg.content) return
        navigator.clipboard?.writeText(msg.content)
        setCopiedMsgId(msg.id)
        setTimeout(() => setCopiedMsgId(null), 1800)
    }

    // Nút đính kèm nhanh tài liệu ngay tại khung nhập chat
    const handleQuickFileUpload = async (event) => {
        const selectedFile = event.target.files?.[0]
        event.target.value = ""
        if (!selectedFile) return

        setQuickUploading(true)
        const formData = new FormData()
        formData.append("file", selectedFile)

        try {
            const res = await fetch("http://localhost:8001/documents/upload", {
                method: "POST",
                body: formData,
            })
            if (res.ok) {
                if (onUploadSuccess) onUploadSuccess()
                setQuestion((prev) =>
                    prev
                        ? `${prev} (Tài liệu: ${selectedFile.name})`
                        : `Hãy tóm tắt nội dung chính của tài liệu ${selectedFile.name} vừa tải lên.`
                )
                textareaRef.current?.focus()
            }
        } catch (err) {
            console.error("Quick upload error:", err)
        } finally {
            setQuickUploading(false)
        }
    }

    // Bắt đầu quy trình Rollback & Chỉnh sửa câu hỏi
    const handleStartRollback = (userMsg) => {
        const msgIndex = messages.findIndex((m) => m.id === userMsg.id)
        if (msgIndex === -1) return

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

        setQuestion(userMsg.content)
        textareaRef.current?.focus()
    }

    const cancelRollback = () => {
        setRollbackState(null)
        setQuestion("")
    }

    const executeSendQuestion = async (overrideText) => {
        const textToSend = typeof overrideText === "string" ? overrideText.trim() : question.trim()
        if (!textToSend || loading) {
            return
        }

        const startTimestamp = performance.now()
        const userMessageId = Date.now()
        const assistantMessageId = userMessageId + 1

        const isRollingBack = Boolean(rollbackState)
        const targetMessageId = rollbackState?.targetMessageId || null
        const targetCheckpointId = rollbackState?.checkpointId || null

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
            content: textToSend,
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
            latencyMs: null,
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
            question: textToSend,
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
                                      content: msg.content || `Xin lỗi, đã xảy ra sự cố: ${event.content}`,
                                      isThinking: false,
                                  }
                                : msg
                        )
                    )
                } else if (event.type === "done") {
                    const elapsed = Math.round(performance.now() - startTimestamp)
                    setMessages((previous) =>
                        previous.map((msg) =>
                            msg.id === assistantMessageId
                                ? {
                                      ...msg,
                                      isThinking: false,
                                      showThoughts: false,
                                      latencyMs: elapsed,
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
            },
        })
    }

    const handleSend = () => {
        executeSendQuestion()
    }

    const handleKeyDown = (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault()
            handleSend()
        }
    }

    return (
        <div className="chat-container">
            {/* Topbar điều khiển Chat */}
            <header className="chat-topbar">
                <div className="chat-topbar-left">
                    {!sidebarOpen && (
                        <button
                            type="button"
                            className="topbar-toggle-sidebar-btn"
                            onClick={onToggleSidebar}
                            title="Mở thanh bên"
                            aria-label="Mở thanh bên"
                        >
                            <PanelLeftOpen size={18} />
                        </button>
                    )}
                    <div className="chat-topbar-info">
                        <h2>{sessionTitle}</h2>
                        <span className="chat-topbar-sub">
                            Phiên: <code>{sessionId.slice(0, 18)}</code>
                        </span>
                    </div>
                </div>

                <div className="chat-topbar-badges">
                    <span className="pipeline-pill">
                        <Zap size={12} />
                        <span>Laya S1 + LangGraph ReAct</span>
                    </span>
                </div>
            </header>

            {/* Khung nội dung hội thoại */}
            <div className="messages">
                {loadingHistory ? (
                    <div className="loading-history-indicator">
                        <div className="typing-dots-row">
                            <div className="typing-dot" />
                            <div className="typing-dot" />
                            <div className="typing-dot" />
                        </div>
                        <span>Đang nạp lịch sử hội thoại...</span>
                    </div>
                ) : messages.length === 0 ? (
                    <div className="empty-chat">
                        <div className="empty-chat-badge">
                            <Sparkles size={14} />
                            <span>AI Document Intelligence Workspace</span>
                        </div>
                        <h2>Khám phá & Đối chiếu Tài liệu Chuyên sâu</h2>
                        <p>
                            Hệ thống kết hợp tìm kiếm <strong>Hybrid (Milvus Vector + BM25 RRF)</strong>, chấm điểm lại với{" "}
                            <strong>BGE Cross-Encoder Reranker</strong> và trích dẫn nguồn tương tác trực tiếp đến từng trang.
                        </p>

                        <div className="starter-prompts-grid">
                            {STARTER_PROMPTS.map((item, idx) => (
                                <button
                                    key={idx}
                                    type="button"
                                    className="starter-prompt-card"
                                    onClick={() => executeSendQuestion(item.prompt)}
                                >
                                    <div className="starter-prompt-icon" aria-hidden="true">
                                        {item.icon}
                                    </div>
                                    <div className="starter-prompt-body">
                                        <span className="starter-prompt-title">{item.title}</span>
                                        <span className="starter-prompt-desc">{item.prompt}</span>
                                    </div>
                                    <ArrowUpRight size={15} className="starter-prompt-arrow" />
                                </button>
                            ))}
                        </div>
                    </div>
                ) : null}

                {messages.map((message) => {
                    const systemThoughts = (message.thoughts || []).filter(
                        (t) => t.step !== "reasoning" && t.step !== "sources" && t.type !== "sources"
                    )
                    const topSourceScore =
                        message.sources && message.sources.length > 0
                            ? Math.max(...message.sources.map((s) => Number(s.score || 0)))
                            : null

                    return (
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
                                aria-hidden="true"
                            >
                                {message.role === "user" ? <User size={15} /> : <Bot size={16} />}
                            </div>

                            <div className="message-content">
                                {/* User Message Bubble & Rollback Action */}
                                {message.role === "user" && (
                                    <div className="user-message-header">
                                        <div className="user-message-bubble">{message.content}</div>
                                        <button
                                            type="button"
                                            className="rollback-msg-btn"
                                            onClick={() => handleStartRollback(message)}
                                            title="Quay lui thời gian (Time-Travel) về câu hỏi này và rẽ nhánh mới"
                                        >
                                            <RotateCcw size={12} />
                                            <span>Quay về & sửa</span>
                                        </button>
                                    </div>
                                )}

                                {/* Khối lịch sử suy luận (Chain-of-Thought Accordion) */}
                                {message.role === "assistant" &&
                                    (systemThoughts.length > 0 ||
                                        message.thinkingContent ||
                                        message.isThinking) && (
                                        <div
                                            className={`thinking-box ${
                                                message.showThoughts ? "expanded" : ""
                                            }`}
                                        >
                                            <button
                                                type="button"
                                                className="thinking-header"
                                                onClick={() => toggleThoughts(message.id)}
                                            >
                                                <div className="thinking-title">
                                                    <span
                                                        className={`thinking-status-indicator ${
                                                            message.isThinking ? "active" : "done"
                                                        }`}
                                                    />
                                                    <span className="thinking-header-text">
                                                        {message.isThinking
                                                            ? "Agent đang suy luận & truy xuất..."
                                                            : "Tiến trình Suy luận & Truy xuất (Chain-of-Thought)"}
                                                    </span>
                                                    <span className="thinking-badge-count">
                                                        {systemThoughts.length} bước
                                                    </span>
                                                    {message.checkpoint_id && (
                                                        <span
                                                            className="checkpoint-pill"
                                                            title={`LangGraph Checkpoint ID: ${message.checkpoint_id}`}
                                                        >
                                                            <GitBranch size={10} />
                                                            {message.checkpoint_id.slice(0, 8)}
                                                        </span>
                                                    )}
                                                </div>
                                                <ChevronDown
                                                    size={15}
                                                    className={`thinking-arrow ${
                                                        message.showThoughts ? "open" : ""
                                                    }`}
                                                />
                                            </button>

                                            {message.showThoughts && (
                                                <div className="thinking-content">
                                                    {/* DeepSeek-style Reasoning Stream */}
                                                    {(message.thinkingContent ||
                                                        (message.isThinking && !message.content)) && (
                                                        <div className="deepseek-reasoning-section">
                                                            <div className="reasoning-section-header">
                                                                <span className="reasoning-section-title">
                                                                    Luồng tư duy nội tại (Internal Monologue)
                                                                </span>
                                                                {message.isThinking && !message.content && (
                                                                    <span className="reasoning-live-badge">
                                                                        Đang suy nghĩ...
                                                                    </span>
                                                                )}
                                                            </div>
                                                            <div className="reasoning-section-body">
                                                                {message.thinkingContent ? (
                                                                    <div className="reasoning-text-stream">
                                                                        {formatPlainText(
                                                                            message.thinkingContent
                                                                        )}
                                                                        {message.isThinking &&
                                                                            !message.content && (
                                                                                <span className="reasoning-cursor" />
                                                                            )}
                                                                    </div>
                                                                ) : (
                                                                    <div className="thinking-typing-placeholder">
                                                                        <span>
                                                                            Mô hình đang phân tích ngữ cảnh
                                                                        </span>
                                                                        <div className="typing-dots-row">
                                                                            <div className="typing-dot" />
                                                                            <div className="typing-dot" />
                                                                            <div className="typing-dot" />
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                    )}

                                                    {/* Các bước Pipeline của Agent */}
                                                    {systemThoughts.map((thought, idx) => {
                                                        const cfg = STEP_CONFIG[thought.step] || {
                                                            label: "Bước xử lý",
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
                                                                            {cfg.label}
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
                                                                            <strong>Tham số gọi Tool:</strong>{" "}
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

                                {/* Nội dung câu trả lời của Assistant */}
                                {message.role === "assistant" &&
                                    (message.content ? (
                                        <div className="assistant-response-card">
                                            <div className="assistant-text-body">
                                                <MarkdownViewer
                                                    content={message.content}
                                                    onCitationClick={handleCitationTrigger}
                                                />
                                            </div>

                                            {/* Khối danh sách nguồn trích dẫn tương tác (Sources Grid) */}
                                            {message.sources && message.sources.length > 0 && (
                                                <div className="sources-container">
                                                    <div className="sources-header">
                                                        <span className="sources-title">
                                                            <BookOpen size={13} />
                                                            <span>
                                                                Nguồn trích dẫn đối chiếu ({message.sources.length})
                                                            </span>
                                                        </span>
                                                        <span className="sources-hint">
                                                            Nhấp để mở khung đối chiếu bên phải
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
                                                                <button
                                                                    type="button"
                                                                    key={sIdx}
                                                                    className="source-card"
                                                                    onClick={() =>
                                                                        handleCitationTrigger({
                                                                            documentName: src.document_name,
                                                                            documentId: src.document_id,
                                                                            page: src.page,
                                                                            snippet: src.snippet,
                                                                        })
                                                                    }
                                                                    title={`Nhấp để đối chiếu trang ${src.page} của ${src.document_name}`}
                                                                >
                                                                    <div className="source-card-top">
                                                                        <span className="source-card-doc">
                                                                            <FileText size={13} />
                                                                            <span>{src.document_name}</span>
                                                                        </span>
                                                                        <span className="source-card-page">
                                                                            Trang {src.page}
                                                                        </span>
                                                                    </div>
                                                                    <div className="source-card-meta">
                                                                        <span
                                                                            className={`source-conf-badge ${confClass}`}
                                                                        >
                                                                            Độ khớp {src.level} (
                                                                            {(src.score * 100).toFixed(0)}%)
                                                                        </span>
                                                                        <span className="source-open-arrow">
                                                                            <span>Đối chiếu</span>
                                                                            <ArrowUpRight size={12} />
                                                                        </span>
                                                                    </div>
                                                                </button>
                                                            )
                                                        })}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Thanh Telemetry & Công cụ dưới mỗi câu trả lời */}
                                            <div className="assistant-telemetry-bar">
                                                <div className="telemetry-metrics">
                                                    {message.latencyMs && (
                                                        <span className="telemetry-chip" title="Tổng thời gian phản hồi">
                                                            <Clock size={11} />
                                                            <span>{(message.latencyMs / 1000).toFixed(2)}s</span>
                                                        </span>
                                                    )}
                                                    {topSourceScore !== null && (
                                                        <span
                                                            className="telemetry-chip"
                                                            title="Điểm Cross-Encoder Reranker cao nhất"
                                                        >
                                                            <Zap size={11} />
                                                            <span>
                                                                Rerank Top: {(topSourceScore * 100).toFixed(0)}%
                                                            </span>
                                                        </span>
                                                    )}
                                                    {message.sources && message.sources.length > 0 && (
                                                        <span className="telemetry-chip">
                                                            <BookOpen size={11} />
                                                            <span>{message.sources.length} nguồn</span>
                                                        </span>
                                                    )}
                                                </div>

                                                <div className="telemetry-actions">
                                                    <button
                                                        type="button"
                                                        className="telemetry-action-btn"
                                                        onClick={() => handleCopyMessage(message)}
                                                        title="Sao chép câu trả lời"
                                                    >
                                                        {copiedMsgId === message.id ? (
                                                            <>
                                                                <Check size={12} />
                                                                <span>Đã chép</span>
                                                            </>
                                                        ) : (
                                                            <>
                                                                <Copy size={12} />
                                                                <span>Sao chép</span>
                                                            </>
                                                        )}
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    ) : message.isThinking ? (
                                        <div className="thinking-typing-placeholder">
                                            <span>AI đang tổng hợp câu trả lời</span>
                                            <div className="typing-dots-row">
                                                <div className="typing-dot" />
                                                <div className="typing-dot" />
                                                <div className="typing-dot" />
                                            </div>
                                        </div>
                                    ) : null)}
                            </div>
                        </div>
                    )
                })}

                <div ref={messagesEndRef} />
            </div>

            {/* Rollback Notification Banner */}
            {rollbackState && (
                <div className="rollback-banner" role="status">
                    <div className="rollback-banner-info">
                        <RotateCcw size={14} />
                        <span>
                            Đang quay lui về câu hỏi: <strong>"{rollbackState.originalText}"</strong>. Chỉnh sửa bên dưới để tạo nhánh hội thoại mới.
                        </span>
                    </div>
                    <button type="button" className="rollback-banner-cancel" onClick={cancelRollback}>
                        <X size={13} />
                        <span>Hủy</span>
                    </button>
                </div>
            )}

            {/* Ô nhập tin nhắn */}
            <div className="chat-input-wrapper">
                <div className="chat-input-box">
                    <input
                        ref={quickFileInputRef}
                        type="file"
                        accept=".pdf,.csv,.txt,.md"
                        onChange={handleQuickFileUpload}
                        hidden
                    />
                    <button
                        type="button"
                        className="attach-button"
                        onClick={() => quickFileInputRef.current?.click()}
                        disabled={quickUploading || loading}
                        title="Tải nhanh tài liệu (.pdf, .csv, .md, .txt) vào cuộc trò chuyện"
                        aria-label="Đính kèm tài liệu"
                    >
                        {quickUploading ? (
                            <Loader2 size={17} className="spin-icon" />
                        ) : (
                            <Paperclip size={17} />
                        )}
                    </button>

                    <textarea
                        ref={textareaRef}
                        value={question}
                        onChange={(event) => setQuestion(event.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={
                            rollbackState
                                ? "Chỉnh sửa câu hỏi để rẽ nhánh hội thoại mới..."
                                : "Đặt câu hỏi về tài liệu của bạn (Nhấn Enter để gửi, Shift+Enter xuống dòng)..."
                        }
                        rows={1}
                    />

                    <button
                        type="button"
                        className="send-button"
                        onClick={handleSend}
                        disabled={!question.trim() || loading}
                        title={rollbackState ? "Gửi và rẽ nhánh" : "Gửi câu hỏi"}
                        aria-label="Gửi câu hỏi"
                    >
                        <Send size={15} />
                    </button>
                </div>

                <p className="input-hint">
                    Hệ thống tự động trích dẫn nguồn từ tài liệu gốc. Nhấp vào nhãn trích dẫn để đối chiếu trực tiếp.
                </p>
            </div>
        </div>
    )
}

export default ChatBox