import { useState } from "react"
import { sendMessage } from "../services/chatAPI"

function ChatBox() {
    const [question, setQuestion] = useState("")

    const [messages, setMessages] = useState([])

    const [loading, setLoading] = useState(false)

    const handleSend = async () => {
        if (!question.trim() || loading) {
            return
        }

        const currentQuestion = question.trim()

        const newMessage = {
            id: Date.now(),
            role: "user",
            content: currentQuestion,
        }

        setMessages((previous) => [
            ...previous,
            newMessage,
        ])

        setQuestion("")
        setLoading(true)

        try {
            const data = await sendMessage(
                currentQuestion
            )

            const assistantMessage = {
                id: Date.now() + 1,
                role: "assistant",
                content: data.answer,
            }

            setMessages((previous) => [
                ...previous,
                assistantMessage,
            ])

        } catch (error) {
            console.error(
                "Failed to send message:",
                error
            )

            const errorMessage = {
                id: Date.now() + 1,
                role: "assistant",
                content:
                    "Sorry, I couldn't process your question.",
            }

            setMessages((previous) => [
                ...previous,
                errorMessage,
            ])

        } finally {
            setLoading(false)
        }
    }

    const handleKeyDown = (event) => {
        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault()
            handleSend()
        }
    }

    return (
        <div className="chat-container">

            <div className="messages">

                {messages.length === 0 && (
                    <div className="empty-chat">

                        <div className="empty-chat-icon">
                            ✦
                        </div>

                        <h2>
                            Ask anything about your documents
                        </h2>

                        <p>
                            Upload your documents and ask questions.
                            The AI will search your knowledge base
                            to find relevant information.
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
                            {message.role === "user"
                                ? "U"
                                : "AI"}
                        </div>

                        <div className="message-content">
                            {message.content}
                        </div>
                    </div>
                ))}

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
                        disabled={
                            !question.trim() ||
                            loading
                        }
                    >
                        {loading ? "..." : "↑"}
                    </button>

                </div>

                <p className="input-hint">
                    AI responses may contain mistakes. Check
                    important information against the source.
                </p>

            </div>

        </div>
    )
}

export default ChatBox