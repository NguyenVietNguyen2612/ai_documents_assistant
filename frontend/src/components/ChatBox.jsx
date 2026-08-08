import { useState } from "react"

function ChatBox() {
    const [question, setQuestion] = useState("")

    const [messages, setMessages] = useState([])

    const handleSend = () => {
        if (!question.trim()) {
            return
        }

        const newMessage = {
            id: Date.now(),
            role: "user",
            content: question,
        }

        setMessages((previous) => [
            ...previous,
            newMessage,
        ])

        console.log("Question:", question)

        setQuestion("")
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
                        className="message user-message"
                        key={message.id}
                    >
                        <div className="avatar user-avatar">
                            U
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
                        disabled={!question.trim()}
                    >
                        ↑
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