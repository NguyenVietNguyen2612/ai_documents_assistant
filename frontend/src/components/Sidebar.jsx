import { useState } from "react"
import DocumentUpload from "./DocumentUpload"
import DocumentList from "./DocumentList"

function Sidebar({
    sessions = [],
    currentSessionId = "default",
    onSelectSession,
    onCreateSession,
    onDeleteSession,
}) {
    const [refreshKey, setRefreshKey] = useState(0)

    const handleUploadSuccess = () => {
        setRefreshKey((prev) => prev + 1)
    }

    return (
        <aside className="sidebar">
            <div className="brand">
                <div className="brand-icon">AI</div>

                <div>
                    <h2>Doc Assistant</h2>
                    <span>RAG-powered</span>
                </div>
            </div>

            {/* Nút Tạo Cuộc Trò Chuyện Mới */}
            <button
                className="new-chat-button"
                onClick={onCreateSession}
                title="Tạo cuộc hội thoại mới"
            >
                <span className="new-chat-icon">+</span>
                <span>Cuộc trò chuyện mới</span>
            </button>

            {/* Danh sách các phiên trò chuyện */}
            <div className="sessions-section">
                <div className="section-header">
                    <h3>Cuộc hội thoại</h3>
                    <span className="session-count">{sessions.length}</span>
                </div>

                <div className="session-list">
                    {sessions.length === 0 ? (
                        <div className="no-sessions-hint">
                            Chưa có cuộc trò chuyện nào
                        </div>
                    ) : (
                        sessions.map((sess) => (
                            <div
                                key={sess.id}
                                className={`session-item ${sess.id === currentSessionId ? "active" : ""}`}
                                onClick={() => onSelectSession && onSelectSession(sess.id)}
                                title={sess.title}
                            >
                                <span className="session-item-title">{sess.title}</span>
                                <button
                                    className="session-delete-button"
                                    onClick={(e) => onDeleteSession && onDeleteSession(sess.id, e)}
                                    title="Xóa cuộc trò chuyện này"
                                >
                                    ✕
                                </button>
                            </div>
                        ))
                    )}
                </div>
            </div>

            <div className="sidebar-divider" />

            <DocumentUpload onUploadSuccess={handleUploadSuccess} />

            <div className="sidebar-divider" />

            <DocumentList refreshKey={refreshKey} />

            <div className="sidebar-footer">
                <span className="status-dot" />
                <span>System ready</span>
            </div>
        </aside>
    )
}

export default Sidebar