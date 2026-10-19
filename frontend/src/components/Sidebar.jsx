import {
    Sparkles,
    MessageSquarePlus,
    MessageSquare,
    Trash2,
    PanelLeftClose,
    Sun,
    Moon,
    Cpu,
} from "lucide-react"
import DocumentUpload from "./DocumentUpload"
import DocumentList from "./DocumentList"

function Sidebar({
    isOpen = true,
    onToggleSidebar,
    sessions = [],
    currentSessionId = "default",
    onSelectSession,
    onCreateSession,
    onDeleteSession,
    refreshDocsKey = 0,
    onUploadSuccess,
    onPreviewDocument,
    activePreviewDocId,
    theme = "light",
    onToggleTheme,
}) {
    return (
        <aside className={`sidebar ${isOpen ? "open" : "closed"}`} aria-label="Điều hướng và Quản lý Tài liệu">
            {/* Brand Header */}
            <div className="sidebar-top">
                <div className="brand">
                    <div className="brand-icon" aria-hidden="true">
                        <Sparkles size={18} strokeWidth={2.2} />
                    </div>
                    <div className="brand-text">
                        <h2>AI Doc Assistant</h2>
                        <span className="brand-badge">
                            <Cpu size={11} />
                            Hybrid RAG v2
                        </span>
                    </div>
                </div>

                <div className="sidebar-top-actions">
                    <button
                        type="button"
                        className="sidebar-icon-btn"
                        onClick={onToggleTheme}
                        title={theme === "light" ? "Chuyển sang giao diện Tối" : "Chuyển sang giao diện Sáng"}
                        aria-label="Chuyển đổi giao diện Sáng/Tối"
                    >
                        {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
                    </button>
                    <button
                        type="button"
                        className="sidebar-icon-btn"
                        onClick={onToggleSidebar}
                        title="Thu gọn thanh bên"
                        aria-label="Thu gọn thanh bên"
                    >
                        <PanelLeftClose size={16} />
                    </button>
                </div>
            </div>

            {/* Nút Tạo Cuộc Trò Chuyện Mới */}
            <button
                type="button"
                className="new-chat-button"
                onClick={onCreateSession}
                title="Bắt đầu cuộc hội thoại mới"
            >
                <MessageSquarePlus size={16} strokeWidth={2.2} />
                <span>Cuộc trò chuyện mới</span>
            </button>

            <div className="sidebar-scrollable">
                {/* Danh sách các phiên trò chuyện */}
                <div className="sessions-section">
                    <div className="section-header">
                        <h3>Lịch sử hội thoại</h3>
                        <span className="section-pill-count">{sessions.length}</span>
                    </div>

                    <div className="session-list" role="list">
                        {sessions.length === 0 ? (
                            <div className="no-sessions-hint">
                                Chưa có cuộc trò chuyện nào
                            </div>
                        ) : (
                            sessions.map((sess) => {
                                const isActive = sess.id === currentSessionId
                                return (
                                    <div
                                        key={sess.id}
                                        role="listitem"
                                        className={`session-item ${isActive ? "active" : ""}`}
                                        onClick={() => onSelectSession && onSelectSession(sess.id)}
                                        title={sess.title}
                                    >
                                        <MessageSquare
                                            size={14}
                                            className="session-item-icon"
                                            aria-hidden="true"
                                        />
                                        <span className="session-item-title">{sess.title}</span>
                                        <button
                                            type="button"
                                            className="session-delete-button"
                                            onClick={(e) => onDeleteSession && onDeleteSession(sess.id, e)}
                                            title="Xóa cuộc trò chuyện này"
                                            aria-label={`Xóa cuộc trò chuyện ${sess.title}`}
                                        >
                                            <Trash2 size={13} />
                                        </button>
                                    </div>
                                )
                            })
                        )}
                    </div>
                </div>

                <div className="sidebar-divider" />

                {/* Khu vực tải tài liệu */}
                <DocumentUpload onUploadSuccess={onUploadSuccess} />

                <div className="sidebar-divider" />

                {/* Danh sách tài liệu trong Kho tri thức */}
                <DocumentList
                    refreshKey={refreshDocsKey}
                    onPreviewDocument={onPreviewDocument}
                    activePreviewDocId={activePreviewDocId}
                />
            </div>

            {/* Footer trạng thái hệ thống */}
            <div className="sidebar-footer">
                <div className="sidebar-status-group">
                    <span className="status-dot" aria-hidden="true" />
                    <span className="status-text">Milvus + BGE Reranker Ready</span>
                </div>
                <span className="status-version">Laya S1</span>
            </div>
        </aside>
    )
}

export default Sidebar