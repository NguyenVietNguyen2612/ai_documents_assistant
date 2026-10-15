import { useState, useEffect } from "react"

function PDFModal({ isOpen, onClose, documentName, documentId: initialDocId, page = 1, snippet = "" }) {
    const [resolvedDocId, setResolvedDocId] = useState(initialDocId)
    const [showSnippet, setShowSnippet] = useState(true)

    // Nếu chưa có documentId, tự động tìm kiếm id từ danh sách documents
    useEffect(() => {
        if (!isOpen) return
        if (initialDocId) {
            setResolvedDocId(initialDocId)
            return
        }

        if (documentName) {
            fetch("http://localhost:8001/documents")
                .then((res) => res.json())
                .then((docs) => {
                    if (Array.isArray(docs)) {
                        const cleanTarget = documentName.toLowerCase().replace(/\.pdf$/i, "").trim()
                        const found = docs.find((d) => {
                            const dName = (d.name || "").toLowerCase().replace(/\.pdf$/i, "").trim()
                            return dName === cleanTarget || dName.includes(cleanTarget) || cleanTarget.includes(dName)
                        })
                        if (found) {
                            setResolvedDocId(found.id)
                        }
                    }
                })
                .catch((err) => console.error("Error resolving doc id:", err))
        }
    }, [isOpen, documentName, initialDocId])

    if (!isOpen) return null

    const targetPage = page || 1
    const fileUrl = resolvedDocId
        ? `http://localhost:8001/documents/${resolvedDocId}/file#page=${targetPage}&view=FitH`
        : null

    const handleOpenInNewTab = () => {
        if (fileUrl) {
            window.open(fileUrl, "_blank", "noopener,noreferrer")
        }
    }

    return (
        <div className="pdf-modal-overlay" onClick={onClose}>
            <div className="pdf-modal-container" onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div className="pdf-modal-header">
                    <div className="pdf-modal-title-group">
                        <span className="pdf-modal-icon">📄</span>
                        <div>
                            <h3 className="pdf-modal-filename">{documentName || "Tài liệu PDF"}</h3>
                            <div className="pdf-modal-meta">
                                <span className="pdf-page-pill">Đang mở: Trang {targetPage}</span>
                                {resolvedDocId && <span className="pdf-doc-id-pill">ID: {resolvedDocId.slice(0, 8)}...</span>}
                            </div>
                        </div>
                    </div>

                    <div className="pdf-modal-actions">
                        {fileUrl && (
                            <button
                                className="pdf-modal-action-btn"
                                onClick={handleOpenInNewTab}
                                title="Mở file trong tab mới"
                            >
                                ↗ Mở tab mới
                            </button>
                        )}
                        <button
                            className="pdf-modal-close-btn"
                            onClick={onClose}
                            title="Đóng (Esc)"
                        >
                            ✕
                        </button>
                    </div>
                </div>

                {/* Đoạn trích dẫn đối chiếu (Snippet Banner) */}
                {snippet && (
                    <div className="pdf-modal-snippet-banner">
                        <div className="snippet-banner-header" onClick={() => setShowSnippet(!showSnippet)}>
                            <span className="snippet-label">
                                💡 Đoạn trích dẫn liên quan được trích xuất từ <strong>Trang {targetPage}</strong>:
                            </span>
                            <span className="snippet-toggle-btn">{showSnippet ? "Ẩn ▲" : "Hiện ▼"}</span>
                        </div>
                        {showSnippet && (
                            <div className="snippet-banner-content">
                                "{snippet}"
                            </div>
                        )}
                    </div>
                )}

                {/* PDF Viewer Frame */}
                <div className="pdf-modal-viewer-body">
                    {resolvedDocId ? (
                        <iframe
                            src={fileUrl}
                            title={`PDF Viewer - ${documentName}`}
                            className="pdf-iframe"
                        />
                    ) : (
                        <div className="pdf-modal-loading-placeholder">
                            <div className="typing-dot" />
                            <div className="typing-dot" />
                            <div className="typing-dot" />
                            <p>Đang định vị tài liệu trên máy chủ...</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default PDFModal
