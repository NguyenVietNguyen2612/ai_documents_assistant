import { useState, useEffect } from "react"
import MarkdownViewer from "./MarkdownViewer"

function parseCsvToMarkdown(csvText) {
    if (!csvText) return ""
    const lines = csvText
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter(Boolean)
    if (lines.length === 0) return ""

    const parseRow = (row) => {
        const result = []
        let current = ""
        let inQuotes = false
        for (let i = 0; i < row.length; i++) {
            const ch = row[i]
            if (ch === '"') {
                inQuotes = !inQuotes
            } else if (ch === "," && !inQuotes) {
                result.push(current.trim())
                current = ""
            } else {
                current += ch
            }
        }
        result.push(current.trim())
        return result
    }

    const headers = parseRow(lines[0])
    const headerLine = "| # | " + headers.join(" | ") + " |"
    const sepLine = "| --- | " + headers.map(() => "---").join(" | ") + " |"

    const bodyLines = lines.slice(1).map((line, idx) => {
        const cols = parseRow(line)
        while (cols.length < headers.length) cols.push("")
        const cleanCols = cols.slice(0, headers.length).map((c) => c.replace(/\|/g, "\\|"))
        return `| ${idx + 1} | ${cleanCols.join(" | ")} |`
    })

    return [headerLine, sepLine, ...bodyLines].join("\n")
}

function PDFModal({ isOpen, onClose, documentName, documentId: initialDocId, page = 1, snippet = "" }) {
    const [resolvedDocId, setResolvedDocId] = useState(initialDocId || null)
    const [resolvedDocName, setResolvedDocName] = useState(documentName || "")
    const [showSnippet, setShowSnippet] = useState(true)
    const [textContent, setTextContent] = useState("")
    const [loadingText, setLoadingText] = useState(false)

    // Tự động đồng bộ documentId và tên file đầy đủ (kèm đuôi mở rộng) từ danh sách documents
    useEffect(() => {
        if (!isOpen) return
        setResolvedDocId(initialDocId || null)
        setResolvedDocName(documentName || "")
        setTextContent("")

        fetch("http://localhost:8001/documents")
            .then((res) => res.json())
            .then((docs) => {
                if (!Array.isArray(docs)) return
                let found = null
                if (initialDocId) {
                    found = docs.find((d) => d.id === initialDocId)
                }
                if (!found && documentName) {
                    const cleanTarget = documentName.toLowerCase().replace(/\.(pdf|csv|txt|md)$/i, "").trim()
                    found = docs.find((d) => {
                        const dName = (d.name || "").toLowerCase().replace(/\.(pdf|csv|txt|md)$/i, "").trim()
                        return dName === cleanTarget || dName.includes(cleanTarget) || cleanTarget.includes(dName)
                    })
                }
                if (found) {
                    setResolvedDocId(found.id)
                    if (found.name) setResolvedDocName(found.name)
                }
            })
            .catch((err) => console.error("Error resolving doc id:", err))
    }, [isOpen, documentName, initialDocId])

    const activeName = (resolvedDocName || documentName || "").toLowerCase()
    const isPdf = activeName.endsWith(".pdf") || (!activeName.endsWith(".csv") && !activeName.endsWith(".md") && !activeName.endsWith(".txt"))
    const isCsv = activeName.endsWith(".csv")

    // Nếu là file văn bản (.md, .csv, .txt), tải nội dung text để hiển thị trực tiếp trong Modal (không dùng iframe gây tải file)
    useEffect(() => {
        if (!isOpen || !resolvedDocId || isPdf) return
        let isMounted = true
        setLoadingText(true)

        fetch(`http://localhost:8001/documents/${resolvedDocId}/file`)
            .then((res) => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`)
                return res.text()
            })
            .then((txt) => {
                if (!isMounted) return
                setTextContent(isCsv ? parseCsvToMarkdown(txt) : txt)
            })
            .catch((err) => {
                console.error("Failed to load text document preview:", err)
                if (isMounted) setTextContent("Không thể tải nội dung xem trước của tài liệu này.")
            })
            .finally(() => {
                if (isMounted) setLoadingText(false)
            })

        return () => {
            isMounted = false
        }
    }, [isOpen, resolvedDocId, isPdf, isCsv])

    if (!isOpen) return null

    const targetPage = page || 1
    const fileUrl = resolvedDocId
        ? isPdf
            ? `http://localhost:8001/documents/${resolvedDocId}/file#page=${targetPage}&view=FitH`
            : `http://localhost:8001/documents/${resolvedDocId}/file`
        : null

    const handleOpenInNewTab = () => {
        if (fileUrl) {
            window.open(fileUrl, "_blank", "noopener,noreferrer")
        }
    }

    const handleDownloadFile = () => {
        if (resolvedDocId) {
            window.open(`http://localhost:8001/documents/${resolvedDocId}/file?download=true`, "_blank", "noopener,noreferrer")
        }
    }

    return (
        <div className="pdf-modal-overlay" onClick={onClose}>
            <div className="pdf-modal-container" onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div className="pdf-modal-header">
                    <div className="pdf-modal-title-group">
                        <span className="pdf-modal-icon">{isCsv ? "📊" : isPdf ? "📄" : "📝"}</span>
                        <div>
                            <h3 className="pdf-modal-filename">{resolvedDocName || documentName || "Tài liệu"}</h3>
                            <div className="pdf-modal-meta">
                                <span className="pdf-page-pill">Đang mở: Trang {targetPage}</span>
                                {resolvedDocId && <span className="pdf-doc-id-pill">ID: {resolvedDocId.slice(0, 8)}...</span>}
                            </div>
                        </div>
                    </div>

                    <div className="pdf-modal-actions">
                        {fileUrl && (
                            <>
                                <button
                                    className="pdf-modal-action-btn"
                                    onClick={handleOpenInNewTab}
                                    title="Mở xem trực tiếp trong tab mới"
                                >
                                    ↗ Mở tab mới
                                </button>
                                <button
                                    className="pdf-modal-action-btn"
                                    onClick={handleDownloadFile}
                                    title="Tải file gốc về máy"
                                >
                                    ⬇ Tải về
                                </button>
                            </>
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

                {/* Document Viewer Body */}
                <div
                    className="pdf-modal-viewer-body"
                    style={!isPdf ? { background: "#ffffff", overflowY: "auto", padding: "24px 32px" } : undefined}
                >
                    {!resolvedDocId || loadingText ? (
                        <div className="pdf-modal-loading-placeholder" style={!isPdf ? { color: "#475569" } : undefined}>
                            <div className="typing-dot" />
                            <div className="typing-dot" />
                            <div className="typing-dot" />
                            <p>Đang tải nội dung tài liệu...</p>
                        </div>
                    ) : isPdf ? (
                        <iframe
                            src={fileUrl}
                            title={`Document Viewer - ${resolvedDocName || documentName}`}
                            className="pdf-iframe"
                        />
                    ) : (
                        <div className="document-text-preview" style={{ color: "#1e293b", lineHeight: 1.65 }}>
                            <MarkdownViewer content={textContent} />
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default PDFModal
