import { useState, useEffect } from "react"
import {
    FileText,
    FileSpreadsheet,
    FileCode,
    ExternalLink,
    Download,
    Maximize2,
    Minimize2,
    X,
    Quote,
    ChevronUp,
    ChevronDown,
} from "lucide-react"
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

function PDFModal({
    isOpen,
    mode = "split",
    onToggleMode,
    onClose,
    documentName,
    documentId: initialDocId,
    page = 1,
    snippet = "",
}) {
    const [resolvedDocId, setResolvedDocId] = useState(initialDocId || null)
    const [resolvedDocName, setResolvedDocName] = useState(documentName || "")
    const [showSnippet, setShowSnippet] = useState(true)
    const [textContent, setTextContent] = useState("")
    const [loadingText, setLoadingText] = useState(false)

    // Đồng bộ documentId và tên file đầy đủ từ danh sách documents
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
    const isPdf =
        activeName.endsWith(".pdf") ||
        (!activeName.endsWith(".csv") && !activeName.endsWith(".md") && !activeName.endsWith(".txt"))
    const isCsv = activeName.endsWith(".csv")

    // Nếu là file văn bản (.md, .csv, .txt), tải nội dung text để hiển thị trực tiếp
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
            window.open(
                `http://localhost:8001/documents/${resolvedDocId}/file?download=true`,
                "_blank",
                "noopener,noreferrer"
            )
        }
    }

    const renderDocTypeIcon = () => {
        if (isCsv) return <FileSpreadsheet size={18} />
        if (isPdf) return <FileText size={18} />
        return <FileCode size={18} />
    }

    const viewerContent = (
        <div
            className={`doc-inspector-container ${mode === "split" ? "split-mode" : "modal-mode"}`}
            onClick={(e) => e.stopPropagation()}
        >
            {/* Header */}
            <div className="pdf-modal-header">
                <div className="pdf-modal-title-group">
                    <span className={`pdf-modal-icon ${isCsv ? "csv" : isPdf ? "pdf" : "md"}`} aria-hidden="true">
                        {renderDocTypeIcon()}
                    </span>
                    <div className="pdf-modal-title-text">
                        <h3 className="pdf-modal-filename" title={resolvedDocName || documentName || "Tài liệu"}>
                            {resolvedDocName || documentName || "Tài liệu"}
                        </h3>
                        <div className="pdf-modal-meta">
                            <span className="pdf-page-pill">Trang {targetPage}</span>
                            {resolvedDocId && <span className="pdf-doc-id-pill">ID: {resolvedDocId.slice(0, 8)}</span>}
                        </div>
                    </div>
                </div>

                <div className="pdf-modal-actions">
                    {fileUrl && (
                        <>
                            <button
                                type="button"
                                className="pdf-modal-action-btn"
                                onClick={handleOpenInNewTab}
                                title="Mở xem trong tab mới"
                            >
                                <ExternalLink size={13} />
                                <span>Tab mới</span>
                            </button>
                            <button
                                type="button"
                                className="pdf-modal-action-btn"
                                onClick={handleDownloadFile}
                                title="Tải file gốc về máy"
                            >
                                <Download size={13} />
                                <span>Tải về</span>
                            </button>
                        </>
                    )}
                    {onToggleMode && (
                        <button
                            type="button"
                            className="pdf-modal-icon-btn"
                            onClick={onToggleMode}
                            title={
                                mode === "split"
                                    ? "Phóng to toàn màn hình (Modal)"
                                    : "Thu về khung đối chiếu song song (Split View)"
                            }
                            aria-label="Chuyển đổi chế độ xem tài liệu"
                        >
                            {mode === "split" ? <Maximize2 size={15} /> : <Minimize2 size={15} />}
                        </button>
                    )}
                    <button
                        type="button"
                        className="pdf-modal-close-btn"
                        onClick={onClose}
                        title="Đóng trình xem tài liệu"
                        aria-label="Đóng trình xem tài liệu"
                    >
                        <X size={16} />
                    </button>
                </div>
            </div>

            {/* Đoạn trích dẫn đối chiếu (Snippet Banner) */}
            {snippet && (
                <div className="pdf-modal-snippet-banner">
                    <div className="snippet-banner-header" onClick={() => setShowSnippet(!showSnippet)}>
                        <span className="snippet-label">
                            <Quote size={13} />
                            <span>
                                Đoạn trích đối chiếu từ <strong>Trang {targetPage}</strong>
                            </span>
                        </span>
                        <span className="snippet-toggle-btn">
                            {showSnippet ? (
                                <>
                                    <span>Thu gọn</span>
                                    <ChevronUp size={13} />
                                </>
                            ) : (
                                <>
                                    <span>Mở rộng</span>
                                    <ChevronDown size={13} />
                                </>
                            )}
                        </span>
                    </div>
                    {showSnippet && <div className="snippet-banner-content">"{snippet}"</div>}
                </div>
            )}

            {/* Document Viewer Body */}
            <div className={`pdf-modal-viewer-body ${!isPdf ? "text-mode" : "pdf-mode"}`}>
                {!resolvedDocId || loadingText ? (
                    <div className="pdf-modal-loading-placeholder">
                        <div className="typing-dots-row">
                            <div className="typing-dot" />
                            <div className="typing-dot" />
                            <div className="typing-dot" />
                        </div>
                        <p>Đang tải nội dung tài liệu...</p>
                    </div>
                ) : isPdf ? (
                    <iframe
                        key={`${resolvedDocId}_${targetPage}`}
                        src={fileUrl}
                        title={`Document Viewer - ${resolvedDocName || documentName}`}
                        className="pdf-iframe"
                    />
                ) : (
                    <div className="document-text-preview">
                        <MarkdownViewer content={textContent} />
                    </div>
                )}
            </div>
        </div>
    )

    if (mode === "modal") {
        return (
            <div className="pdf-modal-overlay" onClick={onClose}>
                {viewerContent}
            </div>
        )
    }

    return <aside className="split-doc-inspector">{viewerContent}</aside>
}

export default PDFModal
