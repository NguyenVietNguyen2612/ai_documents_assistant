import { useState } from "react"
import {
    UploadCloud,
    FileText,
    FileSpreadsheet,
    FileCode,
    CheckCircle2,
    AlertCircle,
    Loader2,
    X,
} from "lucide-react"

function DocumentUpload({ onUploadSuccess }) {
    const [file, setFile] = useState(null)
    const [uploading, setUploading] = useState(false)
    const [dragActive, setDragActive] = useState(false)
    const [feedback, setFeedback] = useState(null)

    const selectFile = (selectedFile) => {
        if (!selectedFile) return
        setFile(selectedFile)
        setFeedback(null)
    }

    const handleFileChange = (event) => {
        selectFile(event.target.files[0])
        event.target.value = ""
    }

    const handleDragOver = (e) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(true)
    }

    const handleDragLeave = (e) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)
    }

    const handleDrop = (e) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            selectFile(e.dataTransfer.files[0])
        }
    }

    const handleUpload = async () => {
        if (!file) return

        setUploading(true)
        setFeedback(null)

        const formData = new FormData()
        formData.append("file", file)

        try {
            const response = await fetch("http://localhost:8001/documents/upload", {
                method: "POST",
                body: formData,
            })

            const data = await response.json().catch(() => ({}))

            if (response.ok && !data.error) {
                const uploadedName = file.name
                setFile(null)
                setFeedback({
                    type: "success",
                    message: `Đã nạp "${uploadedName}" (${data.chunks_count || "nhiều"} phân đoạn)`,
                })
                if (onUploadSuccess) {
                    onUploadSuccess()
                }
                setTimeout(() => setFeedback(null), 4000)
            } else {
                setFeedback({
                    type: "error",
                    message: data.error || "Tải lên thất bại, vui lòng thử lại.",
                })
            }
        } catch (error) {
            console.error("Error uploading file:", error)
            setFeedback({
                type: "error",
                message: "Không thể kết nối tới máy chủ Backend.",
            })
        } finally {
            setUploading(false)
        }
    }

    const renderFileTypeIcon = (filename = "") => {
        const ext = filename.split(".").pop()?.toLowerCase()
        if (ext === "csv") return <FileSpreadsheet size={16} />
        if (ext === "md" || ext === "txt") return <FileCode size={16} />
        return <FileText size={16} />
    }

    return (
        <div className="upload-section">
            <div className="section-header">
                <h3>Nạp tài liệu mới</h3>
                <span className="upload-format-pill">PDF · CSV · MD · TXT</span>
            </div>

            <label
                className={`upload-box ${dragActive ? "drag-active" : ""}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    accept=".pdf,.csv,.txt,.md"
                    onChange={handleFileChange}
                    hidden
                />

                <div className="upload-icon" aria-hidden="true">
                    <UploadCloud size={20} strokeWidth={2} />
                </div>

                <div className="upload-title">
                    Kéo thả hoặc nhấn để chọn tệp
                </div>

                <div className="upload-description">
                    Tự động trích xuất OCR, bảng biểu & đánh chỉ mục Hybrid
                </div>
            </label>

            {file && (
                <div className="selected-file">
                    <div className="file-icon" aria-hidden="true">
                        {renderFileTypeIcon(file.name)}
                    </div>

                    <div className="file-info">
                        <span className="file-name" title={file.name}>
                            {file.name}
                        </span>
                        <span className="file-size">
                            {file.size < 1024 * 1024
                                ? `${Math.max(1, Math.round(file.size / 1024))} KB`
                                : `${(file.size / 1024 / 1024).toFixed(2)} MB`}
                        </span>
                    </div>

                    {!uploading && (
                        <button
                            type="button"
                            className="clear-file-btn"
                            onClick={() => setFile(null)}
                            title="Bỏ chọn tệp"
                            aria-label="Bỏ chọn tệp"
                        >
                            <X size={14} />
                        </button>
                    )}

                    <button
                        type="button"
                        className="upload-button"
                        onClick={handleUpload}
                        disabled={uploading}
                    >
                        {uploading ? (
                            <>
                                <Loader2 size={13} className="spin-icon" />
                                <span>Đang xử lý...</span>
                            </>
                        ) : (
                            <span>Tải lên</span>
                        )}
                    </button>
                </div>
            )}

            {feedback && (
                <div className={`upload-feedback ${feedback.type}`} role="status">
                    {feedback.type === "success" ? (
                        <CheckCircle2 size={14} />
                    ) : (
                        <AlertCircle size={14} />
                    )}
                    <span>{feedback.message}</span>
                </div>
            )}
        </div>
    )
}

export default DocumentUpload