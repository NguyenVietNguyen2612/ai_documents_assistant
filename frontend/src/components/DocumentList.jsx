import { useEffect, useState } from "react"
import {
    FileText,
    FileSpreadsheet,
    FileCode,
    Eye,
    Trash2,
    FolderOpen,
    AlertTriangle,
    X,
} from "lucide-react"

function DocumentList({ refreshKey, onPreviewDocument, activePreviewDocId }) {
    const [documents, setDocuments] = useState([])
    const [docToDelete, setDocToDelete] = useState(null)
    const [isDeleting, setIsDeleting] = useState(false)
    const [deleteError, setDeleteError] = useState(null)

    const fetchDocuments = () => {
        fetch("http://localhost:8001/documents")
            .then((response) => response.json())
            .then((data) => {
                setDocuments(Array.isArray(data) ? data : [])
            })
            .catch((error) => {
                console.error("Failed to fetch documents:", error)
            })
    }

    useEffect(() => {
        fetchDocuments()
    }, [refreshKey])

    const handleDeleteClick = (document, e) => {
        if (e) e.stopPropagation()
        setDocToDelete(document)
        setDeleteError(null)
    }

    const handleConfirmDelete = async () => {
        if (!docToDelete) return
        setIsDeleting(true)
        setDeleteError(null)
        try {
            const res = await fetch(`http://localhost:8001/documents/${docToDelete.id}`, {
                method: "DELETE",
            })
            const data = await res.json()
            if (data.error) {
                setDeleteError(data.error)
            } else {
                setDocToDelete(null)
                fetchDocuments()
            }
        } catch (error) {
            console.error("Failed to delete document:", error)
            setDeleteError("Không thể kết nối đến máy chủ.")
        } finally {
            setIsDeleting(false)
        }
    }

    const getDocBadgeMeta = (filename = "") => {
        const ext = filename.split(".").pop()?.toLowerCase()
        if (ext === "csv") {
            return { icon: <FileSpreadsheet size={15} />, extLabel: "CSV", typeClass: "doc-csv" }
        }
        if (ext === "md" || ext === "txt") {
            return { icon: <FileCode size={15} />, extLabel: ext.toUpperCase(), typeClass: "doc-md" }
        }
        return { icon: <FileText size={15} />, extLabel: "PDF", typeClass: "doc-pdf" }
    }

    return (
        <div className="documents">
            <div className="section-header">
                <h3>Kho tri thức (Vector DB)</h3>
                <span className="section-pill-count">{documents.length}</span>
            </div>

            <div className="document-list" role="list">
                {documents.length === 0 ? (
                    <div className="empty-docs-box">
                        <FolderOpen size={20} strokeWidth={1.6} />
                        <span>Chưa có tài liệu nào được tải lên</span>
                    </div>
                ) : (
                    documents.map((document) => {
                        const meta = getDocBadgeMeta(document.name)
                        const isBeingPreviewed = activePreviewDocId && activePreviewDocId === document.id
                        return (
                            <div
                                className={`document-item ${isBeingPreviewed ? "active-preview" : ""}`}
                                key={document.id}
                                role="listitem"
                                onClick={() =>
                                    onPreviewDocument &&
                                    onPreviewDocument({
                                        documentId: document.id,
                                        documentName: document.name,
                                        page: 1,
                                    })
                                }
                                title={`Nhấn để xem trực tiếp "${document.name}"`}
                            >
                                <div className={`document-icon ${meta.typeClass}`} aria-hidden="true">
                                    {meta.icon}
                                </div>

                                <div className="document-info">
                                    <span className="document-name">{document.name}</span>
                                    <div className="document-meta-row">
                                        <span className="document-ext-tag">{meta.extLabel}</span>
                                        <span className="document-size">{document.size}</span>
                                    </div>
                                </div>

                                <div className="document-item-actions">
                                    <button
                                        type="button"
                                        className="doc-action-btn preview-btn"
                                        onClick={(e) => {
                                            e.stopPropagation()
                                            onPreviewDocument &&
                                                onPreviewDocument({
                                                    documentId: document.id,
                                                    documentName: document.name,
                                                    page: 1,
                                                })
                                        }}
                                        title="Xem tài liệu"
                                        aria-label={`Xem tài liệu ${document.name}`}
                                    >
                                        <Eye size={14} />
                                    </button>
                                    <button
                                        type="button"
                                        className="doc-action-btn delete-document-btn"
                                        onClick={(e) => handleDeleteClick(document, e)}
                                        title="Xóa tài liệu"
                                        aria-label={`Xóa tài liệu ${document.name}`}
                                    >
                                        <Trash2 size={14} />
                                    </button>
                                </div>
                            </div>
                        )
                    })
                )}
            </div>

            {/* Hộp thoại xác nhận xóa tài liệu */}
            {docToDelete && (
                <div className="modal-overlay" onClick={() => !isDeleting && setDocToDelete(null)}>
                    <div className="confirm-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
                        <div className="confirm-modal-header">
                            <div className="confirm-modal-title-wrap">
                                <span className="confirm-warning-icon">
                                    <AlertTriangle size={18} />
                                </span>
                                <h3>Xác nhận xóa tài liệu</h3>
                            </div>
                            <button
                                type="button"
                                className="confirm-modal-close"
                                onClick={() => !isDeleting && setDocToDelete(null)}
                                aria-label="Đóng hộp thoại"
                            >
                                <X size={16} />
                            </button>
                        </div>

                        <p className="confirm-modal-body">
                            Bạn có chắc chắn muốn xóa tài liệu <strong>"{docToDelete.name}"</strong> khỏi cơ sở dữ liệu Milvus Vector & chỉ mục BM25 không?
                        </p>

                        {deleteError && (
                            <div className="confirm-modal-error">
                                {deleteError}
                            </div>
                        )}

                        <div className="confirm-modal-actions">
                            <button
                                type="button"
                                className="modal-btn modal-btn-cancel"
                                onClick={() => setDocToDelete(null)}
                                disabled={isDeleting}
                            >
                                Hủy bỏ
                            </button>
                            <button
                                type="button"
                                className="modal-btn modal-btn-delete"
                                onClick={handleConfirmDelete}
                                disabled={isDeleting}
                            >
                                {isDeleting ? "Đang xóa..." : "Xác nhận xóa"}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default DocumentList