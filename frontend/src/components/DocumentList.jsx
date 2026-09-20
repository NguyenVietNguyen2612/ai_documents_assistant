import { useEffect, useState } from "react"

function DocumentList({ refreshKey }) {
    const [documents, setDocuments] = useState([])

    const fetchDocuments = () => {
        fetch("http://localhost:8001/documents")
            .then((response) => response.json())
            .then((data) => {
                setDocuments(data)
            })
            .catch((error) => {
                console.error(
                    "Failed to fetch documents:",
                    error
                )
            })
    }

    useEffect(() => {
        fetchDocuments()
    }, [refreshKey])

    const [docToDelete, setDocToDelete] = useState(null)
    const [isDeleting, setIsDeleting] = useState(false)
    const [deleteError, setDeleteError] = useState(null)

    const handleDeleteClick = (document) => {
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

    return (
        <div className="documents">
            <div className="section-header">
                <h3>Your documents</h3>

                <span className="document-count">
                    {documents.length}
                </span>
            </div>

            <div className="document-list">
                {documents.map((document) => (
                    <div
                        className="document-item"
                        key={document.id}
                        style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}
                    >
                        <div style={{ display: "flex", alignItems: "center" }}>
                            <div className="document-icon">
                                PDF
                            </div>

                            <div className="document-info" style={{ marginLeft: "10px" }}>
                                <span className="document-name" style={{ display: "block" }}>
                                    {document.name}
                                </span>

                                <span className="document-size">
                                    {document.size}
                                </span>
                            </div>
                        </div>
                        
                        <button 
                            className="delete-document-btn"
                            onClick={() => handleDeleteClick(document)}
                            style={{
                                background: "transparent",
                                border: "none",
                                cursor: "pointer",
                                fontSize: "16px",
                                color: "#888",
                                padding: "4px"
                            }}
                            title="Xóa tài liệu"
                        >
                            ✕
                        </button>
                    </div>
                ))}
            </div>

            {/* Hộp thoại xác nhận xóa tài liệu (Custom Confirmation Modal ở giữa màn hình) */}
            {docToDelete && (
                <div className="modal-overlay" onClick={() => !isDeleting && setDocToDelete(null)}>
                    <div className="confirm-modal" onClick={(e) => e.stopPropagation()}>
                        <div className="confirm-modal-header">
                            <h3>Xác nhận xóa tài liệu</h3>
                            <button 
                                className="confirm-modal-close" 
                                onClick={() => !isDeleting && setDocToDelete(null)}
                            >
                                ✕
                            </button>
                        </div>

                        <p className="confirm-modal-body">
                            Bạn có chắc chắn muốn xóa tài liệu <strong>"{docToDelete.name}"</strong> khỏi cơ sở dữ liệu Vector không?
                        </p>

                        {deleteError && (
                            <div className="confirm-modal-error">
                                {deleteError}
                            </div>
                        )}

                        <div className="confirm-modal-actions">
                            <button 
                                className="modal-btn modal-btn-cancel" 
                                onClick={() => setDocToDelete(null)}
                                disabled={isDeleting}
                            >
                                Hủy bỏ
                            </button>
                            <button 
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