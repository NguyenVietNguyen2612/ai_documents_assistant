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

    const handleDelete = (id) => {
        if (window.confirm("Bạn có chắc chắn muốn xóa tài liệu này khỏi vector DB?")) {
            fetch(`http://localhost:8001/documents/${id}`, {
                method: "DELETE",
            })
                .then((response) => response.json())
                .then((data) => {
                    if (data.error) {
                        alert("Lỗi: " + data.error);
                    } else {
                        fetchDocuments();
                    }
                })
                .catch((error) => {
                    console.error("Failed to delete document:", error)
                })
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
                            onClick={() => handleDelete(document.id)}
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
        </div>
    )
}

export default DocumentList