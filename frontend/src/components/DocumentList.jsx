import { useEffect, useState } from "react"

function DocumentList({ refreshKey }) {
    const [documents, setDocuments] = useState([])

    useEffect(() => {
        fetch("http://localhost:8000/documents")
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
    }, [refreshKey])

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
                    >
                        <div className="document-icon">
                            PDF
                        </div>

                        <div className="document-info">
                            <span className="document-name">
                                {document.name}
                            </span>

                            <span className="document-size">
                                {document.size}
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}

export default DocumentList