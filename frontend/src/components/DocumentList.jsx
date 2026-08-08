function DocumentList() {
    const documents = [
        {
            name: "rag.pdf",
            size: "1.2 MB",
        },
        {
            name: "ai_agents.pdf",
            size: "2.4 MB",
        },
        {
            name: "langgraph.pdf",
            size: "1.8 MB",
        },
    ]

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
                        key={document.name}
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