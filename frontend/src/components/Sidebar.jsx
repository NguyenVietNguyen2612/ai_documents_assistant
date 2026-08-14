import { useState } from "react"
import DocumentUpload from "./DocumentUpload"
import DocumentList from "./DocumentList"

function Sidebar() {
    const [refreshKey, setRefreshKey] = useState(0)

    const handleUploadSuccess = () => {
        setRefreshKey((prev) => prev + 1)
    }

    return (
        <aside className="sidebar">
            <div className="brand">
                <div className="brand-icon">AI</div>

                <div>
                    <h2>Doc Assistant</h2>
                    <span>RAG-powered</span>
                </div>
            </div>

            <DocumentUpload onUploadSuccess={handleUploadSuccess} />

            <div className="sidebar-divider" />

            <DocumentList refreshKey={refreshKey} />

            <div className="sidebar-footer">
                <span className="status-dot" />
                <span>System ready</span>
            </div>
        </aside>
    )
}

export default Sidebar