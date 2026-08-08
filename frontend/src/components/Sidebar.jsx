import DocumentUpload from "./DocumentUpload"
import DocumentList from "./DocumentList"

function Sidebar() {
    return (
        <aside className="sidebar">
            <div className="brand">
                <div className="brand-icon">AI</div>

                <div>
                    <h2>Doc Assistant</h2>
                    <span>RAG-powered</span>
                </div>
            </div>

            <DocumentUpload />

            <div className="sidebar-divider" />

            <DocumentList />

            <div className="sidebar-footer">
                <span className="status-dot" />
                <span>System ready</span>
            </div>
        </aside>
    )
}

export default Sidebar