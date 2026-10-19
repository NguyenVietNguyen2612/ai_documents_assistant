import { useState, useEffect } from "react"
import Sidebar from "./components/Sidebar"
import ChatBox from "./components/ChatBox"
import PDFModal from "./components/PDFModal"
import { getSessions, deleteSession } from "./services/chatAPI"
import "./App.css"

function App() {
  const [sessions, setSessions] = useState([])
  const [currentSessionId, setCurrentSessionId] = useState("default")
  const [refreshDocsKey, setRefreshDocsKey] = useState(0)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("ai_doc_theme") || "light"
  })

  // State cho Split-Screen / Modal Document Inspector
  const [activeDocPreview, setActiveDocPreview] = useState(null)

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme)
    localStorage.setItem("ai_doc_theme", theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"))
  }

  const triggerRefreshDocs = () => {
    setRefreshDocsKey((prev) => prev + 1)
  }

  const loadSessions = async () => {
    try {
      const data = await getSessions()
      setSessions(data || [])
      if (data && data.length > 0) {
        if (!currentSessionId || !data.some((s) => s.id === currentSessionId)) {
          setCurrentSessionId(data[0].id)
        }
      } else {
        setCurrentSessionId(`session_${Date.now()}`)
      }
    } catch (err) {
      console.error("Failed to load sessions:", err)
    }
  }

  useEffect(() => {
    loadSessions()
  }, [])

  const handleCreateSession = () => {
    const newDraftId = `session_${Date.now()}`
    setCurrentSessionId(newDraftId)
  }

  const handleDeleteSession = async (sessionId, e) => {
    if (e) e.stopPropagation()
    try {
      await deleteSession(sessionId)
      const remaining = sessions.filter((s) => s.id !== sessionId)
      setSessions(remaining)
      if (remaining.length === 0) {
        setCurrentSessionId(`session_${Date.now()}`)
      } else if (currentSessionId === sessionId) {
        setCurrentSessionId(remaining[0].id)
      }
    } catch (err) {
      console.error("Failed to delete session:", err)
    }
  }

  const handleOpenDocPreview = (docInfo) => {
    if (!docInfo) return
    setActiveDocPreview((prev) => ({
      isOpen: true,
      documentName: docInfo.documentName || docInfo.name || "Tài liệu",
      documentId: docInfo.documentId || docInfo.id || null,
      page: docInfo.page || 1,
      snippet: docInfo.snippet || "",
      mode: prev?.mode || "split", // Mặc định mở Split-Screen bên phải chuẩn NotebookLM
    }))
  }

  const handleTogglePreviewMode = () => {
    setActiveDocPreview((prev) =>
      prev ? { ...prev, mode: prev.mode === "split" ? "modal" : "split" } : null
    )
  }

  const currentSession = sessions.find((s) => s.id === currentSessionId)

  return (
    <div className={`app ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <Sidebar
        isOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((p) => !p)}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={setCurrentSessionId}
        onCreateSession={handleCreateSession}
        onDeleteSession={handleDeleteSession}
        refreshDocsKey={refreshDocsKey}
        onUploadSuccess={triggerRefreshDocs}
        onPreviewDocument={handleOpenDocPreview}
        activePreviewDocId={activeDocPreview?.documentId}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <main className={`main-workspace ${activeDocPreview?.isOpen && activeDocPreview?.mode === "split" ? "has-split-viewer" : ""}`}>
        <div className="chat-pane">
          <ChatBox
            sessionId={currentSessionId}
            sessionTitle={currentSession?.title || "Cuộc trò chuyện mới"}
            onRefreshSessions={loadSessions}
            onOpenCitation={handleOpenDocPreview}
            onUploadSuccess={triggerRefreshDocs}
            sidebarOpen={sidebarOpen}
            onToggleSidebar={() => setSidebarOpen((p) => !p)}
            theme={theme}
            onToggleTheme={toggleTheme}
            activeDocPreview={activeDocPreview}
          />
        </div>

        {/* Khung xem tài liệu (Hỗ trợ cả Split-Screen bên phải và Full Modal) */}
        <PDFModal
          isOpen={Boolean(activeDocPreview?.isOpen)}
          mode={activeDocPreview?.mode || "split"}
          onToggleMode={handleTogglePreviewMode}
          onClose={() => setActiveDocPreview(null)}
          documentName={activeDocPreview?.documentName}
          documentId={activeDocPreview?.documentId}
          page={activeDocPreview?.page}
          snippet={activeDocPreview?.snippet}
        />
      </main>
    </div>
  )
}

export default App