import { useState, useEffect } from "react"
import Sidebar from "./components/Sidebar"
import ChatBox from "./components/ChatBox"
import { getSessions, createSession, deleteSession } from "./services/chatAPI"
import "./App.css"

function App() {
  const [sessions, setSessions] = useState([])
  const [currentSessionId, setCurrentSessionId] = useState("default")

  const loadSessions = async () => {
    try {
      const data = await getSessions()
      setSessions(data || [])
      if (data && data.length > 0) {
        if (!currentSessionId || !data.some(s => s.id === currentSessionId)) {
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
        // Cho phép danh sách rỗng (bằng 0) và sẵn sàng phiên mới
        setCurrentSessionId(`session_${Date.now()}`)
      } else if (currentSessionId === sessionId) {
        setCurrentSessionId(remaining[0].id)
      }
    } catch (err) {
      console.error("Failed to delete session:", err)
    }
  }

  const currentSession = sessions.find((s) => s.id === currentSessionId)

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={setCurrentSessionId}
        onCreateSession={handleCreateSession}
        onDeleteSession={handleDeleteSession}
      />

      <main className="main-content">
        <ChatBox
          sessionId={currentSessionId}
          sessionTitle={currentSession?.title || "Cuộc trò chuyện"}
          onRefreshSessions={loadSessions}
        />
      </main>
    </div>
  )
}

export default App