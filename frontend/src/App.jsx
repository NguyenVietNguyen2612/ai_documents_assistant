import Sidebar from "./components/Sidebar"
import ChatBox from "./components/ChatBox"
import "./App.css"

function App() {
  return (
    <div className="app">
      <Sidebar />

      <main className="main-content">
        <header className="topbar">
          <div>
            <h1>AI Document Assistant</h1>
            <p>Ask questions about your documents</p>
          </div>

          <button className="icon-button" title="Settings">
            ⚙
          </button>
        </header>

        <ChatBox />
      </main>
    </div>
  )
}

export default App