import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
import heroImg from './assets/hero.png'
import './App.css'

function App() {
  return (
    <div>
      <h1>AI Document Assistant</h1>

      <section>
        <h2>Upload Document</h2>
        <input type="file" />
        <button>Upload</button>
      </section>

      <section>
        <h2>Documents</h2>

        <ul>
          <li>rag.pdf</li>
          <li>ai_agents.pdf</li>
          <li>langgraph.pdf</li>
        </ul>
      </section>

      <section>
        <h2>Chat</h2>

        <input
          type="text"
          placeholder="Ask something about your documents..."
        />

        <button>Send</button>
      </section>
    </div>
  )
}

export default App