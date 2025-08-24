// frontend/react_app/src/App.tsx
import React, { useState } from 'react'
import { CaptionCard } from './components/CaptionCard'
import { VQACard } from './components/VQACard'
import { ChatCard } from './components/ChatCard'
import { HealthStatus } from './components/HealthStatus'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('caption')

  const tabs = [
    { id: 'caption', label: '📸 圖像描述', component: CaptionCard },
    { id: 'vqa', label: '🤔 視覺問答', component: VQACard },
    { id: 'chat', label: '💬 文字聊天', component: ChatCard },
    { id: 'health', label: '🔍 系統狀態', component: HealthStatus }
  ]

  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component || CaptionCard

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎯 VisionQuest - 多模態 AI 工具箱</h1>
        <p>Phase 2: 圖像理解與視覺問答系統</p>
      </header>

      <nav className="tab-navigation">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="main-content">
        <ActiveComponent />
      </main>

      <footer className="app-footer">
        <p>VisionQuest v0.1.0 - Phase 2 Demo</p>
      </footer>
    </div>
  )
}

export default App