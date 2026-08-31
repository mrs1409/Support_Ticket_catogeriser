import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import SingleTicket from '../components/SingleTicket'
import BatchUpload from '../components/BatchUpload'
import ModelInfo from '../components/ModelInfo'

const TABS = [
  { id: 'single', label: '🎫 Single Ticket' },
  { id: 'batch', label: '📂 Batch Upload' },
  { id: 'model', label: '📊 Model Info' },
]

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('single')
  const navigate = useNavigate()

  return (
    <div className="app-wrapper">
      <div className="bg-orbs" />
      <div className="app-container">

        <header className="header glass-strong">
          <div className="header-brand">
            <div className="header-icon">🎫</div>
            <div className="header-text">
              <h1>Support Ticket Classifier</h1>
              <p>NLP-powered routing — predicts category and urgency instantly</p>
            </div>
          </div>
          <div className="header-badges">
            <button className="header-home-link" onClick={() => navigate('/')}>
              ← Home
            </button>
            <span className="header-badge">⚡ FastAPI</span>
            <span className="header-badge">⚛️ React</span>
            <span className="header-badge">🤖 scikit-learn</span>
          </div>
        </header>

        <nav className="tabs-wrapper glass">
          {TABS.map(t => (
            <button
              key={t.id}
              className={`tab-btn ${activeTab === t.id ? 'active' : ''}`}
              onClick={() => setActiveTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {activeTab === 'single' && <SingleTicket />}
        {activeTab === 'batch'  && <BatchUpload />}
        {activeTab === 'model'  && <ModelInfo />}

      </div>
    </div>
  )
}
