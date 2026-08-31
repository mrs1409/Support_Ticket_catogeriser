import { useState } from 'react'
import axios from 'axios'
import UrgencyBadge from './UrgencyBadge'
import ConfidenceBar from './ConfidenceBar'

const API = '/api'

const EXAMPLES = [
  {
    label: '🔴 Access',
    text: 'URGENT: My account is completely locked and I cannot log in at all. I have a client deadline in 2 hours. Please fix immediately!',
  },
  {
    label: '🟠 Hardware',
    text: 'My laptop screen is cracked and the keyboard stopped working after it fell. Need a replacement device urgently.',
  },
  {
    label: '🟡 Storage',
    text: 'My mailbox is almost full and I cannot receive new client emails. Can someone increase my storage quota please?',
  },
  {
    label: '🟢 Purchase',
    text: 'We need to order 5 new laptops and 3 monitors for the new hires joining next week. Please raise a purchase order.',
  },
]

export default function SingleTicket() {
  const [text, setText]       = useState('')
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')

  const classify = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await axios.post(`${API}/classify`, { text })
      setResult(res.data)
    } catch (e) {
      setError(
        e.response?.data?.detail ||
        'Cannot reach API. Make sure uvicorn is running on port 8000.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      {/* Input card */}
      <div className="card glass">
        <div className="card-title">
          <span>✍️</span> Classify a Support Ticket
        </div>

        <div className="example-btns">
          <span style={{ fontSize: 12, color: 'var(--text-muted)',
            alignSelf: 'center', marginRight: 4 }}>
            Try an example:
          </span>
          {EXAMPLES.map(ex => (
            <button
              key={ex.label}
              className="example-btn"
              onClick={() => { setText(ex.text); setResult(null); setError('') }}
            >
              {ex.label}
            </button>
          ))}
        </div>

        <textarea
          className="ticket-textarea"
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Paste or type a customer support ticket here…"
        />

        <button
          className="btn-primary"
          onClick={classify}
          disabled={loading || !text.trim()}
        >
          {loading
            ? <><span className="spinner" /> Classifying…</>
            : <>🔍 Classify Ticket</>}
        </button>

        {error && <div className="error-msg">⚠️ {error}</div>}
      </div>

      {/* Results card */}
      {result && (
        <div className="card glass" style={{ animation: 'fadeIn 0.4s ease' }}>
          <div className="card-title"><span>✅</span> Classification Result</div>

          <div className="results-grid">
            {/* Category */}
            <div className="result-box">
              <div className="result-label">📂 Predicted Category</div>
              <div className="category-name">{result.category}</div>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
              <div className="confidence-text">
                Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
              </div>
            </div>

            {/* Urgency */}
            <div className="result-box">
              <div className="result-label">⚡ Urgency Level</div>
              <UrgencyBadge urgency={result.urgency} />
              <div className="urgency-reason">{result.urgency_reason}</div>
            </div>
          </div>

          <div className="section-divider" />

          {/* All scores */}
          <div className="scores-section">
            <div className="scores-title">All Category Scores</div>
            {Object.entries(result.all_scores).map(([cat, score]) => (
              <ConfidenceBar
                key={cat}
                label={cat}
                value={score}
                isTop={cat === result.category}
              />
            ))}
          </div>

          <div className="model-tag">
            🤖 Model: {result.model_name}
          </div>
        </div>
      )}

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </>
  )
}
