import { useState, useRef } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

export default function BatchUpload() {
  const [file, setFile]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')
  const [done, setDone]       = useState(false)
  const inputRef = useRef()

  const handleFile = e => {
    setFile(e.target.files[0])
    setDone(false)
    setError('')
  }

  const classify = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await axios.post(`${API}/classify-batch`, form, {
        responseType: 'blob',
      })
      const url = window.URL.createObjectURL(new Blob([res.data]))
      const a   = document.createElement('a')
      a.href     = url
      a.download = 'classified_tickets.csv'
      a.click()
      window.URL.revokeObjectURL(url)
      setDone(true)
    } catch {
      setError('Classification failed. Ensure your CSV has a ticket_text column.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card glass">
      <div className="card-title"><span>📂</span> Batch Classification</div>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 24, fontSize: 14 }}>
        Upload a CSV with a <code style={{ background: 'rgba(203,213,225,0.4)',
          padding: '2px 6px', borderRadius: 4 }}>ticket_text</code> column.
        Classified results download automatically as a CSV.
      </p>

      <div className="upload-zone" onClick={() => inputRef.current.click()}>
        <div className="upload-zone-icon">
          {file ? '📄' : '☁️'}
        </div>
        <div className="upload-zone-text">
          {file ? file.name : 'Click to select a CSV file'}
        </div>
        <div className="upload-zone-sub">
          {file
            ? `${(file.size / 1024).toFixed(1)} KB — ready to classify`
            : 'Supports: ticket_text, Ticket Description column names'}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={handleFile}
        />
      </div>

      {file && (
        <button
          className="btn-primary"
          onClick={classify}
          disabled={loading}
          style={{ marginTop: 20 }}
        >
          {loading
            ? <><span className="spinner" /> Classifying all rows…</>
            : '🚀 Classify & Download Results'}
        </button>
      )}

      {done && (
        <div className="success-msg">
          ✅ Done! Check your Downloads folder for <strong>classified_tickets.csv</strong>
        </div>
      )}

      {error && <div className="error-msg">⚠️ {error}</div>}
    </div>
  )
}
