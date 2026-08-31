import { useState, useEffect } from 'react'
import axios from 'axios'

const API = '/api'

export default function ModelInfo() {
  const [info, setInfo]   = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    axios.get(`${API}/model-info`)
      .then(r => setInfo(r.data))
      .catch(() => setError('Cannot reach API. Is uvicorn running on port 8000?'))
  }, [])

  if (error) return <div className="card glass error-msg">{error}</div>
  if (!info)  return (
    <div className="card glass" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 48 }}>
      <span className="spinner" style={{ borderTopColor: 'var(--accent)',
        borderColor: 'rgba(203,190,176,0.5)', width: 28, height: 28,
        borderWidth: 3 }} />
      <p style={{ marginTop: 16 }}>Loading model info…</p>
    </div>
  )

  const best = info.model_comparison?.[0]

  return (
    <>
      {/* Metric cards */}
      {best && (
        <div className="metrics-grid">
          {[
            { value: `${(best.Accuracy * 100).toFixed(1)}%`, label: 'Accuracy' },
            { value: `${(best['F1 Weighted'] * 100).toFixed(1)}%`, label: 'F1 Weighted' },
            { value: best.Model.split(' ')[0], label: 'Best Algorithm' },
          ].map(m => (
            <div key={m.label} className="metric-card glass">
              <div className="metric-value">{m.value}</div>
              <div className="metric-label">{m.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Model comparison table */}
      <div className="card glass">
        <div className="card-title"><span>📊</span> Model Comparison</div>
        {info.model_comparison?.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>F1 Macro</th>
                <th>F1 Weighted</th>
              </tr>
            </thead>
            <tbody>
              {info.model_comparison.map((row, i) => (
                <tr key={i}>
                  <td>{row.Model}</td>
                  <td>{(row.Accuracy * 100).toFixed(2)}%</td>
                  <td>{(row['F1 Macro'] * 100).toFixed(2)}%</td>
                  <td>{(row['F1 Weighted'] * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>
            Run <code>python train.py</code> first to generate comparison data.
          </p>
        )}
      </div>

      {/* Confusion matrix */}
      <div className="card glass">
        <div className="card-title"><span>🗺️</span> Confusion Matrix</div>
        <img
          src={`${API}/confusion-matrix`}
          alt="Confusion Matrix"
          className="confusion-img"
          onError={e => {
            e.target.replaceWith(
              Object.assign(document.createElement('p'), {
                textContent: 'Run train.py to generate the confusion matrix.',
                style: 'color: var(--text-muted); font-size: 14px;'
              })
            )
          }}
        />
      </div>

      {/* How it works */}
      <div className="card glass">
        <div className="card-title"><span>⚙️</span> How It Works</div>
        <div className="how-grid">
          {[
            {
              title: '🧹 Text Cleaning',
              items: ['Lowercase all text', 'Remove URLs & emails', 'Strip punctuation', 'Normalise whitespace'],
            },
            {
              title: '🔢 TF-IDF Features',
              items: ['Unigrams + bigrams', '40,000 max features', 'Sublinear TF scaling', 'Min doc frequency: 1'],
            },
            {
              title: '🤖 ML Pipeline',
              items: ['3 models compared', 'Best by F1 weighted', 'sklearn Pipeline object', 'Saved as .pkl file'],
            },
          ].map(col => (
            <div key={col.title} className="how-card">
              <div className="how-card-title">{col.title}</div>
              {col.items.map(item => (
                <div key={item} className="how-card-item">{item}</div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
