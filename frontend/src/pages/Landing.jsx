import { useNavigate } from 'react-router-dom'

const scrollTo = (id) => (e) => {
  e.preventDefault()
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

// Points to Render backend in prod, localhost in dev
const BACKEND = (import.meta.env.VITE_API_URL || 'https://support-ticket-catogeriser.onrender.com/api')
  .replace('/api', '')

const FEATURES = [
  {
    icon: '🧠',
    title: 'Smart Classification',
    desc: 'TF-IDF + ensemble ML instantly sorts tickets into the right category — no manual triage.',
  },
  {
    icon: '⚡',
    title: 'Urgency Detection',
    desc: 'Weighted keyword rules flag High / Medium / Low urgency so nothing critical slips through.',
  },
  {
    icon: '📂',
    title: 'Batch CSV Processing',
    desc: 'Upload a spreadsheet of tickets and download every prediction classified in seconds.',
  },
  {
    icon: '📊',
    title: 'Confidence Scores',
    desc: 'Full probability breakdown across every category, not just a single black-box label.',
  },
]

const STEPS = [
  {
    n: '01',
    title: 'Paste or upload',
    desc: 'Drop in a single ticket, or upload a CSV of hundreds at once.',
  },
  {
    n: '02',
    title: 'Model predicts',
    desc: 'A trained ensemble (Logistic Regression + Linear SVM) scores every category.',
  },
  {
    n: '03',
    title: 'Act on results',
    desc: 'Get the category, urgency, and confidence — routed and ready for your team.',
  },
]

const CATEGORIES = [
  'Access', 'Hardware', 'HR Support', 'Storage',
  'Purchase', 'Administrative rights', 'Internal Project', 'Miscellaneous',
]

export default function Landing() {
  const navigate = useNavigate()

  return (
    <div className="landing-wrapper">
      <div className="bg-orbs"><div className="bg-orb-mid" /></div>

      {/* NAV */}
      <div className="landing-nav">
        <div className="landing-nav-inner glass-strong">
          <div className="landing-logo">
            <span className="landing-logo-icon">🎫</span>
            TicketSense
          </div>
          <div className="landing-nav-links">
            <a className="landing-nav-link" href="#features" onClick={scrollTo('features')}>Features</a>
            <a className="landing-nav-link" href="#how-it-works" onClick={scrollTo('how-it-works')}>How it works</a>
            <a className="landing-nav-link" href={`${BACKEND}/docs`} target="_blank" rel="noreferrer">API Docs</a>
          </div>
          <button className="btn-primary nav-cta" onClick={() => navigate('/dashboard')}>
            Launch Dashboard →
          </button>
        </div>
      </div>

      {/* HERO */}
      <section className="hero">
        <div>
          <div className="hero-badge glass">
            <span className="hero-badge-dot" />
            Model live · 86.48% accuracy
          </div>
          <h1>
            Route support tickets<br />
            in <span className="accent-text">milliseconds</span>, not minutes.
          </h1>
          <p className="hero-sub">
            An NLP classifier that reads a support ticket, predicts its category,
            scores its urgency, and hands your team a prioritized queue —
            trained on 47,800+ real service-desk tickets.
          </p>
          <div className="hero-actions">
            <button className="btn-primary" onClick={() => navigate('/dashboard')}>
              🚀 Try it now
            </button>
            <a className="btn-secondary" href={`${BACKEND}/docs`} target="_blank" rel="noreferrer">
              View API Docs
            </a>
          </div>
          <div className="hero-trust">
            <div className="hero-trust-item">
              <span className="hero-trust-value">86.4%</span>
              <span className="hero-trust-label">Model accuracy</span>
            </div>
            <div className="hero-trust-item">
              <span className="hero-trust-value">8</span>
              <span className="hero-trust-label">Ticket categories</span>
            </div>
            <div className="hero-trust-item">
              <span className="hero-trust-value">47.8K</span>
              <span className="hero-trust-label">Training tickets</span>
            </div>
          </div>
        </div>

        <div className="hero-visual">
          <div className="hero-demo-card glass-strong">
            <div className="hero-demo-top">
              <span className="hero-demo-dot" style={{ background: '#f87171' }} />
              <span className="hero-demo-dot" style={{ background: '#fbbf24' }} />
              <span className="hero-demo-dot" style={{ background: '#34d399' }} />
              <span style={{ marginLeft: 8, fontSize: 12, color: 'var(--text-muted)', fontWeight: 500 }}>
                classify.request
              </span>
            </div>
            <div className="hero-demo-text">
              "My mailbox is almost full and I can't receive new client
              emails — can someone increase my storage quota?"
            </div>
            <div className="hero-demo-result">
              <div>
                <div className="hero-demo-result-label">Predicted Category</div>
                <div className="hero-demo-result-value">Storage</div>
              </div>
              <div style={{ fontSize: 22 }}>💾</div>
            </div>
            <div className="hero-demo-result">
              <div>
                <div className="hero-demo-result-label">Urgency</div>
                <div className="hero-demo-result-value" style={{ color: '#d97706' }}>Medium</div>
              </div>
              <div style={{ fontSize: 22 }}>🟡</div>
            </div>
          </div>
        </div>
      </section>

      {/* STATS STRIP */}
      <div className="stats-strip">
        <div className="stat-pill glass">
          <div className="stat-pill-value">86.4%</div>
          <div className="stat-pill-label">Accuracy</div>
        </div>
        <div className="stat-pill glass">
          <div className="stat-pill-value">8</div>
          <div className="stat-pill-label">Categories</div>
        </div>
        <div className="stat-pill glass">
          <div className="stat-pill-value">47.8K</div>
          <div className="stat-pill-label">Training tickets</div>
        </div>
        <div className="stat-pill glass">
          <div className="stat-pill-value">4</div>
          <div className="stat-pill-label">Models compared</div>
        </div>
      </div>

      {/* FEATURES */}
      <section className="landing-section" id="features">
        <span className="section-eyebrow">Capabilities</span>
        <h2 className="section-title">Everything a triage queue needs</h2>
        <p className="section-sub">
          One API, four ways to work with it — from a single pasted ticket
          to a full CSV of your backlog.
        </p>
        <div className="features-grid">
          {FEATURES.map(f => (
            <div key={f.title} className="feature-card glass">
              <div className="feature-icon">{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="landing-section" id="how-it-works">
        <span className="section-eyebrow">Process</span>
        <h2 className="section-title">How it works</h2>
        <p className="section-sub">
          Three steps between a raw ticket and an actionable, prioritized queue.
        </p>
        <div className="steps-grid">
          {STEPS.map(s => (
            <div key={s.n} className="step-card glass">
              <div className="step-number">{s.n}</div>
              <h3>{s.title}</h3>
              <p>{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CATEGORIES SHOWCASE */}
      <section className="landing-section">
        <span className="section-eyebrow">Taxonomy</span>
        <h2 className="section-title">Trained across 8 real categories</h2>
        <p className="section-sub">
          Learned from real IT service-desk tickets — not synthetic templates.
        </p>
        <div className="category-pills">
          {CATEGORIES.map(c => (
            <span key={c} className="category-pill glass">{c}</span>
          ))}
        </div>
      </section>

      {/* CTA */}
      <div className="cta-banner">
        <h2>Ready to see it classify a ticket?</h2>
        <p>Jump into the dashboard — paste a ticket or upload a CSV and get results instantly.</p>
        <button className="btn-primary" onClick={() => navigate('/dashboard')}>
          🚀 Launch Dashboard
        </button>
      </div>

      {/* FOOTER */}
      <footer className="landing-footer">
        <span>© {new Date().getFullYear()} TicketSense — built with FastAPI, React &amp; scikit-learn</span>
        <div className="landing-footer-links">
          <a href={`${BACKEND}/docs`} target="_blank" rel="noreferrer">API Docs</a>
          <a href="#features" onClick={scrollTo('features')}>Features</a>
          <a href="#how-it-works" onClick={scrollTo('how-it-works')}>How it works</a>
        </div>
      </footer>
    </div>
  )
}
