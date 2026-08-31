const CONFIG = {
  High:   { color: 'linear-gradient(135deg, #dc2626, #b91c1c)', emoji: '🔴' },
  Medium: { color: 'linear-gradient(135deg, #d97706, #b45309)', emoji: '🟡' },
  Low:    { color: 'linear-gradient(135deg, #16a34a, #15803d)', emoji: '🟢' },
}

export default function UrgencyBadge({ urgency }) {
  const cfg = CONFIG[urgency] || { color: '#6b7280', emoji: '⚪' }
  return (
    <div
      className="urgency-badge"
      style={{ background: cfg.color }}
    >
      <span className="urgency-dot" />
      {urgency?.toUpperCase()}
    </div>
  )
}
