export default function ConfidenceBar({ label, value, isTop }) {
  const pct = (value * 100).toFixed(1)
  const fillColor = isTop
    ? 'linear-gradient(90deg, #e28a5c, #c15f3f)'
    : 'linear-gradient(90deg, #f3c8ab, #eab08a)'

  return (
    <div className="score-row">
      <span className="score-label" title={label}>{label}</span>
      <div className="score-bar-bg">
        <div
          className="score-bar-fill"
          style={{ width: `${pct}%`, background: fillColor }}
        />
      </div>
      <span className="score-pct">{pct}%</span>
    </div>
  )
}
