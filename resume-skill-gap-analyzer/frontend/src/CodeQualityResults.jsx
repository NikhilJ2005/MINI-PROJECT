import "./cssFile/CodeChallenge.css";

const DIMENSIONS = [
  { key: "speed", label: "Speed", icon: "\u26A1", description: "Algorithm efficiency & time complexity" },
  { key: "complexity", label: "Complexity", icon: "\uD83E\uDDE9", description: "Code complexity & readability" },
  { key: "flexibility", label: "Flexibility", icon: "\uD83D\uDD27", description: "Modularity & extensibility" },
  { key: "code_quality", label: "Code Quality", icon: "\u2728", description: "Naming, docs & error handling" },
  { key: "best_practices", label: "Best Practices", icon: "\uD83D\uDCCB", description: "SOLID, DRY & separation of concerns" },
];

function getScoreClass(score) {
  if (score >= 8) return "cq-score-excellent";
  if (score >= 6) return "cq-score-good";
  if (score >= 4) return "cq-score-fair";
  return "cq-score-poor";
}

function getScoreLabel(score) {
  if (score >= 8) return "Excellent";
  if (score >= 6) return "Good";
  if (score >= 4) return "Fair";
  return "Needs Work";
}

function CodeQualityResults({ scores, title = "Code Quality Analysis", compact = false }) {
  if (!scores) return null;

  const overall = scores.overall_score || 0;

  return (
    <div className={`cq-results ${compact ? "cq-results-compact" : ""}`}>
      <h3 className="cq-results-title">{title}</h3>

      <div className="cq-overall">
        <div className={`cq-overall-score ${getScoreClass(overall)}`}>
          <span className="cq-overall-number">{overall.toFixed(1)}</span>
          <span className="cq-overall-max">/10</span>
        </div>
        <div className="cq-overall-label">{getScoreLabel(overall)}</div>
        {scores.summary && <p className="cq-summary">{scores.summary}</p>}
      </div>

      {scores.time_complexity && scores.time_complexity !== "Unknown" && (
        <div className="cq-complexity-badges">
          <span className="cq-badge">Time: {scores.time_complexity}</span>
          {scores.space_complexity && scores.space_complexity !== "Unknown" && (
            <span className="cq-badge">Space: {scores.space_complexity}</span>
          )}
        </div>
      )}

      <div className="cq-dimensions">
        {DIMENSIONS.map((dim) => {
          const dimData = scores[dim.key];
          if (!dimData) return null;
          const score = dimData.score || 0;
          return (
            <div key={dim.key} className="cq-dimension">
              <div className="cq-dim-header">
                <span className="cq-dim-icon">{dim.icon}</span>
                <span className="cq-dim-label">{dim.label}</span>
                <span className={`cq-dim-score ${getScoreClass(score)}`}>{score}/10</span>
              </div>
              <div className="cq-dim-bar-bg">
                <div
                  className={`cq-dim-bar ${getScoreClass(score)}`}
                  style={{ width: `${score * 10}%` }}
                />
              </div>
              {!compact && dimData.notes && (
                <p className="cq-dim-notes">{dimData.notes}</p>
              )}
            </div>
          );
        })}
      </div>

      {scores.correctness && (
        <div className="cq-dimension cq-correctness">
          <div className="cq-dim-header">
            <span className="cq-dim-icon">{"\u2705"}</span>
            <span className="cq-dim-label">Correctness</span>
            <span className={`cq-dim-score ${getScoreClass(scores.correctness.score || 0)}`}>
              {scores.correctness.score || 0}/10
            </span>
          </div>
          <div className="cq-dim-bar-bg">
            <div
              className={`cq-dim-bar ${getScoreClass(scores.correctness.score || 0)}`}
              style={{ width: `${(scores.correctness.score || 0) * 10}%` }}
            />
          </div>
          {!compact && scores.correctness.notes && (
            <p className="cq-dim-notes">{scores.correctness.notes}</p>
          )}
        </div>
      )}

      {!compact && scores.detected_patterns && scores.detected_patterns.length > 0 && (
        <div className="cq-patterns">
          <strong>Detected Patterns:</strong>
          <div className="cq-pattern-tags">
            {scores.detected_patterns.map((p, i) => (
              <span key={i} className="cq-tag">{p}</span>
            ))}
          </div>
        </div>
      )}

      {!compact && scores.improvement_suggestions && scores.improvement_suggestions.length > 0 && (
        <div className="cq-improvements">
          <strong>Suggestions:</strong>
          <ul>
            {scores.improvement_suggestions.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}

      {scores.per_file && scores.per_file.length > 0 && (
        <div className="cq-per-file">
          <strong>Per-File Breakdown ({scores.files_analyzed || scores.per_file.length} files):</strong>
          <div className="cq-file-list">
            {scores.per_file.map((f, i) => (
              <div key={i} className="cq-file-item">
                <span className="cq-file-name">{f.file || f.filename}</span>
                <span className="cq-file-lang">{f.language}</span>
                <span className={`cq-file-score ${getScoreClass(f.scores?.overall_score || 0)}`}>
                  {(f.scores?.overall_score || 0).toFixed(1)}/10
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default CodeQualityResults;
