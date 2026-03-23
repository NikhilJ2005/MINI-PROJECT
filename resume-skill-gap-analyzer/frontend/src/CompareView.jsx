import { useState, useEffect } from "react";
import Role from "./Role";
import "./cssFile/CompareView.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function CompareView() {
  const [candidates, setCandidates] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [targetRole, setTargetRole] = useState("");
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchCandidates = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/candidates?limit=100`);
        if (res.ok) {
          const data = await res.json();
          setCandidates(data.candidates);
        }
      } catch (err) {
        console.error("Failed to load candidates:", err);
        setError("Failed to load candidates list.");
      }
    };
    fetchCandidates();
  }, []);

  const toggleCandidate = (id) => {
    setSelectedIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 5) return prev;
      return [...prev, id];
    });
  };

  const handleCompare = async () => {
    if (selectedIds.length < 2) { setError("Select at least 2 candidates."); return; }
    if (!targetRole) { setError("Select a target role."); return; }

    setLoading(true);
    setError("");
    setComparison(null);

    try {
      const res = await fetch(`${API_BASE_URL}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ candidate_ids: selectedIds, target_role: targetRole }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Comparison failed");
      }
      setComparison(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="compare-view">
      <h2>Compare Candidates</h2>
      <p className="compare-desc">Select 2–5 candidates and a target role to compare side-by-side.</p>

      <div className="compare-setup">
        <div className="candidate-picker">
          <h4>Select Candidates ({selectedIds.length}/5)</h4>
          <div className="candidate-chips">
            {candidates.map((c) => (
              <button
                key={c.id}
                className={`chip ${selectedIds.includes(c.id) ? "selected" : ""}`}
                onClick={() => toggleCandidate(c.id)}
              >
                {c.name || `#${c.id}`}
              </button>
            ))}
          </div>
        </div>

        <div className="role-selector">
          <Role onTargetSet={setTargetRole} />
        </div>

        <button className="submit-btn" onClick={handleCompare} disabled={loading}>
          {loading ? "Comparing..." : "Compare"}
        </button>
      </div>

      {error && <div className="error-msg">{error}</div>}

      {comparison && (
        <div className="comparison-results">
          <h3>Comparison: {comparison.target_role}</h3>

          <div className="comparison-cards">
            {comparison.candidates.map((c) => (
              <div className="compare-card" key={c.candidate_id}>
                <h4>{c.name || `Candidate #${c.candidate_id}`}</h4>
                {c.match_score == null ? (
                  <div className="compare-meta">
                    <span className="no-analysis">No analysis for this role</span>
                  </div>
                ) : (
                  <>
                    <div className="compare-score">
                      <span className={`score-badge ${getScoreClass(c.match_score)}`}>
                        {Math.round(c.match_score)}%
                      </span>
                    </div>
                    <div className="compare-meta">
                      <span>Confidence: {Math.round(c.confidence ?? 0)}%</span>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>

          <div className="skill-matrix-section">
            <h4>Skill Matrix</h4>
            <div className="matrix-wrapper">
              <table className="skill-matrix-table">
                <thead>
                  <tr>
                    <th>Skill</th>
                    {comparison.candidates.map((c) => (
                      <th key={c.candidate_id}>{c.name || `#${c.candidate_id}`}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {comparison.required_skills.map((skill) => (
                    <tr key={skill}>
                      <td className="skill-name">
                        {skill} <span className="req-badge">Required</span>
                      </td>
                      {comparison.candidates.map((c) => {
                        const status = comparison.skill_matrix[skill]?.[String(c.candidate_id)] || "missing";
                        return (
                          <td key={c.candidate_id}>
                            <span className={`status-dot ${status}`}>{statusIcon(status)}</span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                  {comparison.nice_to_have.map((skill) => (
                    <tr key={skill} className="nice-row">
                      <td className="skill-name">
                        {skill} <span className="nice-badge">Nice</span>
                      </td>
                      {comparison.candidates.map((c) => {
                        const status = comparison.skill_matrix[skill]?.[String(c.candidate_id)] || "missing";
                        return (
                          <td key={c.candidate_id}>
                            <span className={`status-dot ${status}`}>{statusIcon(status)}</span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function statusIcon(status) {
  switch (status) {
    case "strong": return "✔✔";
    case "claimed_only": return "✔";
    case "demonstrated_only": return "⚙";
    case "missing": return "✘";
    default: return "?";
  }
}

function getScoreClass(score) {
  if (score >= 75) return "excellent";
  if (score >= 50) return "good";
  if (score >= 25) return "fair";
  return "poor";
}

export default CompareView;
