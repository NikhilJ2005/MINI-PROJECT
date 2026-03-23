import { useState, useEffect, memo } from "react";
import "./cssFile/AnalysisHistory.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function groupByDate(analyses) {
  const now = new Date();
  const today = now.toDateString();
  const yesterday = new Date(now - 86400000).toDateString();
  const weekAgo = new Date(now - 7 * 86400000).getTime();

  const groups = { Today: [], Yesterday: [], "This Week": [], Earlier: [] };

  for (const a of analyses) {
    if (!a.analyzed_at) { groups.Earlier.push(a); continue; }
    const d = new Date(a.analyzed_at * 1000);
    const ds = d.toDateString();
    if (ds === today) groups.Today.push(a);
    else if (ds === yesterday) groups.Yesterday.push(a);
    else if (d.getTime() > weekAgo) groups["This Week"].push(a);
    else groups.Earlier.push(a);
  }

  return Object.entries(groups).filter(([, items]) => items.length > 0);
}

function scoreColor(score) {
  if (score >= 75) return "#22c55e";
  if (score >= 50) return "#eab308";
  return "#ef4444";
}

const AnalysisHistory = memo(function AnalysisHistory({
  refreshKey,
  activeAnalysisId,
  onSelect,
  onNewAnalysis,
  isOpen,
  onToggle,
}) {
  const [analyses, setAnalyses] = useState([]);
  const [loadingId, setLoadingId] = useState(null);
  const [fetchError, setFetchError] = useState(false);

  useEffect(() => {
    fetchHistory();
  }, [refreshKey]);

  async function fetchHistory() {
    try {
      const res = await fetch(`${API_BASE_URL}/analysis-history?limit=50`);
      if (res.ok) {
        const data = await res.json();
        setAnalyses(data.analyses || []);
      }
    } catch {
      setFetchError(true);
    }
  }

  async function handleSelect(analysisId) {
    if (loadingId) return;
    setLoadingId(analysisId);
    try {
      const res = await fetch(`${API_BASE_URL}/analysis/${analysisId}`);
      if (res.ok) {
        const report = await res.json();
        onSelect(report, analysisId);
      }
    } catch (err) {
      console.error("Failed to load analysis:", err);
    } finally {
      setLoadingId(null);
    }
  }

  const grouped = groupByDate(analyses);

  return (
    <>
      <button
        className={`history-toggle ${isOpen ? "open" : ""}`}
        onClick={onToggle}
        title={isOpen ? "Hide history" : "Show history"}
      >
        {isOpen ? "\u2039" : "\u203A"}
      </button>

      <aside className={`history-sidebar ${isOpen ? "open" : "closed"}`}>
        <div className="history-header">
          <h3>History</h3>
          <button className="new-analysis-btn" onClick={onNewAnalysis}>
            + New
          </button>
        </div>

        {fetchError ? (
          <p className="history-empty">Could not load history.</p>
        ) : analyses.length === 0 ? (
          <p className="history-empty">No analyses yet.</p>
        ) : (
          <div className="history-list">
            {grouped.map(([group, items]) => (
              <div key={group}>
                <div className="history-group-header">{group}</div>
                {items.map((a) => (
                  <div
                    key={a.analysis_id}
                    className={`history-entry ${
                      activeAnalysisId === a.analysis_id ? "active" : ""
                    } ${loadingId === a.analysis_id ? "loading" : ""}`}
                    onClick={() => handleSelect(a.analysis_id)}
                  >
                    <div className="history-entry-name">
                      {a.candidate_name || a.resume_filename || "Unknown"}
                    </div>
                    <div className="history-entry-meta">
                      <span className="history-entry-role">{a.target_role}</span>
                      <span
                        className="history-entry-score"
                        style={{ color: scoreColor(a.match_score) }}
                      >
                        {Math.round(a.match_score ?? 0)}%
                      </span>
                    </div>
                    <div className="history-entry-time">
                      {a.analyzed_at
                        ? new Date(a.analyzed_at * 1000).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })
                        : "—"}
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </aside>
    </>
  );
});

export default AnalysisHistory;
