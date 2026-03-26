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
}) {
  const [analyses, setAnalyses] = useState([]);
  const [loadingId, setLoadingId] = useState(null);
  const [fetchError, setFetchError] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    fetchHistory();
  }, [refreshKey]);

  async function fetchHistory() {
    try {
      const res = await fetch(`${API_BASE_URL}/analysis-history?limit=50`);
      if (res.ok) {
        const data = await res.json();
        setAnalyses(data.analyses || []);
        setFetchError(false);
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

  const query = search.trim().toLowerCase();
  const filtered = query
    ? analyses.filter(
        (a) =>
          (a.candidate_name || "").toLowerCase().includes(query) ||
          (a.resume_filename || "").toLowerCase().includes(query) ||
          (a.target_role || "").toLowerCase().includes(query)
      )
    : analyses;

  const grouped = groupByDate(filtered);

  return (
    <div className="history-tab">
      <div className="history-tab-header">
        <div className="history-tab-title">
          <span className="history-tab-icon">{"\uD83D\uDD52"}</span>
          <h2>Analysis History</h2>
          <span className="history-tab-count">{analyses.length} records</span>
        </div>
        <button className="new-analysis-btn" onClick={onNewAnalysis}>
          + New Analysis
        </button>
      </div>

      <div className="history-search-bar">
        <svg className="history-search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
        <input
          type="text"
          className="history-search-input"
          placeholder="Search by name, role, or filename…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        {search && (
          <button className="history-search-clear" onClick={() => setSearch("")} aria-label="Clear search">
            ✕
          </button>
        )}
      </div>

      {fetchError ? (
        <p className="history-empty">Could not load history. Please check your connection.</p>
      ) : analyses.length === 0 ? (
        <p className="history-empty">No analyses yet. Run your first analysis to see it here.</p>
      ) : filtered.length === 0 ? (
        <p className="history-empty">No results for &ldquo;{search}&rdquo;.</p>
      ) : (
        <div className="history-grid">
          {grouped.map(([group, items]) => (
            <div key={group} className="history-group">
              <div className="history-group-header">
                <span className="history-group-label">{group}</span>
                <span className="history-group-count">{items.length}</span>
              </div>
              <div className="history-cards">
                {items.map((a) => (
                  <div
                    key={a.analysis_id}
                    className={`history-card ${activeAnalysisId === a.analysis_id ? "active" : ""} ${loadingId === a.analysis_id ? "loading" : ""}`}
                    onClick={() => handleSelect(a.analysis_id)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => e.key === "Enter" && handleSelect(a.analysis_id)}
                  >
                    <div className="history-card-top">
                      <span className="history-card-name">
                        {a.candidate_name || a.resume_filename || "Unknown"}
                      </span>
                      <span
                        className="history-card-score"
                        style={{ color: scoreColor(a.match_score) }}
                      >
                        {Math.round(a.match_score ?? 0)}%
                      </span>
                    </div>
                    <div className="history-card-role">{a.target_role || "—"}</div>
                    <div className="history-card-time">
                      {a.analyzed_at
                        ? new Date(a.analyzed_at * 1000).toLocaleString([], {
                            month: "short",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })
                        : "—"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

export default AnalysisHistory;
