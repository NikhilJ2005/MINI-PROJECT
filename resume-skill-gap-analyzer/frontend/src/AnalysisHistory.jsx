import { useState, useEffect, useRef, memo } from "react";
import "./cssFile/AnalysisHistory.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function relativeTime(timestamp) {
  if (!timestamp) return "—";
  const now = Date.now();
  const then = timestamp * 1000;
  const diff = now - then;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days}d ago`;
  return new Date(then).toLocaleDateString([], { month: "short", day: "numeric" });
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
  const dropdownRef = useRef(null);

  useEffect(() => {
    fetchHistory();
  }, [refreshKey]);

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;
    function handleClick(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        onToggle();
      }
    }
    function handleEsc(e) {
      if (e.key === "Escape") onToggle();
    }
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleEsc);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleEsc);
    };
  }, [isOpen, onToggle]);

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
        onToggle(); // close dropdown after selection
      }
    } catch (err) {
      console.error("Failed to load analysis:", err);
    } finally {
      setLoadingId(null);
    }
  }

  if (!isOpen) return null;

  return (
    <>
      <div className="history-backdrop" />
      <div className="history-dropdown" ref={dropdownRef}>
        <div className="history-dropdown-header">
          <h3>Recent Analyses</h3>
          <div className="history-dropdown-actions">
            <button className="new-analysis-btn" onClick={() => { onNewAnalysis(); onToggle(); }}>
              + New Analysis
            </button>
            <button className="history-close-btn" onClick={onToggle}>
              ✕
            </button>
          </div>
        </div>

        {fetchError ? (
          <p className="history-empty">Could not load history.</p>
        ) : analyses.length === 0 ? (
          <p className="history-empty">No analyses yet. Upload a resume to get started.</p>
        ) : (
          <div className="history-grid">
            {analyses.map((a) => (
              <div
                key={a.analysis_id}
                className={`history-card ${
                  activeAnalysisId === a.analysis_id ? "active" : ""
                } ${loadingId === a.analysis_id ? "loading" : ""}`}
                onClick={() => handleSelect(a.analysis_id)}
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
                <div className="history-card-role">{a.target_role}</div>
                <div className="history-card-time">{relativeTime(a.analyzed_at)}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
});

export default AnalysisHistory;
