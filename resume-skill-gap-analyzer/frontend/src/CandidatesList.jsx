import { useState, useEffect, useMemo } from "react";
import { useUserRole } from "./UserRoleContext";
import NotAuthorized from "./NotAuthorized";
import Papa from "papaparse";
import { showToast } from "./Toast";
import "./cssFile/CandidatesList.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function CandidatesList() {
  const { userRole } = useUserRole();
  const [candidates, setCandidates] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortKey, setSortKey] = useState("id");
  const [sortDir, setSortDir] = useState("asc");

  const fetchCandidates = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/candidates?limit=100`);
      if (!res.ok) throw new Error("Failed to load candidates");
      const data = await res.json();
      setCandidates(data.candidates);
      setTotal(data.total);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchCandidates(); }, []);

  const viewCandidate = async (id) => {
    setSelected(id);
    try {
      const res = await fetch(`${API_BASE_URL}/candidates/${id}`);
      if (!res.ok) throw new Error("Failed to load candidate");
      setDetail(await res.json());
    } catch (err) {
      setError(err.message);
    }
  };

  const deleteCandidate = async (id) => {
    if (!confirm("Delete this candidate?")) return;
    try {
      const res = await fetch(`${API_BASE_URL}/candidates/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Delete failed");
      setCandidates((prev) => prev.filter((c) => c.id !== id));
      setTotal((prev) => prev - 1);
      if (selected === id) { setSelected(null); setDetail(null); }
    } catch (err) {
      setError(err.message);
    }
  };

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const filtered = useMemo(() => {
    let list = candidates;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(
        (c) =>
          (c.name || "").toLowerCase().includes(q) ||
          (c.email || "").toLowerCase().includes(q) ||
          (c.github_username || "").toLowerCase().includes(q)
      );
    }
    list = [...list].sort((a, b) => {
      const aVal = a[sortKey] ?? "";
      const bVal = b[sortKey] ?? "";
      if (typeof aVal === "number" && typeof bVal === "number") return sortDir === "asc" ? aVal - bVal : bVal - aVal;
      return sortDir === "asc" ? String(aVal).localeCompare(String(bVal)) : String(bVal).localeCompare(String(aVal));
    });
    return list;
  }, [candidates, searchQuery, sortKey, sortDir]);

  const handleExportCSV = () => {
    const csvData = candidates.map((c) => ({
      ID: c.id,
      Name: c.name || "",
      Email: c.email || "",
      GitHub: c.github_username || "",
    }));
    const csv = Papa.unparse(csvData);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "candidates.csv";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast("CSV exported!", "success");
  };

  const sortIcon = (key) => (sortKey === key ? (sortDir === "asc" ? " ▲" : " ▼") : "");

  if (userRole !== "recruiter") {
    return <NotAuthorized message="Recruiter access required" />;
  }

  if (loading) {
    return (
      <div className="candidates-page">
        <h2>Candidates</h2>
        <div className="skeleton-table">
          {[...Array(5)].map((_, i) => (
            <div className="skeleton-row" key={i}><div className="skeleton-cell" /><div className="skeleton-cell wide" /><div className="skeleton-cell wide" /><div className="skeleton-cell" /></div>
          ))}
        </div>
      </div>
    );
  }
  if (error) return <div className="error-msg">{error}</div>;

  return (
    <div className="candidates-page">
      <div className="candidates-header">
        <h2>Candidates ({total})</h2>
        {candidates.length > 0 && (
          <button className="export-btn export-csv" onClick={handleExportCSV}>Export CSV</button>
        )}
      </div>

      {!detail && candidates.length > 0 && (
        <input
          type="text"
          className="search-bar"
          placeholder="Search by name, email, or GitHub..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      )}

      {detail && (
        <div className="candidate-detail-panel">
          <button className="back-btn" onClick={() => { setSelected(null); setDetail(null); }}>
            &larr; Back to list
          </button>
          <h3>{detail.candidate.name || "Unknown"}</h3>
          <div className="detail-info">
            {detail.candidate.email && <p>Email: {detail.candidate.email}</p>}
            {detail.candidate.github_username && <p>GitHub: @{detail.candidate.github_username}</p>}
            {detail.candidate.education && <p>Education: {detail.candidate.education}</p>}
            {detail.candidate.extracted_skills && (
              <div className="skills-list">
                <strong>Extracted Skills:</strong>
                <div className="skill-tags">
                  {(() => {
                    try {
                      return typeof detail.candidate.extracted_skills === "string"
                        ? JSON.parse(detail.candidate.extracted_skills)
                        : (detail.candidate.extracted_skills || []);
                    } catch { return []; }
                  })().map((s, i) => (
                    <span className="skill-tag" key={i}>{s}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
          {detail.analyses && detail.analyses.length > 0 && (
            <div className="analyses-section">
              <h4>Previous Analyses</h4>
              {detail.analyses.map((a, i) => (
                <div className="analysis-card" key={i}>
                  <div className="analysis-header">
                    <span className="analysis-role">{a.target_role}</span>
                    <span className={`score-badge ${getScoreClass(a.match_score)}`}>
                      {Math.round(a.match_score ?? 0)}%
                    </span>
                  </div>
                  <div className="analysis-meta">
                    Confidence: {Math.round(a.confidence ?? 0)}% | Gap: {Math.round(a.gap_score ?? 0)}%
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {!detail && (
        <div className="candidates-list">
          {candidates.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">📋</div>
              <h3>No candidates yet</h3>
              <p>Upload and analyze a resume to see candidates here.</p>
            </div>
          ) : (
            <table className="candidates-table">
              <thead>
                <tr>
                  <th className="sortable" onClick={() => handleSort("id")}>ID{sortIcon("id")}</th>
                  <th className="sortable" onClick={() => handleSort("name")}>Name{sortIcon("name")}</th>
                  <th className="sortable" onClick={() => handleSort("email")}>Email{sortIcon("email")}</th>
                  <th>GitHub</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((c) => (
                  <tr key={c.id}>
                    <td>{c.id}</td>
                    <td>{c.name || "—"}</td>
                    <td>{c.email || "—"}</td>
                    <td>{c.github_username ? `@${c.github_username}` : "—"}</td>
                    <td className="action-cell">
                      <button className="view-btn" onClick={() => viewCandidate(c.id)}>View</button>
                      <button className="delete-btn" onClick={() => deleteCandidate(c.id)}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}

function getScoreClass(score) {
  if (score >= 75) return "excellent";
  if (score >= 50) return "good";
  if (score >= 25) return "fair";
  return "poor";
}

export default CandidatesList;
