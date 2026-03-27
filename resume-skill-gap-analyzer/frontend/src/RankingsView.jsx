import { useState } from "react";
import { useUserRole } from "./UserRoleContext";
import NotAuthorized from "./NotAuthorized";
import Role from "./Role";
import RankingTable from "./RankingTable";
import Papa from "papaparse";
import { showToast } from "./Toast";
import "./cssFile/RankingsView.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function RankingsView() {
  const { userRole } = useUserRole();
  const [targetRole, setTargetRole] = useState("");
  const [rankings, setRankings] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchRankings = async (role) => {
    if (!role) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE_URL}/rankings/${encodeURIComponent(role)}`);
      if (!res.ok) throw new Error("Failed to load rankings");
      const data = await res.json();
      setRankings(data.rankings);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRoleChange = (role) => {
    setTargetRole(role);
    if (role) fetchRankings(role);
  };

  const handleExportCSV = () => {
    if (!rankings) return;
    const csvData = rankings.map((r, i) => ({
      Rank: i + 1,
      Name: r.name || "Unknown",
      "Match Score": `${Math.round(r.match_score ?? 0)}%`,
      Confidence: `${Math.round(r.confidence ?? 0)}%`,
      "Resume Skills": r.resume_skills_count || 0,
      "GitHub Skills": r.github_skills_count || 0,
      Missing: r.missing_count || 0,
    }));
    const csv = Papa.unparse(csvData);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `rankings_${targetRole}.csv`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast("Rankings CSV exported!", "success");
  };

  if (userRole !== "recruiter") {
    return <NotAuthorized message="Recruiter access required" />;
  }

  return (
    <div className="rankings-view">
      <div className="rankings-header">
        <div>
          <h2>Candidate Rankings</h2>
          <p className="rankings-desc">Select a role to see ranked candidates from all previous analyses.</p>
        </div>
        {rankings && rankings.length > 0 && (
          <button className="export-btn export-csv" onClick={handleExportCSV}>Export CSV</button>
        )}
      </div>

      <div className="role-selector">
        <Role onTargetSet={handleRoleChange} />
      </div>

      {loading && <div className="loading-msg">Loading rankings...</div>}
      {error && <div className="error-msg">{error}</div>}

      {!loading && !rankings && !error && (
        <div className="empty-state">
          <div className="empty-icon">🏆</div>
          <h3>Select a role</h3>
          <p>Choose a target role above to see candidate rankings.</p>
        </div>
      )}

      {rankings && (
        <RankingTable
          rankings={rankings.map((r, i) => ({
            rank: r.rank || i + 1,
            candidate_id: r.candidate_id,
            name: r.name,
            match_score: r.match_score,
            composite_score: r.composite_score,
            confidence: r.confidence,
            resume_skills_count: r.resume_skills_count || 0,
            github_skills_count: r.github_skills_count || 0,
            missing_count: r.missing_count || 0,
          }))}
        />
      )}
    </div>
  );
}

export default RankingsView;
