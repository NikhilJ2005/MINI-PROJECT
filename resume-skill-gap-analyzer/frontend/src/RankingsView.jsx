import { useState, useEffect } from "react";
import Role from "./Role";
import RankingTable from "./RankingTable";
import "./cssFile/RankingsView.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function RankingsView() {
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

  return (
    <div className="rankings-view">
      <h2>Candidate Rankings</h2>
      <p className="rankings-desc">Select a role to see ranked candidates from all previous analyses.</p>

      <div className="role-selector">
        <Role onTargetSet={handleRoleChange} />
      </div>

      {loading && <div className="loading-msg">Loading rankings...</div>}
      {error && <div className="error-msg">{error}</div>}

      {rankings && (
        <RankingTable
          rankings={rankings.map((r, i) => ({
            rank: i + 1,
            candidate_id: r.candidate_id,
            name: r.name,
            match_score: r.match_score,
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
