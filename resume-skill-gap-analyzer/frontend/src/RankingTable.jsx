import "./cssFile/RankingTable.css";

function RankingTable({ rankings }) {
  if (!rankings || rankings.length === 0) {
    return <p className="no-data">No candidates ranked yet.</p>;
  }

  return (
    <div className="ranking-table-wrapper">
      <table className="ranking-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Name</th>
            <th>Composite</th>
            <th>Match Score</th>
            <th>Confidence</th>
            <th>Resume Skills</th>
            <th>GitHub Skills</th>
            <th>Missing</th>
          </tr>
        </thead>
        <tbody>
          {rankings.map((r, i) => (
            <tr key={r.candidate_id || i} className={i === 0 ? "top-rank" : ""}>
              <td className="rank-cell">#{r.rank || i + 1}</td>
              <td className="name-cell">{r.name || r.filename || "Unknown"}</td>
              <td>
                <span className={`score-badge ${getScoreClass(r.composite_score ?? r.match_score)}`}>
                  {Math.round(r.composite_score ?? r.match_score)}%
                </span>
              </td>
              <td>{Math.round(r.match_score ?? 0)}%</td>
              <td>{Math.round(r.confidence ?? 0)}%</td>
              <td>{r.resume_skills_count}</td>
              <td>{r.github_skills_count}</td>
              <td>{r.missing_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function getScoreClass(score) {
  if (score >= 75) return "excellent";
  if (score >= 50) return "good";
  if (score >= 25) return "fair";
  return "poor";
}

export default RankingTable;
