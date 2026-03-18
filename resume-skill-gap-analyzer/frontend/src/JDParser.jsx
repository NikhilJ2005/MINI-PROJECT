import { useState } from "react";
import "./cssFile/JDParser.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function JDParser() {
  const [description, setDescription] = useState("");
  const [roleName, setRoleName] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!description.trim()) { setError("Paste a job description."); return; }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch(`${API_BASE_URL}/parse-job-description`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description, role_name: roleName }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Parsing failed");
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="jd-parser">
      <h2>Job Description Parser</h2>
      <p className="jd-desc">Paste a job description to auto-extract skills and create a custom role.</p>

      <form onSubmit={handleSubmit} className="jd-form">
        <div className="form-group">
          <label className="form-label">Role Name (optional)</label>
          <input
            type="text"
            className="text-input"
            placeholder="e.g. Senior ML Engineer"
            value={roleName}
            onChange={(e) => setRoleName(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Job Description</label>
          <textarea
            className="jd-textarea"
            rows={12}
            placeholder="Paste the full job description here..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            maxLength={50000}
          />
        </div>

        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? "Parsing..." : "Parse Job Description"}
        </button>
      </form>

      {error && <div className="error-msg">{error}</div>}

      {result && (
        <div className="jd-results">
          <h3>{result.role_name}</h3>
          <p className="skills-found">{result.total_skills_found} skills extracted</p>
          {result.added_to_roles && (
            <p className="added-notice">This role has been added and is now available for analysis!</p>
          )}

          <div className="skills-columns">
            <div className="skills-col">
              <h4>Required Skills ({result.required_skills.length})</h4>
              <div className="skill-tags">
                {result.required_skills.map((s, i) => (
                  <span className="skill-tag required" key={i}>{s}</span>
                ))}
              </div>
            </div>
            <div className="skills-col">
              <h4>Nice to Have ({result.nice_to_have.length})</h4>
              <div className="skill-tags">
                {result.nice_to_have.map((s, i) => (
                  <span className="skill-tag nice" key={i}>{s}</span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default JDParser;
