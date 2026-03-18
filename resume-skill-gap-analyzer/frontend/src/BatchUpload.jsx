import { useState } from "react";
import Role from "./Role";
import RankingTable from "./RankingTable";
import Papa from "papaparse";
import { showToast } from "./Toast";
import "./cssFile/BatchUpload.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function BatchUpload() {
  const [files, setFiles] = useState([]);
  const [targetRole, setTargetRole] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleFilesChange = (e) => {
    const selected = Array.from(e.target.files);
    const valid = selected.filter((f) => {
      const name = f.name.toLowerCase();
      return name.endsWith(".pdf") || name.endsWith(".txt") || name.endsWith(".docx");
    });
    setFiles(valid);
    if (valid.length < selected.length) {
      setError(`${selected.length - valid.length} file(s) skipped (only .pdf/.docx/.txt allowed)`);
    } else {
      setError("");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (files.length === 0) { setError("Select at least one resume file."); return; }
    if (!targetRole) { setError("Select a target role."); return; }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    files.forEach((f) => formData.append("resume_files", f));
    formData.append("target_role", targetRole);

    try {
      const res = await fetch(`${API_BASE_URL}/analyze-batch`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Batch analysis failed");
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="batch-upload">
      <h2>Batch Resume Analysis</h2>
      <p className="batch-desc">Upload multiple resumes to rank candidates for a target role.</p>

      <form onSubmit={handleSubmit} className="batch-form">
        <div className="form-group">
          <label className="form-label">Resume Files</label>
          <input
            type="file"
            multiple
            accept=".pdf,.txt"
            onChange={handleFilesChange}
            className="file-input"
          />
          {files.length > 0 && (
            <p className="file-count">{files.length} file(s) selected</p>
          )}
        </div>

        <Role onTargetSet={setTargetRole} />

        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? `Analyzing ${files.length} resumes...` : "Analyze Batch"}
        </button>
      </form>

      {error && <div className="error-msg">{error}</div>}

      {loading && (
        <div className="batch-loading">
          <div className="spinner"></div>
          <p>Processing {files.length} resumes... This may take a moment.</p>
        </div>
      )}

      {result && (
        <div className="batch-results">
          <div className="batch-summary">
            <span className="batch-stat">Analyzed: <strong>{result.total_analyzed}</strong></span>
            <span className="batch-stat">Errors: <strong>{result.total_errors}</strong></span>
            <span className="batch-stat">Role: <strong>{result.target_role}</strong></span>
            <button className="export-btn export-csv" onClick={() => {
              const csvData = result.rankings.map((r, i) => ({
                Rank: r.rank || i + 1,
                Name: r.name || r.filename || "Unknown",
                "Match Score": `${Math.round(r.match_score)}%`,
                Confidence: `${Math.round(r.confidence)}%`,
              }));
              const csv = Papa.unparse(csvData);
              const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
              const url = URL.createObjectURL(blob);
              const link = document.createElement("a");
              link.href = url;
              link.download = `batch_${result.target_role}.csv`;
              link.click();
              URL.revokeObjectURL(url);
              showToast("Batch CSV exported!", "success");
            }}>Export CSV</button>
          </div>
          <RankingTable rankings={result.rankings} />
          {result.errors.length > 0 && (
            <div className="batch-errors">
              <h4>Errors</h4>
              {result.errors.map((e, i) => (
                <div key={i} className="error-item">{e.file}: {e.error}</div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default BatchUpload;
