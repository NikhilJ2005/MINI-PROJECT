import { useState, useRef, useCallback } from "react";
import { useUserRole } from "./UserRoleContext";
import NotAuthorized from "./NotAuthorized";
import Role from "./Role";
import RankingTable from "./RankingTable";
import CodeQualityResults from "./CodeQualityResults";
import Papa from "papaparse";
import { showToast } from "./Toast";
import "./cssFile/BatchUpload.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function BatchUpload() {
  const { userRole } = useUserRole();
  const [files, setFiles] = useState([]);
  const [targetRole, setTargetRole] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [codeFiles, setCodeFiles] = useState([]);
  const [isCodeDragging, setIsCodeDragging] = useState(false);
  const fileInputRef = useRef(null);
  const codeFileInputRef = useRef(null);

  const validExtensions = [".pdf", ".txt", ".docx"];
  const codeExtensions = [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".cpp", ".c", ".rb", ".php", ".kt", ".swift"];

  const validateFiles = (fileList) => {
    const valid = [];
    const invalid = [];
    for (const f of fileList) {
      const dotIdx = f.name.lastIndexOf(".");
      const ext = dotIdx >= 0 ? f.name.toLowerCase().slice(dotIdx) : "";
      if (validExtensions.includes(ext)) {
        valid.push(f);
      } else {
        invalid.push(f.name);
      }
    }
    return { valid, invalid };
  };

  // Add new files WITHOUT overwriting existing ones
  const addFiles = useCallback((newFileList) => {
    const { valid, invalid } = validateFiles(Array.from(newFileList));

    if (invalid.length > 0) {
      setError(`${invalid.length} file(s) skipped (only .pdf/.docx/.txt): ${invalid.join(", ")}`);
    } else {
      setError("");
    }

    if (valid.length === 0) return;

    setFiles((prev) => {
      // Deduplicate by name+size to avoid accidental double-adds
      const existing = new Set(prev.map((f) => `${f.name}_${f.size}`));
      const unique = valid.filter((f) => !existing.has(`${f.name}_${f.size}`));
      const dupes = valid.length - unique.length;
      if (dupes > 0) {
        showToast(`${dupes} duplicate file(s) skipped`, "info");
      }
      if (unique.length > 0) {
        showToast(`${unique.length} file(s) added`, "success");
      }
      const combined = [...prev, ...unique];
      if (combined.length > 50) {
        setError("Maximum 50 resumes per batch. Extra files removed.");
        return combined.slice(0, 50);
      }
      return combined;
    });
  }, []);

  const handleFilesChange = (e) => {
    addFiles(e.target.files);
    // Reset input so the same file can be re-selected if needed
    e.target.value = "";
  };

  // Code files handlers
  const addCodeFiles = useCallback((newFileList) => {
    const arr = Array.from(newFileList);
    const valid = [];
    const invalid = [];
    for (const f of arr) {
      const dotIdx = f.name.lastIndexOf(".");
      const ext = dotIdx >= 0 ? f.name.toLowerCase().slice(dotIdx) : "";
      if (codeExtensions.includes(ext)) {
        valid.push(f);
      } else {
        invalid.push(f.name);
      }
    }
    if (invalid.length > 0) {
      showToast(`${invalid.length} non-code file(s) skipped`, "info");
    }
    if (valid.length === 0) return;
    setCodeFiles((prev) => {
      const existing = new Set(prev.map((f) => `${f.name}_${f.size}`));
      const unique = valid.filter((f) => !existing.has(`${f.name}_${f.size}`));
      if (unique.length > 0) showToast(`${unique.length} code file(s) added`, "success");
      return [...prev, ...unique].slice(0, 20);
    });
  }, []);

  const handleCodeFilesChange = (e) => {
    addCodeFiles(e.target.files);
    e.target.value = "";
  };

  const removeCodeFile = (index) => {
    setCodeFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleCodeDrop = (e) => {
    e.preventDefault();
    setIsCodeDragging(false);
    addCodeFiles(e.dataTransfer.files);
  };

  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const clearAllFiles = () => {
    setFiles([]);
    setError("");
  };

  const clearResults = () => {
    setResult(null);
    setError("");
    setFiles([]);
    setCodeFiles([]);
  };

  // Drag & drop handlers
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    addFiles(e.dataTransfer.files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
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
    codeFiles.forEach((f) => formData.append("code_files", f));

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
      showToast(`Batch analysis complete!`, "success");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const totalSize = files.reduce((sum, f) => sum + f.size, 0);

  if (userRole !== "recruiter") {
    return <NotAuthorized message="Recruiter access required" />;
  }

  return (
    <div className="batch-upload">
      <h2>Batch Resume Analysis</h2>
      <p className="batch-desc">Upload multiple resumes to rank candidates for a target role. Add files one by one or all at once — they accumulate without overwriting.</p>

      <form onSubmit={handleSubmit} className="batch-form">
        <div className="form-group">
          <label className="form-label">Resume Files</label>

          {/* Drag & drop zone */}
          <div
            className={`batch-drop-zone ${isDragging ? "dragging" : ""}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
          >
            <span className="drop-zone-icon">+</span>
            <p className="drop-zone-title">
              <strong>Drag & drop resumes here</strong>
            </p>
            <p className="drop-zone-subtitle">
              or click to browse — .pdf, .docx, .txt — add as many as you need
            </p>
          </div>

          <input
            type="file"
            multiple
            accept=".pdf,.txt,.docx"
            onChange={handleFilesChange}
            ref={fileInputRef}
            hidden
          />

          {/* File queue */}
          {files.length > 0 && (
            <div className="file-queue">
              <div className="file-queue-header">
                <span className="file-count-badge">{files.length} file(s) queued ({formatFileSize(totalSize)})</span>
                <button type="button" className="clear-all-btn" onClick={clearAllFiles}>
                  Clear All
                </button>
              </div>
              <div className="file-list">
                {files.map((f, i) => (
                  <div key={`${f.name}_${f.size}_${i}`} className="file-list-item">
                    <span className="file-list-icon">
                      {f.name.endsWith(".pdf") ? "\uD83D\uDCC4" : "\uD83D\uDCDD"}
                    </span>
                    <span className="file-list-name">{f.name}</span>
                    <span className="file-list-size">{formatFileSize(f.size)}</span>
                    <button
                      type="button"
                      className="file-remove-btn"
                      onClick={() => removeFile(i)}
                      title="Remove file"
                    >
                      x
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <Role onTargetSet={setTargetRole} />

        {/* Code Files Upload Section */}
        <div className="form-group">
          <label className="form-label">Code Files (Optional)</label>
          <p className="batch-code-desc">Upload code files to analyze coding quality for candidates in this batch.</p>
          <div
            className={`batch-drop-zone batch-code-drop-zone ${isCodeDragging ? "dragging" : ""}`}
            onDrop={handleCodeDrop}
            onDragOver={(e) => { e.preventDefault(); setIsCodeDragging(true); }}
            onDragLeave={() => setIsCodeDragging(false)}
            onClick={() => codeFileInputRef.current?.click()}
            role="button"
            tabIndex={0}
          >
            <span className="drop-zone-icon">&lt;/&gt;</span>
            <p className="drop-zone-title">
              <strong>Drag & drop code files here</strong>
            </p>
            <p className="drop-zone-subtitle">
              .py, .js, .ts, .java, .go, .rs, .cpp, .rb — max 20 files
            </p>
          </div>
          <input
            type="file"
            multiple
            accept=".py,.js,.ts,.jsx,.tsx,.java,.go,.rs,.cpp,.c,.rb,.php,.kt,.swift"
            onChange={handleCodeFilesChange}
            ref={codeFileInputRef}
            hidden
          />
          {codeFiles.length > 0 && (
            <div className="file-queue">
              <div className="file-queue-header">
                <span className="file-count-badge">{codeFiles.length} code file(s)</span>
                <button type="button" className="clear-all-btn" onClick={() => setCodeFiles([])}>
                  Clear All
                </button>
              </div>
              <div className="file-list">
                {codeFiles.map((f, i) => (
                  <div key={`code_${f.name}_${i}`} className="file-list-item">
                    <span className="file-list-icon">{"\uD83D\uDCBB"}</span>
                    <span className="file-list-name">{f.name}</span>
                    <span className="file-list-size">{formatFileSize(f.size)}</span>
                    <button
                      type="button"
                      className="file-remove-btn"
                      onClick={() => removeCodeFile(i)}
                      title="Remove file"
                    >
                      x
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <button type="submit" className="submit-btn" disabled={loading || files.length === 0}>
          {loading ? `Analyzing ${files.length} resumes...` : `Analyze ${files.length || ""} Resume${files.length !== 1 ? "s" : ""}`}
        </button>
      </form>

      {error && <div className="error-msg">{error}</div>}

      {loading && (
        <div className="batch-loading">
          <div className="spinner"></div>
          <p>Processing {files.length} resumes... This may take a moment.</p>
          <div className="batch-progress-bar">
            <div className="batch-progress-fill"></div>
          </div>
        </div>
      )}

      {result && (
        <div className="batch-results">
          <div className="batch-results-header">
            <h3 className="batch-results-title">Batch Results</h3>
            <button type="button" className="clear-results-btn" onClick={clearResults}>
              Clear Results
            </button>
          </div>
          <div className="batch-summary">
            <span className="batch-stat">Analyzed: <strong>{result.total_analyzed}</strong></span>
            <span className="batch-stat">Errors: <strong>{result.total_errors}</strong></span>
            <span className="batch-stat">Role: <strong>{result.target_role}</strong></span>
            <button className="export-btn export-csv" onClick={() => {
              const csvData = result.rankings.map((r, i) => ({
                Rank: r.rank || i + 1,
                Name: r.name || r.filename || "Unknown",
                Email: r.email || "",
                "Match Score": `${Math.round(r.match_score ?? 0)}%`,
                "Gap Score": `${Math.round(r.gap_score ?? 0)}%`,
                Confidence: `${Math.round(r.confidence ?? 0)}%`,
                "Resume Skills": r.resume_skills_count || 0,
                "GitHub Skills": r.github_skills_count || 0,
                "Missing Skills": r.missing_count || 0,
                "Missing Skills List": (r.missing_required || []).join("; "),
              }));
              const csv = Papa.unparse(csvData);
              const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
              const url = URL.createObjectURL(blob);
              const link = document.createElement("a");
              link.href = url;
              link.download = `batch_${result.target_role}_${new Date().toISOString().slice(0, 10)}.csv`;
              link.click();
              setTimeout(() => URL.revokeObjectURL(url), 1000);
              showToast("Batch CSV exported!", "success");
            }}>Export CSV</button>
            <button
              type="button"
              className="clear-results-btn"
              onClick={() => {
                setResult(null);
                setError("");
                setFiles([]);
                setCodeFiles([]);
                showToast("Batch results cleared.", "info");
              }}
            >
              Clear Results
            </button>
          </div>
          {/* AI Executive Report for Recruiters */}
          {result.ai_executive_report && (
            <div className="batch-ai-report">
              <h4>AI Talent Pool Assessment</h4>
              {result.ai_executive_report.pool_quality && (
                <div className={`pool-quality quality-${result.ai_executive_report.pool_quality.toLowerCase()}`}>
                  Pool Quality: <strong>{result.ai_executive_report.pool_quality}</strong>
                </div>
              )}
              {result.ai_executive_report.summary && (
                <p className="pool-summary">{result.ai_executive_report.summary}</p>
              )}
              {result.ai_executive_report.top_pick_rationale && (
                <p className="top-pick"><strong>Top Pick:</strong> {result.ai_executive_report.top_pick_rationale}</p>
              )}
              {result.ai_executive_report.common_gaps?.length > 0 && (
                <p className="common-gaps"><strong>Common Gaps:</strong> {result.ai_executive_report.common_gaps.join(", ")}</p>
              )}
              {result.ai_executive_report.hiring_advice && (
                <p className="hiring-advice"><strong>Advice:</strong> {result.ai_executive_report.hiring_advice}</p>
              )}
            </div>
          )}

          {/* Code Quality Analysis Results */}
          {result.code_quality && result.code_quality.aggregate && (
            <CodeQualityResults
              scores={result.code_quality.aggregate}
              title="Batch Code Quality Analysis"
            />
          )}

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
