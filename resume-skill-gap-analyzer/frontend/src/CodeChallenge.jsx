import { useState, useEffect, useRef, useCallback } from "react";
import "./cssFile/CodeChallenge.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

const SUPPORTED_LANGUAGES = [
  { label: "Python",     value: "python" },
  { label: "JavaScript", value: "javascript" },
  { label: "Java",       value: "java" },
  { label: "C",          value: "c" },
  { label: "C++",        value: "cpp" },
];

const DEFAULT_PROBLEM_ID = "valid_parentheses";

const VERDICT_CLASS = {
  "Accepted":      "cc-verdict-accepted",
  "Wrong Answer":  "cc-verdict-wrong",
  "Compile Error": "cc-verdict-compile",
  "Runtime Error": "cc-verdict-runtime",
  "Time Limit":    "cc-verdict-tle",
};

function CodeChallenge({ targetRole, candidateId, analysisId }) {
  const [problem, setProblem] = useState(null);
  const [problems, setProblems] = useState([]);
  const [selectedProblemId, setSelectedProblemId] = useState(DEFAULT_PROBLEM_ID);
  const [language, setLanguage] = useState("python");
  const [code, setCode] = useState("");
  const [loadingProblem, setLoadingProblem] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [startTime, setStartTime] = useState(null);
  const timerRef = useRef(null);

  // Timer
  useEffect(() => {
    if (startTime && !(result && result.ok)) {
      timerRef.current = setInterval(() => {
        setElapsed(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [startTime, result]);

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  // Load problem list on mount
  useEffect(() => {
    fetch(`${API_BASE_URL}/challenge/problems`)
      .then((r) => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
      .then((data) => setProblems(data))
      .catch((err) => {
        setError("Could not load problem list. Please try again later.");
        console.error("Failed to fetch challenge problems:", err);
      });
  }, []);

  // Load selected problem
  const fetchProblem = useCallback(async (id) => {
    setLoadingProblem(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API_BASE_URL}/challenge/problem/${id}`);
      if (!res.ok) throw new Error("Failed to fetch problem");
      const data = await res.json();
      setProblem(data);
      setCode(data.starter_templates?.[language] || "");
      setStartTime(Date.now());
      setElapsed(0);
    } catch (err) {
      setError("Failed to load problem: " + err.message);
    } finally {
      setLoadingProblem(false);
    }
  }, [language]);

  // Reload starter template when language changes.
  // `problem` is intentionally omitted: we only want this to fire on language
  // change, not every time problem is re-fetched (fetchProblem already sets code).
  useEffect(() => {
    if (problem) {
      setCode(problem.starter_templates?.[language] || "");
      setResult(null);
    }
  }, [language]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleProblemChange = (id) => {
    setSelectedProblemId(id);
    fetchProblem(id);
  };

  const handleTabKey = (e) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const start = e.target.selectionStart;
      const end = e.target.selectionEnd;
      const val = e.target.value;
      setCode(val.substring(0, start) + "    " + val.substring(end));
      setTimeout(() => {
        e.target.selectionStart = e.target.selectionEnd = start + 4;
      }, 0);
    }
  };

  const submitCode = async (mode) => {
    if (!code.trim()) {
      setError("Please write some code before submitting.");
      return;
    }
    if (!problem) return;
    setSubmitting(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API_BASE_URL}/challenge/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          problem_id: problem.id,
          language,
          code,
          mode,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail || "Submission failed.");
        return;
      }
      setResult(data);
      if (data.ok && timerRef.current) clearInterval(timerRef.current);
    } catch (err) {
      setError("Network error: " + err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="code-challenge">
      <div className="code-challenge-header">
        <h3>Code Challenge</h3>
        {!problem && !loadingProblem && (
          <button
            className="cc-start-btn"
            onClick={() => fetchProblem(selectedProblemId)}
            disabled={loadingProblem}
          >
            Start Challenge
          </button>
        )}
        {problem && (
          <span className="cc-timer">{formatTime(elapsed)}</span>
        )}
      </div>

      {!problem && !loadingProblem && (
        <div className="cc-problem-picker">
          <p style={{ color: "var(--text-gray)", fontSize: "0.9rem" }}>
            Solve function-only coding problems in Python, JavaScript, Java, C, or C++.
            Write only the function — no stdin required.
          </p>
          {problems.length > 1 && (
            <div className="cc-picker-row">
              <label className="cc-editor-label">Problem:</label>
              <select
                className="cc-lang-select"
                value={selectedProblemId}
                onChange={(e) => setSelectedProblemId(e.target.value)}
              >
                {problems.map((p) => (
                  <option key={p.id} value={p.id}>{p.title} ({p.difficulty})</option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}

      {loadingProblem && (
        <div className="cc-analyzing">
          <div className="cc-spinner" />
          <span>Loading problem…</span>
        </div>
      )}

      {error && <div className="error-message" style={{ marginTop: 8 }}>{error}</div>}

      {problem && (
        <div className="cc-problem">
          {/* Problem header */}
          <div className="cc-problem-meta">
            <h4 style={{ margin: 0, fontSize: "1.05rem" }}>{problem.title}</h4>
            <span className={`cc-difficulty cc-difficulty-${problem.difficulty}`}>
              {problem.difficulty}
            </span>
          </div>

          {/* Problem statement */}
          <div className="cc-description">{problem.prompt}</div>

          {/* Examples */}
          {problem.examples && problem.examples.length > 0 && (
            <div className="cc-examples">
              <strong>Examples:</strong>
              {problem.examples.map((ex, i) => (
                <div key={i} className="cc-example">
                  <p><strong>Input:</strong> {ex.input}</p>
                  <p><strong>Output:</strong> {ex.output}</p>
                  {ex.explanation && <p><em>{ex.explanation}</em></p>}
                </div>
              ))}
            </div>
          )}

          {/* Constraints */}
          {problem.constraints && problem.constraints.length > 0 && (
            <div className="cc-constraints">
              <strong>Constraints:</strong>
              <ul>
                {problem.constraints.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Editor */}
          <div className="cc-editor-section">
            <div className="cc-editor-toolbar">
              <span className="cc-editor-label">Language:</span>
              <select
                className="cc-lang-select"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                disabled={submitting}
              >
                {SUPPORTED_LANGUAGES.map((l) => (
                  <option key={l.value} value={l.value}>{l.label}</option>
                ))}
              </select>

              {problems.length > 1 && (
                <>
                  <span className="cc-editor-label" style={{ marginLeft: 16 }}>Problem:</span>
                  <select
                    className="cc-lang-select"
                    value={problem.id}
                    onChange={(e) => handleProblemChange(e.target.value)}
                    disabled={submitting}
                  >
                    {problems.map((p) => (
                      <option key={p.id} value={p.id}>{p.title}</option>
                    ))}
                  </select>
                </>
              )}
            </div>

            <textarea
              className="cc-code-editor"
              value={code}
              onChange={(e) => setCode(e.target.value)}
              onKeyDown={handleTabKey}
              spellCheck={false}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              disabled={submitting}
            />

            {submitting ? (
              <div className="cc-analyzing">
                <div className="cc-spinner" />
                <span>Running tests…</span>
              </div>
            ) : (
              <div className="cc-btn-row">
                <button
                  className="cc-sample-btn"
                  onClick={() => submitCode("sample")}
                  disabled={!code.trim()}
                  title="Run only the visible sample test cases"
                >
                  ▶ Run Sample Tests
                </button>
                <button
                  className="cc-submit-btn"
                  onClick={() => submitCode("all")}
                  disabled={!code.trim()}
                  title="Submit against all test cases including hidden ones"
                >
                  ✔ Submit
                </button>
              </div>
            )}
          </div>

          {/* Result panel */}
          {result && (
            <div className="cc-result-panel">
              <div className="cc-result-header">
                <span className={`cc-verdict ${VERDICT_CLASS[result.verdict] || "cc-verdict-wrong"}`}>
                  {result.verdict}
                </span>
                <span className="cc-passed-count">
                  {result.passed} / {result.total} passed
                </span>
                {result.runtime_ms > 0 && (
                  <span className="cc-runtime">{result.runtime_ms.toFixed(0)} ms</span>
                )}
              </div>

              {/* Failed cases */}
              {result.failed_cases && result.failed_cases.length > 0 && (
                <div className="cc-failed-list">
                  <strong>Failed cases:</strong>
                  {result.failed_cases.map((fc, i) => (
                    <div key={i} className="cc-failed-case">
                      <span className="cc-fc-label">Case #{fc.index + 1}</span>
                      <span>Input: <code>{JSON.stringify(fc.input)}</code></span>
                      <span>Expected: <code>{JSON.stringify(fc.expected)}</code></span>
                      <span>Got: <code>{JSON.stringify(fc.actual)}</code></span>
                      {fc.error && <span className="cc-fc-error">{fc.error}</span>}
                    </div>
                  ))}
                </div>
              )}

              {/* stderr / compile errors */}
              {result.stderr && result.stderr.trim() && (
                <div className="cc-error-panel">
                  <strong>Error output:</strong>
                  <pre className="cc-error-pre">{result.stderr}</pre>
                </div>
              )}

              {/* stdout (debug output) */}
              {result.stdout && result.stdout.trim() && (
                <div className="cc-stdout-panel">
                  <strong>Output:</strong>
                  <pre className="cc-stdout-pre">{result.stdout}</pre>
                </div>
              )}

              <button
                className="cc-new-btn"
                onClick={() => { setResult(null); setStartTime(Date.now()); setElapsed(0); }}
                style={{ marginTop: 12 }}
              >
                Try Again
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default CodeChallenge;
