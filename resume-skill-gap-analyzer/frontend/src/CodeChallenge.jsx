import { useState, useEffect, useRef, useCallback } from "react";
import CodeQualityResults from "./CodeQualityResults";
import "./cssFile/CodeChallenge.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

const LANGUAGES = [
  "Python", "JavaScript", "TypeScript", "Java", "C++", "C", "Go", "Rust", "Ruby",
];

function CodeChallenge({ targetRole, candidateId, analysisId }) {
  const [challenge, setChallenge] = useState(null);
  const [code, setCode] = useState("");
  const [language, setLanguage] = useState("Python");
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [startTime, setStartTime] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);

  // Timer
  useEffect(() => {
    if (startTime && !result) {
      timerRef.current = setInterval(() => {
        setElapsed(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [startTime, result]);

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const fetchChallenge = useCallback(async () => {
    setFetching(true);
    setError("");
    try {
      const params = targetRole ? `?target_role=${encodeURIComponent(targetRole)}` : "";
      const res = await fetch(`${API_BASE_URL}/code-challenge${params}`);
      if (!res.ok) throw new Error("Failed to fetch challenge");
      const data = await res.json();
      setChallenge(data);
      setCode("");
      setResult(null);
      setStartTime(Date.now());
      setElapsed(0);
    } catch (err) {
      setError("Failed to load challenge: " + err.message);
    } finally {
      setFetching(false);
    }
  }, [targetRole]);

  const submitCode = async () => {
    if (!code.trim()) {
      setError("Please write some code before submitting.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE_URL}/code-challenge/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code: code,
          language: language,
          challenge_id: challenge.id,
          candidate_id: candidateId || null,
          analysis_id: analysisId || null,
        }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Submission failed");
      }
      const data = await res.json();
      setResult(data.code_quality);
      if (timerRef.current) clearInterval(timerRef.current);
    } catch (err) {
      setError("Submission failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handle Tab key in textarea
  const handleKeyDown = (e) => {
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

  return (
    <div className="code-challenge">
      <div className="code-challenge-header">
        <h3>Code Challenge</h3>
        {!challenge && (
          <button
            className="cc-start-btn"
            onClick={fetchChallenge}
            disabled={fetching}
          >
            {fetching ? "Loading..." : "Start Challenge"}
          </button>
        )}
      </div>

      {!challenge && !fetching && (
        <p style={{ color: "var(--text-gray)", fontSize: "0.9rem" }}>
          Test your coding skills! Start a challenge to get a problem tailored to your target role.
          Your code will be analyzed for speed, complexity, flexibility, and quality.
        </p>
      )}

      {error && <div className="error-message" style={{ marginTop: 8 }}>{error}</div>}

      {challenge && (
        <div className="cc-problem">
          <div className="cc-problem-meta">
            <h4 style={{ margin: 0, fontSize: "1.05rem" }}>{challenge.title}</h4>
            <span className={`cc-difficulty cc-difficulty-${challenge.difficulty}`}>
              {challenge.difficulty}
            </span>
            {startTime && (
              <span className="cc-timer">
                {formatTime(elapsed)}
                {challenge.time_limit_minutes && ` / ${challenge.time_limit_minutes}:00`}
              </span>
            )}
          </div>

          <div className="cc-description">{challenge.description}</div>

          {challenge.examples && challenge.examples.length > 0 && (
            <div className="cc-examples">
              <strong>Examples:</strong>
              {challenge.examples.map((ex, i) => (
                <div key={i} className="cc-example">
                  <p><strong>Input:</strong> {ex.input}</p>
                  <p><strong>Output:</strong> {ex.output}</p>
                  {ex.explanation && <p><em>{ex.explanation}</em></p>}
                </div>
              ))}
            </div>
          )}

          {challenge.constraints && challenge.constraints.length > 0 && (
            <div className="cc-constraints">
              <strong>Constraints:</strong>
              <ul>
                {challenge.constraints.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </div>
          )}

          {!result && (
            <div className="cc-editor-section">
              <div className="cc-editor-toolbar">
                <span className="cc-editor-label">Language:</span>
                <select
                  className="cc-lang-select"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                >
                  {LANGUAGES.map((l) => (
                    <option key={l} value={l}>{l}</option>
                  ))}
                </select>
              </div>

              <textarea
                className="cc-code-editor"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={`# Write your ${language} solution here...\n\n`}
                spellCheck={false}
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="off"
              />

              {loading ? (
                <div className="cc-analyzing">
                  <div className="cc-spinner" />
                  <span>Analyzing your code...</span>
                </div>
              ) : (
                <button
                  className="cc-submit-btn"
                  onClick={submitCode}
                  disabled={!code.trim()}
                >
                  Submit & Analyze Code
                </button>
              )}
            </div>
          )}

          {result && (
            <>
              <CodeQualityResults
                scores={result}
                title="Your Code Analysis"
              />
              <button
                className="cc-new-btn"
                onClick={fetchChallenge}
                disabled={fetching}
              >
                Try Another Challenge
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default CodeChallenge;
