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
  const [testResults, setTestResults] = useState(null);
  const [runningTests, setRunningTests] = useState(false);
  const [activeLeftTab, setActiveLeftTab] = useState("description");
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
      setTestResults(null);
      setActiveLeftTab("description");
      setStartTime(Date.now());
      setElapsed(0);
    } catch (err) {
      setError("Failed to load challenge: " + err.message);
    } finally {
      setFetching(false);
    }
  }, [targetRole]);

  const runTests = async () => {
    if (!code.trim()) {
      setError("Please write some code before running tests.");
      return;
    }
    setRunningTests(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE_URL}/code-challenge/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code: code,
          language: language,
          challenge_id: challenge.id,
        }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Run failed");
      }
      const data = await res.json();
      setTestResults(data.results);
      setActiveLeftTab("testcases");
    } catch (err) {
      setError("Run failed: " + err.message);
    } finally {
      setRunningTests(false);
    }
  };

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
      setActiveLeftTab("results");
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

  // Not started state
  if (!challenge && !fetching) {
    return (
      <div className="code-challenge">
        <div className="cc-start-screen">
          <div className="cc-start-icon">&lt;/&gt;</div>
          <h3>Code Challenge</h3>
          <p>Test your coding skills with a problem tailored to your target role.
            Your code will be analyzed for speed, complexity, flexibility, and quality.</p>
          <button className="cc-start-btn" onClick={fetchChallenge} disabled={fetching}>
            {fetching ? "Loading..." : "Start Challenge"}
          </button>
        </div>
      </div>
    );
  }

  if (fetching) {
    return (
      <div className="code-challenge">
        <div className="cc-start-screen">
          <div className="cc-spinner" />
          <p>Loading challenge...</p>
        </div>
      </div>
    );
  }

  const passedCount = testResults ? testResults.filter((t) => t.passed).length : 0;
  const totalTests = testResults ? testResults.length : 0;

  return (
    <div className="code-challenge cc-leetcode">
      {error && <div className="cc-error">{error}</div>}

      <div className="cc-split-pane">
        {/* LEFT PANE — Problem */}
        <div className="cc-left-pane">
          <div className="cc-left-tabs">
            <button
              className={`cc-tab ${activeLeftTab === "description" ? "active" : ""}`}
              onClick={() => setActiveLeftTab("description")}
            >
              Description
            </button>
            <button
              className={`cc-tab ${activeLeftTab === "testcases" ? "active" : ""}`}
              onClick={() => setActiveLeftTab("testcases")}
            >
              Test Cases {testResults && (
                <span className={`cc-tab-badge ${passedCount === totalTests ? "all-pass" : "some-fail"}`}>
                  {passedCount}/{totalTests}
                </span>
              )}
            </button>
            {result && (
              <button
                className={`cc-tab ${activeLeftTab === "results" ? "active" : ""}`}
                onClick={() => setActiveLeftTab("results")}
              >
                Results
              </button>
            )}
          </div>

          <div className="cc-left-body">
            {activeLeftTab === "description" && (
              <>
                <div className="cc-problem-header">
                  <h3 className="cc-problem-title">{challenge.title}</h3>
                  <div className="cc-problem-tags">
                    <span className={`cc-difficulty cc-difficulty-${challenge.difficulty}`}>
                      {challenge.difficulty}
                    </span>
                    {startTime && (
                      <span className={`cc-timer ${elapsed > (challenge.time_limit_minutes || 999) * 60 ? "cc-timer-over" : ""}`}>
                        {formatTime(elapsed)}
                        {challenge.time_limit_minutes && ` / ${challenge.time_limit_minutes}:00`}
                      </span>
                    )}
                  </div>
                </div>

                <div className="cc-description">{challenge.description}</div>

                {challenge.examples && challenge.examples.length > 0 && (
                  <div className="cc-examples">
                    {challenge.examples.map((ex, i) => (
                      <div key={i} className="cc-example">
                        <div className="cc-example-label">Example {i + 1}</div>
                        <div className="cc-example-block">
                          <div className="cc-io-row">
                            <span className="cc-io-label">Input:</span>
                            <code className="cc-io-value">{ex.input}</code>
                          </div>
                          <div className="cc-io-row">
                            <span className="cc-io-label">Output:</span>
                            <code className="cc-io-value">{ex.output}</code>
                          </div>
                          {ex.explanation && (
                            <div className="cc-io-row">
                              <span className="cc-io-label">Explanation:</span>
                              <span className="cc-io-explanation">{ex.explanation}</span>
                            </div>
                          )}
                        </div>
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
              </>
            )}

            {activeLeftTab === "testcases" && (
              <div className="cc-test-results">
                {!testResults ? (
                  <div className="cc-no-tests">
                    <p>Click <strong>Run Code</strong> to test against example cases.</p>
                  </div>
                ) : (
                  <>
                    <div className={`cc-test-summary ${passedCount === totalTests ? "all-pass" : "some-fail"}`}>
                      {passedCount === totalTests
                        ? `All ${totalTests} test(s) passed!`
                        : `${passedCount} of ${totalTests} test(s) passed`}
                    </div>
                    {testResults.map((t, i) => (
                      <div key={i} className={`cc-test-case ${t.passed ? "pass" : "fail"}`}>
                        <div className="cc-test-case-header">
                          <span className={`cc-test-icon ${t.passed ? "pass" : "fail"}`}>
                            {t.passed ? "\u2713" : "\u2717"}
                          </span>
                          <span className="cc-test-label">Test Case {i + 1}</span>
                        </div>
                        <div className="cc-test-case-body">
                          <div className="cc-io-row">
                            <span className="cc-io-label">Input:</span>
                            <code className="cc-io-value">{t.input}</code>
                          </div>
                          <div className="cc-io-row">
                            <span className="cc-io-label">Expected:</span>
                            <code className="cc-io-value">{t.expected}</code>
                          </div>
                          <div className="cc-io-row">
                            <span className="cc-io-label">Output:</span>
                            <code className={`cc-io-value ${t.passed ? "" : "cc-io-wrong"}`}>{t.actual}</code>
                          </div>
                        </div>
                      </div>
                    ))}
                  </>
                )}
              </div>
            )}

            {activeLeftTab === "results" && result && (
              <div className="cc-results-pane">
                <CodeQualityResults scores={result} title="Code Analysis" />
                <button className="cc-new-btn" onClick={fetchChallenge} disabled={fetching}>
                  Try Another Challenge
                </button>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT PANE — Editor */}
        <div className="cc-right-pane">
          <div className="cc-editor-toolbar">
            <div className="cc-editor-toolbar-left">
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
            <div className="cc-editor-toolbar-right">
              {startTime && (
                <span className="cc-toolbar-timer">{formatTime(elapsed)}</span>
              )}
            </div>
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
            disabled={!!result}
          />

          <div className="cc-action-bar">
            {loading || runningTests ? (
              <div className="cc-analyzing">
                <div className="cc-spinner" />
                <span>{loading ? "Analyzing your code..." : "Running tests..."}</span>
              </div>
            ) : result ? (
              <div className="cc-submitted-bar">
                <span className="cc-submitted-label">Submitted</span>
                <button className="cc-new-btn" onClick={fetchChallenge} disabled={fetching}>
                  Next Challenge
                </button>
              </div>
            ) : (
              <>
                <button
                  className="cc-run-btn"
                  onClick={runTests}
                  disabled={!code.trim() || runningTests}
                >
                  Run Code
                </button>
                <button
                  className="cc-submit-btn"
                  onClick={submitCode}
                  disabled={!code.trim() || loading}
                >
                  Submit
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default CodeChallenge;
