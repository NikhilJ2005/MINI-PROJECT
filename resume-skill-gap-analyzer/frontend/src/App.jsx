import "./cssFile/App.css";
import { useState, useCallback, useEffect, lazy, Suspense } from "react";
import TabNav from "./TabNav";
import InputSection from "./InputSection";
import Results from "./Results";
import AnalysisHistory from "./AnalysisHistory";
import ToastContainer from "./Toast";

// Lazy-load tab components that aren't shown by default
const BatchUpload = lazy(() => import("./BatchUpload"));
const CandidatesList = lazy(() => import("./CandidatesList"));
const RankingsView = lazy(() => import("./RankingsView"));
const CompareView = lazy(() => import("./CompareView"));
const JDParser = lazy(() => import("./JDParser"));
const DashboardStats = lazy(() => import("./DashboardStats"));

const TabFallback = () => (
  <div className="tab-loading">Loading...</div>
);

const TAB_KEYS = ["analyze", "batch", "candidates", "rankings", "compare", "jd-parser", "dashboard"];

function App() {
  const [activeTab, setActiveTab] = useState(
    () => localStorage.getItem("activeTab") || "analyze"
  );
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [currentAnalysisId, setCurrentAnalysisId] = useState(null);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [historyOpen, setHistoryOpen] = useState(true);

  // Dark mode state
  const [darkMode, setDarkMode] = useState(
    () => localStorage.getItem("theme") === "dark"
  );

  // Apply dark mode on mount and toggle
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", darkMode ? "dark" : "light");
    localStorage.setItem("theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  const toggleDarkMode = useCallback(() => {
    setDarkMode((d) => !d);
  }, []);

  // Persist activeTab to localStorage
  useEffect(() => {
    localStorage.setItem("activeTab", activeTab);
  }, [activeTab]);

  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Alt+1-7: switch tabs
      if (e.altKey && e.key >= "1" && e.key <= "7") {
        e.preventDefault();
        const idx = parseInt(e.key) - 1;
        if (TAB_KEYS[idx]) setActiveTab(TAB_KEYS[idx]);
        return;
      }
      // Alt+N: new analysis
      if (e.altKey && (e.key === "n" || e.key === "N")) {
        e.preventDefault();
        setReport(null);
        setCurrentAnalysisId(null);
        setError("");
        setActiveTab("analyze");
        return;
      }
      // Alt+D: toggle dark mode
      if (e.altKey && (e.key === "d" || e.key === "D")) {
        e.preventDefault();
        toggleDarkMode();
        return;
      }
      // Escape: clear report
      if (e.key === "Escape") {
        if (report) {
          setReport(null);
          setCurrentAnalysisId(null);
        }
        return;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [report, toggleDarkMode]);

  // Called after a new analysis completes
  const handleReportReceived = useCallback((newReport) => {
    setReport(newReport);
    if (newReport?.analysis_id) {
      setCurrentAnalysisId(newReport.analysis_id);
    }
    // Trigger history sidebar refresh
    setHistoryRefreshKey((k) => k + 1);
  }, []);

  // Called when user clicks a history entry
  const handleHistorySelect = useCallback((loadedReport, analysisId) => {
    setReport(loadedReport);
    setCurrentAnalysisId(analysisId);
    setActiveTab("analyze");
    setError("");
  }, []);

  // Called when user clicks "New Analysis"
  const handleNewAnalysis = useCallback(() => {
    setReport(null);
    setCurrentAnalysisId(null);
    setError("");
    setActiveTab("analyze");
  }, []);

  return (
    <>
      <header className="app-header">
        <div className="container header-row">
          <div>
            <h1 className="app-title">Automated Recruiting Platform</h1>
            <p className="app-subtitle">
              Upload resumes, rank candidates, compare skills, and discover skill
              gaps for any target role — powered by Machine Learning.
            </p>
          </div>
          <button
            className="theme-toggle-btn"
            onClick={toggleDarkMode}
            title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {darkMode ? "\u2600\uFE0F" : "\uD83C\uDF19"}
          </button>
        </div>
      </header>

      <TabNav activeTab={activeTab} onTabChange={setActiveTab} />

      <div className="app-layout">
        <AnalysisHistory
          refreshKey={historyRefreshKey}
          activeAnalysisId={currentAnalysisId}
          onSelect={handleHistorySelect}
          onNewAnalysis={handleNewAnalysis}
          isOpen={historyOpen}
          onToggle={() => setHistoryOpen((o) => !o)}
        />

        <main className={`container main-content ${historyOpen ? "with-sidebar" : ""}`}>
          {activeTab === "analyze" && (
            <>
              <section className="input-section">
                <InputSection
                  onError={setError}
                  onAnalyze={setLoading}
                  obtainedReport={handleReportReceived}
                />
              </section>
              {error && <div className="error-message">{error}</div>}
              {loading && (
                <div className="loading-overlay">
                  <div className="spinner-container">
                    <div className="spinner"></div>
                    <p className="loading-text">Analyzing your profile...</p>
                    <p className="loading-subtext">Parsing resume...</p>
                  </div>
                </div>
              )}
              {report && <Results report={report} />}
            </>
          )}

          <Suspense fallback={<TabFallback />}>
            {activeTab === "batch" && <BatchUpload />}
            {activeTab === "candidates" && <CandidatesList />}
            {activeTab === "rankings" && <RankingsView />}
            {activeTab === "compare" && <CompareView />}
            {activeTab === "jd-parser" && <JDParser />}
            {activeTab === "dashboard" && <DashboardStats />}
          </Suspense>
        </main>
      </div>

      <footer className="app-footer">
        <div className="container">
          <p>Automated Recruiting Platform &mdash; Powered by FastAPI, Scikit-learn &amp; spaCy</p>
          <p className="shortcuts-hint">
            Shortcuts: Alt+1-7 tabs | Alt+N new | Alt+D dark mode | Esc clear | Ctrl+Enter submit
          </p>
        </div>
      </footer>

      <ToastContainer />
    </>
  );
}

export default App;
