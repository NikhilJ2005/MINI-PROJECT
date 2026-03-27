import "./cssFile/App.css";
import { useState, useCallback, useEffect, useRef, lazy, Suspense } from "react";
import { useUserRole } from "./UserRoleContext";
import { getVisibleTabs } from "./TabNav";
import TabNav from "./TabNav";
import RoleSelector from "./RoleSelector";
import InputSection from "./InputSection";
import Results from "./Results";
import CodeChallenge from "./CodeChallenge";
import AnalysisHistory from "./AnalysisHistory";
import ToastContainer from "./Toast";
import ErrorBoundary from "./ErrorBoundary";

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

function App() {
  const { userRole } = useUserRole();

  const [activeTab, setActiveTab] = useState(() => {
    // Read userRole directly from localStorage so the initializer
    // can pick up the correct per-role stored tab on first render.
    const role = localStorage.getItem("userRole");
    if (role) {
      return localStorage.getItem(`activeTab_${role}`) || "analyze";
    }
    return "analyze";
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [currentAnalysisId, setCurrentAnalysisId] = useState(null);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);

  // Track the last non-null role to detect actual role switches
  const lastRoleRef = useRef(null);

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

  // Persist activeTab per-role (candidate / recruiter get separate keys)
  useEffect(() => {
    if (!userRole) return;
    localStorage.setItem(`activeTab_${userRole}`, activeTab);
  }, [activeTab, userRole]);

  // Reset view state and restore per-role tab when switching roles
  useEffect(() => {
    if (!userRole) return;
    const stored = localStorage.getItem(`activeTab_${userRole}`);
    setActiveTab(stored || "analyze");
    setReport(null);
    setCurrentAnalysisId(null);
    setError("");
    setLoading(false);
  }, [userRole]);

  // Reset view state when role actually switches (candidate ↔ recruiter)
  useEffect(() => {
    if (userRole) {
      if (lastRoleRef.current !== null && lastRoleRef.current !== userRole) {
        setActiveTab("analyze");
        setReport(null);
        setCurrentAnalysisId(null);
        setError("");
        setLoading(false);
      }
      lastRoleRef.current = userRole;
    }
  }, [userRole]);

  // Guard: if current activeTab isn't visible for this role, reset to "analyze"
  useEffect(() => {
    if (userRole) {
      const visible = getVisibleTabs(userRole);
      const visibleIds = visible.map((t) => t.id);
      if (!visibleIds.includes(activeTab)) {
        setActiveTab("analyze");
      }
    }
  }, [userRole, activeTab]);

  // Global keyboard shortcuts
  useEffect(() => {
    if (!userRole) return;

    const visibleTabKeys = getVisibleTabs(userRole).map((t) => t.id);

    const handleKeyDown = (e) => {
      // Alt+1-9: switch tabs (mapped to visible tabs)
      if (e.altKey && e.key >= "1" && e.key <= "9") {
        e.preventDefault();
        const idx = parseInt(e.key) - 1;
        if (visibleTabKeys[idx]) setActiveTab(visibleTabKeys[idx]);
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
  }, [userRole, report, toggleDarkMode]);

  // Called after a new analysis completes
  const handleReportReceived = useCallback((newReport) => {
    setReport(newReport);
    if (newReport?.analysis_id) {
      setCurrentAnalysisId(newReport.analysis_id);
    }
    setHistoryRefreshKey((k) => k + 1);
  }, []);

  // Called when user clicks a history entry
  const handleHistorySelect = useCallback((loadedReport, analysisId) => {
    setReport(loadedReport);
    setCurrentAnalysisId(analysisId);
    setActiveTab("analyze");
    setError("");
  }, []);

  // Show role selector if no role chosen
  if (!userRole) {
    return <RoleSelector />;
  }

  return (
    <>
      <TabNav
        activeTab={activeTab}
        onTabChange={setActiveTab}
        darkMode={darkMode}
        toggleDarkMode={toggleDarkMode}
      />

      <main className="container main-content">
        {activeTab === "analyze" && (
          <>
            <section className="input-section">
              <ErrorBoundary>
                <InputSection
                  onError={setError}
                  onAnalyze={setLoading}
                  obtainedReport={handleReportReceived}
                />
              </ErrorBoundary>
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
            {report && (
              <ErrorBoundary>
                <Results report={report} />
              </ErrorBoundary>
            )}
            {report && userRole === "candidate" && (
              <ErrorBoundary>
                <CodeChallenge
                  targetRole={report.target_role}
                  candidateId={report.candidate_id}
                  analysisId={report.analysis_id}
                />
              </ErrorBoundary>
            )}
          </>
        )}

        <ErrorBoundary>
          <Suspense fallback={<TabFallback />}>
            {activeTab === "batch" && <BatchUpload />}
            {activeTab === "candidates" && <CandidatesList />}
            {activeTab === "rankings" && <RankingsView />}
            {activeTab === "compare" && <CompareView />}
            {activeTab === "jd-parser" && <JDParser />}
            {activeTab === "dashboard" && <DashboardStats />}
          </Suspense>
        </ErrorBoundary>

        {activeTab === "history" && (
          <ErrorBoundary>
            <AnalysisHistory
              refreshKey={historyRefreshKey}
              activeAnalysisId={currentAnalysisId}
              onSelect={handleHistorySelect}
              onNewAnalysis={() => {
                setReport(null);
                setCurrentAnalysisId(null);
                setError("");
                setActiveTab("analyze");
              }}
            />
          </ErrorBoundary>
        )}
      </main>

      <footer className="app-footer">
        <div className="container">
          <p>Automated Recruiting Platform &mdash; Powered by FastAPI, Scikit-learn &amp; spaCy</p>
          <p className="shortcuts-hint">
            Shortcuts: Alt+1-{getVisibleTabs(userRole).length} tabs | Alt+N new | Alt+D dark mode | Esc clear
          </p>
        </div>
      </footer>

      <ToastContainer />
    </>
  );
}

export default App;
