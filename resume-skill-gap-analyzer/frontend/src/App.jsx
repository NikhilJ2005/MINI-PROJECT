import "./cssFile/App.css";
import { useState, useCallback, useEffect, lazy, Suspense } from "react";
import TabNav from "./TabNav";
import InputSection from "./InputSection";
import Results from "./Results";
import AnalysisHistory from "./AnalysisHistory";

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
  const [activeTab, setActiveTab] = useState(
    () => localStorage.getItem("activeTab") || "analyze"
  );
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [currentAnalysisId, setCurrentAnalysisId] = useState(null);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [historyOpen, setHistoryOpen] = useState(true);

  // Persist activeTab to localStorage
  useEffect(() => {
    localStorage.setItem("activeTab", activeTab);
  }, [activeTab]);

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
        <div className="container">
          <h1 className="app-title">Automated Recruiting Platform</h1>
          <p className="app-subtitle">
            Upload resumes, rank candidates, compare skills, and discover skill
            gaps for any target role — powered by Machine Learning.
          </p>
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
                  err={setError}
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
        </div>
      </footer>
    </>
  );
}

export default App;
