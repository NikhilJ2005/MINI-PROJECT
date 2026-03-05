import "./cssFile/App.css";
import { useState } from "react";
import TabNav from "./TabNav";
import InputSection from "./InputSection";
import Results from "./Results";
import BatchUpload from "./BatchUpload";
import CandidatesList from "./CandidatesList";
import RankingsView from "./RankingsView";
import CompareView from "./CompareView";
import JDParser from "./JDParser";
import DashboardStats from "./DashboardStats";

function App() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);

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

      <main className="container main-content">
        {activeTab === "analyze" && (
          <>
            <section className="input-section">
              <InputSection err={setError} onAnalyze={setLoading} obtainedReport={setReport} />
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

        {activeTab === "batch" && <BatchUpload />}
        {activeTab === "candidates" && <CandidatesList />}
        {activeTab === "rankings" && <RankingsView />}
        {activeTab === "compare" && <CompareView />}
        {activeTab === "jd-parser" && <JDParser />}
        {activeTab === "dashboard" && <DashboardStats />}
      </main>

      <footer className="app-footer">
        <div className="container">
          <p>Automated Recruiting Platform &mdash; Powered by FastAPI, Scikit-learn &amp; spaCy</p>
        </div>
      </footer>
    </>
  );
}

export default App;
