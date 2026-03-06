import { useState, useEffect } from "react";
import "./cssFile/DashboardStats.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function DashboardStats() {
  const [stats, setStats] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [dashRes, metricsRes] = await Promise.all([
          fetch(`${API_BASE_URL}/dashboard`),
          fetch(`${API_BASE_URL}/model-metrics`),
        ]);
        if (!dashRes.ok) throw new Error("Failed to load dashboard");
        setStats(await dashRes.json());
        if (metricsRes.ok) setMetrics(await metricsRes.json());
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return <div className="loading-msg">Loading dashboard...</div>;
  if (error) return <div className="error-msg">{error}</div>;
  if (!stats) return null;

  return (
    <div className="dashboard">
      <h2>Platform Dashboard</h2>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-number">{stats.total_candidates}</div>
          <div className="stat-label">Total Candidates</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{stats.total_analyses}</div>
          <div className="stat-label">Total Analyses</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{stats.avg_match_score != null ? `${Math.round(stats.avg_match_score)}%` : "N/A"}</div>
          <div className="stat-label">Avg Match Score</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{stats.total_batch_jobs || 0}</div>
          <div className="stat-label">Batch Jobs</div>
        </div>
      </div>

      {stats.top_roles && stats.top_roles.length > 0 && (
        <div className="dashboard-section">
          <h3>Top Analyzed Roles</h3>
          <div className="role-list">
            {stats.top_roles.map((r, i) => (
              <div className="role-item" key={i}>
                <span className="role-name">{r.role}</span>
                <span className="role-count">{r.count} analyses</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {metrics && (
        <div className="dashboard-section">
          <h3>ML Model Performance</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-number">{metrics.lr_accuracy}%</div>
              <div className="stat-label">LR Accuracy</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">{metrics.dt_accuracy}%</div>
              <div className="stat-label">DT Accuracy</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">{metrics.dataset_source || "N/A"}</div>
              <div className="stat-label">Dataset</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default DashboardStats;
