import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
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

  if (loading) {
    return (
      <div className="dashboard">
        <h2>Platform Dashboard</h2>
        <div className="stats-grid">
          {[...Array(4)].map((_, i) => (
            <div className="stat-card skeleton-card" key={i}>
              <div className="skeleton-number" />
              <div className="skeleton-label" />
            </div>
          ))}
        </div>
      </div>
    );
  }
  if (error) return <div className="error-msg">{error}</div>;
  if (!stats) return null;

  const roleChartData = (stats.top_roles || []).map((r) => ({
    role: r.role.length > 15 ? r.role.slice(0, 15) + "…" : r.role,
    count: r.count,
  }));

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

      {roleChartData.length > 0 && (
        <div className="dashboard-section">
          <h3>Analyses by Role</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={roleChartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light, #e2e8f0)" />
                <XAxis dataKey="role" tick={{ fill: "var(--text-gray, #64748b)", fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fill: "var(--text-gray, #64748b)", fontSize: 12 }} />
                <Tooltip contentStyle={{ background: "var(--card-bg, #fff)", border: "1px solid var(--border-light, #e2e8f0)", borderRadius: 8 }} />
                <Bar dataKey="count" fill="var(--primary, #2563eb)" radius={[6, 6, 0, 0]} name="Analyses" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {stats.total_candidates === 0 && stats.total_analyses === 0 && (
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h3>No data yet</h3>
          <p>Analyze some resumes to see platform statistics here.</p>
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
