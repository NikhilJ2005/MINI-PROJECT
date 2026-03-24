import { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell,
} from "recharts";
import "./cssFile/DashboardStats.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

const SCORE_COLORS = ["#16a34a", "#eab308", "#ea580c", "#ef4444"];
const SCORE_LABELS = ["Excellent (75-100)", "Good (50-74)", "Fair (25-49)", "Poor (0-24)"];

function timeAgo(ts) {
  if (!ts) return "";
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function scoreClass(score) {
  if (score >= 75) return "excellent";
  if (score >= 50) return "good";
  if (score >= 25) return "fair";
  return "poor";
}

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
    role: r.role.length > 15 ? r.role.slice(0, 15) + "\u2026" : r.role,
    count: r.count,
  }));

  const scoreDist = stats.score_distribution || {};
  const pieData = [
    { name: SCORE_LABELS[0], value: scoreDist.excellent || 0 },
    { name: SCORE_LABELS[1], value: scoreDist.good || 0 },
    { name: SCORE_LABELS[2], value: scoreDist.fair || 0 },
    { name: SCORE_LABELS[3], value: scoreDist.poor || 0 },
  ].filter((d) => d.value > 0);

  const gapData = (stats.top_skill_gaps || []).map((g) => ({
    skill: g.skill.length > 18 ? g.skill.slice(0, 18) + "\u2026" : g.skill,
    count: g.count,
  }));

  const hasData = stats.total_candidates > 0 || stats.total_analyses > 0;

  return (
    <div className="dashboard">
      <h2>Platform Dashboard</h2>

      {/* KPI Cards */}
      <div className="stats-grid stats-grid-4">
        <div className="stat-card accent-blue">
          <div className="stat-icon">👥</div>
          <div className="stat-number">{stats.total_candidates}</div>
          <div className="stat-label">Total Candidates</div>
        </div>
        <div className="stat-card accent-green">
          <div className="stat-icon">📋</div>
          <div className="stat-number">{stats.total_analyses}</div>
          <div className="stat-label">Total Analyses</div>
        </div>
        <div className="stat-card accent-amber">
          <div className="stat-icon">🎯</div>
          <div className="stat-number">
            {stats.average_match_score != null ? `${Math.round(stats.average_match_score)}%` : "N/A"}
          </div>
          <div className="stat-label">Avg Match Score</div>
        </div>
        <div className="stat-card accent-purple">
          <div className="stat-icon">📦</div>
          <div className="stat-number">{stats.total_batches || 0}</div>
          <div className="stat-label">Batch Jobs</div>
        </div>
      </div>

      {!hasData && (
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h3>No data yet</h3>
          <p>Analyze some resumes to see platform statistics here.</p>
        </div>
      )}

      {hasData && (
        <>
          {/* Two-column layout: Charts + Activity */}
          <div className="dashboard-grid">
            {/* Left: Charts */}
            <div className="dashboard-col">
              {roleChartData.length > 0 && (
                <div className="dashboard-section">
                  <h3>Analyses by Role</h3>
                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={roleChartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light, #e2e8f0)" />
                        <XAxis dataKey="role" tick={{ fill: "var(--text-gray, #64748b)", fontSize: 11 }} />
                        <YAxis allowDecimals={false} tick={{ fill: "var(--text-gray, #64748b)", fontSize: 12 }} />
                        <Tooltip contentStyle={{ background: "var(--card-bg, #fff)", border: "1px solid var(--border-light)", borderRadius: 8 }} />
                        <Bar dataKey="count" fill="var(--primary, #2563eb)" radius={[6, 6, 0, 0]} name="Analyses" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {pieData.length > 0 && (
                <div className="dashboard-section">
                  <h3>Score Distribution</h3>
                  <div className="chart-container pie-chart-container">
                    <ResponsiveContainer width="100%" height={220}>
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={80}
                          paddingAngle={3}
                          dataKey="value"
                        >
                          {pieData.map((entry, i) => {
                            const origIdx = SCORE_LABELS.indexOf(entry.name);
                            return <Cell key={i} fill={SCORE_COLORS[origIdx]} />;
                          })}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="pie-legend">
                      {pieData.map((d, i) => {
                        const origIdx = SCORE_LABELS.indexOf(d.name);
                        return (
                          <div key={i} className="pie-legend-item">
                            <span className="pie-dot" style={{ background: SCORE_COLORS[origIdx] }} />
                            <span className="pie-label">{d.name}</span>
                            <span className="pie-value">{d.value}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Right: Activity + Top Candidates */}
            <div className="dashboard-col">
              {(stats.recent_activity || []).length > 0 && (
                <div className="dashboard-section">
                  <h3>Recent Activity</h3>
                  <div className="activity-feed">
                    {(stats.recent_activity || []).map((a, i) => (
                      <div key={i} className="activity-item">
                        <div className="activity-dot" />
                        <div className="activity-content">
                          <div className="activity-name">{a.name || "Unknown"}</div>
                          <div className="activity-role">{a.role}</div>
                        </div>
                        <div className="activity-right">
                          <span className={`activity-score ${scoreClass(a.match_score)}`}>
                            {Math.round(a.match_score ?? 0)}%
                          </span>
                          <span className="activity-time">{timeAgo(a.analyzed_at)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {(stats.top_candidates || []).length > 0 && (
                <div className="dashboard-section">
                  <h3>Top Candidates</h3>
                  <div className="top-candidates-list">
                    {(stats.top_candidates || []).map((c, i) => (
                      <div key={i} className="top-candidate-item">
                        <div className="top-candidate-rank">#{i + 1}</div>
                        <div className="top-candidate-info">
                          <div className="top-candidate-name">{c.name || "Unknown"}</div>
                          <div className="top-candidate-role">{c.role}</div>
                        </div>
                        <div className={`top-candidate-score ${scoreClass(c.match_score)}`}>
                          {Math.round(c.match_score ?? 0)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ML Model Performance (compact) */}
              {metrics && (
                <div className="dashboard-section">
                  <h3>ML Model Performance</h3>
                  <div className="ml-compact-grid">
                    <div className="ml-compact-card">
                      <div className="ml-compact-value">{metrics.lr_accuracy != null ? `${metrics.lr_accuracy}%` : "N/A"}</div>
                      <div className="ml-compact-label">Logistic Regression</div>
                    </div>
                    <div className="ml-compact-card">
                      <div className="ml-compact-value">{metrics.dt_accuracy != null ? `${metrics.dt_accuracy}%` : "N/A"}</div>
                      <div className="ml-compact-label">Decision Tree</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Full-width: Top Skill Gaps */}
          {gapData.length > 0 && (
            <div className="dashboard-section">
              <h3>Most Common Skill Gaps</h3>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={gapData} layout="vertical" margin={{ top: 5, right: 20, bottom: 5, left: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light, #e2e8f0)" />
                    <XAxis type="number" allowDecimals={false} tick={{ fill: "var(--text-gray)", fontSize: 12 }} />
                    <YAxis type="category" dataKey="skill" tick={{ fill: "var(--text-dark)", fontSize: 12 }} width={80} />
                    <Tooltip contentStyle={{ background: "var(--card-bg, #fff)", border: "1px solid var(--border-light)", borderRadius: 8 }} />
                    <Bar dataKey="count" fill="#ef4444" radius={[0, 6, 6, 0]} name="Candidates Missing" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default DashboardStats;
