import { memo, useRef, useCallback } from "react";
import { useUserRole } from "./UserRoleContext";
import ScoreCard from "./ScoreCard";
import Summary from "./Summary";
import SkillTable from "./SkillTable";
import SkillRadarChart from "./SkillRadarChart";
import { showToast } from "./Toast";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import Papa from "papaparse";
import "./cssFile/Results.css";

const Results = memo(function Results({ report }) {
    if (!report) return null;
    const { userRole } = useUserRole();
    const isCandidate = userRole === "candidate";
    const isRecruiter = userRole === "recruiter";
    const ml_insights = report.ml_insights || {};
    const git_insights = report.github_insights || {};
    const maxLang = git_insights?.top_languages?.[0]?.bytes || 1;
    const resultsRef = useRef(null);

    const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

    // --- PDF Export (server-side for large reports, client-side fallback) ---
    const handleExportPDF = useCallback(async () => {
        const analysisId = report.analysis_id;

        // Try server-side PDF first (handles large reports reliably)
        if (analysisId) {
            try {
                showToast("Generating PDF on server...", "info");
                const res = await fetch(`${API_BASE_URL}/export/pdf/${analysisId}`);
                if (res.ok) {
                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement("a");
                    link.href = url;
                    const name = report.candidate_info?.name || "Candidate";
                    const role = report.target_role || "Role";
                    link.download = `Report_${name}_${role}.pdf`;
                    link.click();
                    setTimeout(() => URL.revokeObjectURL(url), 1000);
                    showToast("PDF downloaded!", "success");
                    return;
                }
            } catch (err) {
                console.warn("Server PDF failed, falling back to client-side:", err);
            }
        }

        // Fallback: client-side PDF with html2canvas
        if (!resultsRef.current) return;
        try {
            showToast("Generating PDF...", "info");
            const canvas = await html2canvas(resultsRef.current, {
                scale: 2,
                useCORS: true,
                backgroundColor: "#ffffff",
            });
            const imgData = canvas.toDataURL("image/png");
            const pdf = new jsPDF("p", "mm", "a4");
            const pdfWidth = pdf.internal.pageSize.getWidth();
            const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

            // Handle multi-page PDFs for large reports
            const pageHeight = pdf.internal.pageSize.getHeight();
            if (pdfHeight > pageHeight) {
                let yOffset = 0;
                while (yOffset < canvas.height) {
                    if (yOffset > 0) pdf.addPage();
                    const srcY = yOffset;
                    const srcH = Math.min(canvas.height - yOffset, (pageHeight / pdfWidth) * canvas.width);
                    const tmpCanvas = document.createElement("canvas");
                    tmpCanvas.width = canvas.width;
                    tmpCanvas.height = srcH;
                    const ctx = tmpCanvas.getContext("2d");
                    ctx.drawImage(canvas, 0, srcY, canvas.width, srcH, 0, 0, canvas.width, srcH);
                    const pageImg = tmpCanvas.toDataURL("image/png");
                    const h = (srcH * pdfWidth) / canvas.width;
                    pdf.addImage(pageImg, "PNG", 0, 0, pdfWidth, h);
                    yOffset += srcH;
                }
            } else {
                pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
            }

            const name = report.candidate_info?.name || "Candidate";
            const role = report.target_role || "Role";
            pdf.save(`SkillGap_${name}_${role}.pdf`);
            showToast("PDF downloaded!", "success");
        } catch (err) {
            console.error("PDF generation error:", err);
            showToast("PDF generation failed", "error");
        }
    }, [report]);

    // --- CSV Export ---
    const handleExportCSV = useCallback(() => {
        const allSkills = [
            ...(report.skill_breakdown?.required_analysis || []),
            ...(report.skill_breakdown?.nice_to_have_analysis || []),
        ];
        const skillRows = allSkills.map((s) => ({
            Skill: s.skill,
            Status: s.status,
            "In Resume": s.in_resume ? "Yes" : "No",
            "On GitHub": s.in_github ? "Yes" : "No",
            "ML Confidence": s.probability != null ? `${Math.round(s.probability * 100)}%` : "N/A",
            Category: s.category || "required",
        }));

        let csv = Papa.unparse(skillRows);

        // Executive Summary section
        const es = report.executive_summary;
        if (es) {
            csv += "\n\n--- Executive Summary ---\n";
            csv += Papa.unparse([{
                "Match Score": `${es.match_score?.toFixed(1)}%`,
                "Match Label": es.match_label || "",
                "Confidence": `${es.confidence_score?.toFixed(1)}%`,
                "Resume Skills": es.total_resume_skills,
                "GitHub Skills": es.total_github_skills,
                "Missing Critical": es.missing_critical_skills,
            }]);
        }

        // Recommendations section
        const recs = report.recommendations || [];
        if (recs.length > 0) {
            csv += "\n\n--- Recommendations ---\n";
            csv += Papa.unparse(recs.map((r) => ({
                Skill: r.skill,
                Priority: r.priority,
                Action: r.action,
                Difficulty: r.difficulty || "",
                "Estimated Time": r.estimated_time || "",
                "Resource Hint": r.resource_hint || "",
            })));
        }

        // AI Credibility section
        const cred = report.ai_skill_credibility;
        if (cred) {
            csv += "\n\n--- Skill Credibility ---\n";
            csv += Papa.unparse([{
                "Credibility Score": cred.overall_credibility_score != null ? `${cred.overall_credibility_score}/10` : "N/A",
                "Verified Skills": (cred.verified_skills || []).join("; "),
                "Questionable Skills": (cred.questionable_skills || []).join("; "),
            }]);
        }

        // Culture Fit section
        const culture = report.ai_culture_fit;
        if (culture) {
            csv += "\n\n--- Culture & Soft Skills ---\n";
            csv += Papa.unparse([{
                "Soft Skills": (culture.soft_skills || []).join("; "),
                "Communication Score": culture.communication_score != null ? `${culture.communication_score}/10` : "N/A",
                "Team Fit": culture.team_fit_notes || "",
            }]);
        }

        const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        const name = report.candidate_info?.name || "Candidate";
        link.download = `SkillGap_${name}_skills.csv`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        showToast("CSV exported!", "success");
    }, [report]);

    const skillBreakdown = report.skill_breakdown || {};
    const recommendations = report.recommendations || [];

    return (
        <>
            {/* Export buttons — outside ref so they don't appear in PDF */}
            <div className="export-bar">
                <button className="export-btn export-pdf" onClick={handleExportPDF}>
                    Export PDF
                </button>
                <button className="export-btn export-csv" onClick={handleExportCSV}>
                    Export CSV
                </button>
            </div>

            <section className="results-section" ref={resultsRef}>
                {report.generated_at && (
                    <div className="report-timestamp">
                        Report generated: {new Date(report.generated_at).toLocaleString()}
                    </div>
                )}
                <ScoreCard report={report} />
                {report.executive_summary && <Summary summary={report.executive_summary} />}

                {/* Skill Radar Chart */}
                <SkillRadarChart report={report} />

                {(skillBreakdown.required_analysis || skillBreakdown.nice_to_have_analysis) && (
                    <div className="skill-breakdown">
                        <h2>Skill Breakdown</h2>
                        <SkillTable title="Required Skills" analysis={skillBreakdown.required_analysis} />
                        <SkillTable title="Nice to have Skills" analysis={skillBreakdown.nice_to_have_analysis} />
                    </div>
                )}

                {recommendations.length > 0 && (
                    <div className="recommendations">
                        <h3>Recommendations</h3>
                        {recommendations.map((item, index) => (
                            <div className="recommendation-item" key={index}>
                                <span className={`badge badge-${item.priority || "info"}`}>{item.priority || "Info"}</span>
                                <div className="recommendation-content">
                                    <div className="recommended-action">{item.action}</div>
                                    <div className="recommendation-meta">
                                        {item.difficulty && <span className="rec-difficulty">{item.difficulty}</span>}
                                        {item.estimated_time && <span className="rec-time">{item.estimated_time}</span>}
                                    </div>
                                    {Array.isArray(item.learn_first) && item.learn_first.length > 0 && (
                                        <div className="rec-prereqs">Learn first: {item.learn_first.join(", ")}</div>
                                    )}
                                    <div className="recommended-hints">{item.resource_hint}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {report.learning_path?.length > 0 && !report.ai_learning_path?.length && (
                    <div className="learning-path-section">
                        <h3>Skill Learning Path</h3>
                        <div className="learning-path-list">
                            {report.learning_path.map((item, i) => (
                                <div key={i} className={`learning-path-card lp-${(item.priority || "info").toLowerCase()}`}>
                                    <div className="lp-header">
                                        <span className="lp-skill">{item.skill}</span>
                                        <span className={`badge badge-${item.priority || "info"}`}>{item.priority || "Info"}</span>
                                    </div>
                                    <div className="lp-meta">
                                        {item.difficulty && <span className="lp-difficulty">{item.difficulty}</span>}
                                        {item.estimated_time && <span className="lp-time">{item.estimated_time}</span>}
                                    </div>
                                    {item.suggested_path && (
                                        <div className="lp-path">{item.suggested_path}</div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {ml_insights.lr_accuracy != null && (
                    <div className="ml-insights">
                        <h3>ML Model Insights</h3>
                        <div className="ml-insights-container">
                            <div className="ml-grid ml-grid-3">
                                <div className="ml-metric">
                                    <div className="metric-value">{ml_insights.lr_accuracy}%</div>
                                    <div className="metric-label">Logistic Regression</div>
                                </div>
                                <div className="ml-metric">
                                    <div className="metric-value">{ml_insights.dt_accuracy}%</div>
                                    <div className="metric-label">Decision Tree</div>
                                </div>
                                <div className="ml-metric">
                                    <div className="metric-value">{ml_insights.feature_names?.length ?? 11}</div>
                                    <div className="metric-label">Features Used</div>
                                </div>
                            </div>
                            {ml_insights.ensemble_explanation && (
                                <div className="ml-explanation">
                                    <strong>Ensemble: </strong>
                                    {ml_insights.ensemble_explanation}
                                </div>
                            )}
                            {ml_insights.model_explanation && (
                                <div className="ml-explanation">
                                    <strong>How it works: </strong>
                                    {ml_insights.model_explanation}
                                </div>
                            )}
                            {ml_insights.feature_importance?.dt_importance && (
                                <div className="feature-importance-section">
                                    <h4>Feature Importance (Decision Tree)</h4>
                                    <div className="feature-bars">
                                        {Object.entries(ml_insights.feature_importance.dt_importance)
                                            .sort(([,a], [,b]) => b - a)
                                            .slice(0, 6)
                                            .map(([name, value]) => (
                                                <div className="feature-bar-row" key={name}>
                                                    <span className="feature-bar-name">{name.replace(/_/g, ' ')}</span>
                                                    <div className="feature-bar-track">
                                                        <div className="feature-bar-fill" style={{ width: `${Math.min(value * 100 / 0.5, 100)}%` }}></div>
                                                    </div>
                                                    <span className="feature-bar-val">{(value * 100).toFixed(1)}%</span>
                                                </div>
                                            ))
                                        }
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* AI Candidate Summary (recruiter-facing) */}
                {isRecruiter && report.ai_candidate_summary && (
                    <div className="ai-candidate-summary-section">
                        <h3>AI Candidate Assessment</h3>
                        <div className="ai-summary-container">
                            {report.ai_candidate_summary.headline && (
                                <p className="ai-headline">{report.ai_candidate_summary.headline}</p>
                            )}
                            {report.ai_candidate_summary.executive_summary && (
                                <p className="ai-exec-summary">{report.ai_candidate_summary.executive_summary}</p>
                            )}
                            {report.ai_candidate_summary.hiring_recommendation && (
                                <div className={`hiring-recommendation rec-${report.ai_candidate_summary.hiring_recommendation.toLowerCase().replace(/\s+/g, '-')}`}>
                                    Recommendation: <strong>{report.ai_candidate_summary.hiring_recommendation}</strong>
                                </div>
                            )}
                            {report.ai_candidate_summary.top_strengths?.length > 0 && (
                                <div className="ai-strengths-list">
                                    <h4>Top Strengths</h4>
                                    <ul>{report.ai_candidate_summary.top_strengths.map((s, i) => <li key={i}>{s}</li>)}</ul>
                                </div>
                            )}
                            {report.ai_candidate_summary.risk_factors?.length > 0 && (
                                <div className="ai-risks-list">
                                    <h4>Risk Factors</h4>
                                    <ul>{report.ai_candidate_summary.risk_factors.map((r, i) => <li key={i}>{r}</li>)}</ul>
                                </div>
                            )}
                            {report.ai_candidate_summary.salary_positioning && (
                                <p className="salary-note"><strong>Level:</strong> {report.ai_candidate_summary.salary_positioning}</p>
                            )}
                        </div>
                    </div>
                )}

                {/* AI Skill Credibility Assessment (recruiter-facing) */}
                {isRecruiter && report.ai_skill_credibility && (
                    <div className="ai-credibility-section">
                        <h3>Skill Credibility Assessment</h3>
                        <div className="credibility-container">
                            {report.ai_skill_credibility.overall_credibility_score != null && (
                                <div className="credibility-score-badge">
                                    Credibility Score: <strong>{report.ai_skill_credibility.overall_credibility_score}/10</strong>
                                </div>
                            )}
                            {report.ai_skill_credibility.assessment && (
                                <p className="ai-exec-summary">{report.ai_skill_credibility.assessment}</p>
                            )}
                            {report.ai_skill_credibility.verified_skills?.length > 0 && (
                                <div className="credibility-group">
                                    <h4>Verified Claims</h4>
                                    <div className="credibility-tags verified">
                                        {report.ai_skill_credibility.verified_skills.map((s, i) => (
                                            <span key={i} className="cred-tag cred-verified">{s}</span>
                                        ))}
                                    </div>
                                </div>
                            )}
                            {report.ai_skill_credibility.questionable_skills?.length > 0 && (
                                <div className="credibility-group">
                                    <h4>Needs Verification</h4>
                                    <div className="credibility-tags questionable">
                                        {report.ai_skill_credibility.questionable_skills.map((s, i) => (
                                            <span key={i} className="cred-tag cred-questionable">{s}</span>
                                        ))}
                                    </div>
                                </div>
                            )}
                            {report.ai_skill_credibility.recommendations?.length > 0 && (
                                <div className="ai-tips">
                                    <h4>Verification Recommendations</h4>
                                    <ul>
                                        {report.ai_skill_credibility.recommendations.map((r, i) => (
                                            <li key={i}>{r}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* AI Role-Fit Narrative (recruiter-facing) */}
                {isRecruiter && report.ai_role_fit_narrative && (
                    <div className="ai-rolefit-section">
                        <h3>Role-Fit Analysis</h3>
                        <div className="rolefit-container">
                            {report.ai_role_fit_narrative.fit_score != null && (
                                <div className={`rolefit-score-badge ${report.ai_role_fit_narrative.fit_score >= 7 ? 'fit-strong' : report.ai_role_fit_narrative.fit_score >= 5 ? 'fit-moderate' : 'fit-weak'}`}>
                                    Role Fit: <strong>{report.ai_role_fit_narrative.fit_score}/10</strong>
                                </div>
                            )}
                            {report.ai_role_fit_narrative.narrative && (
                                <p className="ai-exec-summary">{report.ai_role_fit_narrative.narrative}</p>
                            )}
                            {report.ai_role_fit_narrative.standout_qualities?.length > 0 && (
                                <div className="rolefit-qualities">
                                    <h4>Standout Qualities</h4>
                                    <ul>
                                        {report.ai_role_fit_narrative.standout_qualities.map((q, i) => (
                                            <li key={i}>{q}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            {report.ai_role_fit_narrative.growth_areas?.length > 0 && (
                                <div className="rolefit-growth">
                                    <h4>Growth Areas</h4>
                                    <ul>
                                        {report.ai_role_fit_narrative.growth_areas.map((g, i) => (
                                            <li key={i}>{g}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            {report.ai_role_fit_narrative.onboarding_estimate && (
                                <p className="onboarding-note">
                                    <strong>Estimated Onboarding:</strong> {report.ai_role_fit_narrative.onboarding_estimate}
                                </p>
                            )}
                        </div>
                    </div>
                )}

                {/* AI Culture Fit & Soft Skills (recruiter-facing) */}
                {isRecruiter && report.ai_culture_fit && (
                    <div className="ai-culture-section">
                        <h3>Culture & Soft Skills</h3>
                        <div className="culture-container">
                            {report.ai_culture_fit.soft_skills?.length > 0 && (
                                <div className="soft-skills-tags">
                                    {report.ai_culture_fit.soft_skills.map((s, i) => (
                                        <span key={i} className="soft-skill-tag">{s}</span>
                                    ))}
                                </div>
                            )}
                            {report.ai_culture_fit.communication_score && (
                                <div className="comm-score">
                                    Communication Score: <strong>{report.ai_culture_fit.communication_score}/10</strong>
                                </div>
                            )}
                            {report.ai_culture_fit.leadership_indicators?.length > 0 && (
                                <div className="leadership-list">
                                    <h4>Leadership Indicators</h4>
                                    <ul>{report.ai_culture_fit.leadership_indicators.map((l, i) => <li key={i}>{l}</li>)}</ul>
                                </div>
                            )}
                            {report.ai_culture_fit.team_fit_notes && (
                                <p className="team-fit"><strong>Team Fit:</strong> {report.ai_culture_fit.team_fit_notes}</p>
                            )}
                        </div>
                    </div>
                )}

                {/* AI Resume Coach (candidate-facing) */}
                {isCandidate && report.ai_feedback && (
                    <div className="ai-feedback-section">
                        <h3>AI Resume Coach</h3>
                        <div className="ai-feedback-container">
                            {report.ai_feedback.overall_advice && (
                                <p className="ai-overall-advice">{report.ai_feedback.overall_advice}</p>
                            )}
                            {report.ai_feedback.resume_tips?.length > 0 && (
                                <div className="ai-tips">
                                    <h4>Improvement Tips</h4>
                                    <ul>
                                        {report.ai_feedback.resume_tips.map((tip, i) => (
                                            <li key={i}>{tip}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            {report.ai_feedback.bullet_suggestions?.length > 0 && (
                                <div className="ai-bullets">
                                    <h4>Suggested Bullet Points to Add</h4>
                                    <ul>
                                        {report.ai_feedback.bullet_suggestions.map((bullet, i) => (
                                            <li key={i}>{bullet}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            {report.ai_feedback.keyword_suggestions?.length > 0 && (
                                <div className="ai-keywords">
                                    <h4>ATS Keywords to Add</h4>
                                    <div className="keyword-tags">
                                        {report.ai_feedback.keyword_suggestions.map((kw, i) => (
                                            <span key={i} className="keyword-tag">{kw}</span>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* AI Interview Prep (candidate-facing only) */}
                {isCandidate && report.ai_interview_questions?.length > 0 && (
                    <div className="ai-interview-section">
                        <h3>AI Interview Prep</h3>
                        <div className="interview-questions">
                            {report.ai_interview_questions.map((q, i) => (
                                <div key={i} className="interview-question-card">
                                    <div className="question-header">
                                        <span className={`badge badge-${q.difficulty === 'hard' ? 'Critical' : q.difficulty === 'medium' ? 'Recommended' : 'info'}`}>
                                            {q.difficulty}
                                        </span>
                                        <span className="question-skill">{q.skill}</span>
                                    </div>
                                    <p className="question-text">{q.question}</p>
                                    <p className="prep-hint"><strong>Prep hint:</strong> {q.prep_hint}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* AI Learning Path (candidate-facing) */}
                {isCandidate && report.ai_learning_path?.length > 0 && (
                    <div className="ai-learning-section">
                        <h3>AI Learning Path</h3>
                        <div className="learning-path-timeline">
                            {report.ai_learning_path.map((item, i) => (
                                <div key={i} className="learning-path-item">
                                    <div className="learning-week">Week {item.week || i + 1}</div>
                                    <div className="learning-content">
                                        <h4>{item.skill}</h4>
                                        {item.resources?.length > 0 && (
                                            <ul className="resource-list">
                                                {item.resources.map((r, j) => (
                                                    <li key={j}>
                                                        {r.url ? (
                                                            <a href={r.url} target="_blank" rel="noopener noreferrer">{r.name || r}</a>
                                                        ) : (
                                                            <span>{typeof r === 'string' ? r : r.name}</span>
                                                        )}
                                                    </li>
                                                ))}
                                            </ul>
                                        )}
                                        {item.project_idea && (
                                            <p className="project-idea"><strong>Project:</strong> {item.project_idea}</p>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                <div className="github-insights">
                    <h3>GitHub Insights</h3>
                    <div className="github-insights-container">
                        <div className="github-stat">
                            <span className="github-stat-label">Repositories Analyzed</span>
                            <span className="github-stat-value">{git_insights.repos_analyzed || 0}</span>
                        </div>
                        {git_insights?.top_languages?.length > 0 && (
                            <>
                                <h4>Top Languages</h4>
                                {git_insights.top_languages.map((item, index) => (
                                    <div className="language-bar" key={index}>
                                        <span className="language-bar-name">{item.language}</span>
                                        <div className="language-bar-track">
                                            <div className="language-bar-fill" style={{ width: `${(item.bytes / maxLang) * 100}%` }}></div>
                                        </div>
                                        <span className="language-bar-bytes">{(item.bytes / 1000).toFixed(1)} KB</span>
                                    </div>
                                ))}
                            </>
                        )}
                    </div>
                </div>
            </section>
        </>
    );
});
export default Results;
