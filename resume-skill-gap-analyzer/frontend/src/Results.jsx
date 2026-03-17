import { memo, useRef, useCallback } from "react";
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
    const ml_insights = report.ml_insights || {};
    const git_insights = report.github_insights || {};
    const maxLang = git_insights?.top_languages?.[0]?.bytes || 1;
    const resultsRef = useRef(null);

    // --- PDF Export ---
    const handleExportPDF = useCallback(async () => {
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
            pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
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
        const csvData = allSkills.map((s) => ({
            Skill: s.skill,
            Status: s.status,
            "In Resume": s.in_resume ? "Yes" : "No",
            "On GitHub": s.in_github ? "Yes" : "No",
            "ML Confidence": s.probability != null ? `${Math.round(s.probability * 100)}%` : "N/A",
            Category: s.category || "required",
        }));
        const csv = Papa.unparse(csvData);
        const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        const name = report.candidate_info?.name || "Candidate";
        link.download = `SkillGap_${name}_skills.csv`;
        link.click();
        URL.revokeObjectURL(url);
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
                                <span className={`badge badge-${item.priority}`}>{item.priority}</span>
                                <div className="recommendation-content">
                                    <div className="recommended-action">{item.action}</div>
                                    <div className="recommended-hints">{item.resource_hint}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {ml_insights.lr_accuracy != null && (
                    <div className="ml-insights">
                        <h3>ML Model Insights</h3>
                        <div className="ml-insights-container">
                            <div className="ml-grid">
                                <div className="ml-metric">
                                    <div className="metric-value">{ml_insights.lr_accuracy}%</div>
                                    <div className="metric-label">Logistic Regression Accuracy</div>
                                </div>
                                <div className="ml-metric">
                                    <div className="metric-value">{ml_insights.dt_accuracy}%</div>
                                    <div className="metric-label">Decision Tree Accuracy</div>
                                </div>
                            </div>
                            {ml_insights.model_explanation && (
                                <div className="ml-explanation">
                                    <strong>How it works: </strong>
                                    {ml_insights.model_explanation}
                                </div>
                            )}
                            {ml_insights.lr_explanation && (
                                <div className="ml-explanation">
                                    <strong>Logistic Regression: </strong>
                                    {ml_insights.lr_explanation}
                                </div>
                            )}
                            {ml_insights.dt_explanation && (
                                <div className="ml-explanation">
                                    <strong>Decision Tree: </strong>
                                    {ml_insights.dt_explanation}
                                </div>
                            )}
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
