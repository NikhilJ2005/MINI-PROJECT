import "./cssFile/SkillTable.css";
import ConfidenceBar from "./ConfidenceBar";

function SkillTable({ title, analysis }) {
    if (!analysis) return null;
    const showConfidence = analysis.some((item) => item.probability !== undefined);

    return (
        <div className="card">
            <h3>{title}</h3>
            <div className="table-container">
                <table className="skill-table">
                    <thead>
                        <tr>
                            <th>Skill</th>
                            <th>Status</th>
                            <th>In Resume</th>
                            <th>On GitHub</th>
                            {showConfidence && <th>ML Confidence</th>}
                        </tr>
                    </thead>
                    <tbody>
                        {analysis.map((item, index) => (
                            <tr key={index}>
                                <td>{item.skill}</td>
                                <td>
                                    <span
                                        className={`status-pill status-${item.status}`}
                                        title={getStatusTooltip(item.status)}
                                    >
                                        <span className="status-icon">{getStatusIcon(item.status)}</span>
                                        {formatStatus(item.status)}
                                    </span>
                                </td>
                                <td className="check-cell">
                                    {item.in_resume ? (
                                        <span className="indicator indicator-yes">&#10003;</span>
                                    ) : (
                                        <span className="indicator indicator-no">&#10007;</span>
                                    )}
                                </td>
                                <td className="check-cell">
                                    {item.in_github ? (
                                        <span className="indicator indicator-yes">&#10003;</span>
                                    ) : (
                                        <span className="indicator indicator-no">&#10007;</span>
                                    )}
                                </td>
                                {showConfidence && (
                                    <td>
                                        {item.probability != null ? (
                                            <ConfidenceBar probability={item.probability} />
                                        ) : (
                                            "-"
                                        )}
                                    </td>
                                )}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function getStatusIcon(status) {
    switch (status) {
        case "strong": return "✓";
        case "claimed_only": return "◐";
        case "demonstrated_only": return "◑";
        case "missing": return "✕";
        default: return "?";
    }
}

function getStatusTooltip(status) {
    switch (status) {
        case "strong": return "Found in both resume and GitHub — strong evidence";
        case "claimed_only": return "Listed on resume but not found on GitHub";
        case "demonstrated_only": return "Found on GitHub but not listed on resume";
        case "missing": return "Not found in resume or GitHub — skill gap";
        default: return "";
    }
}

function formatStatus(status) {
    switch (status) {
        case "strong": return "Strong";
        case "claimed_only": return "Claimed";
        case "missing": return "Missing";
        case "demonstrated_only": return "Demo Only";
        default: return status;
    }
}

export default SkillTable;
