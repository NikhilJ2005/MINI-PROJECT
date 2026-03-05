/**
 * =============================================================================
 *  Automated Recruiting Platform — Frontend Application
 * =============================================================================
 *  Handles:
 *    1. Tab navigation (single, batch, candidates, rankings, compare, JD parser)
 *    2. Single resume analysis with auto-detected GitHub
 *    3. Batch multi-resume upload with candidate ranking table
 *    4. Candidate database browsing
 *    5. Role-based ranking views
 *    6. Side-by-side candidate comparison
 *    7. Job description parsing for custom roles
 *    8. Dashboard stats in header
 * =============================================================================
 */

const API_BASE_URL = "http://localhost:8000";

// =====================================================================
//  TAB NAVIGATION
// =====================================================================
document.querySelectorAll(".tab-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
        document.querySelectorAll(".tab-btn").forEach(function (b) { b.classList.remove("active"); });
        document.querySelectorAll(".tab-content").forEach(function (t) { t.classList.remove("active"); });
        btn.classList.add("active");
        var tabId = "tab-" + btn.dataset.tab;
        document.getElementById(tabId).classList.add("active");

        // Load data when switching tabs
        if (btn.dataset.tab === "candidates") loadCandidates();
    });
});

// =====================================================================
//  INITIALIZATION
// =====================================================================
async function fetchJobRoles() {
    try {
        var response = await fetch(API_BASE_URL + "/job-roles");
        if (!response.ok) throw new Error("Server error: " + response.status);
        var data = await response.json();

        var selects = ["target-role", "batch-role", "ranking-role", "compare-role"];
        selects.forEach(function (id) {
            var sel = document.getElementById(id);
            if (!sel) return;
            sel.innerHTML = '<option value="">Select a target role...</option>';
            data.job_roles.forEach(function (role) {
                var opt = document.createElement("option");
                opt.value = role.name;
                opt.textContent = role.name;
                sel.appendChild(opt);
            });
        });
    } catch (error) {
        console.error("Failed to fetch job roles:", error);
        showError("Cannot connect to the API. Make sure the backend is running on " + API_BASE_URL);
    }
}

async function fetchDashboardStats() {
    try {
        var response = await fetch(API_BASE_URL + "/dashboard");
        if (!response.ok) return;
        var stats = await response.json();
        var container = document.getElementById("dashboard-stats");
        container.innerHTML =
            '<div class="dash-stat"><span class="dash-num">' + stats.total_candidates + '</span> Candidates</div>' +
            '<div class="dash-stat"><span class="dash-num">' + stats.total_analyses + '</span> Analyses</div>' +
            '<div class="dash-stat"><span class="dash-num">' + stats.average_match_score + '%</span> Avg Score</div>';
    } catch (e) { /* ignore */ }
}

document.addEventListener("DOMContentLoaded", function () {
    fetchJobRoles();
    fetchDashboardStats();
});

// =====================================================================
//  FILE UPLOAD HELPERS
// =====================================================================
var resumeFileInput = document.getElementById("resume-file");
var fileNameDisplay = document.getElementById("file-name-display");
var dropZone = document.getElementById("drop-zone");

resumeFileInput.addEventListener("change", function () {
    if (this.files.length > 0) fileNameDisplay.textContent = "Selected: " + this.files[0].name;
    else fileNameDisplay.textContent = "";
});

dropZone.addEventListener("dragover", function (e) { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", function () { dropZone.classList.remove("drag-over"); });
dropZone.addEventListener("drop", function (e) {
    e.preventDefault(); dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
        resumeFileInput.files = e.dataTransfer.files;
        fileNameDisplay.textContent = "Selected: " + e.dataTransfer.files[0].name;
    }
});

// Batch file upload
var batchFilesInput = document.getElementById("batch-files");
var batchFileCount = document.getElementById("batch-file-count");
var batchDropZone = document.getElementById("batch-drop-zone");

batchFilesInput.addEventListener("change", function () {
    if (this.files.length > 0) batchFileCount.textContent = this.files.length + " file(s) selected";
    else batchFileCount.textContent = "";
});

batchDropZone.addEventListener("dragover", function (e) { e.preventDefault(); batchDropZone.classList.add("drag-over"); });
batchDropZone.addEventListener("dragleave", function () { batchDropZone.classList.remove("drag-over"); });
batchDropZone.addEventListener("drop", function (e) {
    e.preventDefault(); batchDropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
        batchFilesInput.files = e.dataTransfer.files;
        batchFileCount.textContent = e.dataTransfer.files.length + " file(s) selected";
    }
});

// =====================================================================
//  SINGLE ANALYSIS
// =====================================================================
document.getElementById("analyze-form").addEventListener("submit", async function (e) {
    e.preventDefault();
    hideError();

    var file = resumeFileInput.files[0];
    var githubUsername = document.getElementById("github-username").value.trim();
    var targetRole = document.getElementById("target-role").value;

    if (!file) { showError("Please upload a resume file."); return; }
    if (!targetRole) { showError("Please select a target job role."); return; }

    var formData = new FormData();
    formData.append("resume_file", file);
    formData.append("github_username", githubUsername);
    formData.append("target_role", targetRole);

    showLoading("Analyzing resume...");

    try {
        var response = await fetch(API_BASE_URL + "/analyze", { method: "POST", body: formData });
        if (!response.ok) {
            var errData = await response.json();
            throw new Error(errData.detail || "Server error: " + response.status);
        }
        var report = await response.json();
        hideLoading();
        renderResults(report);
        fetchDashboardStats();
    } catch (error) {
        hideLoading();
        showError("Analysis failed: " + error.message);
    }
});

// =====================================================================
//  BATCH ANALYSIS
// =====================================================================
document.getElementById("batch-form").addEventListener("submit", async function (e) {
    e.preventDefault();
    hideError();

    var files = batchFilesInput.files;
    var targetRole = document.getElementById("batch-role").value;

    if (!files || files.length === 0) { showError("Please upload at least one resume."); return; }
    if (!targetRole) { showError("Please select a target job role."); return; }

    var formData = new FormData();
    for (var i = 0; i < files.length; i++) {
        formData.append("resume_files", files[i]);
    }
    formData.append("target_role", targetRole);

    showLoading("Analyzing " + files.length + " resumes...");

    try {
        var response = await fetch(API_BASE_URL + "/analyze-batch", { method: "POST", body: formData });
        if (!response.ok) {
            var errData = await response.json();
            throw new Error(errData.detail || "Server error");
        }
        var result = await response.json();
        hideLoading();
        renderBatchResults(result);
        fetchDashboardStats();
    } catch (error) {
        hideLoading();
        showError("Batch analysis failed: " + error.message);
    }
});

function renderBatchResults(result) {
    document.getElementById("batch-results").style.display = "block";

    var summary = document.getElementById("batch-summary");
    summary.innerHTML =
        '<div class="batch-stat">Role: <strong>' + result.target_role + '</strong></div>' +
        '<div class="batch-stat">Analyzed: <strong>' + result.total_analyzed + '/' + result.total_submitted + '</strong></div>' +
        '<div class="batch-stat">Batch ID: <strong>#' + result.batch_id + '</strong></div>';

    var tbody = document.getElementById("rankings-body");
    tbody.innerHTML = "";

    result.rankings.forEach(function (r) {
        var scoreColor = r.match_score >= 75 ? "var(--color-strong)" :
                         r.match_score >= 50 ? "var(--color-claimed)" :
                         r.match_score >= 25 ? "var(--color-score-fair)" : "var(--color-missing)";
        var row = document.createElement("tr");
        row.innerHTML =
            '<td><span class="rank-badge">#' + r.rank + '</span></td>' +
            '<td><strong>' + (r.name || "—") + '</strong></td>' +
            '<td>' + (r.email || "—") + '</td>' +
            '<td>' + (r.github_username ? "@" + r.github_username : "—") + '</td>' +
            '<td><span style="color:' + scoreColor + ';font-weight:700;">' + r.match_score + '%</span></td>' +
            '<td>' + r.missing_count + ' missing</td>' +
            '<td>' + r.confidence.toFixed(1) + '%</td>' +
            '<td><button class="btn-sm" onclick="viewCandidate(' + r.candidate_id + ')">View</button></td>';
        tbody.appendChild(row);
    });

    // Show errors if any
    if (result.errors && result.errors.length > 0) {
        document.getElementById("batch-errors-card").style.display = "block";
        var errHtml = "";
        result.errors.forEach(function (err) {
            errHtml += '<div class="error-item">' + err.file + ': ' + err.error + '</div>';
        });
        document.getElementById("batch-errors").innerHTML = errHtml;
    }
}

// =====================================================================
//  CANDIDATES TAB
// =====================================================================
async function loadCandidates() {
    try {
        var response = await fetch(API_BASE_URL + "/candidates?limit=100");
        if (!response.ok) return;
        var data = await response.json();
        var tbody = document.getElementById("candidates-body");
        tbody.innerHTML = "";

        data.candidates.forEach(function (c) {
            var date = new Date(c.created_at * 1000).toLocaleDateString();
            var skills = c.extracted_skills || [];
            var row = document.createElement("tr");
            row.innerHTML =
                '<td>' + c.id + '</td>' +
                '<td><strong>' + (c.name || "—") + '</strong></td>' +
                '<td>' + (c.email || "—") + '</td>' +
                '<td>' + (c.github_username ? "@" + c.github_username : "—") + '</td>' +
                '<td>' + skills.length + '</td>' +
                '<td class="truncate">' + (c.education || "—") + '</td>' +
                '<td>' + date + '</td>' +
                '<td>' +
                    '<button class="btn-sm" onclick="viewCandidate(' + c.id + ')">View</button> ' +
                    '<button class="btn-sm btn-danger" onclick="deleteCandidate(' + c.id + ')">Delete</button>' +
                '</td>';
            tbody.appendChild(row);
        });

        if (data.candidates.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--color-text-secondary);">No candidates yet. Upload resumes to get started.</td></tr>';
        }
    } catch (e) {
        console.error("Failed to load candidates:", e);
    }
}

document.getElementById("refresh-candidates-btn").addEventListener("click", loadCandidates);

async function viewCandidate(id) {
    try {
        var response = await fetch(API_BASE_URL + "/candidates/" + id);
        if (!response.ok) { alert("Candidate not found."); return; }
        var data = await response.json();

        // Switch to single analysis tab and show the latest analysis report
        document.querySelectorAll(".tab-btn").forEach(function (b) { b.classList.remove("active"); });
        document.querySelectorAll(".tab-content").forEach(function (t) { t.classList.remove("active"); });
        document.querySelector('[data-tab="single"]').classList.add("active");
        document.getElementById("tab-single").classList.add("active");

        if (data.analyses && data.analyses.length > 0) {
            var report = data.analyses[0].report_json;
            if (report && report.target_role) {
                renderResults(report);
            }
        }
    } catch (e) {
        alert("Error loading candidate: " + e.message);
    }
}

async function deleteCandidate(id) {
    if (!confirm("Delete candidate #" + id + "? This cannot be undone.")) return;
    try {
        await fetch(API_BASE_URL + "/candidates/" + id, { method: "DELETE" });
        loadCandidates();
        fetchDashboardStats();
    } catch (e) {
        alert("Error deleting candidate.");
    }
}

// =====================================================================
//  RANKINGS TAB
// =====================================================================
document.getElementById("load-rankings-btn").addEventListener("click", async function () {
    var role = document.getElementById("ranking-role").value;
    if (!role) { showError("Select a role first."); return; }

    try {
        var response = await fetch(API_BASE_URL + "/rankings/" + encodeURIComponent(role));
        if (!response.ok) throw new Error("Failed to load rankings");
        var data = await response.json();

        document.getElementById("role-rankings-results").style.display = "block";
        document.getElementById("role-rankings-title").textContent = "Rankings for " + role;

        var tbody = document.getElementById("role-rankings-body");
        tbody.innerHTML = "";

        if (data.rankings.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">No candidates analyzed for this role yet.</td></tr>';
            return;
        }

        data.rankings.forEach(function (r) {
            var scoreColor = r.match_score >= 75 ? "var(--color-strong)" :
                             r.match_score >= 50 ? "var(--color-claimed)" : "var(--color-missing)";
            var missing = r.missing_skills || [];
            var row = document.createElement("tr");
            row.innerHTML =
                '<td><span class="rank-badge">#' + r.rank + '</span></td>' +
                '<td><strong>' + (r.name || "—") + '</strong></td>' +
                '<td>' + (r.email || "—") + '</td>' +
                '<td>' + (r.github_username ? "@" + r.github_username : "—") + '</td>' +
                '<td><span style="color:' + scoreColor + ';font-weight:700;">' + r.match_score + '%</span></td>' +
                '<td>' + missing.length + ' missing</td>' +
                '<td>' + r.confidence.toFixed(1) + '%</td>';
            tbody.appendChild(row);
        });
    } catch (e) {
        showError("Failed to load rankings: " + e.message);
    }
});

// =====================================================================
//  COMPARE TAB
// =====================================================================
document.getElementById("compare-btn").addEventListener("click", async function () {
    hideError();
    var idsStr = document.getElementById("compare-ids").value.trim();
    var role = document.getElementById("compare-role").value;

    if (!idsStr || !role) { showError("Enter candidate IDs and select a role."); return; }

    var ids = idsStr.split(",").map(function (s) { return parseInt(s.trim()); }).filter(function (n) { return !isNaN(n); });
    if (ids.length < 2) { showError("Enter at least 2 candidate IDs."); return; }

    showLoading("Comparing candidates...");

    try {
        var response = await fetch(API_BASE_URL + "/compare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ candidate_ids: ids, target_role: role }),
        });
        if (!response.ok) {
            var errData = await response.json();
            throw new Error(errData.detail || "Comparison failed");
        }
        var data = await response.json();
        hideLoading();
        renderComparison(data);
    } catch (e) {
        hideLoading();
        showError("Comparison failed: " + e.message);
    }
});

function renderComparison(data) {
    var container = document.getElementById("compare-results");
    container.style.display = "block";

    // Score overview
    var html = '<div class="card"><h2 class="section-title">Comparison: ' + data.target_role + '</h2>';
    html += '<div class="compare-scores">';
    data.candidates.forEach(function (c) {
        var score = c.match_score || 0;
        var scoreColor = score >= 75 ? "var(--color-strong)" :
                         score >= 50 ? "var(--color-claimed)" : "var(--color-missing)";
        html += '<div class="compare-card">' +
            '<div class="compare-name">' + (c.name || "Candidate #" + c.candidate_id) + '</div>' +
            '<div class="compare-score" style="color:' + scoreColor + '">' + score + '%</div>' +
            '<div class="compare-meta">' + (c.email || "") + '</div>' +
            '<div class="compare-meta">@' + (c.github_username || "—") + '</div>' +
            '</div>';
    });
    html += '</div></div>';

    // Skill matrix table
    html += '<div class="card"><h2 class="section-title">Skill Comparison Matrix</h2>';
    html += '<div class="table-container"><table class="skill-table"><thead><tr><th>Skill</th><th>Type</th>';
    data.candidates.forEach(function (c) {
        html += '<th>' + (c.name || "#" + c.candidate_id) + '</th>';
    });
    html += '</tr></thead><tbody>';

    var allSkills = data.required_skills.concat(data.nice_to_have || []);
    allSkills.forEach(function (skill) {
        var isRequired = data.required_skills.indexOf(skill) !== -1;
        html += '<tr><td><strong>' + skill + '</strong></td>';
        html += '<td>' + (isRequired ? '<span class="badge badge-strong">Required</span>' : '<span class="badge badge-claimed_only">Nice</span>') + '</td>';

        var matrix = data.skill_matrix[skill] || {};
        data.candidates.forEach(function (c) {
            var status = matrix[String(c.candidate_id)] || "missing";
            var icon = status === "strong" ? '<span class="check-mark" title="Strong">&#10003;&#10003;</span>' :
                       status === "claimed_only" ? '<span style="color:var(--color-claimed)" title="Resume only">&#10003;</span>' :
                       status === "demonstrated_only" ? '<span style="color:var(--color-demonstrated)" title="GitHub only">G</span>' :
                       '<span class="x-mark" title="Missing">&#10007;</span>';
            html += '<td style="text-align:center">' + icon + '</td>';
        });
        html += '</tr>';
    });

    html += '</tbody></table></div></div>';
    container.innerHTML = html;
}

// =====================================================================
//  JD PARSER TAB
// =====================================================================
document.getElementById("jd-form").addEventListener("submit", async function (e) {
    e.preventDefault();
    hideError();

    var roleName = document.getElementById("jd-role-name").value.trim();
    var description = document.getElementById("jd-text").value.trim();

    if (!description) { showError("Paste a job description first."); return; }

    showLoading("Parsing job description...");

    try {
        var response = await fetch(API_BASE_URL + "/parse-job-description", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ description: description, role_name: roleName }),
        });
        if (!response.ok) throw new Error("Parsing failed");
        var data = await response.json();
        hideLoading();

        var container = document.getElementById("jd-results");
        container.style.display = "block";

        var html = '<div class="card">';
        html += '<h2 class="section-title">Parsed Role: ' + data.role_name + '</h2>';
        html += '<p>Found <strong>' + data.total_skills_found + '</strong> skills in the job description.</p>';

        html += '<h3 style="margin-top:16px;">Required Skills (' + data.required_skills.length + ')</h3>';
        html += '<div class="skill-tags">';
        data.required_skills.forEach(function (s) {
            html += '<span class="skill-tag skill-tag-required">' + s + '</span>';
        });
        html += '</div>';

        html += '<h3 style="margin-top:16px;">Nice-to-Have (' + data.nice_to_have.length + ')</h3>';
        html += '<div class="skill-tags">';
        data.nice_to_have.forEach(function (s) {
            html += '<span class="skill-tag skill-tag-nice">' + s + '</span>';
        });
        html += '</div>';

        if (data.added_to_roles) {
            html += '<p style="margin-top:16px;color:var(--color-strong);font-weight:600;">Role "' + data.role_name + '" has been added. You can now use it for analysis.</p>';
        }

        html += '</div>';
        container.innerHTML = html;

        // Refresh role dropdowns
        fetchJobRoles();
    } catch (e) {
        hideLoading();
        showError("JD parsing failed: " + e.message);
    }
});

// =====================================================================
//  RENDER SINGLE ANALYSIS RESULTS
// =====================================================================
function renderResults(report) {
    var resultsSection = document.getElementById("results-section");
    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

    // Candidate info
    if (report.candidate_info) {
        var info = report.candidate_info;
        var infoCard = document.getElementById("candidate-info-card");
        var infoContainer = document.getElementById("candidate-info-container");
        var hasInfo = info.name || info.email || info.phone || info.education;

        if (hasInfo) {
            infoCard.style.display = "block";
            var html = '<div class="candidate-info-grid">';
            if (info.name) html += '<div class="info-item"><span class="info-label">Name</span><span class="info-value">' + info.name + '</span></div>';
            if (info.email) html += '<div class="info-item"><span class="info-label">Email</span><span class="info-value">' + info.email + '</span></div>';
            if (info.phone) html += '<div class="info-item"><span class="info-label">Phone</span><span class="info-value">' + info.phone + '</span></div>';
            if (info.github_url) html += '<div class="info-item"><span class="info-label">GitHub</span><span class="info-value">' + info.github_url + '</span></div>';
            if (info.linkedin_url) html += '<div class="info-item"><span class="info-label">LinkedIn</span><span class="info-value">' + info.linkedin_url + '</span></div>';
            if (info.education) html += '<div class="info-item info-wide"><span class="info-label">Education</span><span class="info-value">' + info.education + '</span></div>';
            html += '</div>';
            infoContainer.innerHTML = html;
        }
    }

    renderScoreCard(report);
    renderExecutiveSummary(report.executive_summary);
    renderSkillTable("required-skills-body", report.skill_breakdown.required_analysis, true);
    renderSkillTable("nice-skills-body", report.skill_breakdown.nice_to_have_analysis, false);
    renderRecommendations(report.recommendations);
    renderMLInsights(report.ml_insights);
    renderGitHubInsights(report.github_insights);
}

function renderScoreCard(report) {
    var score = report.skill_breakdown.match_score;
    var label = report.executive_summary.match_label;
    document.getElementById("score-value").textContent = Math.round(score);
    document.getElementById("score-label").textContent = label;
    document.getElementById("result-target-role").textContent = report.target_role;

    var color = score >= 75 ? "var(--color-score-excellent)" :
                score >= 50 ? "var(--color-score-good)" :
                score >= 25 ? "var(--color-score-fair)" : "var(--color-score-poor)";

    document.getElementById("score-circle").style.borderColor = color;
    document.getElementById("score-value").style.color = color;
    document.getElementById("score-label").style.color = color;
}

function renderExecutiveSummary(summary) {
    document.getElementById("summary-grid").innerHTML =
        '<div class="summary-card"><div class="stat-value">' + summary.total_resume_skills + '</div><div class="stat-label">Resume Skills</div></div>' +
        '<div class="summary-card"><div class="stat-value">' + summary.total_github_skills + '</div><div class="stat-label">GitHub Skills</div></div>' +
        '<div class="summary-card"><div class="stat-value">' + summary.missing_critical_skills + '</div><div class="stat-label">Missing Critical</div></div>' +
        '<div class="summary-card"><div class="stat-value">' + summary.confidence_score.toFixed(1) + '%</div><div class="stat-label">ML Confidence</div></div>';
}

function renderSkillTable(tbodyId, skills, showConfidence) {
    var tbody = document.getElementById(tbodyId);
    tbody.innerHTML = "";

    skills.forEach(function (skill, index) {
        var row = document.createElement("tr");
        row.style.animationDelay = (index * 0.08) + "s";

        var statusBadge = '<span class="badge badge-' + skill.status + '">' + formatStatus(skill.status) + '</span>';
        var resumeMark = skill.in_resume ? '<span class="check-mark">&#10003;</span>' : '<span class="x-mark">&#10007;</span>';
        var githubMark = skill.in_github ? '<span class="check-mark">&#10003;</span>' : '<span class="x-mark">&#10007;</span>';

        var html = '<td><strong>' + skill.skill + '</strong></td><td>' + statusBadge + '</td><td>' + resumeMark + '</td><td>' + githubMark + '</td>';

        if (showConfidence && skill.probability !== undefined) {
            var prob = Math.round(skill.probability * 100);
            var barColor = prob >= 70 ? "var(--color-strong)" : prob >= 40 ? "var(--color-claimed)" : "var(--color-missing)";
            html += '<td><div class="confidence-bar-container"><div class="confidence-bar"><div class="confidence-bar-fill" style="width:' + prob + '%;background:' + barColor + ';"></div></div><span class="confidence-text">' + prob + '%</span></div></td>';
        }

        row.innerHTML = html;
        tbody.appendChild(row);
    });
}

function formatStatus(status) {
    var labels = { "strong": "Strong", "claimed_only": "Claimed Only", "demonstrated_only": "GitHub Only", "missing": "Missing" };
    return labels[status] || status;
}

function renderRecommendations(recommendations) {
    var container = document.getElementById("recommendations-container");
    if (recommendations.length === 0) {
        container.innerHTML = '<p style="color:var(--color-strong);font-weight:600;">No gaps found!</p>';
        return;
    }
    var html = "";
    recommendations.forEach(function (rec) {
        var cls = rec.priority === "Critical" ? "priority-critical" : "priority-recommended";
        html += '<div class="recommendation-item"><span class="priority-badge ' + cls + '">' + rec.priority + '</span><div class="recommendation-content"><div class="recommendation-action">' + rec.action + '</div><div class="recommendation-hint">' + rec.resource_hint + '</div></div></div>';
    });
    container.innerHTML = html;
}

function renderMLInsights(insights) {
    document.getElementById("ml-insights-container").innerHTML =
        '<div class="ml-grid"><div class="ml-metric"><div class="metric-value">' + insights.lr_accuracy + '%</div><div class="metric-label">Logistic Regression Accuracy</div></div><div class="ml-metric"><div class="metric-value">' + insights.dt_accuracy + '%</div><div class="metric-label">Decision Tree Accuracy</div></div></div>' +
        '<div class="ml-explanation"><strong>How it works:</strong> ' + insights.model_explanation + '</div>' +
        '<div class="ml-explanation" style="margin-top:8px;"><strong>Logistic Regression:</strong> ' + insights.lr_explanation + '</div>' +
        '<div class="ml-explanation" style="margin-top:8px;"><strong>Decision Tree:</strong> ' + insights.dt_explanation + '</div>';
}

function renderGitHubInsights(insights) {
    var container = document.getElementById("github-insights-container");

    if (insights.error && insights.repos_analyzed === 0) {
        container.innerHTML = '<div class="error-message" style="display:block;">GitHub: ' + insights.error + '</div>';
        return;
    }

    var html = '<div class="github-stat"><span class="github-stat-label">Repositories Analyzed</span><span class="github-stat-value">' + insights.repos_analyzed + '</span></div>';

    // Commit activity
    if (insights.commit_activity) {
        var ca = insights.commit_activity;
        html += '<div class="github-stat"><span class="github-stat-label">Recent Commits</span><span class="github-stat-value">' + (ca.total_commits || 0) + '</span></div>';
        html += '<div class="github-stat"><span class="github-stat-label">Active Days (recent)</span><span class="github-stat-value">' + (ca.active_days || 0) + '</span></div>';
    }

    if (insights.top_languages && insights.top_languages.length > 0) {
        html += '<h3 style="margin:16px 0 8px;font-size:0.95rem;">Top Languages</h3>';
        var maxBytes = insights.top_languages[0].bytes || 1;
        insights.top_languages.forEach(function (lang) {
            var pct = Math.round((lang.bytes / maxBytes) * 100);
            html += '<div class="language-bar"><span class="language-bar-name">' + lang.language + '</span><div class="language-bar-track"><div class="language-bar-fill" style="width:' + pct + '%;"></div></div><span class="language-bar-bytes">' + formatBytes(lang.bytes) + '</span></div>';
        });
    }

    if (insights.hidden_strengths && insights.hidden_strengths.length > 0) {
        html += '<div class="github-stat" style="margin-top:12px;"><span class="github-stat-label">Hidden Strengths (GitHub only)</span><span class="github-stat-value">' + insights.hidden_strengths.join(", ") + '</span></div>';
    }

    container.innerHTML = html;
}

function formatBytes(bytes) {
    if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + " MB";
    if (bytes >= 1024) return (bytes / 1024).toFixed(1) + " KB";
    return bytes + " B";
}

// =====================================================================
//  UTILITY FUNCTIONS
// =====================================================================
function showLoading(msg) {
    document.getElementById("loading-overlay").style.display = "flex";
    document.getElementById("loading-step").textContent = msg || "Please wait...";
}
function hideLoading() { document.getElementById("loading-overlay").style.display = "none"; }
function showError(msg) {
    var el = document.getElementById("error-display");
    el.textContent = msg; el.style.display = "block";
}
function hideError() { document.getElementById("error-display").style.display = "none"; }
