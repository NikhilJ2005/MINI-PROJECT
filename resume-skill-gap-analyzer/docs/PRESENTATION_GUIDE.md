# SkillSync — Presentation Guide: Code Explanation & Feature Walkthrough

---

## 1. PROJECT OVERVIEW — What Is SkillSync?

**One-liner:** An AI-powered recruiting platform that analyzes resumes against job roles, ranks candidates, and identifies skill gaps — using ML models, LLM intelligence, and GitHub portfolio analysis.

**Tech Stack:**
- **Backend:** Python + FastAPI (async REST API)
- **Frontend:** React 18 + Vite (SPA with 7 tabs)
- **ML:** scikit-learn (Logistic Regression + Decision Tree ensemble)
- **AI/LLM:** Groq API (Llama models) for deep analysis
- **Database:** SQLite with WAL mode
- **PDF Generation:** PyMuPDF (server-side)
- **Charts:** Recharts (radar, bar, pie/donut)
- **Deployment:** Railway

---

## 2. ARCHITECTURE — The 6-Stage Pipeline

Every resume goes through this pipeline:

```
Resume Upload → [1] Parse → [2] GitHub Analyze → [3] Feature Engineer → [4] ML Predict → [5] Gap Analyze → [6] Report Generate
```

**Stage 1 — Resume Parser** (`modules/resume_parser.py`)
- Accepts PDF, DOCX, TXT (up to 10MB)
- Extracts text using PyMuPDF / python-docx
- Skill extraction via regex word-boundary matching + spaCy NLP noun chunks
- Auto-detects: name (spaCy NER + first-line heuristic), email, phone, GitHub URL, LinkedIn, education
- Handles skill aliases (e.g., "JS" → "JavaScript", "k8s" → "Kubernetes")

**Stage 2 — GitHub Analyzer** (`modules/github_analyzer.py`)
- Async HTTP with connection pooling (up to 50 concurrent repos)
- Analyzes: languages, topics, 10+ dependency file types (package.json, requirements.txt, Cargo.toml, etc.)
- 400+ package-to-skill mappings (e.g., Flask → Python, React → JavaScript)
- Parses Dockerfiles and docker-compose for infrastructure skills
- Proficiency scoring: logarithmic scale based on byte count per language
- Retry logic with exponential backoff + semaphore rate limiting

**Stage 3 — Feature Engineering** (`modules/feature_engineering.py`)
- Creates 13 ML-ready features per skill:
  - 3 binary: `in_resume`, `in_github`, `is_required`
  - 10 continuous: `resume_skill_ratio`, `github_skill_ratio`, `skill_source_agreement`, `github_evidence_strength`, `profile_consistency_score`, `category_match_score`, `skill_rarity_score`, `both_sources`, etc.
- Outputs a DataFrame (one row per skill, 13 columns)

**Stage 4 — ML Prediction** (`modules/ml_model.py`)
- **Ensemble model:** 35% Logistic Regression + 65% Decision Tree
- LR pipeline: StandardScaler → PolynomialFeatures (degree=2, interactions only) → LR (C=1.5, balanced weights)
- DT: max_depth=15, min_samples_split=5, balanced weights
- Training: 80/20 stratified split, 5-fold cross-validation
- Output: probability per skill (0-1), binary prediction, feature importance rankings
- Models persisted to disk with joblib

**Stage 5 — Skill Gap Analyzer** (`modules/skill_gap_analyzer.py`)
- Classifies each required skill as:
  - **Strong** = in both resume AND GitHub (verified)
  - **Claimed Only** = resume only (unverified)
  - **Demonstrated Only** = GitHub only (hidden strength)
  - **Missing** = neither source
- **Composite Score formula** (the master ranking metric):
  ```
  45% match_score + 15% nice_to_have + 15% ML_confidence + 10% github_bonus + 10% strong_ratio + 5% claim_penalty
  ```

**Stage 6 — Report Generator** (`modules/report_generator.py`)
- Builds structured JSON report with: executive summary, skill breakdown, recommendations, learning path
- Priority assignment: Urgent (required + missing), High (required + claimed only), Medium, Low
- Topological sort for learning path (respects skill prerequisites like "Kubernetes requires Docker + Linux")
- Score labels: "Excellent Match" (75+), "Good Match" (50-74), "Fair Match" (25-49), "Poor Match" (<25)

---

## 3. AI/LLM FEATURES — Groq Integration (`modules/groq_llm.py`)

All AI features gracefully degrade if no API key is set (the app works without AI, just less rich).

| Feature | What It Does | Presentation Talking Point |
|---------|-------------|---------------------------|
| **Enhanced Skill Extraction** | Catches implied skills regex misses (e.g., "REST API with auth" → implies OAuth, JWT) | "AI catches contextual skills humans write between the lines" |
| **AI Resume Coach** | Personalized tips, ATS keyword suggestions, bullet point rewrites | "Helps candidates improve their resume for this specific role" |
| **Interview Question Generator** | 5-7 questions targeting claimed-but-unproven skills, with difficulty levels and prep hints | "Generates verification questions for recruiters to validate claims" |
| **AI Learning Path** | Week-by-week study plan with real resource URLs (enriched via Serper web search) | "Creates a personalized upskilling roadmap with verified links" |
| **Skill Credibility Assessment** | Analyzes resume text for evidence of real experience vs keyword stuffing | "Flags questionable claims vs credible ones — like a lie detector for resumes" |
| **Role-Fit Narrative** | Fit score (1-10), standout qualities, growth areas, onboarding estimate | "Gives recruiters a narrative story, not just numbers" |
| **Culture & Soft Skills** | Communication score, leadership indicators, team fit, work style analysis | "Goes beyond technical skills to assess soft skills from resume language" |
| **Candidate Summary** | One-line headline, hiring recommendation (Strong Hire/Hire/Maybe/Pass), salary positioning | "The recruiter gets a TL;DR with a clear recommendation" |
| **Batch Executive Report** | Pool quality assessment, common gaps, hiring strategy advice across all candidates | "When you upload 50 resumes, AI summarizes the entire talent pool" |

**Technical highlights to mention:**
- Model fallback chain: Llama 4 Scout → Llama 3.3 → Llama 3.1 (automatic retry on failure)
- In-memory LRU cache (100 entries, 1-hour TTL) to avoid duplicate API calls
- JSON schema validation on all LLM responses
- Temperature tuning per task (0.1 for precise extraction, 0.5 for creative narratives)

---

## 4. FRONTEND — 7 Tabs Explained

### Tab 1: Analyze
- Upload resume (drag & drop or paste), enter GitHub username, select target role
- Returns a full analysis report with 17 visual sections
- Key sections: Score Card (circular gauge), Radar Chart (skill coverage), Skill Table (color-coded status), Recommendations, AI sections

### Tab 2: Batch
- Upload up to 50 resumes at once for a target role
- Returns ranked candidate table sorted by composite score
- AI generates an executive report for the entire talent pool
- CSV export of rankings

### Tab 3: Candidates
- Database of all analyzed candidates
- Search by name/email/GitHub
- Click into any candidate to see all their analyses across different roles
- Delete candidates (cascade removes analyses)

### Tab 4: Rankings
- Select a role → see all candidates ranked by composite score
- Persistent rankings from database (not recalculated)

### Tab 5: Compare
- Select 2-5 candidates + a role
- Side-by-side skill matrix showing who has what
- Status indicators: Strong/Claimed/Demonstrated/Missing per candidate per skill

### Tab 6: JD Parser
- Paste any job description text
- AI + regex extract required and nice-to-have skills
- Creates a custom role saved to the database for future analyses

### Tab 7: Dashboard
- 4 KPI cards: total candidates, analyses, avg match score, batch jobs
- Charts: analyses by role (bar), score distribution (donut), top skill gaps (horizontal bar)
- Recent activity feed with fuzzy timestamps
- Top candidates list
- ML model accuracy display

### Cross-cutting UI features:
- Dark mode toggle (Alt+D, persisted in localStorage)
- Keyboard shortcuts (Alt+1-7 for tabs, Ctrl+Enter to submit, Escape to clear)
- Toast notifications (success/error/info)
- Error boundaries wrapping every tab
- Analysis history sidebar (grouped by Today/Yesterday/This Week/Earlier)
- PDF export (server-side with client-side html2canvas fallback)
- CSV export with AI enrichment (recommendations, credibility, culture data)

---

## 5. DATABASE DESIGN (`modules/database.py`)

4 tables in SQLite (WAL mode for concurrent reads):

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `candidates` | Store candidate profiles | name, email, github_username, resume_text, extracted_skills (JSON) |
| `analyses` | Per-role analysis results | candidate_id (FK), target_role, match_score, composite_score, report_json (full report) |
| `batch_jobs` | Batch upload tracking | target_role, total_candidates, status, results (JSON) |
| `batch_results` | Links batch → candidates | batch_id, candidate_id, analysis_id, rank |

Indexed on: composite_score DESC, match_score DESC, candidate_id, email, github_username

---

## 6. EXPORT FEATURES

### PDF Export (Server-side, PyMuPDF)
- Professional A4 formatting with color-coded sections
- Skill status badges: `[OK]` green, `[RESUME]` yellow, `[GITHUB]` blue, `[MISSING]` red
- All 16+ sections: candidate info, executive summary, AI assessment, skill breakdown, recommendations, culture fit, learning path, ML insights, credibility, role-fit, resume coach, interview prep, AI learning path, GitHub insights
- Auto page breaks, text wrapping, timestamp footer

### CSV Export
- **Single analysis:** Skills + executive summary + recommendations + credibility + culture fit
- **Batch:** Ranked table with composite score, match score, confidence, missing skills per candidate

---

## 7. KEY TALKING POINTS FOR PRESENTATION

1. **"Not just keyword matching"** — Uses ML ensemble (LR + DT) with 13 engineered features and polynomial interactions to predict skill proficiency, not just string matching.

2. **"Two sources of truth"** — Cross-references resume claims with GitHub evidence. Skills verified in both sources are "Strong"; resume-only claims are flagged for interview verification.

3. **"AI that degrades gracefully"** — All LLM features are optional. Without Groq API key, the platform still works with ML + regex. With it, you get 9 additional AI analysis layers.

4. **"Composite scoring is multi-signal"** — The ranking formula weighs 6 factors (match, nice-to-have, confidence, GitHub bonus, verification ratio, claim penalty) — not just a simple percentage.

5. **"Real-world skill prerequisites"** — Learning paths use topological sort to respect dependencies (e.g., learn Docker before Kubernetes, learn JavaScript before React).

6. **"Handles scale"** — Batch analysis processes up to 50 resumes concurrently, async GitHub API calls with rate limiting, and generates ranked output with AI executive summary.

7. **"Custom roles from any JD"** — Paste any job description and the system extracts skills automatically, creating a reusable role for future analyses.

8. **"Full export pipeline"** — Everything visible on screen can be exported as PDF or CSV, including all AI insights.

---

## 8. API ENDPOINTS SUMMARY (20 endpoints)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | System status + capability check |
| `/dashboard` | GET | Analytics aggregation (7 SQL queries) |
| `/job-roles` | GET | Available roles with skill requirements |
| `/skills-master` | GET | Complete skills taxonomy |
| `/analyze` | POST | Single resume analysis (file upload) |
| `/analyze-text` | POST | Single resume analysis (text input) |
| `/analyze-batch` | POST | Multi-resume batch analysis (up to 50) |
| `/batch/{id}` | GET | Retrieve batch results |
| `/candidates` | GET | Paginated candidate list |
| `/candidates/{id}` | GET | Candidate detail + all analyses |
| `/candidates/{id}` | DELETE | Remove candidate (cascade) |
| `/rankings/{role}` | GET | Ranked candidates for a role |
| `/compare` | POST | Side-by-side candidate comparison |
| `/parse-job-description` | POST | Extract skills from JD text |
| `/analysis-history` | GET | Recent analyses list |
| `/analysis/{id}` | GET | Full analysis report by ID |
| `/model-metrics` | GET | ML model performance stats |
| `/model-retrain` | GET | Force model retraining |
| `/dataset-status` | GET | Training data metadata |
| `/export/pdf/{id}` | GET | PDF report generation |
| `/export/batch-csv/{id}` | GET | Batch CSV export |

Rate limiting applied: 10/min for analysis, 3/min for batch, 1/hour for retrain.

---

## 9. WHAT MAKES THIS "FRONTIER"

- **Multi-modal analysis**: Resume text + GitHub API + LLM reasoning = three independent signals
- **Ensemble ML with polynomial feature interactions**: Not just logistic regression — captures non-linear relationships between features
- **AI credibility scoring**: Detects resume padding by analyzing contextual evidence
- **Web search enrichment**: AI learning paths get real, verified URLs via Serper API (not hallucinated)
- **Production patterns**: Rate limiting, retry with backoff, graceful degradation, caching, error boundaries, async processing
