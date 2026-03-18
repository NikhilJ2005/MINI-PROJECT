# Resume Skill Gap Analyzer

An AI-powered recruiting platform that analyzes resumes and GitHub profiles to identify skill gaps for target job roles. Features ML-based skill prediction, Groq LLM integration (Llama 4 Scout), and a modern React dashboard.

---

## Key Features

- **Multi-format Resume Parsing**: PDF, DOCX, and TXT support with spaCy NLP + regex extraction
- **GitHub Deep Analysis**: Analyzes repos, languages, topics, dependencies, and READMEs
- **300+ Skills Database**: Comprehensive skills master with alias/synonym resolution (JS→JavaScript, k8s→Kubernetes)
- **20 Job Roles**: From Data Scientist to Blockchain Developer
- **ML Skill Prediction**: Logistic Regression + Decision Tree ensemble with cross-validation
- **Groq LLM Integration** (optional): Llama 4 Scout for AI-enhanced skill extraction, resume coaching, interview prep, and learning paths
- **Batch Analysis**: Upload multiple resumes, rank candidates automatically
- **Candidate Comparison**: Side-by-side radar chart comparison
- **PDF/CSV Export**: Download analysis reports
- **Dark Mode**: Full theme support with keyboard shortcuts
- **Docker Support**: One-command deployment with docker-compose
- **Rate Limiting**: API protection with slowapi

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REACT FRONTEND (Vite + JSX)                       │
│   Upload Resume + GitHub Username + Target Role  →  [Analyze]        │
│   Tabs: Analyze│Batch│Candidates│Rankings│Compare│JD Parser│Dashboard │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  POST /analyze
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND (main.py)                       │
│                                                                      │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐    │
│  │ Resume Parser │   │ GitHub Analyzer  │   │  Skills Master    │    │
│  │(spaCy+Regex+ │   │ (REST API v3 +   │   │  300+ skills      │    │
│  │ DOCX+Aliases)│   │  Deep Analysis)  │   │  + 20 Job Roles   │    │
│  └──────┬───────┘   └────────┬─────────┘   └─────────┬─────────┘    │
│         │                    │                        │              │
│         ▼                    ▼                        ▼              │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │         Feature Engineering → ML Models → Gap Analysis       │    │
│  └──────────────────────────┬───────────────────────────────────┘    │
│                              │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐    │
│  │     Groq LLM (Optional) — Llama 4 Scout via Groq Cloud      │    │
│  │  AI Skill Extraction│Resume Coach│Interview Prep│Learn Path  │    │
│  └──────────────────────────┬───────────────────────────────────┘    │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │              Report Generator + SQLite Database               │    │
│  └──────────────────────────┬───────────────────────────────────┘    │
└──────────────────────────────┼──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RESULTS DASHBOARD                                │
│  Score Card│Radar Chart│Skill Table│AI Coach│Interview Prep│Learn   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Setup Instructions

### Option A: Railway (Recommended for Hosting — $5/mo)

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub Repo**
2. Connect your GitHub repo

**Deploy Backend:**
- Click **Add Service** → **GitHub Repo** → set **Root Directory** to `resume-skill-gap-analyzer/backend`
- Railway auto-detects Python via `nixpacks.toml` (installs deps + spaCy model)
- Add environment variables: `GITHUB_TOKEN`, `GROQ_API_KEY` (both optional)
- Deploy — note the public URL (e.g., `https://resume-analyzer-api-production.up.railway.app`)

**Deploy Frontend:**
- Click **Add Service** → **GitHub Repo** → set **Root Directory** to `resume-skill-gap-analyzer/frontend`
- Set build command: `npm install && npm run build`, publish dir: `dist`
- Add env var: `VITE_API_BASE_URL` = your backend URL from above
- Deploy — your app is live!

> Railway gives $5 free trial credit (no card needed). The $5/mo Hobby plan provides 8GB RAM and no cold starts.

### Option B: Docker (Local)

```bash
git clone <repo-url>
cd resume-skill-gap-analyzer
# Edit backend/.env with your API keys
docker-compose up --build
```
- Backend: http://localhost:8000
- Frontend: http://localhost:5173

### Option C: Render (Free Tier)

Use the `render.yaml` Blueprint at the repo root for one-click deployment. Note: free tier has 60-90s cold starts after 15 min idle.

### Option D: Manual Setup

#### 1. Backend (Manual)
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure environment
cp .env.example .env
# Edit .env: add GITHUB_TOKEN and GROQ_API_KEY

uvicorn main:app --reload --port 8000
```

#### 2. Frontend (Manual)
```bash
cd frontend
npm install
npm run dev
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | No | GitHub personal access token (60→5000 req/hr) |
| `GROQ_API_KEY` | No | Groq Cloud API key for LLM features |
| `CORS_ORIGINS` | No | Comma-separated allowed origins (default: `*`) |

---

## API Endpoints

| Method | Path | Rate Limit | Description |
|--------|------|-----------|-------------|
| GET | `/` | - | Health check + LLM status |
| GET | `/job-roles` | - | Available job roles (20+) |
| GET | `/skills-master` | - | Full skills list (300+) |
| POST | `/analyze` | 10/min | Analyze resume file upload |
| POST | `/analyze-text` | - | Analyze raw resume text |
| POST | `/analyze-batch` | 3/min | Batch multi-resume analysis |
| GET | `/candidates` | - | List all candidates |
| GET | `/candidates/{id}` | - | Candidate detail + history |
| POST | `/compare` | - | Side-by-side comparison |
| POST | `/parse-job-description` | - | Extract skills from JD |
| GET | `/rankings/{role}` | - | Ranked candidates for role |
| GET | `/dashboard` | - | Analytics dashboard |
| GET | `/model-metrics` | - | ML model performance |
| GET | `/model-retrain` | 1/hr | Retrain ML models |

---

## Groq LLM Integration

When `GROQ_API_KEY` is set, the app activates AI-powered features using **Llama 4 Scout** (17B params, 460+ tokens/sec):

1. **Smart Skill Extraction**: Detects implied skills regex misses (e.g., "built distributed system" → Kafka, microservices)
2. **AI Resume Coach**: Personalized improvement tips, bullet point suggestions, ATS keyword recommendations
3. **Interview Prep**: Role-specific questions based on skill gaps and unproven claims
4. **Learning Path**: Prioritized weekly plan with specific resources and project ideas

All LLM features are **optional** — the app works fully without them.

---

## How the ML Models Work

### Ensemble Approach
- **Logistic Regression**: Calibrated confidence scores (0-100%) per skill
- **Decision Tree** (max depth 4): Interpretable rules + feature importance
- Both trained on HuggingFace datasets with cross-validation

### Skill Gap Scoring
1. **Match Score**: `(present_required / total_required) × 100`
2. **Per-Skill Status**: Strong | Claimed Only | Demonstrated Only | Missing
3. **Confidence**: Average ML probability across required skills

---

## Project Structure

```
resume-skill-gap-analyzer/
├── backend/
│   ├── main.py                      # FastAPI app with 19 endpoints
│   ├── requirements.txt             # Python dependencies
│   ├── .env                         # API keys (gitignored)
│   ├── modules/
│   │   ├── resume_parser.py         # PDF/DOCX/TXT + skill extraction + aliases
│   │   ├── github_analyzer.py       # GitHub API deep analysis
│   │   ├── feature_engineering.py   # ML feature vectors
│   │   ├── ml_model.py             # LR + DT ensemble
│   │   ├── skill_gap_analyzer.py   # Gap computation
│   │   ├── report_generator.py     # Report compilation
│   │   ├── groq_llm.py            # Groq LLM integration (Llama 4 Scout)
│   │   └── database.py            # SQLite persistence
│   └── data/
│       ├── skills_master.json      # 300+ skills across 6 categories
│       ├── job_roles.json          # 20 job roles with requirements
│       ├── skill_aliases.json      # 150+ alias→canonical mappings
│       └── dataset_loader.py       # HuggingFace dataset integration
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Main app with 7 tabs + dark mode
│   │   ├── Results.jsx             # Analysis results + AI sections
│   │   ├── ErrorBoundary.jsx       # Error recovery component
│   │   └── ... (18 components)
│   └── package.json
├── Dockerfile                       # Backend Docker image
├── docker-compose.yml              # Full stack deployment
├── sample_resumes/                 # Test resume files
└── README.md
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Alt+1-7` | Switch tabs |
| `Alt+N` | New analysis |
| `Alt+D` | Toggle dark mode |
| `Ctrl+Enter` | Submit form |
| `Escape` | Clear results |
