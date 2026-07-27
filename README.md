# Resume Skill Gap Analyzer — Mini Project

AI-assisted resume & GitHub profile analyzer that detects skill gaps relative to target job roles, ranks candidates, and generates actionable coaching and learning plans.

- Backend: FastAPI (Python) — resume parsing, GitHub analysis, ML models, report generation
- Frontend: Vite + React — upload resumes, view results, compare candidates
- Optional LLM features via Groq (Llama 4 Scout) for enhanced extraction, coaching, and interview prep
- Docker + docker-compose for one-command local deployment

---

## Quick demo / TL;DR

- Upload a resume (PDF/DOCX/TXT) and provide a GitHub username and target role → the app returns:
  - Detected skills and confidence scores
  - Per-skill status (Demonstrated / Claimed only / Missing)
  - Overall match & gap score vs the role
  - Suggested learning path and interview prep (LLM-powered if enabled)

---

## Key features

- Multi-format resume parsing (PDF, DOCX, TXT) using spaCy + regex + alias mapping
- Deep GitHub analysis (repos, languages, topics, dependencies, README parsing)
- 300+ skills master with alias/synonym resolution
- 20 job role templates with required/desired skills
- ML ensemble: Logistic Regression + Decision Tree for skill prediction & confidence
- Optional LLM (Groq Llama 4 Scout) for advanced extraction, personalized coaching, and question generation
- Batch analysis, candidate comparison, PDF/CSV export
- Docker-ready + deployment blueprints (Railway / Render)

---

## Stack

- Languages: Python (backend), JavaScript/React (frontend), CSS
- Backend: FastAPI, Uvicorn
- Frontend: Vite + React
- ML / NLP: spaCy, scikit-learn (Logistic Regression + Decision Tree), HuggingFace dataset integration
- Optional LLM: Groq Llama 4 Scout
- Persistence: SQLite (local)
- Deployment: Docker & docker-compose, Railway & Render blueprints included

---

## Repository layout

```
.
├── resume-skill-gap-analyzer/
│   ├── backend/
│   │   ├── main.py                  # FastAPI app (endpoints & business logic)
│   │   ├── requirements.txt
│   │   ├── modules/
│   │   │   ├── resume_parser.py
│   │   │   ├── github_analyzer.py
│   │   │   ├── feature_engineering.py
│   │   │   ├─�� ml_model.py
│   │   │   ├── skill_gap_analyzer.py
│   │   │   ├── report_generator.py
│   │   │   ├── groq_llm.py
│   │   │   └── database.py
│   │   └── data/
│   │       ├── skills_master.json
│   │       ├── job_roles.json
│   │       └── skill_aliases.json
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── App.jsx
│   │   │   ├── Results.jsx
│   │   │   └── ...components
│   │   └── package.json
│   ├── Dockerfile
│   └── docker-compose.yml
├── sample_resumes/
└── README.md (this file)
```

---

## How it fits together

1. Frontend (Vite + React) provides an upload form and UI to enter GitHub username and target role.
2. Frontend calls the FastAPI backend (/analyze, /analyze-text, /analyze-batch).
3. Backend:
   - Parses resume text and extracts candidate claims (spaCy + regex + aliases)
   - Analyzes GitHub profile/repos for demonstrated skills
   - Featurizes inputs and runs ensemble ML models for per-skill probability
   - Computes gap analysis and generates reports
   - If GROQ_API_KEY is present, invokes LLM augmentation for coaching and interview prep
4. Results are returned to the frontend for visualization (radar charts, tables, reports).

---

## Quick Start — Local (Docker)

1. Clone:
```bash
git clone https://github.com/NikhilJ2005/MINI-PROJECT.git
cd MINI-PROJECT/resume-skill-gap-analyzer
```

2. Build & run:
```bash
# from resume-skill-gap-analyzer/
docker-compose up --build
```

3. Open:
- Backend: http://localhost:8000
- Frontend: http://localhost:5173

---

## Quick Start — Manual (dev)

Backend
```bash
cd resume-skill-gap-analyzer/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# copy example env (if present) and edit
cp .env.example .env
# set GITHUB_TOKEN (optional) and GROQ_API_KEY (optional)

uvicorn main:app --reload --port 8000
```

Frontend
```bash
cd resume-skill-gap-analyzer/frontend
npm install
npm run dev
```

---

## Environment variables

At minimum you can run the app without API keys, but these enable useful integrations:

- GITHUB_TOKEN (optional) — GitHub API token increase rate limits for repo analysis
- GROQ_API_KEY (optional) — enable Groq LLM features (Llama 4 Scout)
- CORS_ORIGINS (optional) — comma-separated list; default `*`

---

## Notable API endpoints (backend)

- GET /               ��� health check
- POST /analyze       — analyze uploaded resume file
- POST /analyze-text  — analyze raw resume text
- POST /analyze-batch — analyze multiple resumes
- GET /job-roles      — list supported job roles
- GET /skills-master  — list all skills
- GET /candidates     — list candidates
- GET /dashboard      — analytics dashboard

(See resume-skill-gap-analyzer/backend/main.py for full endpoint list and rate limits.)

---

## ML & LLM notes

- Ensemble: Logistic Regression (calibrated) + Decision Tree (max depth 4) — used together for probabilities and explainability.
- Skill gap scoring includes match percentage, per-skill status, and averaged confidence.
- LLM features are optional and only enabled when GROQ_API_KEY is supplied; without it, the system uses deterministic parsing + ML.

---

## Development notes

- Resume parsing relies on spaCy and pattern-based extraction; add aliases to data/skill_aliases.json to improve detection.
- GitHub analysis uses REST calls — supply GITHUB_TOKEN to avoid low rate limits.
- Tests / sample resumes are in sample_resumes/ — use them for quick local checks.

---

## Deployments & hosting

- Railway: project includes a `railway.json` and recommended instructions to deploy frontend & backend as separate services.
- Render: `render.yaml` blueprint included for one-click deploy on Render.
- Docker Compose: full local stack for development & testing.

---

## Contributing

Contributions welcome — please open issues for feature requests or bugs. Suggested workflow:
1. Fork the repo
2. Create a feature branch
3. Add tests where appropriate
4. Open a pull request with a clear description

Please add or update skill aliases and job role JSONs carefully — they directly affect model inputs and scoring.

---

## License & contact

- License: (If you have one, add it here — e.g., MIT). If you want, I can add an MIT license file.
- Author / Contact: NikhilJ2005 (GitHub) — open issues or PRs for questions.

---
