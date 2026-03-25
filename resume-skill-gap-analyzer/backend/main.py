"""
=============================================================================
 Automated Recruiting Platform — FastAPI Backend
=============================================================================
 Full-stack recruiting platform that analyzes resumes and GitHub profiles
 against target job roles using ML. Supports:
   - Single resume analysis
   - Batch multi-resume upload with candidate ranking
   - Candidate database with SQLite persistence
   - Side-by-side candidate comparison
   - Auto-extraction of GitHub URLs from resumes
   - Deep GitHub repo analysis (deps, READMEs, commit activity)
   - Job description parsing for custom roles
   - Dashboard analytics

 Run with: uvicorn main:app --reload --port 8000
=============================================================================
"""

import json
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Import all pipeline modules
from modules.resume_parser import ResumeParser, compile_skill_patterns, set_compiled_patterns, get_compiled_patterns, set_flat_skills, set_skill_aliases
from modules.github_analyzer import GitHubAnalyzer
from modules.feature_engineering import FeatureEngineer
from modules.ml_model import SkillGapMLModel
from modules.skill_gap_analyzer import SkillGapAnalyzer
from modules.report_generator import ReportGenerator
from modules.database import Database
from modules.groq_llm import (
    is_available as groq_available,
    extract_skills_with_llm,
    generate_ai_feedback,
    generate_interview_questions,
    generate_learning_path,
    generate_candidate_summary,
    generate_batch_executive_report,
    generate_jd_skills_extraction,
    generate_culture_fit_analysis,
    generate_skill_credibility_assessment,
    generate_role_fit_narrative,
)
from modules.web_search import (
    is_available as serper_available,
    enrich_learning_path,
)
from data.dataset_loader import DatasetLoader

# ---------------------------------------------------------------------------
#  Load Environment Variables
# ---------------------------------------------------------------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
#  Application State — all pipeline components in one place
# ---------------------------------------------------------------------------
class AppState:
    """Holds all pipeline components. Initialized once at startup."""
    def __init__(self):
        self.job_roles_data: Dict = {}
        self.skills_master: Dict[str, List[str]] = {}
        self.resume_parser: ResumeParser = None          # type: ignore
        self.github_analyzer: GitHubAnalyzer = None      # type: ignore
        self.feature_engineer: FeatureEngineer = None    # type: ignore
        self.ml_model: SkillGapMLModel = None            # type: ignore
        self.skill_gap_analyzer: SkillGapAnalyzer = None # type: ignore
        self.report_generator: ReportGenerator = None    # type: ignore
        self.dataset_loader: DatasetLoader = None        # type: ignore
        self.db: Database = None                         # type: ignore
        self.last_retrain_time: float = 0.0

state = AppState()


# ---------------------------------------------------------------------------
#  Application Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AUTOMATED RECRUITING PLATFORM — Starting Up")

    # Load data files
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    with open(os.path.join(data_dir, "job_roles.json"), "r") as f:
        state.job_roles_data = json.load(f)
    # Validate job roles structure
    for role_name, role_data in state.job_roles_data.items():
        if "required_skills" not in role_data:
            logger.warning(f"Job role '{role_name}' missing 'required_skills' — adding empty list")
            role_data["required_skills"] = []
        if "nice_to_have" not in role_data:
            role_data["nice_to_have"] = []
    logger.info(f"Loaded {len(state.job_roles_data)} job roles.")

    with open(os.path.join(data_dir, "skills_master.json"), "r") as f:
        state.skills_master = json.load(f)
    total_skills = sum(len(v) for v in state.skills_master.values())
    logger.info(f"Loaded {total_skills} skills across {len(state.skills_master)} categories.")

    # Load skill aliases
    skill_aliases = {}
    aliases_path = os.path.join(data_dir, "skill_aliases.json")
    if os.path.exists(aliases_path):
        with open(aliases_path, "r") as f:
            skill_aliases = json.load(f)
        set_skill_aliases(skill_aliases)
        logger.info(f"Loaded {len(skill_aliases)} skill aliases.")

    # Pre-compile regex patterns for skill matching (used across all modules)
    compiled_patterns = compile_skill_patterns(state.skills_master, skill_aliases)
    set_compiled_patterns(compiled_patterns)
    logger.info(f"Pre-compiled {len(compiled_patterns)} skill regex patterns.")

    # Flatten skills_master once for reuse across modules
    set_flat_skills(state.skills_master)

    # Initialize pipeline modules
    logger.info("Initializing pipeline modules...")
    state.resume_parser = ResumeParser()
    state.github_analyzer = GitHubAnalyzer(github_token=GITHUB_TOKEN if GITHUB_TOKEN else None)
    state.feature_engineer = FeatureEngineer()
    state.ml_model = SkillGapMLModel()
    state.skill_gap_analyzer = SkillGapAnalyzer(skills_master=state.skills_master)
    state.report_generator = ReportGenerator()
    state.dataset_loader = DatasetLoader()
    state.db = Database()

    # Load saved models or train fresh
    models_loaded = state.ml_model.load_models()
    if not models_loaded:
        logger.info("Training ML models from scratch...")
        X, y, source = state.dataset_loader.load_training_data()
        state.ml_model.train(X, y, dataset_source=source, use_cross_validation=True)
    else:
        logger.info("Using cached models — skipping retraining")

    # Check Groq LLM availability
    llm_status = "enabled" if groq_available() else "disabled (set GROQ_API_KEY to enable)"
    logger.info(f"Groq LLM integration: {llm_status}")

    logger.info(f"Server ready | LR acc: {state.ml_model.lr_accuracy}% | "
                f"DT acc: {state.ml_model.dt_accuracy}% | "
                f"Roles: {len(state.job_roles_data)} | Skills: {total_skills}")

    yield

    if state.github_analyzer:
        await state.github_analyzer.close()
    logger.info("Shutting down Automated Recruiting Platform.")


# ---------------------------------------------------------------------------
#  Create FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Automated Recruiting Platform",
    description="Analyzes resumes and GitHub profiles to rank candidates for target job roles using ML.",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow Railway domain + any env-configured origins
_default_origins = [
    "https://mini-project-production-b783.up.railway.app",
    "http://localhost:5173",   # Vite dev server
    "http://localhost:8000",   # Local backend
]
_env_origins = os.getenv("CORS_ORIGINS", "")
if _env_origins == "*":
    logger.warning("[Security] CORS_ORIGINS='*' is unsafe with credentials; using defaults")
    cors_origins = _default_origins
elif _env_origins:
    cors_origins = list(set(_default_origins + _env_origins.split(",")))
else:
    cors_origins = _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["Content-Disposition"],
)


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self' https://api.github.com https://google.serper.dev; "
        "frame-ancestors 'none'; "
        "upgrade-insecure-requests"
    )
    return response


# ---------------------------------------------------------------------------
#  Pydantic Models
# ---------------------------------------------------------------------------
class TextAnalyzeRequest(BaseModel):
    resume_text: str = Field(max_length=500_000)
    github_username: str = Field(max_length=39)
    target_role: str = Field(max_length=200)


class CompareRequest(BaseModel):
    candidate_ids: List[int]
    target_role: str = Field(max_length=200)


class JobDescriptionRequest(BaseModel):
    description: str = Field(max_length=100_000)
    role_name: str = Field(default="", max_length=200)


# ---------------------------------------------------------------------------
#  Helper: Run full analysis pipeline for one candidate
# ---------------------------------------------------------------------------
async def _run_single_analysis(
    resume_text: str,
    claimed_skills: List[str],
    github_username: str,
    target_role: str,
    personal_info: Dict = None,
    filename: str = "",
) -> Dict:
    """Run the full pipeline for a single candidate. Returns report + metadata."""

    # GitHub analysis — auto-detect username from resume if not provided
    if not github_username and personal_info and personal_info.get("github_username"):
        github_username = personal_info["github_username"]

    github_result = {"demonstrated_skills": [], "repos_analyzed": 0,
                     "raw_languages": {}, "raw_topics": [],
                     "commit_activity": {}, "error": "No GitHub username provided"}
    demonstrated_skills = []

    # Validate GitHub username format (1-39 chars, alphanumeric + hyphens)
    _gh_user_re = re.compile(r'^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,37}[a-zA-Z0-9])?$')
    if github_username and not _gh_user_re.match(github_username):
        logger.warning(f"[Security] Invalid GitHub username format, skipping: {github_username!r}")
        github_username = ""

    if github_username:
        try:
            github_result = await state.github_analyzer.analyze_github_profile(github_username, state.skills_master)
            demonstrated_skills = github_result["demonstrated_skills"]
        except Exception as e:
            github_result["error"] = str(e)

    # LLM-enhanced skill extraction (if Groq is available)
    if groq_available():
        try:
            llm_extra_skills = extract_skills_with_llm(resume_text, claimed_skills)
            if llm_extra_skills:
                # Only add skills that are in the master list
                all_master = [s for cat in state.skills_master.values() for s in cat]
                valid_extras = [s for s in llm_extra_skills if s in all_master]
                claimed_skills = list(set(claimed_skills) | set(valid_extras))
                logger.info(f"[GroqLLM] Added {len(valid_extras)} LLM-detected skills")
        except Exception as e:
            logger.warning(f"[GroqLLM] Skill extraction failed (non-critical): {e}")

    # Feature engineering
    role_data = state.job_roles_data.get(target_role)
    if not role_data:
        raise ValueError(f"Unknown target role: '{target_role}'")
    skill_matrix = state.feature_engineer.create_skill_matrix(
        claimed_skills,
        demonstrated_skills,
        role_data.get("required_skills") or [],
        role_data.get("nice_to_have") or [],
        repos_analyzed=github_result.get("repos_analyzed", 0),
        skills_master=state.skills_master,
    )
    X, y = state.feature_engineer.encode_for_model(skill_matrix)

    # ML predictions
    predictions = state.ml_model.predict(X)
    ensemble_probabilities = predictions["ensemble_probabilities"]

    # Skill gap analysis
    analysis = state.skill_gap_analyzer.analyze(
        claimed_skills=claimed_skills,
        demonstrated_skills=demonstrated_skills,
        target_role=target_role,
        job_roles_data=state.job_roles_data,
        ml_predictions=predictions,
        probabilities=ensemble_probabilities,
        skill_matrix=skill_matrix,
    )

    # Report
    report = state.report_generator.generate_report(
        analysis_result=analysis,
        target_role=target_role,
        github_username=github_username,
        resume_skills=claimed_skills,
        github_skills=demonstrated_skills,
        model_summary=state.ml_model.get_model_summary(),
        github_insights_data=github_result,
    )

    # Add personal info and GitHub deep insights to report
    if personal_info:
        report["candidate_info"] = personal_info

    if github_result.get("commit_activity"):
        report["github_insights"]["commit_activity"] = github_result["commit_activity"]

    # --- Groq LLM-powered enhancements (optional, runs only if GROQ_API_KEY is set) ---
    if groq_available():
        try:
            # AI Resume Coach feedback
            ai_feedback = generate_ai_feedback(
                resume_text=resume_text,
                target_role=target_role,
                missing_skills=analysis["missing_required"],
                strengths=analysis["strengths"],
                match_score=analysis["match_score"],
            )
            if ai_feedback:
                report["ai_feedback"] = ai_feedback

            # AI Interview Questions
            ai_questions = generate_interview_questions(
                target_role=target_role,
                claimed_skills=claimed_skills,
                missing_skills=analysis["missing_required"],
                claims_not_proven=analysis.get("claims_not_proven", []),
            )
            if ai_questions:
                report["ai_interview_questions"] = ai_questions

            # AI Learning Path
            all_candidate_skills = list(set(claimed_skills) | set(demonstrated_skills))
            ai_learning_path = generate_learning_path(
                target_role=target_role,
                missing_skills=analysis["missing_required"] + analysis.get("missing_nice_to_have", []),
                current_skills=all_candidate_skills,
            )
            if ai_learning_path:
                # Enrich with real URLs from Serper if available
                if serper_available():
                    try:
                        ai_learning_path = enrich_learning_path(ai_learning_path)
                    except Exception as e:
                        logger.warning(f"[WebSearch] Failed to enrich learning path: {e}")
                report["ai_learning_path"] = ai_learning_path
                report.pop("learning_path", None)  # AI version supersedes static version

            # AI Candidate Summary (recruiter-facing)
            candidate_name = (personal_info or {}).get("name", filename or "Candidate")
            ai_summary = generate_candidate_summary(
                candidate_name=candidate_name,
                resume_text=resume_text,
                target_role=target_role,
                match_score=analysis["match_score"],
                strengths=analysis["strengths"],
                missing_skills=analysis["missing_required"],
                github_insights=report.get("github_insights"),
            )
            if ai_summary:
                report["ai_candidate_summary"] = ai_summary

            # AI Culture Fit & Soft Skills
            ai_culture = generate_culture_fit_analysis(
                resume_text=resume_text,
                target_role=target_role,
            )
            if ai_culture:
                report["ai_culture_fit"] = ai_culture

            # AI Skill Credibility Assessment (validates resume-only claims)
            if analysis["claims_not_proven"]:
                ai_credibility = generate_skill_credibility_assessment(
                    resume_text=resume_text,
                    claimed_skills=claimed_skills,
                    demonstrated_skills=demonstrated_skills,
                    claims_not_proven=analysis["claims_not_proven"],
                )
                if ai_credibility:
                    report["ai_skill_credibility"] = ai_credibility

            # AI Role-Fit Narrative (detailed fit analysis)
            ai_role_fit = generate_role_fit_narrative(
                target_role=target_role,
                match_score=analysis["match_score"],
                strengths=analysis["strengths"],
                missing_skills=analysis["missing_required"],
                claims_not_proven=analysis.get("claims_not_proven", []),
                hidden_strengths=analysis.get("hidden_strengths", []),
                github_insights=report.get("github_insights"),
            )
            if ai_role_fit:
                report["ai_role_fit_narrative"] = ai_role_fit

        except Exception as e:
            logger.warning(f"[GroqLLM] Enhancement failed (non-critical): {e}")

    return {
        "report": report,
        "analysis": analysis,
        "github_result": github_result,
        "claimed_skills": claimed_skills,
        "demonstrated_skills": demonstrated_skills,
    }


# ---------------------------------------------------------------------------
#  ENDPOINT: Health Check
# ---------------------------------------------------------------------------
@app.get("/health")
@app.get("/api/health")
async def health_check():
    stats = state.db.get_dashboard_stats()
    return {
        "status": "running",
        "message": "Automated Recruiting Platform API",
        "version": "2.0.0",
        "available_roles": list(state.job_roles_data.keys()),
        "total_candidates": stats["total_candidates"],
        "total_analyses": stats["total_analyses"],
        "llm_enabled": groq_available(),
        "serper_enabled": serper_available(),
    }


# ---------------------------------------------------------------------------
#  ENDPOINT: App Info (public, AI/crawler-friendly)
# ---------------------------------------------------------------------------
@app.get("/api/info")
async def app_info():
    """
    Public endpoint that returns structured metadata about this application.
    Designed for AI tools, web crawlers, and external service discovery.
    No authentication required.
    """
    return {
        "app": "Automated Recruiting Platform",
        "description": "Resume analysis and skill gap detection using ML and LLM",
        "status": "online",
        "version": "2.0.0",
        "features": [
            "Resume parsing (PDF and plain text)",
            "Skill gap analysis against target job roles",
            "Job matching with ML-based scoring",
            "GitHub profile analysis for demonstrated skills",
            "Batch multi-resume upload with candidate ranking",
            "LLM-powered AI feedback and interview questions",
            "AI-generated learning paths for skill improvement",
            "Candidate database with SQLite persistence",
            "Side-by-side candidate comparison",
            "Job description parsing for custom roles",
            "Dashboard analytics and reporting",
        ],
        "endpoints": [
            {"path": "/api/info",        "method": "GET",  "description": "App metadata (this endpoint)"},
            {"path": "/health",          "method": "GET",  "description": "Health check with system stats"},
            {"path": "/api/health",      "method": "GET",  "description": "Health check (API-prefixed alias)"},
            {"path": "/dashboard",       "method": "GET",  "description": "Dashboard analytics"},
            {"path": "/job-roles",       "method": "GET",  "description": "List available job roles"},
            {"path": "/skills-master",   "method": "GET",  "description": "Full skills taxonomy"},
            {"path": "/analyze",         "method": "POST", "description": "Analyze a single resume (file upload)"},
            {"path": "/analyze-text",    "method": "POST", "description": "Analyze a single resume (raw text)"},
            {"path": "/analyze-batch",   "method": "POST", "description": "Batch analyze multiple resumes"},
            {"path": "/analyze-github",  "method": "POST", "description": "Analyze a GitHub profile only"},
            {"path": "/parse-jd",        "method": "POST", "description": "Extract skills from a job description"},
            {"path": "/candidates",      "method": "GET",  "description": "List all stored candidates"},
            {"path": "/compare",         "method": "POST", "description": "Side-by-side candidate comparison"},
            {"path": "/rankings",        "method": "GET",  "description": "Ranked candidate leaderboard"},
            {"path": "/docs",            "method": "GET",  "description": "Interactive API documentation (Swagger UI)"},
            {"path": "/redoc",           "method": "GET",  "description": "API documentation (ReDoc)"},
        ],
        "tech_stack": {
            "backend": "FastAPI (Python 3.11)",
            "frontend": "React + Vite",
            "ml": "scikit-learn (Logistic Regression, Decision Tree)",
            "llm": "Groq API",
            "database": "SQLite",
        },
    }


# ---------------------------------------------------------------------------
#  ENDPOINT: Dashboard Stats
# ---------------------------------------------------------------------------
@app.get("/dashboard")
async def get_dashboard():
    stats = state.db.get_dashboard_stats()
    return stats


# ---------------------------------------------------------------------------
#  ENDPOINT: Get Available Job Roles
# ---------------------------------------------------------------------------
@app.get("/job-roles")
async def get_job_roles():
    roles = []
    for role_name, role_data in state.job_roles_data.items():
        roles.append({
            "name": role_name,
            "required_skills": role_data["required_skills"],
            "nice_to_have": role_data.get("nice_to_have", []),
        })
    return {"job_roles": roles}


# ---------------------------------------------------------------------------
#  ENDPOINT: Get Skills Master List
# ---------------------------------------------------------------------------
@app.get("/skills-master")
async def get_skills_master():
    return {"skills_master": state.skills_master}


# ---------------------------------------------------------------------------
#  ENDPOINT: Analyze Single (File Upload)
# ---------------------------------------------------------------------------
@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze(
    request: Request,
    resume_file: UploadFile = File(...),
    github_username: str = Form(""),
    target_role: str = Form(...),
):
    logger.info(f"ANALYSIS | File: {resume_file.filename} | "
                f"GitHub: {github_username} | Role: {target_role}")

    filename = resume_file.filename or ""
    if not filename.lower().endswith((".pdf", ".txt", ".docx")):
        raise HTTPException(400, "Unsupported file type. Upload .pdf, .docx, or .txt.")

    if target_role not in state.job_roles_data:
        raise HTTPException(400, f"Unknown role: '{target_role}'. Available: {list(state.job_roles_data.keys())}")

    # Parse resume (with size limit)
    file_bytes = await resume_file.read()
    if len(file_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(400, "File too large. Maximum size is 10 MB.")
    if len(file_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty.")
    try:
        resume_result = state.resume_parser.parse(file_bytes, filename, state.skills_master)
    except Exception as e:
        raise HTTPException(422, f"Resume parsing failed: {e}")

    claimed_skills = resume_result["extracted_skills"]
    personal_info = resume_result.get("personal_info", {})

    # Auto-detect GitHub from resume if not provided
    if not github_username and personal_info.get("github_username"):
        github_username = personal_info["github_username"]
        logger.info(f"Auto-detected GitHub username: {github_username}")

    # Run pipeline
    try:
        result = await _run_single_analysis(
            resume_text=resume_result["raw_text"],
            claimed_skills=claimed_skills,
            github_username=github_username,
            target_role=target_role,
            personal_info=personal_info,
            filename=filename,
        )
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")

    # Save to database
    candidate_id = state.db.insert_candidate({
        "name": personal_info.get("name", ""),
        "email": personal_info.get("email", ""),
        "phone": personal_info.get("phone", ""),
        "education": personal_info.get("education", ""),
        "github_username": github_username,
        "github_url": personal_info.get("github_url", ""),
        "linkedin_url": personal_info.get("linkedin_url", ""),
        "resume_text": resume_result["raw_text"],
        "resume_filename": filename,
        "extracted_skills": claimed_skills,
    })

    analysis_id = state.db.insert_analysis({
        "candidate_id": candidate_id,
        "target_role": target_role,
        "match_score": result["analysis"]["match_score"],
        "gap_score": result["analysis"]["gap_score"],
        "confidence": result["analysis"]["confidence"],
        "report": result["report"],
        "composite_score": result["analysis"].get("composite_score", 0),
        "github_skills": result["demonstrated_skills"],
        "missing_skills": result["analysis"]["missing_required"],
    })

    report = result["report"]
    report["candidate_id"] = candidate_id
    report["analysis_id"] = analysis_id

    logger.info(f"Analysis complete! Score: {result['analysis']['match_score']}% | "
                f"Candidate #{candidate_id}")
    return report


# ---------------------------------------------------------------------------
#  ENDPOINT: Analyze Text (No File Upload)
# ---------------------------------------------------------------------------
@app.post("/analyze-text")
@limiter.limit("10/minute")
async def analyze_text(request: Request, body: TextAnalyzeRequest):
    logger.info(f"TEXT ANALYSIS | GitHub: {body.github_username} | Role: {body.target_role}")

    if body.target_role not in state.job_roles_data:
        raise HTTPException(400, f"Unknown role: '{body.target_role}'.")
    if not body.resume_text.strip():
        raise HTTPException(400, "Resume text cannot be empty.")

    claimed_skills = state.resume_parser.extract_skills(body.resume_text, state.skills_master)
    personal_info = state.resume_parser.extract_personal_info(body.resume_text)

    github_username = body.github_username
    if not github_username and personal_info.get("github_username"):
        github_username = personal_info["github_username"]

    try:
        result = await _run_single_analysis(
            resume_text=body.resume_text,
            claimed_skills=claimed_skills,
            github_username=github_username,
            target_role=body.target_role,
            personal_info=personal_info,
        )
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")

    # Save to database
    candidate_id = state.db.insert_candidate({
        "name": personal_info.get("name", ""),
        "email": personal_info.get("email", ""),
        "phone": personal_info.get("phone", ""),
        "education": personal_info.get("education", ""),
        "github_username": github_username,
        "github_url": personal_info.get("github_url", ""),
        "linkedin_url": personal_info.get("linkedin_url", ""),
        "resume_text": body.resume_text,
        "extracted_skills": claimed_skills,
    })

    analysis_id = state.db.insert_analysis({
        "candidate_id": candidate_id,
        "target_role": body.target_role,
        "match_score": result["analysis"]["match_score"],
        "gap_score": result["analysis"]["gap_score"],
        "confidence": result["analysis"]["confidence"],
        "report": result["report"],
        "composite_score": result["analysis"].get("composite_score", 0),
        "github_skills": result["demonstrated_skills"],
        "missing_skills": result["analysis"]["missing_required"],
    })

    report = result["report"]
    report["candidate_id"] = candidate_id
    report["analysis_id"] = analysis_id
    return report


# ---------------------------------------------------------------------------
#  ENDPOINT: Batch Upload — Multiple Resumes
# ---------------------------------------------------------------------------
@app.post("/analyze-batch")
@limiter.limit("3/minute")
async def analyze_batch(
    request: Request,
    resume_files: List[UploadFile] = File(...),
    target_role: str = Form(...),
):
    """
    Analyze multiple resumes at once. Returns a ranked list of candidates.
    GitHub usernames are auto-extracted from resume content.
    """
    if target_role not in state.job_roles_data:
        raise HTTPException(400, f"Unknown role: '{target_role}'.")

    if len(resume_files) > 50:
        raise HTTPException(400, "Maximum 50 resumes per batch.")

    logger.info(f"BATCH ANALYSIS | {len(resume_files)} resumes | Role: {target_role}")

    batch_id = state.db.create_batch_job(target_role, len(resume_files))
    results = []
    errors = []

    for i, resume_file in enumerate(resume_files):
        filename = resume_file.filename or f"resume_{i}.txt"
        logger.info(f"  Batch [{i+1}/{len(resume_files)}]: {filename}")

        if not filename.lower().endswith((".pdf", ".txt", ".docx")):
            errors.append({"file": filename, "error": "Unsupported file type"})
            continue

        try:
            file_bytes = await resume_file.read()
            if len(file_bytes) > 10 * 1024 * 1024:
                errors.append({"file": filename, "error": "File too large (max 10 MB)"})
                continue
            if len(file_bytes) == 0:
                errors.append({"file": filename, "error": "File is empty"})
                continue
            resume_result = state.resume_parser.parse(file_bytes, filename, state.skills_master)
            claimed_skills = resume_result["extracted_skills"]
            personal_info = resume_result.get("personal_info", {})
            github_username = personal_info.get("github_username", "")

            result = await _run_single_analysis(
                resume_text=resume_result["raw_text"],
                claimed_skills=claimed_skills,
                github_username=github_username,
                target_role=target_role,
                personal_info=personal_info,
                filename=filename,
            )

            # Save candidate
            candidate_id = state.db.insert_candidate({
                "name": personal_info.get("name", filename),
                "email": personal_info.get("email", ""),
                "phone": personal_info.get("phone", ""),
                "education": personal_info.get("education", ""),
                "github_username": github_username,
                "github_url": personal_info.get("github_url", ""),
                "linkedin_url": personal_info.get("linkedin_url", ""),
                "resume_text": resume_result["raw_text"],
                "resume_filename": filename,
                "extracted_skills": claimed_skills,
            })

            analysis_id = state.db.insert_analysis({
                "candidate_id": candidate_id,
                "target_role": target_role,
                "match_score": result["analysis"]["match_score"],
                "gap_score": result["analysis"]["gap_score"],
                "confidence": result["analysis"]["confidence"],
                "composite_score": result["analysis"].get("composite_score", 0),
                "report": result["report"],
                "github_skills": result["demonstrated_skills"],
                "missing_skills": result["analysis"]["missing_required"],
            })

            results.append({
                "candidate_id": candidate_id,
                "analysis_id": analysis_id,
                "name": personal_info.get("name", filename),
                "email": personal_info.get("email", ""),
                "github_username": github_username,
                "filename": filename,
                "match_score": result["analysis"]["match_score"],
                "gap_score": result["analysis"]["gap_score"],
                "confidence": result["analysis"]["confidence"],
                "composite_score": result["analysis"].get("composite_score", 0),
                "missing_required": result["analysis"].get("missing_required", []),
                "missing_count": len(result["analysis"].get("missing_required", [])),
                "resume_skills_count": len(claimed_skills),
                "github_skills_count": len(result["demonstrated_skills"]),
            })

        except Exception as e:
            logger.error(f"  Batch error for {filename}: {e}")
            errors.append({"file": filename, "error": str(e)})

    # Sort by composite_score descending (consistent with /rankings endpoint)
    results.sort(key=lambda x: (-x.get("composite_score", 0), -x["match_score"], -x["confidence"]))

    # Assign ranks and save batch results
    for rank, r in enumerate(results, 1):
        r["rank"] = rank
        state.db.add_batch_result(batch_id, r["candidate_id"], r["analysis_id"], rank)

    state.db.complete_batch_job(batch_id)

    logger.info(f"Batch complete! {len(results)} analyzed, {len(errors)} errors")

    # AI-powered batch executive report for recruiters
    ai_batch_report = None
    if groq_available() and results:
        try:
            ai_batch_report = generate_batch_executive_report(target_role, results)
        except Exception as e:
            logger.warning(f"[GroqLLM] Batch report generation failed (non-critical): {e}")

    response = {
        "batch_id": batch_id,
        "target_role": target_role,
        "total_submitted": len(resume_files),
        "total_analyzed": len(results),
        "total_errors": len(errors),
        "rankings": results,
        "errors": errors,
    }
    if ai_batch_report:
        response["ai_executive_report"] = ai_batch_report

    return response


# ---------------------------------------------------------------------------
#  ENDPOINT: Get Batch Job Results
# ---------------------------------------------------------------------------
@app.get("/batch/{batch_id}")
async def get_batch(batch_id: int):
    job = state.db.get_batch_job(batch_id)
    if not job:
        raise HTTPException(404, "Batch job not found.")
    return job


# ---------------------------------------------------------------------------
#  ENDPOINT: Candidates List
# ---------------------------------------------------------------------------
@app.get("/candidates")
async def get_candidates(limit: int = 100, offset: int = 0):
    limit = min(max(1, limit), 500)
    offset = max(0, offset)
    candidates = state.db.get_all_candidates(limit, offset)
    total = state.db.get_candidate_count()
    return {"candidates": candidates, "total": total}


# ---------------------------------------------------------------------------
#  ENDPOINT: Single Candidate Detail
# ---------------------------------------------------------------------------
@app.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: int):
    candidate = state.db.get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(404, "Candidate not found.")
    analyses = state.db.get_analyses_for_candidate(candidate_id)
    return {"candidate": candidate, "analyses": analyses}


# ---------------------------------------------------------------------------
#  ENDPOINT: Delete Candidate
# ---------------------------------------------------------------------------
@app.delete("/candidates/{candidate_id}")
async def delete_candidate(candidate_id: int):
    if state.db.delete_candidate(candidate_id):
        return {"status": "deleted", "candidate_id": candidate_id}
    raise HTTPException(404, "Candidate not found.")


# ---------------------------------------------------------------------------
#  ENDPOINT: Ranked Candidates for a Role
# ---------------------------------------------------------------------------
@app.get("/rankings/{target_role}")
async def get_rankings(target_role: str, limit: int = 50):
    limit = min(max(1, limit), 500)
    if target_role not in state.job_roles_data:
        raise HTTPException(400, f"Unknown role: '{target_role}'.")
    rankings = state.db.get_ranked_candidates(target_role, limit)
    return {"target_role": target_role, "rankings": rankings}


# ---------------------------------------------------------------------------
#  ENDPOINT: Compare Candidates Side-by-Side
# ---------------------------------------------------------------------------
@app.post("/compare")
@limiter.limit("15/minute")
async def compare_candidates(request: Request, body: CompareRequest):
    if len(body.candidate_ids) < 2:
        raise HTTPException(400, "Provide at least 2 candidate IDs.")
    if len(body.candidate_ids) > 5:
        raise HTTPException(400, "Maximum 5 candidates for comparison.")
    if body.target_role not in state.job_roles_data:
        raise HTTPException(400, f"Unknown role: '{body.target_role}'.")

    comparisons = state.db.get_candidates_comparison(
        body.candidate_ids, body.target_role
    )

    # Warn about candidates with no analysis for this role
    missing_analysis = [c for c in comparisons if c.get("match_score") is None]
    if missing_analysis:
        names = [c.get("name", f"#{c['candidate_id']}") for c in missing_analysis]
        logger.warning(f"[Compare] Candidates without analysis for {body.target_role}: {names}")

    # Filter to only candidates with analysis data for the skill matrix
    analyzed = [c for c in comparisons if c.get("match_score") is not None]

    # Build comparison matrix
    role_data = state.job_roles_data[body.target_role]
    all_skills = role_data["required_skills"] + role_data.get("nice_to_have", [])

    skill_matrix = {}
    for skill in all_skills:
        skill_matrix[skill] = {}
        for comp in analyzed:
            cid = comp["candidate_id"]
            resume_skills = comp.get("extracted_skills") or []
            github_skills = comp.get("github_skills") or []
            has_resume = skill in resume_skills
            has_github = skill in github_skills
            if has_resume and has_github:
                status = "strong"
            elif has_resume:
                status = "claimed_only"
            elif has_github:
                status = "demonstrated_only"
            else:
                status = "missing"
            skill_matrix[skill][str(cid)] = status

    return {
        "target_role": body.target_role,
        "candidates": comparisons,
        "skill_matrix": skill_matrix,
        "required_skills": role_data["required_skills"],
        "nice_to_have": role_data.get("nice_to_have", []),
    }


# ---------------------------------------------------------------------------
#  ENDPOINT: Parse Job Description
# ---------------------------------------------------------------------------
@app.post("/parse-job-description")
@limiter.limit("10/minute")
async def parse_job_description(request: Request, body: JobDescriptionRequest):
    """
    Parse a job description text to auto-extract required skills.
    Can be used to create custom roles on-the-fly.
    """
    if not body.description.strip():
        raise HTTPException(400, "Job description cannot be empty.")

    text = body.description
    text_lower = text.lower()

    # Find skills mentioned in the JD using pre-compiled patterns
    found_skills = set()
    patterns = get_compiled_patterns()
    if patterns:
        for skill, pattern in patterns.items():
            if pattern.search(text_lower):
                found_skills.add(skill)
    else:
        all_skills = []
        for category, skills in state.skills_master.items():
            all_skills.extend(skills)
        for skill in all_skills:
            pat = r"\b" + re.escape(skill.lower()) + r"\b"
            if skill in ("C++", "C#"):
                pat = re.escape(skill.lower())
            if re.search(pat, text_lower):
                found_skills.add(skill)

    # Heuristic: skills in "required" sections vs "nice to have" sections
    required_skills = []
    nice_to_have = []

    required_section = ""
    nice_section = ""
    lines = text.split("\n")
    current_section = "required"

    for line in lines:
        line_lower = line.lower().strip()
        if any(kw in line_lower for kw in ["nice to have", "preferred", "bonus", "plus", "optional"]):
            current_section = "nice"
        elif any(kw in line_lower for kw in ["required", "must have", "essential", "qualifications", "requirements"]):
            current_section = "required"

        if current_section == "required":
            required_section += line + "\n"
        else:
            nice_section += line + "\n"

    required_lower = required_section.lower()
    nice_lower = nice_section.lower()
    for skill in found_skills:
        pat = patterns.get(skill) if patterns else re.compile(r"\b" + re.escape(skill.lower()) + r"\b")
        in_required = pat.search(required_lower)
        in_nice = pat.search(nice_lower)

        if in_nice and not in_required:
            nice_to_have.append(skill)
        else:
            required_skills.append(skill)

    role_name = (body.role_name or "Custom Role").strip()[:100]

    if role_name and required_skills:
        state.job_roles_data[role_name] = {
            "required_skills": required_skills,
            "nice_to_have": nice_to_have,
        }

    # Enhance with LLM-powered extraction if available
    ai_jd_analysis = None
    if groq_available():
        try:
            ai_jd_analysis = generate_jd_skills_extraction(text)
            if ai_jd_analysis:
                # Merge LLM-detected skills with regex-detected ones
                llm_required = ai_jd_analysis.get("required_skills", [])
                llm_nice = ai_jd_analysis.get("nice_to_have", [])
                all_master = [s for cat in state.skills_master.values() for s in cat]
                for s in llm_required:
                    if s in all_master and s not in required_skills:
                        required_skills.append(s)
                for s in llm_nice:
                    if s in all_master and s not in nice_to_have and s not in required_skills:
                        nice_to_have.append(s)
        except Exception as e:
            logger.warning(f"[GroqLLM] JD parsing enhancement failed: {e}")

    # Update roles data with final merged skills
    if role_name and required_skills:
        state.job_roles_data[role_name] = {
            "required_skills": required_skills,
            "nice_to_have": nice_to_have,
        }

    result = {
        "role_name": role_name,
        "required_skills": sorted(required_skills),
        "nice_to_have": sorted(nice_to_have),
        "total_skills_found": len(set(required_skills + nice_to_have)),
        "added_to_roles": role_name in state.job_roles_data,
    }
    if ai_jd_analysis:
        result["ai_analysis"] = {
            "experience_level": ai_jd_analysis.get("experience_level", ""),
            "role_summary": ai_jd_analysis.get("role_summary", ""),
        }
    return result


# ---------------------------------------------------------------------------
#  ENDPOINT: Analysis History (for sidebar)
# ---------------------------------------------------------------------------
@app.get("/analysis-history")
async def get_analysis_history(limit: int = 50):
    """Get recent analyses for the history sidebar."""
    limit = min(max(1, limit), 200)
    analyses = state.db.get_recent_analyses(limit)
    return {"analyses": analyses}


@app.get("/analysis/{analysis_id}")
async def get_analysis_detail(analysis_id: int):
    """Get full analysis report by ID."""
    analysis = state.db.get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found.")
    # Return the stored report_json as the full report
    report = analysis["report_json"]
    report["candidate_id"] = analysis["candidate_id"]
    report["analysis_id"] = analysis["id"]
    return report


# ---------------------------------------------------------------------------
#  ENDPOINT: Model Metrics
# ---------------------------------------------------------------------------
@app.get("/model-metrics")
async def get_model_metrics():
    summary = state.ml_model.get_model_summary()
    summary["feature_importance"] = state.ml_model.get_feature_importance()
    return summary


# ---------------------------------------------------------------------------
#  ENDPOINT: Model Retrain
# ---------------------------------------------------------------------------
@app.get("/model-retrain")
async def retrain_model():
    now = time.time()
    if now - state.last_retrain_time < 3600:
        remaining = int(3600 - (now - state.last_retrain_time))
        raise HTTPException(429, f"Retrain rate limited. Try again in {remaining}s.")

    logger.info("Retraining ML models with fresh data...")
    state.last_retrain_time = now

    X, y, source = state.dataset_loader.load_training_data()
    metrics = state.ml_model.train(X, y, dataset_source=source, use_cross_validation=True)
    return {"status": "retrained", "metrics": metrics}


# ---------------------------------------------------------------------------
#  ENDPOINT: Dataset Status
# ---------------------------------------------------------------------------
@app.get("/dataset-status")
async def get_dataset_status():
    return {
        "training_data_source": state.ml_model.dataset_source,
        "model_trained": state.ml_model.is_trained,
        "dataset_details": state.dataset_loader.get_status(),
    }


# ---------------------------------------------------------------------------
#  ENDPOINT: Export Candidate Report as PDF (server-side generation)
# ---------------------------------------------------------------------------
from fastapi.responses import StreamingResponse, FileResponse
import io


@app.get("/export/pdf/{analysis_id}")
async def export_pdf_report(analysis_id: int):
    """
    Generate a professional PDF report for a candidate analysis.
    Server-side PDF generation handles large reports that crash browser-based
    html2canvas. Uses structured text layout for recruiter-friendly output.
    """
    analysis = state.db.get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found.")

    report = analysis["report_json"]
    candidate = state.db.get_candidate(analysis["candidate_id"])

    try:
        import fitz  # PyMuPDF

        doc = fitz.open()
        WIDTH, HEIGHT = fitz.paper_size("a4")
        MARGIN = 50
        TEXT_WIDTH = WIDTH - 2 * MARGIN

        def new_page():
            page = doc.new_page(width=WIDTH, height=HEIGHT)
            return page, MARGIN + 40

        page, y = new_page()

        def write_text(text, fontsize=10, bold=False, color=(0, 0, 0)):
            nonlocal page, y
            font = "helv" if not bold else "hebo"
            lines = text.split("\n")
            for line in lines:
                if y > HEIGHT - MARGIN - 20:
                    page, y = new_page()
                page.insert_text(
                    fitz.Point(MARGIN, y), line,
                    fontsize=fontsize, fontname=font, color=color,
                )
                y += fontsize + 4

        def write_separator():
            nonlocal page, y
            if y > HEIGHT - MARGIN - 20:
                page, y = new_page()
            page.draw_line(
                fitz.Point(MARGIN, y), fitz.Point(WIDTH - MARGIN, y),
                color=(0.8, 0.8, 0.8), width=0.5,
            )
            y += 12

        # --- Title ---
        write_text("SKILL GAP ANALYSIS REPORT", fontsize=18, bold=True, color=(0.1, 0.2, 0.5))
        y += 5
        write_text(f"Target Role: {report.get('target_role', 'N/A')}", fontsize=12, bold=True)
        y += 3

        # --- Candidate Info ---
        if candidate:
            name = candidate.get("name", "Unknown")
            email = candidate.get("email", "")
            github = candidate.get("github_username", "")
            write_text(f"Candidate: {name}", fontsize=11)
            if email:
                write_text(f"Email: {email}", fontsize=10)
            if github:
                write_text(f"GitHub: {github}", fontsize=10)
        y += 5
        write_separator()

        # --- Executive Summary ---
        es = report.get("executive_summary", {})
        write_text("EXECUTIVE SUMMARY", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
        y += 3
        write_text(f"Match Score: {es.get('match_score', 0):.1f}% ({es.get('match_label', '')})", fontsize=11, bold=True)
        write_text(f"Resume Skills: {es.get('total_resume_skills', 0)}  |  GitHub Skills: {es.get('total_github_skills', 0)}", fontsize=10)
        write_text(f"Missing Critical Skills: {es.get('missing_critical_skills', 0)}", fontsize=10)
        write_text(f"Confidence: {es.get('confidence_score', 0):.1f}% — {es.get('confidence_rating', '')}", fontsize=10)
        y += 5

        # --- AI Candidate Summary (if available) ---
        ai_sum = report.get("ai_candidate_summary")
        if ai_sum:
            write_separator()
            write_text("AI CANDIDATE ASSESSMENT", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
            y += 3
            if ai_sum.get("headline"):
                write_text(ai_sum["headline"], fontsize=11, bold=True)
            if ai_sum.get("executive_summary"):
                write_text(ai_sum["executive_summary"], fontsize=10)
            if ai_sum.get("hiring_recommendation"):
                write_text(f"Recommendation: {ai_sum['hiring_recommendation']}", fontsize=10, bold=True,
                           color=(0.0, 0.5, 0.0) if "Hire" in ai_sum["hiring_recommendation"] else (0.7, 0.0, 0.0))
            y += 5

        # --- Skill Breakdown ---
        write_separator()
        write_text("SKILL BREAKDOWN", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
        y += 3

        sb = report.get("skill_breakdown", {})
        for skill_data in (sb.get("required_analysis") or []):
            status = skill_data.get("status", "missing")
            symbol = {"strong": "[OK]", "claimed_only": "[RESUME]", "demonstrated_only": "[GITHUB]", "missing": "[MISSING]"}.get(status, "[?]")
            color = {"strong": (0, 0.5, 0), "claimed_only": (0.7, 0.5, 0), "demonstrated_only": (0, 0.4, 0.7), "missing": (0.8, 0, 0)}.get(status, (0, 0, 0))
            write_text(f"  {symbol} {skill_data.get('skill', '?')} — {status}", fontsize=9, color=color)

        y += 5
        if sb.get("nice_to_have_analysis"):
            write_text("Nice-to-Have Skills:", fontsize=10, bold=True)
            for skill_data in sb["nice_to_have_analysis"]:
                status = skill_data.get("status", "missing")
                write_text(f"  {skill_data.get('skill', '?')} — {status}", fontsize=9)
            y += 5

        # --- Recommendations ---
        recs = report.get("recommendations", [])
        if recs:
            write_separator()
            write_text("RECOMMENDATIONS", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
            y += 3
            for rec in recs[:10]:
                priority_color = (0.8, 0, 0) if rec.get("priority") == "Urgent" else (0.6, 0.4, 0)
                write_text(f"  [{rec.get('priority', '')}] {rec.get('action', '')}", fontsize=9, color=priority_color)
            y += 5

        # --- AI Culture Fit (if available) ---
        culture = report.get("ai_culture_fit")
        if culture:
            write_separator()
            write_text("CULTURE & SOFT SKILLS", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
            y += 3
            if culture.get("soft_skills"):
                write_text(f"Soft Skills: {', '.join(culture['soft_skills'])}", fontsize=10)
            if culture.get("communication_score"):
                write_text(f"Communication Score: {culture['communication_score']}/10", fontsize=10)
            if culture.get("team_fit_notes"):
                write_text(f"Team Fit: {culture['team_fit_notes']}", fontsize=10)
            y += 5

        # --- Learning Path ---
        learning_path = report.get("learning_path") or []
        if learning_path and not report.get("ai_learning_path"):
            write_separator()
            write_text("LEARNING PATH", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
            y += 3
            for lp in learning_path:
                priority_color = (0.8, 0, 0) if lp.get("priority") == "Critical" else (0.6, 0.4, 0)
                meta = f" ({lp.get('difficulty', '')}, {lp.get('estimated_time', '')})" if lp.get("difficulty") else ""
                write_text(f"  [{lp.get('priority', '')}] {lp.get('skill', '')}{meta}", fontsize=9, color=priority_color)
                if lp.get("suggested_path"):
                    write_text(f"    {lp['suggested_path']}", fontsize=8, color=(0.3, 0.3, 0.3))
            y += 5

        # --- ML Insights ---
        ml = report.get("ml_insights") or {}
        if ml.get("lr_accuracy") is not None:
            write_separator()
            write_text("ML MODEL INSIGHTS", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
            y += 3
            write_text(f"Logistic Regression Accuracy: {ml.get('lr_accuracy', 0)}%", fontsize=10)
            write_text(f"Decision Tree Accuracy: {ml.get('dt_accuracy', 0)}%", fontsize=10)
            if ml.get("ensemble_explanation"):
                write_text(f"Ensemble: {ml['ensemble_explanation']}", fontsize=9)
            if ml.get("model_explanation"):
                write_text(f"How it works: {ml['model_explanation']}", fontsize=9)
            y += 5

        # --- Skill Credibility (if available) ---
        cred = report.get("ai_skill_credibility")
        if cred:
            write_separator()
            write_text("SKILL CREDIBILITY ASSESSMENT", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
            y += 3
            if cred.get("overall_credibility_score") is not None:
                write_text(f"Credibility Score: {cred['overall_credibility_score']}/10", fontsize=11, bold=True)
            if cred.get("assessment"):
                write_text(cred["assessment"], fontsize=9)
            if cred.get("verified_skills"):
                write_text(f"Verified: {', '.join(cred['verified_skills'])}", fontsize=9, color=(0, 0.5, 0))
            if cred.get("questionable_skills"):
                write_text(f"Needs Verification: {', '.join(cred['questionable_skills'])}", fontsize=9, color=(0.8, 0, 0))
            y += 5

        # --- Role-Fit Analysis (if available) ---
        rolefit = report.get("ai_role_fit_narrative")
        if rolefit:
            write_separator()
            write_text("ROLE-FIT ANALYSIS", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
            y += 3
            if rolefit.get("fit_score") is not None:
                write_text(f"Role Fit Score: {rolefit['fit_score']}/10", fontsize=11, bold=True)
            if rolefit.get("narrative"):
                write_text(rolefit["narrative"], fontsize=9)
            if rolefit.get("standout_qualities"):
                write_text(f"Standout Qualities: {', '.join(rolefit['standout_qualities'])}", fontsize=9, color=(0, 0.5, 0))
            if rolefit.get("growth_areas"):
                write_text(f"Growth Areas: {', '.join(rolefit['growth_areas'])}", fontsize=9, color=(0.7, 0.5, 0))
            if rolefit.get("onboarding_estimate"):
                write_text(f"Estimated Onboarding: {rolefit['onboarding_estimate']}", fontsize=9)
            y += 5

        # --- AI Resume Coach (if available) ---
        feedback = report.get("ai_feedback")
        if feedback:
            write_separator()
            write_text("AI RESUME COACH", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
            y += 3
            if feedback.get("overall_advice"):
                write_text(feedback["overall_advice"], fontsize=9)
            if feedback.get("resume_tips"):
                write_text("Improvement Tips:", fontsize=10, bold=True)
                for tip in feedback["resume_tips"][:5]:
                    write_text(f"  - {tip}", fontsize=9)
            if feedback.get("keyword_suggestions"):
                write_text(f"ATS Keywords to Add: {', '.join(feedback['keyword_suggestions'])}", fontsize=9)
            y += 5

        # --- AI Interview Prep (if available) ---
        questions = report.get("ai_interview_questions") or []
        if questions:
            write_separator()
            write_text("AI INTERVIEW PREP", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
            y += 3
            for q in questions[:5]:
                diff_color = (0.8, 0, 0) if q.get("difficulty") == "hard" else (0.6, 0.4, 0) if q.get("difficulty") == "medium" else (0, 0.5, 0)
                write_text(f"  [{q.get('difficulty', '')}] ({q.get('skill', '')}) {q.get('question', '')}", fontsize=9, color=diff_color)
                if q.get("prep_hint"):
                    write_text(f"    Prep: {q['prep_hint']}", fontsize=8, color=(0.3, 0.3, 0.3))
            y += 5

        # --- AI Learning Path (if available) ---
        ai_lp = report.get("ai_learning_path") or []
        if ai_lp:
            write_separator()
            write_text("AI LEARNING PATH", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
            y += 3
            for item in ai_lp[:8]:
                week = item.get("week", "")
                write_text(f"  Week {week}: {item.get('skill', '')}", fontsize=10, bold=True)
                if item.get("resources"):
                    for r in item["resources"][:3]:
                        name_str = r.get("name", r) if isinstance(r, dict) else str(r)
                        write_text(f"    - {name_str}", fontsize=8, color=(0.3, 0.3, 0.3))
                if item.get("project_idea"):
                    write_text(f"    Project: {item['project_idea']}", fontsize=8, color=(0.3, 0.3, 0.3))
            y += 5

        # --- GitHub Insights ---
        gi = report.get("github_insights") or {}
        if gi.get("repos_analyzed"):
            write_separator()
            write_text("GITHUB INSIGHTS", fontsize=13, bold=True, color=(0.1, 0.2, 0.5))
            y += 3
            write_text(f"Repositories Analyzed: {gi['repos_analyzed']}", fontsize=10)
            if gi.get("total_topics"):
                write_text(f"Topics Found: {gi['total_topics']}", fontsize=10)
            if gi.get("hidden_strengths"):
                write_text(f"Hidden Strengths: {', '.join(gi['hidden_strengths'])}", fontsize=9, color=(0, 0.4, 0.7))
            for lang in (gi.get("top_languages") or [])[:5]:
                kb = lang.get("bytes", 0) / 1000
                write_text(f"  {lang.get('language', '?')}: {kb:.1f} KB", fontsize=9)
            y += 5

        # --- Footer ---
        write_separator()
        generated_at = report.get("generated_at", "")
        if generated_at:
            write_text(f"Generated: {generated_at}", fontsize=8, color=(0.5, 0.5, 0.5))
        write_text("Generated by Automated Recruiting Platform", fontsize=8, color=(0.5, 0.5, 0.5))

        # Save to buffer
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        buf.seek(0)

        name = (candidate or {}).get("name", "Candidate")
        role = report.get("target_role", "Role")
        filename = f"Report_{name}_{role}.pdf".replace(" ", "_")

        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(500, f"PDF generation failed: {e}")


# ---------------------------------------------------------------------------
#  ENDPOINT: Export Batch Results as CSV (server-side, handles big data)
# ---------------------------------------------------------------------------
def _csv_safe(val):
    """Prefix strings that start with formula-triggering chars to prevent CSV injection."""
    if isinstance(val, str) and val and val[0] in "=+@-":
        return "'" + val
    return val


@app.get("/export/batch-csv/{batch_id}")
async def export_batch_csv(batch_id: int):
    """
    Generate a comprehensive CSV export for a batch job.
    Handles large batches that would be slow to export client-side.
    """
    job = state.db.get_batch_job(batch_id)
    if not job:
        raise HTTPException(404, "Batch job not found.")

    import csv
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header
    writer.writerow([
        "Rank", "Name", "Email", "GitHub", "Match Score (%)",
        "Composite Score", "Gap Score (%)", "Confidence (%)",
        "Resume Skills Count", "GitHub Skills Count",
        "Missing Skills Count", "Missing Skills", "Filename",
    ])

    rankings = job.get("results", [])
    for r in rankings:
        writer.writerow([
            r.get("rank", ""),
            _csv_safe(r.get("name", "")),
            _csv_safe(r.get("email", "")),
            _csv_safe(r.get("github_username", "")),
            f"{r.get('match_score', 0):.1f}",
            f"{r.get('composite_score', 0):.1f}",
            f"{r.get('gap_score', 0):.1f}",
            f"{r.get('confidence', 0):.1f}",
            r.get("resume_skills_count", 0),
            r.get("github_skills_count", 0),
            r.get("missing_count", 0),
            _csv_safe("; ".join(r.get("missing_required") or [])),
            _csv_safe(r.get("filename", "")),
        ])

    buf.seek(0)
    role = job.get("target_role", "Role")
    filename = f"Batch_{role}_{batch_id}.csv".replace(" ", "_")

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
#  ENDPOINT: Export All Candidates as CSV
# ---------------------------------------------------------------------------
@app.get("/export/candidates-csv")
async def export_candidates_csv(target_role: str = ""):
    """Export all candidates (optionally filtered by role) as CSV for ATS import."""
    import csv

    if target_role and target_role in state.job_roles_data:
        rankings = state.db.get_ranked_candidates(target_role, limit=500)
    else:
        candidates_raw = state.db.get_all_candidates(limit=500, offset=0)
        rankings = candidates_raw

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Candidate ID", "Name", "Email", "Phone", "GitHub",
        "LinkedIn", "Education", "Skills Count",
    ])

    for c in rankings:
        writer.writerow([
            c.get("id", c.get("candidate_id", "")),
            _csv_safe(c.get("name", "")),
            _csv_safe(c.get("email", "")),
            _csv_safe(c.get("phone", "")),
            _csv_safe(c.get("github_username", "")),
            _csv_safe(c.get("linkedin_url", "")),
            _csv_safe(c.get("education", "")),
            c.get("resume_skills_count", len(c.get("extracted_skills", []))),
        ])

    buf.seek(0)
    filename = f"Candidates_{target_role or 'All'}.csv".replace(" ", "_")

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
#  Serve React Frontend (static files from unified Docker build)
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    # Serve /assets (JS, CSS, images)
    assets_dir = os.path.join(static_dir, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="static-assets")

    from pathlib import Path as _Path
    _static_root = _Path(static_dir).resolve()

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Catch-all: serve index.html for React SPA routing."""
        requested = (_static_root / full_path).resolve()
        if requested.is_relative_to(_static_root) and requested.is_file():
            return FileResponse(requested)
        return FileResponse(_static_root / "index.html")


# ---------------------------------------------------------------------------
#  Run with Uvicorn (if executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
