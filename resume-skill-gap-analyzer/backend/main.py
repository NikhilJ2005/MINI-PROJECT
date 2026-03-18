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
from pydantic import BaseModel
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
from modules.groq_llm import is_available as groq_available, extract_skills_with_llm, generate_ai_feedback, generate_interview_questions, generate_learning_path
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
    state.skill_gap_analyzer = SkillGapAnalyzer()
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

# CORS — use env var for production, default to permissive for dev
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
#  Pydantic Models
# ---------------------------------------------------------------------------
class TextAnalyzeRequest(BaseModel):
    resume_text: str
    github_username: str
    target_role: str


class CompareRequest(BaseModel):
    candidate_ids: List[int]
    target_role: str


class JobDescriptionRequest(BaseModel):
    description: str
    role_name: str = ""


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
        role_data["required_skills"],
        role_data.get("nice_to_have", []),
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
                report["ai_learning_path"] = ai_learning_path

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
@app.get("/")
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
async def analyze_text(request: TextAnalyzeRequest):
    logger.info(f"TEXT ANALYSIS | GitHub: {request.github_username} | Role: {request.target_role}")

    if request.target_role not in state.job_roles_data:
        raise HTTPException(400, f"Unknown role: '{request.target_role}'.")
    if not request.resume_text.strip():
        raise HTTPException(400, "Resume text cannot be empty.")

    claimed_skills = state.resume_parser.extract_skills(request.resume_text, state.skills_master)
    personal_info = state.resume_parser.extract_personal_info(request.resume_text)

    github_username = request.github_username
    if not github_username and personal_info.get("github_username"):
        github_username = personal_info["github_username"]

    try:
        result = await _run_single_analysis(
            resume_text=request.resume_text,
            claimed_skills=claimed_skills,
            github_username=github_username,
            target_role=request.target_role,
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
        "resume_text": request.resume_text,
        "extracted_skills": claimed_skills,
    })

    analysis_id = state.db.insert_analysis({
        "candidate_id": candidate_id,
        "target_role": request.target_role,
        "match_score": result["analysis"]["match_score"],
        "gap_score": result["analysis"]["gap_score"],
        "confidence": result["analysis"]["confidence"],
        "report": result["report"],
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
                "missing_required": result["analysis"]["missing_required"],
                "missing_count": len(result["analysis"]["missing_required"]),
                "resume_skills_count": len(claimed_skills),
                "github_skills_count": len(result["demonstrated_skills"]),
            })

        except Exception as e:
            logger.error(f"  Batch error for {filename}: {e}")
            errors.append({"file": filename, "error": str(e)})

    # Sort by match_score descending
    results.sort(key=lambda x: (-x["match_score"], -x["confidence"]))

    # Assign ranks and save batch results
    for rank, r in enumerate(results, 1):
        r["rank"] = rank
        state.db.add_batch_result(batch_id, r["candidate_id"], r["analysis_id"], rank)

    state.db.complete_batch_job(batch_id)

    logger.info(f"Batch complete! {len(results)} analyzed, {len(errors)} errors")

    return {
        "batch_id": batch_id,
        "target_role": target_role,
        "total_submitted": len(resume_files),
        "total_analyzed": len(results),
        "total_errors": len(errors),
        "rankings": results,
        "errors": errors,
    }


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
    if target_role not in state.job_roles_data:
        raise HTTPException(400, f"Unknown role: '{target_role}'.")
    rankings = state.db.get_ranked_candidates(target_role, limit)
    return {"target_role": target_role, "rankings": rankings}


# ---------------------------------------------------------------------------
#  ENDPOINT: Compare Candidates Side-by-Side
# ---------------------------------------------------------------------------
@app.post("/compare")
async def compare_candidates(request: CompareRequest):
    if len(request.candidate_ids) < 2:
        raise HTTPException(400, "Provide at least 2 candidate IDs.")
    if len(request.candidate_ids) > 5:
        raise HTTPException(400, "Maximum 5 candidates for comparison.")
    if request.target_role not in state.job_roles_data:
        raise HTTPException(400, f"Unknown role: '{request.target_role}'.")

    comparisons = state.db.get_candidates_comparison(
        request.candidate_ids, request.target_role
    )

    # Build comparison matrix
    role_data = state.job_roles_data[request.target_role]
    all_skills = role_data["required_skills"] + role_data.get("nice_to_have", [])

    skill_matrix = {}
    for skill in all_skills:
        skill_matrix[skill] = {}
        for comp in comparisons:
            cid = comp["candidate_id"]
            resume_skills = comp["extracted_skills"]
            github_skills = comp["github_skills"]
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
        "target_role": request.target_role,
        "candidates": comparisons,
        "skill_matrix": skill_matrix,
        "required_skills": role_data["required_skills"],
        "nice_to_have": role_data.get("nice_to_have", []),
    }


# ---------------------------------------------------------------------------
#  ENDPOINT: Parse Job Description
# ---------------------------------------------------------------------------
@app.post("/parse-job-description")
async def parse_job_description(request: JobDescriptionRequest):
    """
    Parse a job description text to auto-extract required skills.
    Can be used to create custom roles on-the-fly.
    """
    if not request.description.strip():
        raise HTTPException(400, "Job description cannot be empty.")

    text = request.description
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

    role_name = request.role_name or "Custom Role"

    if role_name and required_skills:
        state.job_roles_data[role_name] = {
            "required_skills": required_skills,
            "nice_to_have": nice_to_have,
        }

    return {
        "role_name": role_name,
        "required_skills": sorted(required_skills),
        "nice_to_have": sorted(nice_to_have),
        "total_skills_found": len(found_skills),
        "added_to_roles": role_name in state.job_roles_data,
    }


# ---------------------------------------------------------------------------
#  ENDPOINT: Analysis History (for sidebar)
# ---------------------------------------------------------------------------
@app.get("/analysis-history")
async def get_analysis_history(limit: int = 50):
    """Get recent analyses for the history sidebar."""
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
    return {"status": "retrained", "dataset_source": source, "metrics": metrics}


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
#  Run with Uvicorn (if executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
