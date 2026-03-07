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

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

# Import all pipeline modules
from modules.resume_parser import ResumeParser, compile_skill_patterns, set_compiled_patterns, get_compiled_patterns, set_flat_skills
from modules.github_analyzer import GitHubAnalyzer
from modules.feature_engineering import FeatureEngineer
from modules.ml_model import SkillGapMLModel
from modules.skill_gap_analyzer import SkillGapAnalyzer
from modules.report_generator import ReportGenerator
from modules.database import Database
from data.dataset_loader import DatasetLoader

# ---------------------------------------------------------------------------
#  Load Environment Variables
# ---------------------------------------------------------------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# ---------------------------------------------------------------------------
#  Global State (loaded on startup)
# ---------------------------------------------------------------------------
job_roles_data: Dict = {}
skills_master: Dict[str, List[str]] = {}
resume_parser: ResumeParser = None      # type: ignore
github_analyzer: GitHubAnalyzer = None  # type: ignore
feature_engineer: FeatureEngineer = None  # type: ignore
ml_model: SkillGapMLModel = None        # type: ignore
skill_gap_analyzer: SkillGapAnalyzer = None  # type: ignore
report_generator: ReportGenerator = None    # type: ignore
dataset_loader: DatasetLoader = None    # type: ignore
db: Database = None                      # type: ignore

_last_retrain_time: float = 0.0
_extractor_validation: dict = {}


# ---------------------------------------------------------------------------
#  Background Task: Validate Skill Extractor
# ---------------------------------------------------------------------------
async def _validate_extractor_background():
    global _extractor_validation
    try:
        results = dataset_loader.validate_skill_extractor(
            parser=resume_parser, sample_size=50
        )
        _extractor_validation = results
        logger.info(
            f"Skill extractor validation: "
            f"avg {results.get('avg_skills_per_resume', 0)} skills/resume"
        )
    except Exception as e:
        logger.warning(f"Extractor validation failed: {e}")
        _extractor_validation = {"validated": False, "reason": str(e)}


# ---------------------------------------------------------------------------
#  Startup Banner
# ---------------------------------------------------------------------------
def _print_startup_banner():
    lr_metrics = ml_model.metrics.get('lr', {})
    dt_metrics = ml_model.metrics.get('dt', {})
    total_skills = sum(len(v) for v in skills_master.values())

    lr_acc = f"{ml_model.lr_accuracy}%"
    lr_f1 = f"{lr_metrics.get('f1', 0):.3f}"
    dt_acc = f"{ml_model.dt_accuracy}%"
    dt_f1 = f"{dt_metrics.get('f1', 0):.3f}"
    source = ml_model.dataset_source
    candidates = db.get_candidate_count()

    logger.info("")
    logger.info("+" + "=" * 56 + "+")
    logger.info("|     Automated Recruiting Platform  v2.0.0              |")
    logger.info("+" + "=" * 56 + "+")
    logger.info(f"|  Status     : Running                                  |")
    logger.info(f"|  Dataset    : {source:<42s}|")
    logger.info(f"|  LR Model   : Acc {lr_acc:<6s} F1 {lr_f1:<28s}|")
    logger.info(f"|  DT Model   : Acc {dt_acc:<6s} F1 {dt_f1:<28s}|")
    logger.info(f"|  Skills     : {total_skills} loaded{' ' * (36 - len(str(total_skills)))}|")
    logger.info(f"|  Roles      : {len(job_roles_data)} loaded{' ' * (36 - len(str(len(job_roles_data))))}|")
    logger.info(f"|  Candidates : {candidates} in database{' ' * (31 - len(str(candidates)))}|")
    logger.info(f"|  Docs       : http://localhost:8000/docs               |")
    logger.info("+" + "=" * 56 + "+")
    logger.info("")


# ---------------------------------------------------------------------------
#  Application Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global job_roles_data, skills_master
    global resume_parser, github_analyzer, feature_engineer
    global ml_model, skill_gap_analyzer, report_generator
    global dataset_loader, db

    logger.info("AUTOMATED RECRUITING PLATFORM — Starting Up")

    # Load data files
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    with open(os.path.join(data_dir, "job_roles.json"), "r") as f:
        job_roles_data = json.load(f)
    logger.info(f"Loaded {len(job_roles_data)} job roles.")

    with open(os.path.join(data_dir, "skills_master.json"), "r") as f:
        skills_master = json.load(f)
    total_skills = sum(len(v) for v in skills_master.values())
    logger.info(f"Loaded {total_skills} skills across {len(skills_master)} categories.")

    # Pre-compile regex patterns for skill matching (used across all modules)
    compiled_patterns = compile_skill_patterns(skills_master)
    set_compiled_patterns(compiled_patterns)
    logger.info(f"Pre-compiled {len(compiled_patterns)} skill regex patterns.")

    # Flatten skills_master once for reuse across modules
    set_flat_skills(skills_master)

    # Initialize modules
    logger.info("Initializing pipeline modules...")
    resume_parser = ResumeParser()
    github_analyzer = GitHubAnalyzer(github_token=GITHUB_TOKEN if GITHUB_TOKEN else None)
    feature_engineer = FeatureEngineer()
    ml_model = SkillGapMLModel()
    skill_gap_analyzer = SkillGapAnalyzer()
    report_generator = ReportGenerator()
    dataset_loader = DatasetLoader()
    db = Database()

    # Load or train models
    models_loaded = ml_model.load_models()
    if not models_loaded:
        logger.info("Training ML models from scratch...")
        X, y, source = dataset_loader.load_training_data()
        ml_model.train(X, y, dataset_source=source, use_cross_validation=True, tune_hyperparameters=False)
    else:
        logger.info("Using cached models — skipping retraining")

    asyncio.create_task(_validate_extractor_background())
    _print_startup_banner()

    yield

    # Close async HTTP client
    if github_analyzer:
        await github_analyzer.close()
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
            github_result = await github_analyzer.analyze_github_profile(github_username, skills_master)
            demonstrated_skills = github_result["demonstrated_skills"]
        except Exception as e:
            github_result["error"] = str(e)

    # Feature engineering
    role_data = job_roles_data[target_role]
    skill_matrix = feature_engineer.create_skill_matrix(
        claimed_skills,
        demonstrated_skills,
        role_data["required_skills"],
        role_data.get("nice_to_have", []),
    )
    X, y = feature_engineer.encode_for_model(skill_matrix)

    # ML predictions
    predictions = ml_model.predict(X)
    lr_probabilities = predictions["lr_probabilities"]

    # Skill gap analysis
    analysis = skill_gap_analyzer.analyze(
        claimed_skills=claimed_skills,
        demonstrated_skills=demonstrated_skills,
        target_role=target_role,
        job_roles_data=job_roles_data,
        ml_predictions=predictions,
        lr_probabilities=lr_probabilities,
        skill_matrix=skill_matrix,
    )

    # Report
    report = report_generator.generate_report(
        analysis_result=analysis,
        target_role=target_role,
        github_username=github_username,
        resume_skills=claimed_skills,
        github_skills=demonstrated_skills,
        model_summary=ml_model.get_model_summary(),
        github_insights_data=github_result,
    )

    # Add personal info and GitHub deep insights to report
    if personal_info:
        report["candidate_info"] = personal_info

    if github_result.get("commit_activity"):
        report["github_insights"]["commit_activity"] = github_result["commit_activity"]

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
    stats = db.get_dashboard_stats()
    return {
        "status": "running",
        "message": "Automated Recruiting Platform API",
        "version": "2.0.0",
        "available_roles": list(job_roles_data.keys()),
        "total_candidates": stats["total_candidates"],
        "total_analyses": stats["total_analyses"],
    }


# ---------------------------------------------------------------------------
#  ENDPOINT: Dashboard Stats
# ---------------------------------------------------------------------------
@app.get("/dashboard")
async def get_dashboard():
    stats = db.get_dashboard_stats()
    return stats


# ---------------------------------------------------------------------------
#  ENDPOINT: Get Available Job Roles
# ---------------------------------------------------------------------------
@app.get("/job-roles")
async def get_job_roles():
    roles = []
    for role_name, role_data in job_roles_data.items():
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
    return {"skills_master": skills_master}


# ---------------------------------------------------------------------------
#  ENDPOINT: Analyze Single (File Upload)
# ---------------------------------------------------------------------------
@app.post("/analyze")
async def analyze(
    resume_file: UploadFile = File(...),
    github_username: str = Form(""),
    target_role: str = Form(...),
):
    logger.info(f"ANALYSIS | File: {resume_file.filename} | "
                f"GitHub: {github_username} | Role: {target_role}")

    filename = resume_file.filename or ""
    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(400, "Unsupported file type. Upload .pdf or .txt.")

    if target_role not in job_roles_data:
        raise HTTPException(400, f"Unknown role: '{target_role}'. Available: {list(job_roles_data.keys())}")

    # Parse resume
    file_bytes = await resume_file.read()
    try:
        resume_result = resume_parser.parse(file_bytes, filename, skills_master)
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
    candidate_id = db.insert_candidate({
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

    analysis_id = db.insert_analysis({
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

    if request.target_role not in job_roles_data:
        raise HTTPException(400, f"Unknown role: '{request.target_role}'.")
    if not request.resume_text.strip():
        raise HTTPException(400, "Resume text cannot be empty.")

    claimed_skills = resume_parser.extract_skills(request.resume_text, skills_master)
    personal_info = resume_parser.extract_personal_info(request.resume_text)

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
    candidate_id = db.insert_candidate({
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

    analysis_id = db.insert_analysis({
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
async def analyze_batch(
    resume_files: List[UploadFile] = File(...),
    target_role: str = Form(...),
):
    """
    Analyze multiple resumes at once. Returns a ranked list of candidates.
    GitHub usernames are auto-extracted from resume content.
    """
    if target_role not in job_roles_data:
        raise HTTPException(400, f"Unknown role: '{target_role}'.")

    if len(resume_files) > 50:
        raise HTTPException(400, "Maximum 50 resumes per batch.")

    logger.info(f"BATCH ANALYSIS | {len(resume_files)} resumes | Role: {target_role}")

    batch_id = db.create_batch_job(target_role, len(resume_files))
    results = []
    errors = []

    for i, resume_file in enumerate(resume_files):
        filename = resume_file.filename or f"resume_{i}.txt"
        logger.info(f"  Batch [{i+1}/{len(resume_files)}]: {filename}")

        if not filename.lower().endswith((".pdf", ".txt")):
            errors.append({"file": filename, "error": "Unsupported file type"})
            continue

        try:
            file_bytes = await resume_file.read()
            resume_result = resume_parser.parse(file_bytes, filename, skills_master)
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
            candidate_id = db.insert_candidate({
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

            analysis_id = db.insert_analysis({
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
        db.add_batch_result(batch_id, r["candidate_id"], r["analysis_id"], rank)

    db.complete_batch_job(batch_id)

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
    job = db.get_batch_job(batch_id)
    if not job:
        raise HTTPException(404, "Batch job not found.")
    return job


# ---------------------------------------------------------------------------
#  ENDPOINT: Candidates List
# ---------------------------------------------------------------------------
@app.get("/candidates")
async def get_candidates(limit: int = 100, offset: int = 0):
    candidates = db.get_all_candidates(limit, offset)
    total = db.get_candidate_count()
    return {"candidates": candidates, "total": total}


# ---------------------------------------------------------------------------
#  ENDPOINT: Single Candidate Detail
# ---------------------------------------------------------------------------
@app.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: int):
    candidate = db.get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(404, "Candidate not found.")
    analyses = db.get_analyses_for_candidate(candidate_id)
    return {"candidate": candidate, "analyses": analyses}


# ---------------------------------------------------------------------------
#  ENDPOINT: Delete Candidate
# ---------------------------------------------------------------------------
@app.delete("/candidates/{candidate_id}")
async def delete_candidate(candidate_id: int):
    if db.delete_candidate(candidate_id):
        return {"status": "deleted", "candidate_id": candidate_id}
    raise HTTPException(404, "Candidate not found.")


# ---------------------------------------------------------------------------
#  ENDPOINT: Ranked Candidates for a Role
# ---------------------------------------------------------------------------
@app.get("/rankings/{target_role}")
async def get_rankings(target_role: str, limit: int = 50):
    if target_role not in job_roles_data:
        raise HTTPException(400, f"Unknown role: '{target_role}'.")
    rankings = db.get_ranked_candidates(target_role, limit)
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
    if request.target_role not in job_roles_data:
        raise HTTPException(400, f"Unknown role: '{request.target_role}'.")

    comparisons = db.get_candidates_comparison(
        request.candidate_ids, request.target_role
    )

    # Build comparison matrix
    role_data = job_roles_data[request.target_role]
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
        for category, skills in skills_master.items():
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
        job_roles_data[role_name] = {
            "required_skills": required_skills,
            "nice_to_have": nice_to_have,
        }

    return {
        "role_name": role_name,
        "required_skills": sorted(required_skills),
        "nice_to_have": sorted(nice_to_have),
        "total_skills_found": len(found_skills),
        "added_to_roles": role_name in job_roles_data,
    }


# ---------------------------------------------------------------------------
#  ENDPOINT: Analysis History (for sidebar)
# ---------------------------------------------------------------------------
@app.get("/analysis-history")
async def get_analysis_history(limit: int = 50):
    """Get recent analyses for the history sidebar."""
    analyses = db.get_recent_analyses(limit)
    return {"analyses": analyses}


@app.get("/analysis/{analysis_id}")
async def get_analysis_detail(analysis_id: int):
    """Get full analysis report by ID."""
    analysis = db.get_analysis_by_id(analysis_id)
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
    summary = ml_model.get_model_summary()
    summary["feature_importance"] = ml_model.get_feature_importance()
    return summary


# ---------------------------------------------------------------------------
#  ENDPOINT: Model Retrain
# ---------------------------------------------------------------------------
@app.get("/model-retrain")
async def retrain_model(tune: bool = False):
    global _last_retrain_time

    now = time.time()
    if now - _last_retrain_time < 3600:
        remaining = int(3600 - (now - _last_retrain_time))
        raise HTTPException(429, f"Retrain rate limited. Try again in {remaining}s.")

    logger.info("Retraining ML models with fresh data...")
    _last_retrain_time = now

    X, y, source = dataset_loader.load_training_data()
    metrics = ml_model.train(
        X, y, dataset_source=source,
        use_cross_validation=True, tune_hyperparameters=tune,
    )
    return {"status": "retrained", "dataset_source": source, "metrics": metrics}


# ---------------------------------------------------------------------------
#  ENDPOINT: Dataset Status
# ---------------------------------------------------------------------------
@app.get("/dataset-status")
async def get_dataset_status():
    cache_dir = dataset_loader.cache_dir
    cached_datasets = []
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            if os.path.isdir(os.path.join(cache_dir, item)):
                cached_datasets.append(item)

    return {
        "huggingface_available": len(cached_datasets) > 0,
        "cached_datasets": cached_datasets,
        "training_data_source": ml_model.dataset_source,
        "cache_dir": cache_dir,
        "extractor_validation": _extractor_validation,
        "model_trained": ml_model.is_trained,
    }


# ---------------------------------------------------------------------------
#  Run with Uvicorn (if executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
