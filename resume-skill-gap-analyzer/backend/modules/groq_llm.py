"""
=============================================================================
 Groq LLM Integration Module
=============================================================================
 Provides AI-powered analysis using Groq Cloud API with Llama 4 Scout.
 All functions gracefully return None if GROQ_API_KEY is not set,
 allowing the app to work without LLM integration.

 Features:
   - Enhanced skill extraction (catches implied/contextual skills)
   - AI Resume Coach (personalized improvement suggestions)
   - Interview question generation
   - Smart learning path generation
   - Candidate comparison summary (recruiter-facing)
   - Job description skill extraction
   - Batch executive summary for recruiters
   - Culture fit & soft skills analysis
   - Skill credibility scoring (validates resume-only claims)
   - Role-fit narrative (detailed fit analysis)
   - Retry logic with exponential backoff
   - In-memory LRU response cache with SHA256 keys
   - Model fallback chain
   - Response schema validation
=============================================================================
"""

import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from loguru import logger

# Groq client -- initialized lazily
_client = None
_available = False

# Model fallback chain -- try primary first, then fallbacks
MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

# Simple LRU cache (max 100 entries)
_cache: OrderedDict = OrderedDict()
_CACHE_MAX = 100
_CACHE_TTL = 3600  # 1 hour


def _get_client():
    """Get or create the Groq client. Returns None if API key not set."""
    global _client, _available
    if _client is not None:
        return _client
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        _available = False
        return None
    try:
        from groq import Groq
        _client = Groq(api_key=api_key)
        _available = True
        logger.info("[GroqLLM] Client initialized with model fallback chain.")
        return _client
    except Exception as e:
        logger.warning(f"[GroqLLM] Failed to initialize: {e}")
        _available = False
        return None


def is_available() -> bool:
    """Check if LLM integration is available."""
    _get_client()
    return _available


def _cache_key(system_prompt: str, user_prompt: str) -> str:
    """Generate a cache key from prompts using SHA256."""
    content = f"{system_prompt}||{user_prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    """Get from cache if not expired."""
    if key in _cache:
        result, ts = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            _cache.move_to_end(key)
            return result
        else:
            del _cache[key]
    return None


def _cache_set(key: str, value: str):
    """Store in cache with LRU eviction."""
    _cache[key] = (value, time.time())
    if len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def _validate_response(
    data: Any,
    required_keys: List[str],
    type_hints: Optional[Dict[str, type]] = None,
) -> bool:
    """Validate that the parsed JSON response contains expected keys and types."""
    if not isinstance(data, dict):
        return False
    if not all(k in data for k in required_keys):
        return False
    if type_hints:
        for key, expected_type in type_hints.items():
            if key in data and not isinstance(data[key], expected_type):
                return False
    return True


def _llm_call(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    use_cache: bool = True,
) -> Optional[str]:
    """
    Make an LLM call with retry logic, model fallback, and caching.

    Retries: 3 attempts with 1s, 2s, 4s backoff on transient errors.
    Fallback: Tries each model in MODELS list before giving up.
    Cache: In-memory LRU cache with 1-hour TTL using SHA256 keys.
    """
    client = _get_client()
    if not client:
        return None

    # Check cache first
    if use_cache:
        key = _cache_key(system_prompt, user_prompt)
        cached = _cache_get(key)
        if cached:
            logger.debug("[GroqLLM] Cache hit")
            return cached

    # Try each model in fallback chain
    for model_idx, model in enumerate(MODELS):
        # Retry loop with exponential backoff
        for attempt in range(3):
            try:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = client.chat.completions.create(**kwargs)
                result = response.choices[0].message.content

                # Cache the result
                if use_cache and result:
                    _cache_set(_cache_key(system_prompt, user_prompt), result)

                if model_idx > 0:
                    logger.info(f"[GroqLLM] Succeeded with fallback model: {model}")
                return result

            except Exception as e:
                error_str = str(e).lower()
                is_transient = any(kw in error_str for kw in [
                    "rate_limit", "rate limit", "timeout", "503", "529",
                    "overloaded", "too many requests",
                ])

                if is_transient and attempt < 2:
                    wait = (2 ** attempt)  # 1s, 2s, 4s
                    logger.warning(f"[GroqLLM] Transient error (attempt {attempt+1}/3), retrying in {wait}s: {e}")
                    time.sleep(wait)
                    continue
                elif model_idx < len(MODELS) - 1:
                    logger.warning(f"[GroqLLM] Model {model} failed, trying fallback: {e}")
                    break  # Try next model
                else:
                    logger.error(f"[GroqLLM] All models failed: {e}")
                    return None
    return None


# =========================================================================
#  Skill Extraction
# =========================================================================

def extract_skills_with_llm(resume_text: str, known_skills: List[str]) -> List[str]:
    """
    Use LLM to extract skills from resume text, including implied/contextual ones.
    Returns list of additional skills not already detected by regex.
    """
    if not is_available():
        return []

    system_prompt = (
        "You are a technical recruiter AI. Extract ALL technical skills from the resume text. "
        "Include explicitly mentioned skills AND implied skills (e.g., if someone mentions "
        "'built a REST API with authentication', imply OAuth, JWT, API Design). "
        "Return ONLY a JSON object with key 'skills' containing an array of skill name strings. "
        "Use standard canonical names (e.g., 'JavaScript' not 'JS', 'Kubernetes' not 'k8s')."
    )
    user_prompt = f"Resume text:\n{resume_text[:4000]}"

    result = _llm_call(system_prompt, user_prompt, json_mode=True, temperature=0.1)
    if not result:
        return []

    try:
        data = json.loads(result)
        if not _validate_response(data, ["skills"], {"skills": list}):
            logger.warning("[GroqLLM] Skill extraction response missing or invalid 'skills' key")
            return []
        llm_skills = data.get("skills", [])
        # Return only skills not already found by regex
        new_skills = [s for s in llm_skills if isinstance(s, str) and s not in known_skills]
        logger.info(f"[GroqLLM] Extracted {len(new_skills)} additional skills via LLM")
        return new_skills
    except (json.JSONDecodeError, KeyError):
        logger.warning("[GroqLLM] Failed to parse skill extraction response")
        return []


# =========================================================================
#  AI Resume Coach
# =========================================================================

def generate_ai_feedback(
    resume_text: str,
    target_role: str,
    missing_skills: List[str],
    strengths: List[str],
    match_score: float,
) -> Optional[Dict]:
    """
    Generate personalized AI resume coaching feedback.
    Returns dict with resume_tips, bullet_suggestions, overall_advice, keyword_suggestions.
    """
    if not is_available():
        return None

    system_prompt = (
        "You are an expert career coach and resume reviewer. Provide actionable, specific advice "
        "to help the candidate improve their resume for the target role. Be encouraging but honest. "
        "Return a JSON object with keys: "
        "'resume_tips' (array of 3-5 specific improvement tips), "
        "'bullet_suggestions' (array of 2-3 bullet points the candidate could add to strengthen gaps), "
        "'overall_advice' (a brief 2-3 sentence encouraging summary), "
        "'keyword_suggestions' (array of 5-8 keywords to add for ATS optimization), "
        "'formatting_tips' (array of 2-3 resume formatting improvements)."
    )
    user_prompt = (
        f"Target Role: {target_role}\n"
        f"Match Score: {match_score}%\n"
        f"Strengths: {', '.join(strengths[:10])}\n"
        f"Missing Skills: {', '.join(missing_skills[:10])}\n"
        f"Resume excerpt:\n{resume_text[:3000]}"
    )

    result = _llm_call(system_prompt, user_prompt, json_mode=True, temperature=0.4)
    if not result:
        return None

    try:
        data = json.loads(result)
        if not _validate_response(data, ["resume_tips", "overall_advice"], {"resume_tips": list}):
            logger.warning("[GroqLLM] AI feedback response missing required keys")
            return None
        logger.info("[GroqLLM] Generated AI resume feedback")
        return data
    except json.JSONDecodeError:
        logger.warning("[GroqLLM] Failed to parse AI feedback response")
        return None


# =========================================================================
#  Interview Questions
# =========================================================================

def generate_interview_questions(
    target_role: str,
    claimed_skills: List[str],
    missing_skills: List[str],
    claims_not_proven: List[str],
) -> Optional[List[Dict]]:
    """
    Generate likely interview questions based on the candidate's profile.
    Returns list of {question, skill, difficulty, prep_hint, why_asked}.
    """
    if not is_available():
        return None

    system_prompt = (
        "You are a senior technical interviewer. Generate interview questions tailored to "
        "this candidate's skill profile. Focus on skills they claim but haven't demonstrated "
        "(to verify claims) and skills critical to the role. "
        "Return a JSON object with key 'questions' containing an array of objects, each with: "
        "'question' (the interview question), "
        "'skill' (the skill being tested), "
        "'difficulty' ('easy'/'medium'/'hard'), "
        "'prep_hint' (brief hint on how to prepare for this question), "
        "'why_asked' (brief explanation of why this question is important for the role)."
    )
    user_prompt = (
        f"Target Role: {target_role}\n"
        f"Claimed skills: {', '.join(claimed_skills[:15])}\n"
        f"Unproven claims (on resume but not GitHub): {', '.join(claims_not_proven[:10])}\n"
        f"Missing critical skills: {', '.join(missing_skills[:10])}\n"
        f"Generate 5-7 targeted interview questions."
    )

    result = _llm_call(system_prompt, user_prompt, json_mode=True, temperature=0.5)
    if not result:
        return None

    try:
        data = json.loads(result)
        if not _validate_response(data, ["questions"], {"questions": list}):
            logger.warning("[GroqLLM] Interview questions response missing 'questions' key")
            return None
        questions = data.get("questions", [])
        if not isinstance(questions, list):
            return None
        logger.info(f"[GroqLLM] Generated {len(questions)} interview questions")
        return questions
    except json.JSONDecodeError:
        logger.warning("[GroqLLM] Failed to parse interview questions response")
        return None


# =========================================================================
#  Learning Path
# =========================================================================

def generate_learning_path(
    target_role: str,
    missing_skills: List[str],
    current_skills: List[str],
) -> Optional[List[Dict]]:
    """
    Generate a personalized learning path with specific resources.
    Returns list of {skill, week, resources, project_idea, prerequisites}.
    """
    if not is_available():
        return None

    system_prompt = (
        "You are a technical learning advisor. Create a personalized, prioritized learning plan. "
        "Consider skill dependencies (e.g., learn Docker before Kubernetes). "
        "Return a JSON object with key 'learning_path' containing an array of objects, each with: "
        "'skill' (skill to learn), "
        "'week' (suggested week number, 1-8), "
        "'resources' (array of 2-3 learning resources, each as an object with 'name' (resource title/description) "
        "and 'url' (the actual URL -- only include a URL if you are confident it is correct, otherwise omit the url field)), "
        "'project_idea' (a specific mini-project to demonstrate this skill), "
        "'prerequisites' (array of skills that should be learned first, can be empty), "
        "'estimated_hours' (estimated hours to reach basic competency, as integer)."
    )
    user_prompt = (
        f"Target Role: {target_role}\n"
        f"Skills to learn: {', '.join(missing_skills[:12])}\n"
        f"Current skills: {', '.join(current_skills[:15])}\n"
        f"Create a structured learning plan for the missing skills."
    )

    result = _llm_call(system_prompt, user_prompt, json_mode=True, max_tokens=3000, temperature=0.3)
    if not result:
        return None

    try:
        data = json.loads(result)
        if not _validate_response(data, ["learning_path"], {"learning_path": list}):
            logger.warning("[GroqLLM] Learning path response missing 'learning_path' key")
            return None
        path = data.get("learning_path", [])
        if not isinstance(path, list):
            return None
        logger.info(f"[GroqLLM] Generated learning path with {len(path)} items")
        return path
    except json.JSONDecodeError:
        logger.warning("[GroqLLM] Failed to parse learning path response")
        return None


# =========================================================================
#  Skill Credibility Scoring (NEW)
# =========================================================================

def generate_skill_credibility_assessment(
    resume_text: str,
    claimed_skills: List[str],
    demonstrated_skills: List[str],
    claims_not_proven: List[str],
) -> Optional[Dict]:
    """
    Use LLM to assess the credibility of resume-only skill claims by
    analyzing context clues in the resume text.

    Returns dict with per-skill credibility scores and overall assessment.
    """
    if not is_available():
        return None

    if not claims_not_proven:
        return None

    system_prompt = (
        "You are a senior technical recruiter specializing in skill verification. "
        "Analyze the resume text to assess how credible each unverified skill claim is. "
        "Look for contextual evidence: project descriptions, experience duration, "
        "specific version numbers, concrete accomplishments, and related skill clusters. "
        "Return a JSON object with EXACTLY these keys: "
        "'overall_credibility_score' (integer 1-10 rating of overall resume credibility), "
        "'assessment' (1-2 sentence summary of overall credibility assessment), "
        "'verified_skills' (array of skill name strings that appear credible based on resume context — "
        "skills with strong project evidence, version numbers, or detailed accomplishments), "
        "'questionable_skills' (array of skill name strings that lack sufficient evidence — "
        "skills mentioned without context, projects, or detail), "
        "'recommendations' (array of 2-4 actionable string recommendations for how to verify "
        "the questionable skills during the interview process)."
    )
    user_prompt = (
        f"Skills confirmed on GitHub: {', '.join(demonstrated_skills[:15])}\n"
        f"Skills ONLY on resume (unverified): {', '.join(claims_not_proven[:12])}\n"
        f"All claimed skills: {', '.join(claimed_skills[:20])}\n"
        f"Resume text:\n{resume_text[:3500]}"
    )

    result = _llm_call(system_prompt, user_prompt, json_mode=True, max_tokens=2000, temperature=0.2)
    if not result:
        return None

    try:
        data = json.loads(result)
        if not _validate_response(data, ["overall_credibility_score", "assessment"]):
            logger.warning("[GroqLLM] Skill credibility response missing required keys")
            return None
        # Ensure arrays exist even if LLM omitted them
        if "verified_skills" not in data:
            data["verified_skills"] = []
        if "questionable_skills" not in data:
            data["questionable_skills"] = []
        if "recommendations" not in data:
            data["recommendations"] = []
        logger.info(f"[GroqLLM] Generated skill credibility assessment for {len(claims_not_proven)} skills")
        return data
    except json.JSONDecodeError:
        logger.warning("[GroqLLM] Failed to parse skill credibility response")
        return None


# =========================================================================
#  Role-Fit Narrative (NEW)
# =========================================================================

def generate_role_fit_narrative(
    target_role: str,
    match_score: float,
    strengths: List[str],
    missing_skills: List[str],
    claims_not_proven: List[str],
    hidden_strengths: List[str],
    github_insights: Dict = None,
) -> Optional[Dict]:
    """
    Generate a detailed role-fit narrative that explains WHY the candidate
    is or isn't a good fit, beyond just listing skills.

    Returns dict with narrative sections for different audiences.
    """
    if not is_available():
        return None

    github_context = ""
    if github_insights:
        repos = github_insights.get("repos_analyzed", 0)
        langs = [l.get("language", "") for l in github_insights.get("top_languages", [])]
        github_context = f"\nGitHub: {repos} repos, top languages: {', '.join(langs)}"

    system_prompt = (
        "You are a talent analytics specialist. Write a comprehensive role-fit analysis "
        "that explains the candidate's fit for the role in detail. Go beyond listing skills -- "
        "analyze patterns, identify strengths, and explain gaps in context. "
        "Return a JSON object with EXACTLY these keys: "
        "'fit_score' (integer 1-10 rating of overall role fit, where 10 is perfect fit), "
        "'narrative' (2-3 sentence explanation of what the match score means in practice "
        "and how well the candidate fits this specific role), "
        "'standout_qualities' (array of 3-5 short string descriptions of the candidate's "
        "most impressive qualities relevant to this role — e.g. 'Strong React ecosystem experience'), "
        "'growth_areas' (array of 2-4 short string descriptions of areas where the candidate "
        "needs development — e.g. 'Database design and SQL optimization'), "
        "'onboarding_estimate' (string estimating ramp-up time — e.g. '2-4 weeks for core tasks, "
        "2-3 months for full productivity')."
    )
    user_prompt = (
        f"Target Role: {target_role}\n"
        f"Match Score: {match_score}%\n"
        f"Verified Strengths (resume + GitHub): {', '.join(strengths[:10])}\n"
        f"Unverified Claims (resume only): {', '.join(claims_not_proven[:8])}\n"
        f"Hidden Strengths (GitHub only): {', '.join(hidden_strengths[:5])}\n"
        f"Missing Critical Skills: {', '.join(missing_skills[:8])}"
        f"{github_context}"
    )

    result = _llm_call(system_prompt, user_prompt, json_mode=True, max_tokens=2000, temperature=0.3)
    if not result:
        return None

    try:
        data = json.loads(result)
        if not _validate_response(data, ["fit_score", "narrative"]):
            logger.warning("[GroqLLM] Role-fit narrative response missing required keys")
            return None
        # Ensure arrays exist even if LLM omitted them
        if "standout_qualities" not in data:
            data["standout_qualities"] = []
        if "growth_areas" not in data:
            data["growth_areas"] = []
        if "onboarding_estimate" not in data:
            data["onboarding_estimate"] = ""
        logger.info("[GroqLLM] Generated role-fit narrative")
        return data
    except json.JSONDecodeError:
        logger.warning("[GroqLLM] Failed to parse role-fit narrative response")
        return None


# =========================================================================
#  Recruiter-Focused AI Features
# =========================================================================

def generate_candidate_summary(
    candidate_name: str,
    resume_text: str,
    target_role: str,
    match_score: float,
    strengths: List[str],
    missing_skills: List[str],
    github_insights: Dict = None,
) -> Optional[Dict]:
    """
    Generate a recruiter-facing executive summary for a single candidate.
    This is the 'elevator pitch' a recruiter can paste into their ATS or email.
    """
    if not is_available():
        return None

    github_context = ""
    if github_insights:
        repos = github_insights.get("repos_analyzed", 0)
        langs = [l.get("language", "") for l in github_insights.get("top_languages", [])]
        github_context = f"\nGitHub: {repos} repos analyzed, top languages: {', '.join(langs)}"

    system_prompt = (
        "You are a senior technical recruiter writing a candidate assessment. "
        "Write a concise, professional summary that another recruiter or hiring manager "
        "can quickly scan to decide whether to move forward with the candidate. "
        "Return a JSON object with keys: "
        "'headline' (one-line candidate headline, e.g. 'Strong Python backend dev with ML experience'), "
        "'executive_summary' (3-4 sentence paragraph assessing overall fit), "
        "'top_strengths' (array of 3 strongest selling points as strings), "
        "'risk_factors' (array of 1-3 potential concerns or gaps), "
        "'hiring_recommendation' ('Strong Hire' / 'Hire' / 'Maybe' / 'Pass'), "
        "'salary_positioning' (brief note on where they'd likely land: junior/mid/senior based on skills)."
    )
    user_prompt = (
        f"Candidate: {candidate_name}\n"
        f"Target Role: {target_role}\n"
        f"Match Score: {match_score}%\n"
        f"Key Strengths: {', '.join(strengths[:10])}\n"
        f"Missing Skills: {', '.join(missing_skills[:8])}{github_context}\n"
        f"Resume excerpt:\n{resume_text[:2500]}"
    )

    result = _llm_call(system_prompt, user_prompt, json_mode=True, max_tokens=1500, temperature=0.2)
    if not result:
        return None

    try:
        data = json.loads(result)
        if not _validate_response(data, ["headline", "executive_summary", "hiring_recommendation"]):
            logger.warning("[GroqLLM] Candidate summary response missing required keys")
            return None
        logger.info(f"[GroqLLM] Generated candidate summary for {candidate_name}")
        return data
    except json.JSONDecodeError:
        logger.warning("[GroqLLM] Failed to parse candidate summary response")
        return None


def generate_batch_executive_report(
    target_role: str,
    rankings: List[Dict],
) -> Optional[Dict]:
    """
    Generate an executive summary for an entire batch of candidates.
    Helps recruiters quickly understand the talent pool quality.
    """
    if not is_available():
        return None

    # Build a compact summary of top candidates
    top_candidates = []
    for r in rankings[:10]:
        top_candidates.append(
            f"#{r.get('rank', '?')} {r.get('name', 'Unknown')} -- "
            f"{r.get('match_score', 0):.0f}% match, "
            f"{r.get('missing_count', 0)} gaps"
        )

    avg_score = sum(r.get("match_score", 0) for r in rankings) / max(len(rankings), 1)

    system_prompt = (
        "You are a recruiting analytics AI. Analyze this batch of candidates and provide "
        "an executive summary for the hiring team. "
        "Return a JSON object with keys: "
        "'pool_quality' ('Excellent' / 'Good' / 'Fair' / 'Weak'), "
        "'summary' (2-3 sentence overview of the talent pool), "
        "'top_pick_rationale' (why the #1 candidate stands out), "
        "'common_gaps' (array of skills most candidates are missing), "
        "'hiring_advice' (1-2 sentences of actionable advice for the recruiter), "
        "'diversity_of_skills' (brief note on whether candidates bring varied or similar backgrounds), "
        "'market_difficulty' (1 sentence on how hard this role is to fill based on the candidate pool)."
    )
    user_prompt = (
        f"Target Role: {target_role}\n"
        f"Total Candidates: {len(rankings)}\n"
        f"Average Match Score: {avg_score:.1f}%\n"
        f"Top Candidates:\n" + "\n".join(top_candidates)
    )

    result = _llm_call(system_prompt, user_prompt, json_mode=True, max_tokens=1500, temperature=0.3)
    if not result:
        return None

    try:
        data = json.loads(result)
        if not _validate_response(data, ["pool_quality", "summary"]):
            logger.warning("[GroqLLM] Batch report response missing required keys")
            return None
        logger.info(f"[GroqLLM] Generated batch executive report for {len(rankings)} candidates")
        return data
    except json.JSONDecodeError:
        logger.warning("[GroqLLM] Failed to parse batch executive report")
        return None


def generate_jd_skills_extraction(
    job_description: str,
) -> Optional[Dict]:
    """
    Use LLM to intelligently parse a job description and extract structured skills.
    Better than regex -- understands context, synonyms, and implied requirements.
    """
    if not is_available():
        return None

    system_prompt = (
        "You are an expert job description parser for a recruiting platform. "
        "Extract all technical skills, tools, frameworks, and concepts from the job description. "
        "Classify them as required vs nice-to-have based on context clues. "
        "Return a JSON object with keys: "
        "'required_skills' (array of must-have skill strings), "
        "'nice_to_have' (array of preferred/bonus skill strings), "
        "'experience_level' ('Junior' / 'Mid' / 'Senior' / 'Staff' / 'Principal'), "
        "'role_summary' (one sentence describing the role), "
        "'team_size_hint' (estimated team size if mentioned, otherwise 'unknown'), "
        "'key_responsibilities' (array of 3-5 main responsibilities)."
    )
    user_prompt = f"Job Description:\n{job_description[:4000]}"

    result = _llm_call(system_prompt, user_prompt, json_mode=True, max_tokens=2000, temperature=0.1)
    if not result:
        return None

    try:
        data = json.loads(result)
        if not _validate_response(data, ["required_skills"], {"required_skills": list}):
            logger.warning("[GroqLLM] JD extraction response missing required keys")
            return None
        logger.info(f"[GroqLLM] Extracted skills from JD: "
                     f"{len(data.get('required_skills', []))} required, "
                     f"{len(data.get('nice_to_have', []))} nice-to-have")
        return data
    except json.JSONDecodeError:
        logger.warning("[GroqLLM] Failed to parse JD skill extraction")
        return None


def generate_culture_fit_analysis(
    resume_text: str,
    target_role: str,
) -> Optional[Dict]:
    """
    Analyze soft skills and cultural indicators from the resume.
    Helps recruiters assess beyond just technical skills.
    """
    if not is_available():
        return None

    system_prompt = (
        "You are a talent assessment specialist. Analyze the resume for soft skills, "
        "leadership indicators, communication style, and cultural signals. "
        "Return a JSON object with keys: "
        "'soft_skills' (array of detected soft skills like 'Leadership', 'Collaboration', etc.), "
        "'communication_score' (1-10 rating of how well the resume communicates), "
        "'leadership_indicators' (array of specific examples from the resume showing leadership), "
        "'team_fit_notes' (1-2 sentences on likely team dynamics), "
        "'work_style' (brief assessment: 'independent'/'collaborative'/'mixed'), "
        "'red_flags' (array of any concerning patterns, empty array if none)."
    )
    user_prompt = (
        f"Target Role: {target_role}\n"
        f"Resume:\n{resume_text[:3500]}"
    )

    result = _llm_call(system_prompt, user_prompt, json_mode=True, max_tokens=1500, temperature=0.3)
    if not result:
        return None

    try:
        data = json.loads(result)
        if not _validate_response(data, ["soft_skills", "communication_score"]):
            logger.warning("[GroqLLM] Culture fit response missing required keys")
            return None
        logger.info("[GroqLLM] Generated culture fit analysis")
        return data
    except json.JSONDecodeError:
        logger.warning("[GroqLLM] Failed to parse culture fit analysis")
        return None
