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
=============================================================================
"""

import json
import os
from typing import Dict, List, Optional

from loguru import logger

# Groq client — initialized lazily
_client = None
_available = False

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


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
        logger.info("[GroqLLM] Client initialized with Llama 4 Scout model.")
        return _client
    except Exception as e:
        logger.warning(f"[GroqLLM] Failed to initialize: {e}")
        _available = False
        return None


def is_available() -> bool:
    """Check if LLM integration is available."""
    _get_client()
    return _available


def _llm_call(system_prompt: str, user_prompt: str, json_mode: bool = False, max_tokens: int = 2048) -> Optional[str]:
    """Make a single LLM call. Returns None on failure."""
    client = _get_client()
    if not client:
        return None
    try:
        kwargs = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"[GroqLLM] API call failed: {e}")
        return None


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

    result = _llm_call(system_prompt, user_prompt, json_mode=True)
    if not result:
        return []

    try:
        data = json.loads(result)
        llm_skills = data.get("skills", [])
        # Return only skills not already found by regex
        new_skills = [s for s in llm_skills if s not in known_skills]
        logger.info(f"[GroqLLM] Extracted {len(new_skills)} additional skills via LLM")
        return new_skills
    except (json.JSONDecodeError, KeyError):
        logger.warning("[GroqLLM] Failed to parse skill extraction response")
        return []


def generate_ai_feedback(
    resume_text: str,
    target_role: str,
    missing_skills: List[str],
    strengths: List[str],
    match_score: float,
) -> Optional[Dict]:
    """
    Generate personalized AI resume coaching feedback.
    Returns dict with resume_tips, bullet_suggestions, and overall_advice.
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
        "'keyword_suggestions' (array of 5-8 keywords to add for ATS optimization)."
    )
    user_prompt = (
        f"Target Role: {target_role}\n"
        f"Match Score: {match_score}%\n"
        f"Strengths: {', '.join(strengths[:10])}\n"
        f"Missing Skills: {', '.join(missing_skills[:10])}\n"
        f"Resume excerpt:\n{resume_text[:3000]}"
    )

    result = _llm_call(system_prompt, user_prompt, json_mode=True)
    if not result:
        return None

    try:
        data = json.loads(result)
        logger.info("[GroqLLM] Generated AI resume feedback")
        return data
    except json.JSONDecodeError:
        logger.warning("[GroqLLM] Failed to parse AI feedback response")
        return None


def generate_interview_questions(
    target_role: str,
    claimed_skills: List[str],
    missing_skills: List[str],
    claims_not_proven: List[str],
) -> Optional[List[Dict]]:
    """
    Generate likely interview questions based on the candidate's profile.
    Returns list of {question, skill, difficulty, prep_hint}.
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
        "'prep_hint' (brief hint on how to prepare for this question)."
    )
    user_prompt = (
        f"Target Role: {target_role}\n"
        f"Claimed skills: {', '.join(claimed_skills[:15])}\n"
        f"Unproven claims (on resume but not GitHub): {', '.join(claims_not_proven[:10])}\n"
        f"Missing critical skills: {', '.join(missing_skills[:10])}\n"
        f"Generate 5-7 targeted interview questions."
    )

    result = _llm_call(system_prompt, user_prompt, json_mode=True)
    if not result:
        return None

    try:
        data = json.loads(result)
        questions = data.get("questions", [])
        logger.info(f"[GroqLLM] Generated {len(questions)} interview questions")
        return questions
    except json.JSONDecodeError:
        logger.warning("[GroqLLM] Failed to parse interview questions response")
        return None


def generate_learning_path(
    target_role: str,
    missing_skills: List[str],
    current_skills: List[str],
) -> Optional[List[Dict]]:
    """
    Generate a personalized learning path with specific resources.
    Returns list of {skill, week, resources, project_idea}.
    """
    if not is_available():
        return None

    system_prompt = (
        "You are a technical learning advisor. Create a personalized, prioritized learning plan. "
        "Consider skill dependencies (e.g., learn Docker before Kubernetes). "
        "Return a JSON object with key 'learning_path' containing an array of objects, each with: "
        "'skill' (skill to learn), "
        "'week' (suggested week number, 1-8), "
        "'resources' (array of 2-3 specific free learning resources with names and URLs), "
        "'project_idea' (a specific mini-project to demonstrate this skill)."
    )
    user_prompt = (
        f"Target Role: {target_role}\n"
        f"Skills to learn: {', '.join(missing_skills[:12])}\n"
        f"Current skills: {', '.join(current_skills[:15])}\n"
        f"Create a structured learning plan for the missing skills."
    )

    result = _llm_call(system_prompt, user_prompt, json_mode=True, max_tokens=3000)
    if not result:
        return None

    try:
        data = json.loads(result)
        path = data.get("learning_path", [])
        logger.info(f"[GroqLLM] Generated learning path with {len(path)} items")
        return path
    except json.JSONDecodeError:
        logger.warning("[GroqLLM] Failed to parse learning path response")
        return None
