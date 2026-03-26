"""
=============================================================================
 Code Quality Analyzer Module
=============================================================================
 Analyzes source code for quality metrics using the Groq LLM integration.
 Evaluates code on five dimensions:
   - Speed: Algorithm efficiency, time complexity
   - Complexity: Cyclomatic complexity, nesting depth, space complexity
   - Flexibility: Modularity, design patterns, extensibility
   - Code Quality: Naming conventions, documentation, error handling
   - Best Practices: SOLID principles, DRY, separation of concerns

 Returns structured scores (1-10) with textual feedback per dimension.
 Gracefully returns None if Groq LLM is unavailable.
=============================================================================
"""

import json
from typing import Dict, List, Optional

from loguru import logger

from modules.groq_llm import is_available, _llm_call, _validate_response


# Supported source code file extensions
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".rb", ".php", ".kt", ".swift", ".cs", ".scala",
}

# Directories to skip when selecting files for analysis
SKIP_DIRS = {
    "node_modules", "vendor", "dist", "build", "__pycache__",
    ".git", ".venv", "venv", "env", ".tox", "coverage",
    "migrations", ".next", "out", "target", "bin", "obj",
}

# Files to skip
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "Pipfile.lock",
    "poetry.lock", "go.sum",
}

MAX_CODE_LENGTH = 15000  # chars per file sent to LLM
MAX_LINES = 500  # max lines per file


def is_code_file(filename: str) -> bool:
    """Check if a filename has a supported code extension."""
    dot_idx = filename.rfind(".")
    if dot_idx < 0:
        return False
    return filename[dot_idx:].lower() in CODE_EXTENSIONS


def should_skip_path(path: str) -> bool:
    """Check if a file path is in a directory we should skip."""
    parts = path.lower().split("/")
    for part in parts[:-1]:  # check directories, not the filename
        if part in SKIP_DIRS:
            return True
    # Skip test files
    filename = parts[-1] if parts else ""
    if filename in SKIP_FILES:
        return True
    if filename.startswith("test_") or filename.endswith("_test.py"):
        return True
    if ".test." in filename or ".spec." in filename:
        return True
    if filename.startswith("__") and filename.endswith("__.py"):
        return True
    return False


def detect_language(filename: str) -> str:
    """Detect programming language from file extension."""
    ext_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".jsx": "JavaScript (React)", ".tsx": "TypeScript (React)",
        ".java": "Java", ".go": "Go", ".rs": "Rust",
        ".cpp": "C++", ".c": "C", ".rb": "Ruby", ".php": "PHP",
        ".kt": "Kotlin", ".swift": "Swift", ".cs": "C#", ".scala": "Scala",
    }
    dot_idx = filename.rfind(".")
    if dot_idx < 0:
        return "Unknown"
    return ext_map.get(filename[dot_idx:].lower(), "Unknown")


def analyze_code_quality(
    code: str,
    language: str = "",
    context: str = "general",
    challenge_description: str = "",
    filename: str = "",
) -> Optional[Dict]:
    """
    Analyze a single code snippet for quality metrics using LLM.

    Args:
        code: The source code to analyze.
        language: Programming language (e.g., "Python", "JavaScript").
        context: Analysis context — "github", "challenge", or "uploaded".
        challenge_description: If context is "challenge", the problem statement.
        filename: Original filename for context.

    Returns:
        Dict with scores and feedback, or None if LLM unavailable.
    """
    if not is_available():
        return None

    if not code or not code.strip():
        return None

    # Truncate if too long
    code_trimmed = code[:MAX_CODE_LENGTH]

    challenge_ctx = ""
    if challenge_description:
        challenge_ctx = (
            f"\n\nThe code was written to solve this problem:\n{challenge_description[:1500]}\n"
            "Also evaluate correctness — does the code solve the stated problem?"
        )

    system_prompt = (
        "You are an expert code reviewer and software quality analyst. "
        "Analyze the provided source code and evaluate it on five quality dimensions. "
        "Be specific — reference actual code patterns, function names, and line-level observations. "
        "Return a JSON object with EXACTLY these keys:\n"
        "'speed': {'score': <int 1-10>, 'notes': '<specific observations about algorithm efficiency>'}, "
        "'complexity': {'score': <int 1-10>, 'notes': '<observations about code complexity, nesting, readability>'}, "
        "'flexibility': {'score': <int 1-10>, 'notes': '<observations about modularity, extensibility, patterns>'}, "
        "'code_quality': {'score': <int 1-10>, 'notes': '<observations about naming, docs, error handling>'}, "
        "'best_practices': {'score': <int 1-10>, 'notes': '<observations about SOLID, DRY, separation of concerns>'}, "
        "'overall_score': <float, weighted average of all 5 scores>, "
        "'time_complexity': '<estimated Big-O time complexity, e.g. O(n log n)>', "
        "'space_complexity': '<estimated Big-O space complexity, e.g. O(n)>', "
        "'summary': '<2-3 sentence overall assessment>', "
        "'detected_patterns': [<array of design patterns or algorithms identified>], "
        "'improvement_suggestions': [<array of 2-4 specific actionable improvements>]"
    )

    if challenge_description:
        system_prompt += (
            ", 'correctness': {'score': <int 1-10>, 'notes': '<does it solve the problem correctly?>'}"
        )

    user_prompt = (
        f"Language: {language or 'Auto-detect'}\n"
        f"Context: {context}\n"
        f"{'Filename: ' + filename if filename else ''}"
        f"{challenge_ctx}\n\n"
        f"Code:\n```\n{code_trimmed}\n```"
    )

    result = _llm_call(
        system_prompt, user_prompt,
        json_mode=True, max_tokens=2048,
        temperature=0.2, model_tier="reasoning",
    )
    if not result:
        return None

    try:
        data = json.loads(result)
        required = ["speed", "complexity", "flexibility", "code_quality", "best_practices"]
        if not all(k in data for k in required):
            logger.warning("[CodeQuality] Response missing required dimension keys")
            return None

        # Ensure all dimension scores are valid
        for dim in required:
            if isinstance(data[dim], dict):
                score = data[dim].get("score", 5)
                data[dim]["score"] = max(1, min(10, int(score)))
            else:
                data[dim] = {"score": 5, "notes": "Unable to assess"}

        # Compute overall if missing or invalid
        dim_scores = [data[d]["score"] for d in required]
        data["overall_score"] = round(sum(dim_scores) / len(dim_scores), 1)

        # Ensure optional fields exist
        data.setdefault("time_complexity", "Unknown")
        data.setdefault("space_complexity", "Unknown")
        data.setdefault("summary", "")
        data.setdefault("detected_patterns", [])
        data.setdefault("improvement_suggestions", [])
        data["source"] = context
        data["language"] = language or "Unknown"

        logger.info(f"[CodeQuality] Analyzed code — overall: {data['overall_score']}/10")
        return data

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"[CodeQuality] Failed to parse LLM response: {e}")
        return None


def analyze_code_batch(
    files: List[Dict],
    context: str = "uploaded",
) -> Optional[Dict]:
    """
    Analyze multiple code files and return per-file + aggregate scores.

    Args:
        files: List of dicts with keys "filename", "content", "language" (optional).
        context: Analysis context — "github", "challenge", or "uploaded".

    Returns:
        Dict with "per_file" results and "aggregate" scores, or None.
    """
    if not is_available():
        return None

    if not files:
        return None

    per_file_results = []
    total_scores = {"speed": 0, "complexity": 0, "flexibility": 0,
                    "code_quality": 0, "best_practices": 0}
    valid_count = 0

    for file_info in files[:15]:  # cap at 15 files
        filename = file_info.get("filename", "unknown")
        content = file_info.get("content", "")
        language = file_info.get("language", "") or detect_language(filename)

        if not content.strip():
            continue

        result = analyze_code_quality(
            code=content,
            language=language,
            context=context,
            filename=filename,
        )

        if result:
            per_file_results.append({
                "filename": filename,
                "language": result.get("language", language),
                "scores": result,
            })
            for dim in total_scores:
                total_scores[dim] += result.get(dim, {}).get("score", 0)
            valid_count += 1

    if valid_count == 0:
        return None

    # Compute aggregate
    aggregate = {}
    for dim in total_scores:
        avg = round(total_scores[dim] / valid_count, 1)
        aggregate[dim] = {"score": avg, "notes": f"Average across {valid_count} files"}

    dim_scores = [aggregate[d]["score"] for d in total_scores]
    aggregate["overall_score"] = round(sum(dim_scores) / len(dim_scores), 1)
    aggregate["files_analyzed"] = valid_count

    logger.info(f"[CodeQuality] Batch analysis complete — {valid_count} files, "
                f"overall: {aggregate['overall_score']}/10")

    return {
        "per_file": per_file_results,
        "aggregate": aggregate,
    }
