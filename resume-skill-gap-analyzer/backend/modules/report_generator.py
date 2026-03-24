"""
=============================================================================
 Report Generator Module
=============================================================================
 Role in the pipeline:
   This is the SIXTH and final stage. It takes all the analysis results
   from previous stages and compiles them into a comprehensive, structured
   report that the frontend can render.

 The report includes:
   - Executive summary with key metrics
   - Detailed skill breakdown with status badges
   - Actionable recommendations for missing skills
   - ML model insights and accuracy metrics
   - GitHub profile insights
   - A suggested learning path
=============================================================================
"""

from datetime import datetime, timezone
from typing import Dict, List

from loguru import logger


class ReportGenerator:
    """Compiles analysis results into a structured, presentation-ready report."""

    def __init__(self) -> None:
        """Initialize the report generator."""
        logger.info("[ReportGenerator] Initialized.")

    # -----------------------------------------------------------------
    #  Score Label Helper
    # -----------------------------------------------------------------
    @staticmethod
    def _get_score_label(score: float) -> str:
        """
        Convert a numeric match score to a human-readable label.

        Thresholds:
          75-100 → Excellent (strong candidate)
          50-74  → Good (some gaps but viable)
          25-49  → Fair (significant gaps)
          0-24   → Poor (major skills missing)

        Args:
            score: The match score (0-100).

        Returns:
            A label string.
        """
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Strong"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        elif score >= 20:
            return "Weak"
        else:
            return "Poor"

    # -----------------------------------------------------------------
    #  Confidence Rating Helper
    # -----------------------------------------------------------------
    @staticmethod
    def _get_confidence_rating(confidence: float) -> str:
        """
        Convert a confidence score to a qualitative rating with 6 granular bins.

        Args:
            confidence: The confidence percentage (0-100).

        Returns:
            A rating string.
        """
        if confidence >= 90:
            return "Very High -- ML model is highly confident in all skill assessments"
        elif confidence >= 75:
            return "High -- strong confidence, most skills well-evidenced"
        elif confidence >= 60:
            return "Medium-High -- good confidence, a few skills need verification"
        elif confidence >= 45:
            return "Medium -- moderate confidence, several skills lack strong evidence"
        elif confidence >= 30:
            return "Low -- limited evidence for many skills, verification recommended"
        else:
            return "Very Low -- insufficient data for reliable assessment"

    # -----------------------------------------------------------------
    #  Skill Prerequisites (learning dependency order)
    # -----------------------------------------------------------------
    SKILL_PREREQUISITES = {
        "Kubernetes": ["Docker", "Linux"],
        "Django": ["Python"],
        "Flask": ["Python"],
        "FastAPI": ["Python"],
        "React": ["JavaScript", "HTML", "CSS"],
        "Next.js": ["React", "JavaScript"],
        "Vue": ["JavaScript", "HTML", "CSS"],
        "Angular": ["TypeScript", "JavaScript"],
        "Spring Boot": ["Java", "Spring"],
        "Spring": ["Java"],
        "TensorFlow": ["Python", "Machine Learning"],
        "PyTorch": ["Python", "Machine Learning"],
        "Keras": ["Python", "Deep Learning"],
        "Scikit-learn": ["Python", "Statistics"],
        "Pandas": ["Python"],
        "NumPy": ["Python"],
        "Terraform": ["AWS", "Linux"],
        "Helm": ["Kubernetes", "Docker"],
        "ArgoCD": ["Kubernetes", "CI/CD"],
        "Airflow": ["Python", "Docker"],
        "Spark": ["Python", "SQL"],
        "LangChain": ["Python", "LLM"],
        "RAG": ["LLM", "Python"],
        "GraphQL": ["REST API", "JavaScript"],
        "CI/CD": ["Git", "Docker"],
        "Microservices": ["REST API", "Docker"],
        "Deep Learning": ["Machine Learning", "Python"],
        "Machine Learning": ["Python", "Statistics"],
    }

    # -----------------------------------------------------------------
    #  Skill Difficulty Ratings
    # -----------------------------------------------------------------
    SKILL_DIFFICULTY = {
        "Git": 1, "HTML": 1, "CSS": 1, "SQL": 1, "Markdown": 1,
        "Python": 2, "JavaScript": 2, "Docker": 2, "REST API": 2, "React": 2,
        "TypeScript": 2, "Java": 2, "Node.js": 2, "Express": 2, "Vue": 2,
        "Angular": 2, "Flask": 2, "Django": 2, "FastAPI": 2, "C#": 2,
        "Go": 2, "Ruby": 2, "PHP": 2, "Kotlin": 2, "Swift": 2,
        "Kubernetes": 3, "Machine Learning": 3, "System Design": 3, "AWS": 3,
        "Azure": 3, "GCP": 3, "Terraform": 3, "CI/CD": 3, "GraphQL": 3,
        "Spring Boot": 3, "Microservices": 3, "Spark": 3, "Airflow": 3,
        "Deep Learning": 4, "Distributed Systems": 4, "LLM": 3,
        "TensorFlow": 3, "PyTorch": 3, "RAG": 3, "LangChain": 3,
    }

    DIFFICULTY_LABELS = {1: "Beginner", 2: "Intermediate", 3: "Advanced", 4: "Expert"}
    DIFFICULTY_WEEKS = {1: "1-2 weeks", 2: "2-4 weeks", 3: "4-12 weeks", 4: "3-6 months"}

    # -----------------------------------------------------------------
    #  Resource Hint Helper (with difficulty)
    # -----------------------------------------------------------------
    @classmethod
    def _get_resource_hint(cls, skill: str) -> str:
        """
        Suggest what type of learning resource would help for a given skill.

        Args:
            skill: The skill name.

        Returns:
            A resource suggestion string.
        """
        programming_langs = {"Python", "JavaScript", "Java", "C++", "TypeScript", "Go",
                             "Rust", "R", "Kotlin", "Swift", "Scala", "C#", "PHP", "Ruby", "SQL"}
        frameworks = {"React", "Angular", "Vue", "Django", "Flask", "FastAPI", "Spring",
                      "Node.js", "Express", "TensorFlow", "PyTorch", "Keras", "Scikit-learn",
                      "Pandas", "NumPy", "Next.js"}
        tools = {"Docker", "Kubernetes", "Jenkins", "Terraform", "Ansible", "Git",
                 "Webpack", "Maven", "MLflow"}
        cloud = {"AWS", "Azure", "GCP", "Heroku", "Firebase"}
        concepts = {"Machine Learning", "Deep Learning", "NLP", "Computer Vision",
                    "Data Science", "Statistics", "Microservices", "CI/CD", "REST API",
                    "GraphQL", "DevOps", "Agile"}

        if skill in programming_langs:
            return f"Online course + practice problems (e.g., LeetCode in {skill})"
        elif skill in frameworks:
            return f"Official {skill} documentation + build a small project"
        elif skill in tools:
            return f"Hands-on tutorial + set up {skill} in a personal project"
        elif skill in cloud:
            return f"{skill} free tier + certification study path"
        elif skill in concepts:
            return f"Structured course on {skill} + implement a portfolio project"
        else:
            return f"Self-study + practice project using {skill}"

    @classmethod
    def _get_difficulty_info(cls, skill: str) -> Dict:
        """Return difficulty rating info for a skill."""
        level = cls.SKILL_DIFFICULTY.get(skill, 2)
        return {
            "level": level,
            "label": cls.DIFFICULTY_LABELS.get(level, "Intermediate"),
            "estimated_time": cls.DIFFICULTY_WEEKS.get(level, "2-4 weeks"),
        }

    @classmethod
    def _get_missing_prerequisites(cls, skill: str, all_missing: set) -> List[str]:
        """Return prerequisites for a skill that the candidate is also missing."""
        prereqs = cls.SKILL_PREREQUISITES.get(skill, [])
        return [p for p in prereqs if p in all_missing]

    @classmethod
    def _topological_sort_skills(cls, skills: List[str]) -> List[str]:
        """Sort skills so prerequisites come before dependents."""
        skill_set = set(skills)
        sorted_list = []
        visited = set()

        def visit(skill):
            if skill in visited:
                return
            visited.add(skill)
            for prereq in cls.SKILL_PREREQUISITES.get(skill, []):
                if prereq in skill_set:
                    visit(prereq)
            sorted_list.append(skill)

        for s in skills:
            visit(s)
        return sorted_list

    # -----------------------------------------------------------------
    #  Generate Full Report
    # -----------------------------------------------------------------
    def generate_report(
        self,
        analysis_result: Dict,
        target_role: str,
        github_username: str,
        resume_skills: List[str],
        github_skills: List[str],
        model_summary: Dict,
        github_insights_data: Dict,
    ) -> Dict:
        """
        Compile all analysis results into a comprehensive report.

        This is the main output of the entire pipeline — everything the
        frontend needs to display a full analysis dashboard.

        Args:
            analysis_result:     Output from SkillGapAnalyzer.analyze()
            target_role:         The target job role name
            github_username:     The analyzed GitHub username
            resume_skills:       List of skills from the resume
            github_skills:       List of skills from GitHub
            model_summary:       Output from SkillGapMLModel.get_model_summary()
            github_insights_data: Raw GitHub analysis data (languages, repos, etc.)

        Returns:
            A fully structured report dict ready for JSON serialization.
        """
        logger.info(f"[ReportGenerator] Generating report for: {target_role}")

        match_score = analysis_result["match_score"]
        confidence = analysis_result["confidence"]

        # --- Executive Summary ---
        executive_summary = {
            "match_score": match_score,
            "match_label": self._get_score_label(match_score),
            "total_resume_skills": len(resume_skills),
            "total_github_skills": len(github_skills),
            "missing_critical_skills": len(analysis_result["missing_required"]),
            "confidence_rating": self._get_confidence_rating(confidence),
            "confidence_score": confidence,
        }

        # --- Recommendations ---
        # Generate actionable recommendations with prerequisite ordering
        recommendations = []
        all_missing = set(analysis_result["missing_required"]) | set(analysis_result["missing_nice_to_have"])

        # Sort required skills by prerequisite order (learn foundations first)
        sorted_required = self._topological_sort_skills(analysis_result["missing_required"])

        for skill in sorted_required:
            difficulty = self._get_difficulty_info(skill)
            missing_prereqs = self._get_missing_prerequisites(skill, all_missing)
            recommendations.append({
                "skill": skill,
                "priority": "Urgent",
                "action": f"Learn {skill} -- required for {target_role}",
                "resource_hint": self._get_resource_hint(skill),
                "difficulty": difficulty["label"],
                "estimated_time": difficulty["estimated_time"],
                "learn_first": missing_prereqs if missing_prereqs else None,
            })

        # Also add recommendations for missing nice-to-have skills (lower priority)
        sorted_nice = self._topological_sort_skills(analysis_result["missing_nice_to_have"])
        for skill in sorted_nice:
            difficulty = self._get_difficulty_info(skill)
            missing_prereqs = self._get_missing_prerequisites(skill, all_missing)
            recommendations.append({
                "skill": skill,
                "priority": "Recommended",
                "action": f"Consider learning {skill} -- nice to have for {target_role}",
                "resource_hint": self._get_resource_hint(skill),
                "difficulty": difficulty["label"],
                "estimated_time": difficulty["estimated_time"],
                "learn_first": missing_prereqs if missing_prereqs else None,
            })

        # --- ML Insights ---
        ml_insights = {
            "models_used": model_summary.get("models_used", []),
            "lr_accuracy": model_summary.get("lr_accuracy", 0),
            "dt_accuracy": model_summary.get("dt_accuracy", 0),
            "feature_importance": model_summary.get("feature_importance", {}),
            "model_explanation": model_summary.get("training_explanation", ""),
            "lr_explanation": model_summary.get("lr_explanation", ""),
            "dt_explanation": model_summary.get("dt_explanation", ""),
            "ensemble_explanation": model_summary.get("ensemble_explanation", ""),
        }

        # --- GitHub Insights ---
        # Sort languages by byte count to find the top 5
        raw_languages = github_insights_data.get("raw_languages", {})
        sorted_languages = sorted(raw_languages.items(), key=lambda x: x[1], reverse=True)
        top_languages = [{"language": lang, "bytes": bytes_count} for lang, bytes_count in sorted_languages[:5]]

        github_insights = {
            "repos_analyzed": github_insights_data.get("repos_analyzed", 0),
            "top_languages": top_languages,
            "hidden_strengths": analysis_result.get("hidden_strengths", []),
            "total_topics": len(github_insights_data.get("raw_topics", [])),
            "error": github_insights_data.get("error", ""),
        }

        # --- Compile the Full Report ---
        report = {
            "title": "Skill Gap Analysis Report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_role": target_role,
            "github_username": github_username,
            "executive_summary": executive_summary,
            "skill_breakdown": {
                "match_score": analysis_result["match_score"],
                "gap_score": analysis_result["gap_score"],
                "confidence": analysis_result["confidence"],
                "required_analysis": analysis_result["required_analysis"],
                "nice_to_have_analysis": analysis_result["nice_to_have_analysis"],
                "missing_required": analysis_result["missing_required"],
                "missing_nice_to_have": analysis_result["missing_nice_to_have"],
                "strengths": analysis_result["strengths"],
                "claims_not_proven": analysis_result["claims_not_proven"],
                "hidden_strengths": analysis_result["hidden_strengths"],
            },
            "recommendations": recommendations,
            "learning_path": self.generate_learning_path(
                analysis_result["missing_required"],
                analysis_result["missing_nice_to_have"],
            ),
            "ml_insights": ml_insights,
            "github_insights": github_insights,
            "resume_skills": resume_skills,
            "github_skills": github_skills,
        }

        logger.info(f"[ReportGenerator] Report generated — Match: {match_score}% ({executive_summary['match_label']})")
        return report

    # -----------------------------------------------------------------
    #  Generate Learning Path
    # -----------------------------------------------------------------
    def generate_learning_path(
        self, missing_required: List[str], missing_nice_to_have: List[str] = None
    ) -> List[Dict]:
        """
        Create a prioritized learning path for the candidate's missing skills.

        Args:
            missing_required:     Required skills the candidate is missing.
            missing_nice_to_have: Nice-to-have skills the candidate is missing.

        Returns:
            A list of dicts with skill, priority, and suggested_path.
        """
        learning_path = []
        all_missing = set(missing_required) | set(missing_nice_to_have or [])

        sorted_required = self._topological_sort_skills(missing_required)
        for skill in sorted_required:
            difficulty = self._get_difficulty_info(skill)
            learning_path.append({
                "skill": skill,
                "priority": "Critical",
                "suggested_path": self._get_resource_hint(skill),
                "difficulty": difficulty["label"],
                "estimated_time": difficulty["estimated_time"],
            })

        sorted_nice = self._topological_sort_skills(missing_nice_to_have or [])
        for skill in sorted_nice:
            difficulty = self._get_difficulty_info(skill)
            learning_path.append({
                "skill": skill,
                "priority": "Recommended",
                "suggested_path": self._get_resource_hint(skill),
                "difficulty": difficulty["label"],
                "estimated_time": difficulty["estimated_time"],
            })

        logger.info(f"[ReportGenerator] Learning path with {len(learning_path)} items.")
        return learning_path
