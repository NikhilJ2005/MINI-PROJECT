"""
=============================================================================
 Feature Engineering Module
=============================================================================
 Role in the pipeline:
   This is the THIRD stage. It transforms the raw skill lists (from the
   resume parser and GitHub analyzer) into structured feature vectors that
   can be fed into the ML models.

 Features per skill (9 total):
   Binary:
   - in_resume       (0/1) -- found in the candidate's resume
   - in_github       (0/1) -- found on the candidate's GitHub
   - is_required     (0/1) -- required for the target job role
   Continuous:
   - resume_skill_ratio      (0-1) -- fraction of role skills claimed on resume
   - github_skill_ratio      (0-1) -- fraction of role skills on GitHub
   - skill_source_agreement  (0-1) -- how consistent resume/github are across role skills
   - resume_claim_density    (0-1) -- resume skills / (resume + github skills)
   - github_evidence_strength(0-1) -- depth of GitHub presence (repos analyzed)
   - category_match_score    (0-1) -- how many candidate skills share this skill's category
=============================================================================
"""

from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger


# The 9 features used by the ML model
FEATURE_NAMES = [
    "in_resume", "in_github", "is_required",
    "resume_skill_ratio", "github_skill_ratio",
    "skill_source_agreement", "resume_claim_density",
    "github_evidence_strength", "category_match_score",
]


class FeatureEngineer:
    """Transforms raw skill data into structured features for ML models."""

    def __init__(self) -> None:
        """Initialize the feature engineer."""
        logger.info("[FeatureEngineer] Initialized.")

    # -----------------------------------------------------------------
    #  Create Skill Matrix (Main Method)
    # -----------------------------------------------------------------
    def create_skill_matrix(
        self,
        claimed_skills: List[str],
        demonstrated_skills: List[str],
        required_skills: List[str],
        nice_to_have_skills: List[str],
        repos_analyzed: int = 0,
        skills_master: Dict[str, List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create a skill-by-skill matrix used for ML prediction and analysis.

        Each row represents one skill from the target role's requirements.
        Includes 9 features: 3 binary + 6 continuous profile/skill-level signals.
        """
        claimed_set = set(claimed_skills)
        demonstrated_set = set(demonstrated_skills)
        all_role_skills = required_skills + nice_to_have_skills

        # -- Profile-level continuous features (same for every row) --
        n_role = max(len(all_role_skills), 1)
        resume_in_role = sum(1 for s in all_role_skills if s in claimed_set)
        github_in_role = sum(1 for s in all_role_skills if s in demonstrated_set)

        resume_skill_ratio = resume_in_role / n_role
        github_skill_ratio = github_in_role / n_role

        # Agreement: fraction of role skills where resume & github status match
        agreements = sum(
            1 for s in all_role_skills
            if (s in claimed_set) == (s in demonstrated_set)
        )
        skill_source_agreement = agreements / n_role

        # Resume claim density: proportion of candidate skills from resume
        total_candidate = max(len(claimed_set) + len(demonstrated_set), 1)
        resume_claim_density = len(claimed_set) / total_candidate

        # GitHub evidence strength: normalized repos count
        github_evidence_strength = min(repos_analyzed / 20.0, 1.0)

        # Build skill-to-category map for category_match_score
        skill_to_cat = {}
        if skills_master:
            for cat, cat_skills in skills_master.items():
                for s in cat_skills:
                    skill_to_cat[s] = cat

        # All candidate skills (from either source)
        all_candidate_skills = claimed_set | demonstrated_set

        rows = []
        for skill in all_role_skills:
            in_resume = 1 if skill in claimed_set else 0
            in_github = 1 if skill in demonstrated_set else 0
            is_req = 1 if skill in required_skills else 0

            # Per-skill: category match score
            cat = skill_to_cat.get(skill)
            if cat and all_candidate_skills:
                same_cat_count = sum(
                    1 for s in all_candidate_skills
                    if skill_to_cat.get(s) == cat and s != skill
                )
                cat_score = min(same_cat_count / max(len(all_candidate_skills), 1), 1.0)
            else:
                cat_score = 0.0

            rows.append({
                "skill_name": skill,
                "category": "required" if is_req else "nice_to_have",
                "in_resume": in_resume,
                "in_github": in_github,
                "combined": 1 if (in_resume or in_github) else 0,
                "both_confirmed": 1 if (in_resume and in_github) else 0,
                "is_required": is_req,
                "resume_skill_ratio": round(resume_skill_ratio, 4),
                "github_skill_ratio": round(github_skill_ratio, 4),
                "skill_source_agreement": round(skill_source_agreement, 4),
                "resume_claim_density": round(resume_claim_density, 4),
                "github_evidence_strength": round(github_evidence_strength, 4),
                "category_match_score": round(cat_score, 4),
            })

        df = pd.DataFrame(rows)
        logger.debug(f"[FeatureEngineer] Created skill matrix with {len(df)} skills "
                     f"(9 features per skill).")
        return df

    # -----------------------------------------------------------------
    #  Encode for ML Model
    # -----------------------------------------------------------------
    def encode_for_model(
        self,
        skill_matrix: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract the feature matrix (X) and label vector (y) from the skill matrix.

        X contains the 9 input features the model uses for prediction.
        y is the target label: combined (1 if skill present in either source).
        """
        X = skill_matrix[FEATURE_NAMES].copy()

        # Label: whether the skill is "present" (found in at least one source)
        y = skill_matrix["combined"].copy()

        logger.debug(f"[FeatureEngineer] Encoded {len(X)} samples for model (9 features).")
        return X, y
