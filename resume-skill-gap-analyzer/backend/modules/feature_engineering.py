"""
=============================================================================
 Feature Engineering Module
=============================================================================
 Role in the pipeline:
   This is the THIRD stage. It transforms the raw skill lists (from the
   resume parser and GitHub analyzer) into structured feature vectors that
   can be fed into the ML models.

 Why feature engineering matters:
   ML models can't understand raw text — they need numerical features.
   We create binary (0/1) features indicating whether each skill was
   found in the resume, on GitHub, or both.

 Features per skill:
   - in_resume       (0/1) — found in the candidate's resume
   - in_github       (0/1) — found on the candidate's GitHub
   - both_confirmed  (0/1) — found in BOTH sources (strongest signal)
   - is_required     (0/1) — required for the target job role
=============================================================================
"""

from typing import List, Tuple

import pandas as pd
from loguru import logger


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
    ) -> pd.DataFrame:
        """
        Create a skill-by-skill matrix used for ML prediction and analysis.

        Each row represents one skill from the target role's requirements.
        Columns indicate:
          - skill_name:       The skill identifier
          - category:         "required" or "nice_to_have"
          - in_resume:        1 if found in resume, 0 otherwise
          - in_github:        1 if found on GitHub, 0 otherwise
          - combined:         1 if found in either source
          - both_confirmed:   1 if found in BOTH sources (strongest evidence)
          - is_required:      1 if skill is required for the role

        This matrix is what gets fed into the ML model for predictions
        and is also used directly for gap analysis.
        """
        rows = []

        # Process required skills first
        for skill in required_skills:
            in_resume = 1 if skill in claimed_skills else 0
            in_github = 1 if skill in demonstrated_skills else 0
            rows.append({
                "skill_name": skill,
                "category": "required",
                "in_resume": in_resume,
                "in_github": in_github,
                "combined": 1 if (in_resume or in_github) else 0,
                "both_confirmed": 1 if (in_resume and in_github) else 0,
                "is_required": 1,
            })

        # Then process nice-to-have skills
        for skill in nice_to_have_skills:
            in_resume = 1 if skill in claimed_skills else 0
            in_github = 1 if skill in demonstrated_skills else 0
            rows.append({
                "skill_name": skill,
                "category": "nice_to_have",
                "in_resume": in_resume,
                "in_github": in_github,
                "combined": 1 if (in_resume or in_github) else 0,
                "both_confirmed": 1 if (in_resume and in_github) else 0,
                "is_required": 0,
            })

        df = pd.DataFrame(rows)
        logger.debug(f"[FeatureEngineer] Created skill matrix with {len(df)} skills "
                     f"(4 features per skill).")
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

        X contains the 4 input features the model uses for prediction.
        y is the target label: combined (1 if skill present in either source).
        """
        # Features: 4-dimensional feature vector per skill
        X = skill_matrix[["in_resume", "in_github",
                          "both_confirmed", "is_required"]].copy()

        # Label: whether the skill is "present" (found in at least one source)
        y = skill_matrix["combined"].copy()

        logger.debug(f"[FeatureEngineer] Encoded {len(X)} samples for model (4 features).")
        return X, y
