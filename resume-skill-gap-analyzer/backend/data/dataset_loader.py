"""
Dataset Loader — Synthetic Training Data Generator
====================================================
Generates training data for the ML models.

The training data simulates realistic patterns of how skills
appear in resumes and GitHub profiles:
  - Some candidates have skills in BOTH sources (strongest signal)
  - Some have skills in NEITHER (clear gap)
  - Some claim skills on resume but don't show them on GitHub
  - Some demonstrate skills on GitHub but don't list them on resume

This gives the ML model diverse examples to learn from.
"""

import pandas as pd
import numpy as np
from loguru import logger


class DatasetLoader:
    def __init__(self, cache_dir: str = "./datasets/hf_cache"):
        """Initialize the dataset loader."""
        self.loaded_sources = []

    # -----------------------------------------------------------------
    #  Public: Load Training Data
    # -----------------------------------------------------------------
    def load_training_data(self) -> tuple:
        """
        Generate training data for the ML models.

        Returns:
            X: DataFrame with columns [in_resume, in_github, both_confirmed, is_required]
            y: Series with labels (1=skill present, 0=skill gap)
            source: string describing where data came from
        """
        X, y = self._generate_synthetic_data(n_samples=800)
        return X, y, "Synthetic training data (800 samples)"

    # -----------------------------------------------------------------
    #  Private: Generate Synthetic Training Data
    # -----------------------------------------------------------------
    def _generate_synthetic_data(self, n_samples: int = 800):
        """
        Generate synthetic training data with realistic distributions.

        We create 4 segments that mirror real-world patterns:

        Segment 1 (30%): Skill in BOTH resume AND GitHub
          → Label = 1 (definitely has the skill)
          → This is the strongest evidence a candidate has a skill

        Segment 2 (20%): Skill in NEITHER resume NOR GitHub
          → Label = 0 (skill gap)
          → Candidate clearly doesn't have this skill

        Segment 3 (25%): Skill in RESUME ONLY (not on GitHub)
          → Label = 1 with 70% probability
          → Claimed but not demonstrated — might be real or inflated

        Segment 4 (25%): Skill on GITHUB ONLY (not in resume)
          → Label = 1 with 80% probability
          → Demonstrated in code — strong signal even without resume claim

        Finally, 5% random noise is added to simulate real-world messiness
        (typos, misclassifications, etc.)
        """
        np.random.seed(42)
        rows = []

        # --- Segment 1: Both confirmed (30%) — strongest positive signal ---
        n1 = int(n_samples * 0.30)
        for _ in range(n1):
            rows.append({
                'in_resume': 1, 'in_github': 1,
                'both_confirmed': 1,
                'is_required': np.random.choice([0, 1], p=[0.3, 0.7]),
                'label': 1
            })

        # --- Segment 2: Both absent (20%) — clear gap ---
        n2 = int(n_samples * 0.20)
        for _ in range(n2):
            rows.append({
                'in_resume': 0, 'in_github': 0,
                'both_confirmed': 0,
                'is_required': np.random.choice([0, 1], p=[0.4, 0.6]),
                'label': 0
            })

        # --- Segment 3: Resume only (25%) — claimed but not demonstrated ---
        n3 = int(n_samples * 0.25)
        for _ in range(n3):
            label = np.random.choice([0, 1], p=[0.30, 0.70])
            rows.append({
                'in_resume': 1, 'in_github': 0,
                'both_confirmed': 0,
                'is_required': np.random.choice([0, 1], p=[0.35, 0.65]),
                'label': label
            })

        # --- Segment 4: GitHub only (25%) — demonstrated but not listed ---
        n4 = n_samples - n1 - n2 - n3
        for _ in range(n4):
            label = np.random.choice([0, 1], p=[0.20, 0.80])
            rows.append({
                'in_resume': 0, 'in_github': 1,
                'both_confirmed': 0,
                'is_required': np.random.choice([0, 1], p=[0.3, 0.7]),
                'label': label
            })

        df = pd.DataFrame(rows)

        # Add 5% noise to simulate real-world data imperfections
        noise_idx = df.sample(frac=0.05, random_state=42).index
        df.loc[noise_idx, 'label'] = 1 - df.loc[noise_idx, 'label']

        X = df[['in_resume', 'in_github', 'both_confirmed', 'is_required']]
        y = df['label']

        logger.info(f"Generated {len(df)} synthetic training samples | "
                    f"Positive rate: {y.mean():.1%}")
        return X, y
