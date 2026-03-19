"""
=============================================================================
 Dataset Loader — Synthetic Training Data
=============================================================================
 Generates synthetic training data in the 4-feature format the ML models
 expect. Pre-trained models are shipped with the app to avoid expensive
 dataset downloads at startup.

 Feature format (per skill):
   - in_resume       (0/1) — skill found in candidate's resume
   - in_github       (0/1) — skill found on candidate's GitHub
   - both_confirmed  (0/1) — skill found in BOTH sources
   - is_required     (0/1) — skill is required for the target job role
=============================================================================
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class DatasetLoader:
    """Generates synthetic training data for the skill gap ML models."""

    def __init__(self):
        self.loaded_sources: List[str] = []

    # -----------------------------------------------------------------
    #  Public: Load Training Data
    # -----------------------------------------------------------------
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series, str]:
        """
        Load training data from synthetic generation.

        Returns:
            X: DataFrame with columns [in_resume, in_github, both_confirmed, is_required]
            y: Series with labels (1=skill present, 0=skill gap)
            source: string describing where data came from
        """
        combined = self._generate_synthetic_data(n_samples=1200)
        combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

        X = combined[['in_resume', 'in_github', 'both_confirmed', 'is_required']]
        y = combined['label']

        source_str = f"Synthetic ({len(combined)} samples)"
        self.loaded_sources = [source_str]

        logger.info(f"[DatasetLoader] Training data ready: {source_str}")
        logger.info(f"[DatasetLoader] Class distribution: "
                    f"positive={y.sum()}/{len(y)} ({y.mean():.1%})")

        return X, y, source_str

    # -----------------------------------------------------------------
    #  Synthetic Data Generation
    # -----------------------------------------------------------------
    def _generate_synthetic_data(self, n_samples: int = 1200) -> pd.DataFrame:
        """
        Generate synthetic training data with realistic distributions.

        Creates 4 segments that mirror real-world patterns:

        Segment 1 (30%): Skill in BOTH resume AND GitHub
          → Label = 1 (definitely has the skill)

        Segment 2 (20%): Skill in NEITHER resume NOR GitHub
          → Label = 0 (skill gap)

        Segment 3 (25%): Skill in RESUME ONLY (not on GitHub)
          → Label = 1 with 70% probability

        Segment 4 (25%): Skill on GITHUB ONLY (not in resume)
          → Label = 1 with 80% probability

        5% noise added to simulate real-world data imperfections.
        """
        np.random.seed(42)
        rows = []

        # Segment 1: Both confirmed (30%)
        n1 = int(n_samples * 0.30)
        for _ in range(n1):
            rows.append({
                'in_resume': 1, 'in_github': 1,
                'both_confirmed': 1,
                'is_required': np.random.choice([0, 1], p=[0.3, 0.7]),
                'label': 1
            })

        # Segment 2: Both absent (20%)
        n2 = int(n_samples * 0.20)
        for _ in range(n2):
            rows.append({
                'in_resume': 0, 'in_github': 0,
                'both_confirmed': 0,
                'is_required': np.random.choice([0, 1], p=[0.4, 0.6]),
                'label': 0
            })

        # Segment 3: Resume only (25%)
        n3 = int(n_samples * 0.25)
        for _ in range(n3):
            label = np.random.choice([0, 1], p=[0.30, 0.70])
            rows.append({
                'in_resume': 1, 'in_github': 0,
                'both_confirmed': 0,
                'is_required': np.random.choice([0, 1], p=[0.35, 0.65]),
                'label': label
            })

        # Segment 4: GitHub only (25%)
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

        # Add 5% noise
        noise_idx = df.sample(frac=0.05, random_state=42).index
        df.loc[noise_idx, 'label'] = 1 - df.loc[noise_idx, 'label']

        logger.info(f"[DatasetLoader] Generated {len(df)} synthetic samples | "
                    f"Positive rate: {df['label'].mean():.1%}")
        return df

    # -----------------------------------------------------------------
    #  Status / Info
    # -----------------------------------------------------------------
    def get_status(self) -> Dict:
        """Return information about loaded data sources."""
        return {
            "loaded_sources": self.loaded_sources,
        }
