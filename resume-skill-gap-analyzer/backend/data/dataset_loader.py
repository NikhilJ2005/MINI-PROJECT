"""
=============================================================================
 Dataset Loader — HuggingFace + Synthetic Training Data
=============================================================================
 Loads real-world resume and job-skill data from HuggingFace Hub, transforms
 it into the 4-feature format the ML models expect, and blends with synthetic
 data for robust training.

 HuggingFace datasets used:
   1. datasetmaster/resumes     — structured resumes with skills sections
   2. batuhanmtl/job-skill-set  — job categories with extracted skill sets
   3. fazni/roles-based-on-skills — roles mapped to required skills

 Feature format (per skill):
   - in_resume       (0/1) — skill found in candidate's resume
   - in_github       (0/1) — skill found on candidate's GitHub
   - both_confirmed  (0/1) — skill found in BOTH sources
   - is_required     (0/1) — skill is required for the target job role

 Fallback: If HuggingFace datasets are unavailable (network error, etc.),
 falls back to enhanced synthetic data generation.
=============================================================================
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Master skills loaded at init time for matching
_SKILLS_MASTER_PATH = Path(__file__).parent / "skills_master.json"
_JOB_ROLES_PATH = Path(__file__).parent / "job_roles.json"


def _load_master_skills() -> Set[str]:
    """Load the master skills list and return as a lowercase→original mapping."""
    try:
        with open(_SKILLS_MASTER_PATH) as f:
            data = json.load(f)
        skills = set()
        for category_skills in data.values():
            skills.update(category_skills)
        return skills
    except Exception:
        return set()


def _load_job_roles() -> Dict:
    """Load job roles with their required and nice-to-have skills."""
    try:
        with open(_JOB_ROLES_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


class DatasetLoader:
    """Loads and transforms training data from HuggingFace + synthetic sources."""

    def __init__(self, cache_dir: str = "./datasets/hf_cache"):
        """Initialize with cache directory for HuggingFace datasets."""
        self.cache_dir = cache_dir
        self.loaded_sources: List[str] = []
        self.master_skills = _load_master_skills()
        self.job_roles = _load_job_roles()
        # Build lowercase lookup for fuzzy matching
        self._skill_lower_map = {s.lower(): s for s in self.master_skills}

    # -----------------------------------------------------------------
    #  Public: Load Training Data
    # -----------------------------------------------------------------
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series, str]:
        """
        Load training data from all available sources.

        Strategy:
          1. Try loading real-world data from HuggingFace datasets
          2. Generate enhanced synthetic data
          3. Blend both for maximum diversity and robustness

        Returns:
            X: DataFrame with columns [in_resume, in_github, both_confirmed, is_required]
            y: Series with labels (1=skill present, 0=skill gap)
            source: string describing where data came from
        """
        all_dfs = []
        sources = []

        # --- 1. HuggingFace real-world datasets ---
        hf_df = self._load_huggingface_data()
        if hf_df is not None and len(hf_df) > 0:
            all_dfs.append(hf_df)
            sources.append(f"HuggingFace ({len(hf_df)} real-world samples)")
            logger.info(f"[DatasetLoader] Loaded {len(hf_df)} samples from HuggingFace datasets")

        # --- 2. Enhanced synthetic data (always included for coverage) ---
        synth_df = self._generate_synthetic_data(n_samples=800)
        all_dfs.append(synth_df)
        sources.append(f"Synthetic ({len(synth_df)} samples)")

        # --- 3. Blend all sources ---
        combined = pd.concat(all_dfs, ignore_index=True)

        # Shuffle
        combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

        X = combined[['in_resume', 'in_github', 'both_confirmed', 'is_required']]
        y = combined['label']

        source_str = " + ".join(sources) + f" = {len(combined)} total"
        self.loaded_sources = sources

        logger.info(f"[DatasetLoader] Training data ready: {source_str}")
        logger.info(f"[DatasetLoader] Class distribution: "
                    f"positive={y.sum()}/{len(y)} ({y.mean():.1%})")

        return X, y, source_str

    # -----------------------------------------------------------------
    #  HuggingFace Data Loading
    # -----------------------------------------------------------------
    def _load_huggingface_data(self) -> pd.DataFrame:
        """
        Load and transform real-world data from HuggingFace datasets.

        Tries multiple datasets and combines results. Each dataset is
        transformed into the 4-feature format independently.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.warning("[DatasetLoader] 'datasets' package not installed. "
                           "Run: pip install datasets")
            return None

        all_rows = []

        # --- Dataset 1: datasetmaster/resumes ---
        # Structured resumes with skills sections
        try:
            logger.info("[DatasetLoader] Loading datasetmaster/resumes from HuggingFace...")
            ds = load_dataset(
                "datasetmaster/resumes",
                split="train",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            rows = self._transform_resumes_dataset(ds)
            all_rows.extend(rows)
            logger.info(f"[DatasetLoader] datasetmaster/resumes → {len(rows)} training samples")
        except Exception as e:
            logger.warning(f"[DatasetLoader] Could not load datasetmaster/resumes: {e}")

        # --- Dataset 2: batuhanmtl/job-skill-set ---
        # Job categories with extracted skill sets from LinkedIn postings
        try:
            logger.info("[DatasetLoader] Loading batuhanmtl/job-skill-set from HuggingFace...")
            ds = load_dataset(
                "batuhanmtl/job-skill-set",
                split="train",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            rows = self._transform_job_skill_dataset(ds)
            all_rows.extend(rows)
            logger.info(f"[DatasetLoader] batuhanmtl/job-skill-set → {len(rows)} training samples")
        except Exception as e:
            logger.warning(f"[DatasetLoader] Could not load batuhanmtl/job-skill-set: {e}")

        # --- Dataset 3: fazni/roles-based-on-skills ---
        # Roles mapped to skills
        try:
            logger.info("[DatasetLoader] Loading fazni/roles-based-on-skills from HuggingFace...")
            ds = load_dataset(
                "fazni/roles-based-on-skills",
                split="train",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            rows = self._transform_roles_skills_dataset(ds)
            all_rows.extend(rows)
            logger.info(f"[DatasetLoader] fazni/roles-based-on-skills → {len(rows)} training samples")
        except Exception as e:
            logger.warning(f"[DatasetLoader] Could not load fazni/roles-based-on-skills: {e}")

        # --- Dataset 4: jacob-hugging-face/job-descriptions ---
        # Job descriptions with responsibilities and requirements
        try:
            logger.info("[DatasetLoader] Loading jacob-hugging-face/job-descriptions...")
            ds = load_dataset(
                "jacob-hugging-face/job-descriptions",
                split="train",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            rows = self._transform_job_descriptions_dataset(ds)
            all_rows.extend(rows)
            logger.info(f"[DatasetLoader] jacob-hugging-face/job-descriptions → {len(rows)} samples")
        except Exception as e:
            logger.warning(f"[DatasetLoader] Could not load jacob-hugging-face/job-descriptions: {e}")

        # --- Dataset 5: MikePfunk28/resume-training-dataset ---
        # 22k+ curated resume samples with skills and role data
        try:
            logger.info("[DatasetLoader] Loading MikePfunk28/resume-training-dataset...")
            ds = load_dataset(
                "MikePfunk28/resume-training-dataset",
                split="train",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            rows = self._transform_resume_training_dataset(ds)
            all_rows.extend(rows)
            logger.info(f"[DatasetLoader] MikePfunk28/resume-training-dataset → {len(rows)} samples")
        except Exception as e:
            logger.warning(f"[DatasetLoader] Could not load MikePfunk28/resume-training-dataset: {e}")

        # --- Dataset 6: NxtGenIntern/job_titles_and_descriptions ---
        # IT job roles with required skills and descriptions
        try:
            logger.info("[DatasetLoader] Loading NxtGenIntern/job_titles_and_descriptions...")
            ds = load_dataset(
                "NxtGenIntern/job_titles_and_descriptions",
                split="train",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            rows = self._transform_job_titles_dataset(ds)
            all_rows.extend(rows)
            logger.info(f"[DatasetLoader] NxtGenIntern/job_titles_and_descriptions → {len(rows)} samples")
        except Exception as e:
            logger.warning(f"[DatasetLoader] Could not load NxtGenIntern/job_titles_and_descriptions: {e}")

        # --- Dataset 7: InferencePrince55/machine-learning-dataset ---
        # ML-specific resumes with technical skill sections
        try:
            logger.info("[DatasetLoader] Loading InferencePrince55/machine-learning-dataset...")
            ds = load_dataset(
                "InferencePrince55/machine-learning-dataset",
                split="train",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            rows = self._transform_generic_skills_dataset(ds, seed=48)
            all_rows.extend(rows)
            logger.info(f"[DatasetLoader] InferencePrince55/machine-learning-dataset → {len(rows)} samples")
        except Exception as e:
            logger.warning(f"[DatasetLoader] Could not load InferencePrince55/machine-learning-dataset: {e}")

        # --- Dataset 8: lukebarousse/data_jobs ---
        # Data science/analytics job postings with real skills
        try:
            logger.info("[DatasetLoader] Loading lukebarousse/data_jobs...")
            ds = load_dataset(
                "lukebarousse/data_jobs",
                split="train",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            rows = self._transform_generic_skills_dataset(ds, seed=49)
            all_rows.extend(rows)
            logger.info(f"[DatasetLoader] lukebarousse/data_jobs → {len(rows)} samples")
        except Exception as e:
            logger.warning(f"[DatasetLoader] Could not load lukebarousse/data_jobs: {e}")

        if not all_rows:
            logger.warning("[DatasetLoader] No HuggingFace data loaded — using synthetic only")
            return None

        df = pd.DataFrame(all_rows)
        logger.info(f"[DatasetLoader] Total HuggingFace samples: {len(df)}")
        return df

    # -----------------------------------------------------------------
    #  Dataset Transformers
    # -----------------------------------------------------------------
    def _match_skills(self, text: str) -> Set[str]:
        """
        Extract known skills from text by matching against the master skills list.
        Uses word-boundary regex for precise matching.
        """
        if not text:
            return set()
        text_lower = text.lower()
        found = set()
        for skill in self.master_skills:
            if skill in ("C++", "C#"):
                pattern = re.escape(skill.lower())
            else:
                pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(pattern, text_lower):
                found.add(skill)
        return found

    def _get_required_skills_for_category(self, category: str) -> Set[str]:
        """
        Map a job category/title to known required skills from our job_roles.json.
        Uses fuzzy matching against role names.
        """
        cat_lower = category.lower()
        best_match_skills = set()

        for role_name, role_data in self.job_roles.items():
            role_lower = role_name.lower()
            # Check if category matches or overlaps with any known role
            if (role_lower in cat_lower or cat_lower in role_lower
                    or any(word in cat_lower for word in role_lower.split()
                           if len(word) > 3)):
                required = set(role_data.get("required_skills", []))
                nice = set(role_data.get("nice_to_have", []))
                best_match_skills.update(required | nice)

        return best_match_skills

    def _transform_resumes_dataset(self, ds) -> List[dict]:
        """
        Transform datasetmaster/resumes into training rows.

        Each resume has structured skills. We simulate the resume/GitHub
        split by treating listed skills as resume_skills and a subset
        (programming languages + frameworks) as GitHub-demonstrable skills.
        """
        rows = []
        np.random.seed(42)

        # Process up to 500 resumes for manageable training size
        max_samples = min(len(ds), 500)

        for i in range(max_samples):
            try:
                record = ds[i]

                # Extract skills from various fields
                resume_skills = set()

                # Try common field names for skills
                for field in ['skills', 'Skills', 'technical_skills',
                              'programming_languages', 'frameworks']:
                    val = record.get(field)
                    if val:
                        if isinstance(val, list):
                            for item in val:
                                if isinstance(item, str):
                                    resume_skills.update(self._match_skills(item))
                                elif isinstance(item, dict):
                                    # Nested structure like {category: [skills]}
                                    for v in item.values():
                                        if isinstance(v, list):
                                            for s in v:
                                                resume_skills.update(
                                                    self._match_skills(str(s)))
                                        elif isinstance(v, str):
                                            resume_skills.update(self._match_skills(v))
                        elif isinstance(val, str):
                            resume_skills.update(self._match_skills(val))

                # Also try extracting from full text/summary
                for field in ['summary', 'experience', 'text', 'resume_text']:
                    val = record.get(field)
                    if val and isinstance(val, str):
                        resume_skills.update(self._match_skills(val))

                if not resume_skills:
                    continue

                # Simulate GitHub skills: programming languages + frameworks
                # have higher chance of being on GitHub
                github_skills = set()
                language_skills = {"Python", "JavaScript", "Java", "C++",
                                   "TypeScript", "Go", "Rust", "Ruby", "PHP",
                                   "Kotlin", "Swift", "Scala", "C#", "R"}
                framework_skills = {"React", "Angular", "Vue", "Django",
                                    "Flask", "FastAPI", "TensorFlow", "PyTorch",
                                    "Node.js", "Express", "Spring", "Next.js"}

                for skill in resume_skills:
                    # Languages/frameworks: 75% chance of appearing on GitHub
                    if skill in language_skills or skill in framework_skills:
                        if np.random.random() < 0.75:
                            github_skills.add(skill)
                    # Other skills: 35% chance of appearing on GitHub
                    elif np.random.random() < 0.35:
                        github_skills.add(skill)

                # Also add some GitHub-only skills (demonstrated but not listed)
                for skill in self.master_skills:
                    if skill not in resume_skills and np.random.random() < 0.05:
                        github_skills.add(skill)

                # Determine which skills are required for a matching role
                required_skills = set()
                for field in ['role', 'title', 'job_title', 'position']:
                    val = record.get(field)
                    if val and isinstance(val, str):
                        required_skills = self._get_required_skills_for_category(val)
                        break

                if not required_skills:
                    # Fallback: pick a random role's requirements
                    if self.job_roles:
                        random_role = list(self.job_roles.values())[
                            np.random.randint(len(self.job_roles))]
                        required_skills = set(random_role.get("required_skills", []))

                # Generate one training row per skill in the union of all sources
                all_skills = resume_skills | github_skills | required_skills
                for skill in all_skills:
                    in_resume = 1 if skill in resume_skills else 0
                    in_github = 1 if skill in github_skills else 0
                    both = 1 if (in_resume and in_github) else 0
                    is_req = 1 if skill in required_skills else 0
                    # Label: candidate has the skill if it appears in any source
                    label = 1 if (in_resume or in_github) else 0
                    rows.append({
                        'in_resume': in_resume,
                        'in_github': in_github,
                        'both_confirmed': both,
                        'is_required': is_req,
                        'label': label,
                    })

            except Exception:
                continue

        return rows

    def _transform_job_skill_dataset(self, ds) -> List[dict]:
        """
        Transform batuhanmtl/job-skill-set into training rows.

        This dataset has job categories with skill sets extracted from real
        LinkedIn postings. We use it to create realistic required/nice-to-have
        skill patterns and simulate candidates with varying skill coverage.
        """
        rows = []
        np.random.seed(43)

        max_samples = min(len(ds), 1000)

        for i in range(max_samples):
            try:
                record = ds[i]

                # Extract skills from the record
                job_skills = set()
                category = ""

                for field in ['job_skill_set', 'skills', 'skill_set',
                              'required_skills']:
                    val = record.get(field)
                    if val:
                        if isinstance(val, list):
                            for s in val:
                                job_skills.update(self._match_skills(str(s)))
                        elif isinstance(val, str):
                            job_skills.update(self._match_skills(val))

                for field in ['category', 'job_category', 'title', 'role']:
                    val = record.get(field)
                    if val and isinstance(val, str):
                        category = val
                        break

                # Also extract from job description text
                for field in ['job_description', 'description', 'text']:
                    val = record.get(field)
                    if val and isinstance(val, str):
                        job_skills.update(self._match_skills(val))

                if not job_skills:
                    continue

                # These are "required" skills for the job
                required_skills = job_skills

                # Simulate 3-5 candidates with varying coverage
                n_candidates = np.random.randint(3, 6)
                for _ in range(n_candidates):
                    # Random coverage: candidate knows 30-80% of required skills
                    coverage = np.random.uniform(0.3, 0.8)
                    candidate_skills = set(
                        np.random.choice(
                            list(required_skills),
                            size=max(1, int(len(required_skills) * coverage)),
                            replace=False
                        )
                    ) if required_skills else set()

                    # Split candidate skills into resume vs GitHub
                    resume_skills = set()
                    github_skills = set()
                    for skill in candidate_skills:
                        # 85% chance skill appears on resume
                        if np.random.random() < 0.85:
                            resume_skills.add(skill)
                        # 60% chance skill appears on GitHub
                        if np.random.random() < 0.60:
                            github_skills.add(skill)

                    # Generate rows for ALL required skills (present or missing)
                    for skill in required_skills:
                        in_resume = 1 if skill in resume_skills else 0
                        in_github = 1 if skill in github_skills else 0
                        both = 1 if (in_resume and in_github) else 0
                        label = 1 if (in_resume or in_github) else 0
                        rows.append({
                            'in_resume': in_resume,
                            'in_github': in_github,
                            'both_confirmed': both,
                            'is_required': 1,
                            'label': label,
                        })

            except Exception:
                continue

        return rows

    def _transform_roles_skills_dataset(self, ds) -> List[dict]:
        """
        Transform fazni/roles-based-on-skills into training rows.

        Maps roles to their required skills, then simulates realistic
        candidate profiles with varying skill completeness.
        """
        rows = []
        np.random.seed(44)

        max_samples = min(len(ds), 500)

        for i in range(max_samples):
            try:
                record = ds[i]

                role_skills = set()
                for field in ['skills', 'required_skills', 'skill_set',
                              'Skills', 'Skill']:
                    val = record.get(field)
                    if val:
                        if isinstance(val, list):
                            for s in val:
                                role_skills.update(self._match_skills(str(s)))
                        elif isinstance(val, str):
                            role_skills.update(self._match_skills(val))

                if not role_skills:
                    continue

                # Simulate 4 candidate profiles per role
                for profile_type in range(4):
                    resume_skills = set()
                    github_skills = set()

                    for skill in role_skills:
                        if profile_type == 0:
                            # Strong candidate: most skills in both
                            if np.random.random() < 0.85:
                                resume_skills.add(skill)
                            if np.random.random() < 0.80:
                                github_skills.add(skill)
                        elif profile_type == 1:
                            # Resume-heavy: claims many, demonstrates few
                            if np.random.random() < 0.80:
                                resume_skills.add(skill)
                            if np.random.random() < 0.30:
                                github_skills.add(skill)
                        elif profile_type == 2:
                            # GitHub-heavy: demonstrates but doesn't list
                            if np.random.random() < 0.30:
                                resume_skills.add(skill)
                            if np.random.random() < 0.75:
                                github_skills.add(skill)
                        else:
                            # Weak candidate: has few skills
                            if np.random.random() < 0.25:
                                resume_skills.add(skill)
                            if np.random.random() < 0.20:
                                github_skills.add(skill)

                    for skill in role_skills:
                        in_resume = 1 if skill in resume_skills else 0
                        in_github = 1 if skill in github_skills else 0
                        both = 1 if (in_resume and in_github) else 0
                        label = 1 if (in_resume or in_github) else 0
                        rows.append({
                            'in_resume': in_resume,
                            'in_github': in_github,
                            'both_confirmed': both,
                            'is_required': 1,
                            'label': label,
                        })

            except Exception:
                continue

        return rows

    def _transform_job_descriptions_dataset(self, ds) -> List[dict]:
        """
        Transform jacob-hugging-face/job-descriptions into training rows.

        Extracts skills from job description text and simulates candidates
        with varying coverage of those skills.
        """
        rows = []
        np.random.seed(45)
        max_samples = min(len(ds), 500)

        for i in range(max_samples):
            try:
                record = ds[i]
                job_skills = set()

                # Extract skills from all text fields
                for field in ['description', 'job_description', 'text',
                              'requirements', 'qualifications', 'skills']:
                    val = record.get(field)
                    if val and isinstance(val, str):
                        job_skills.update(self._match_skills(val))

                if not job_skills:
                    continue

                # Simulate 3 candidates per job posting
                for _ in range(3):
                    coverage = np.random.uniform(0.25, 0.85)
                    n_skills = max(1, int(len(job_skills) * coverage))
                    candidate_has = set(np.random.choice(
                        list(job_skills), size=n_skills, replace=False
                    ))

                    for skill in job_skills:
                        has_skill = skill in candidate_has
                        in_resume = 1 if has_skill and np.random.random() < 0.80 else 0
                        in_github = 1 if has_skill and np.random.random() < 0.55 else 0
                        both = 1 if (in_resume and in_github) else 0
                        label = 1 if (in_resume or in_github) else 0
                        rows.append({
                            'in_resume': in_resume, 'in_github': in_github,
                            'both_confirmed': both, 'is_required': 1,
                            'label': label,
                        })
            except Exception:
                continue
        return rows

    def _transform_resume_training_dataset(self, ds) -> List[dict]:
        """
        Transform MikePfunk28/resume-training-dataset into training rows.

        Contains conversational resume samples. We extract skills mentioned
        in the conversations and map them to training features.
        """
        rows = []
        np.random.seed(46)
        max_samples = min(len(ds), 800)

        for i in range(max_samples):
            try:
                record = ds[i]
                resume_skills = set()

                # Extract from all text fields — this dataset may have
                # conversations or structured text
                for field in ['text', 'content', 'conversations', 'input',
                              'output', 'instruction', 'response', 'skills']:
                    val = record.get(field)
                    if val:
                        if isinstance(val, str):
                            resume_skills.update(self._match_skills(val))
                        elif isinstance(val, list):
                            for item in val:
                                if isinstance(item, str):
                                    resume_skills.update(self._match_skills(item))
                                elif isinstance(item, dict):
                                    for v in item.values():
                                        if isinstance(v, str):
                                            resume_skills.update(
                                                self._match_skills(v))

                if not resume_skills:
                    continue

                # Pick a random target role for required skills context
                if self.job_roles:
                    role_data = list(self.job_roles.values())[
                        np.random.randint(len(self.job_roles))]
                    required = set(role_data.get("required_skills", []))
                else:
                    required = set()

                all_skills = resume_skills | required
                github_skills = {s for s in resume_skills
                                 if np.random.random() < 0.55}

                for skill in all_skills:
                    in_resume = 1 if skill in resume_skills else 0
                    in_github = 1 if skill in github_skills else 0
                    both = 1 if (in_resume and in_github) else 0
                    is_req = 1 if skill in required else 0
                    label = 1 if (in_resume or in_github) else 0
                    rows.append({
                        'in_resume': in_resume, 'in_github': in_github,
                        'both_confirmed': both, 'is_required': is_req,
                        'label': label,
                    })
            except Exception:
                continue
        return rows

    def _transform_job_titles_dataset(self, ds) -> List[dict]:
        """
        Transform NxtGenIntern/job_titles_and_descriptions into training rows.

        IT job roles with curated skills lists. We extract the skills and
        simulate candidate profiles.
        """
        rows = []
        np.random.seed(47)
        max_samples = min(len(ds), 500)

        for i in range(max_samples):
            try:
                record = ds[i]
                job_skills = set()

                for field in ['skills', 'Skills', 'required_skills',
                              'description', 'job_description']:
                    val = record.get(field)
                    if val:
                        if isinstance(val, str):
                            job_skills.update(self._match_skills(val))
                        elif isinstance(val, list):
                            for s in val:
                                job_skills.update(self._match_skills(str(s)))

                if not job_skills:
                    continue

                # Simulate 3 candidates
                for _ in range(3):
                    coverage = np.random.uniform(0.3, 0.9)
                    n_skills = max(1, int(len(job_skills) * coverage))
                    candidate_has = set(np.random.choice(
                        list(job_skills), size=n_skills, replace=False
                    ))

                    for skill in job_skills:
                        has_skill = skill in candidate_has
                        in_resume = 1 if has_skill and np.random.random() < 0.85 else 0
                        in_github = 1 if has_skill and np.random.random() < 0.50 else 0
                        both = 1 if (in_resume and in_github) else 0
                        label = 1 if (in_resume or in_github) else 0
                        rows.append({
                            'in_resume': in_resume, 'in_github': in_github,
                            'both_confirmed': both, 'is_required': 1,
                            'label': label,
                        })
            except Exception:
                continue
        return rows

    def _transform_generic_skills_dataset(self, ds, seed: int = 48) -> List[dict]:
        """
        Generic transformer for datasets with text fields containing skills.
        Works with any dataset that has text-like columns — extracts skills
        from all string fields and simulates candidate profiles.
        """
        rows = []
        np.random.seed(seed)
        max_samples = min(len(ds), 600)

        for i in range(max_samples):
            try:
                record = ds[i]
                found_skills = set()

                # Try to extract skills from every string field in the record
                for key, val in record.items() if isinstance(record, dict) else []:
                    if val and isinstance(val, str) and len(val) > 20:
                        found_skills.update(self._match_skills(val))
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                found_skills.update(self._match_skills(item))

                if not found_skills or len(found_skills) < 2:
                    continue

                # Simulate 2 candidate profiles
                for _ in range(2):
                    coverage = np.random.uniform(0.3, 0.85)
                    n_skills = max(1, int(len(found_skills) * coverage))
                    candidate_has = set(np.random.choice(
                        list(found_skills), size=min(n_skills, len(found_skills)), replace=False
                    ))

                    # Pick a random role for required context
                    required = set()
                    if self.job_roles:
                        role_data = list(self.job_roles.values())[
                            np.random.randint(len(self.job_roles))]
                        required = set(role_data.get("required_skills", []))

                    all_skills = found_skills | required
                    for skill in all_skills:
                        has_skill = skill in candidate_has
                        in_resume = 1 if has_skill and np.random.random() < 0.82 else 0
                        in_github = 1 if has_skill and np.random.random() < 0.50 else 0
                        both = 1 if (in_resume and in_github) else 0
                        is_req = 1 if skill in required else 0
                        label = 1 if (in_resume or in_github) else 0
                        rows.append({
                            'in_resume': in_resume, 'in_github': in_github,
                            'both_confirmed': both, 'is_required': is_req,
                            'label': label,
                        })
            except Exception:
                continue
        return rows

    # -----------------------------------------------------------------
    #  Enhanced Synthetic Data (always included)
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
            "master_skills_count": len(self.master_skills),
            "job_roles_count": len(self.job_roles),
            "cache_dir": self.cache_dir,
        }
