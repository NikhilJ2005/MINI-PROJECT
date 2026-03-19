#!/usr/bin/env python3
"""
=============================================================================
 Build-Time Dataset Processor
=============================================================================
 Generates a large, realistic training dataset by:
   1. Trying to download real HuggingFace resume/job datasets (if network available)
   2. Falling back to comprehensive simulation using skills_master.json + job_roles.json
      to create realistic candidate profiles with varied skill distributions

 Outputs 9-feature training rows (3 binary + 6 continuous) with feature-conditioned
 labels for high-accuracy ML training.

 Outputs: datasets/hf_processed.csv

 Usage:
   cd backend
   python build_datasets.py
=============================================================================
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
#  Load skills master list & job roles
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "datasets"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(DATA_DIR / "skills_master.json") as f:
    SKILLS_MASTER = json.load(f)

with open(DATA_DIR / "job_roles.json") as f:
    JOB_ROLES = json.load(f)

# Build flat skill list
ALL_SKILLS = []
for category, skills in SKILLS_MASTER.items():
    ALL_SKILLS.extend(skills)
ALL_SKILLS = list(set(ALL_SKILLS))

# Build required skills set
REQUIRED_SKILLS = set()
for role_data in JOB_ROLES.values():
    REQUIRED_SKILLS.update(role_data.get("required_skills", []))

# Build skill-to-category mapping
SKILL_CATEGORY = {}
for category, skills in SKILLS_MASTER.items():
    for s in skills:
        SKILL_CATEGORY[s] = category

# Skills commonly found together (co-occurrence clusters)
SKILL_CLUSTERS = {
    "python_ds": ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Jupyter", "Statistics"],
    "python_ml": ["Python", "TensorFlow", "PyTorch", "Keras", "Deep Learning", "Machine Learning", "Scikit-learn"],
    "python_web": ["Python", "Django", "Flask", "FastAPI", "REST API", "SQL", "PostgreSQL"],
    "js_frontend": ["JavaScript", "React", "HTML", "CSS", "TypeScript", "Node.js", "Git"],
    "js_fullstack": ["JavaScript", "React", "Node.js", "Express", "MongoDB", "REST API", "Git", "Docker"],
    "java_backend": ["Java", "Spring", "Spring Boot", "Maven", "SQL", "Docker", "Git", "REST API"],
    "devops": ["Docker", "Kubernetes", "CI/CD", "AWS", "Linux", "Terraform", "Git", "Ansible"],
    "cloud_aws": ["AWS", "EC2", "S3", "Lambda", "Docker", "Linux", "Terraform", "CloudFormation"],
    "cloud_gcp": ["GCP", "BigQuery", "Cloud Run", "Docker", "Kubernetes", "Python", "SQL"],
    "data_eng": ["Python", "SQL", "Spark", "Airflow", "Docker", "AWS", "Kafka", "ETL"],
    "mobile": ["React Native", "JavaScript", "TypeScript", "Git", "REST API", "Mobile Development"],
    "go_backend": ["Go", "Docker", "Kubernetes", "REST API", "gRPC", "PostgreSQL", "Redis", "Git"],
    "rust_systems": ["Rust", "Linux", "Git", "Docker", "System Design"],
    "ai_llm": ["Python", "LLM", "Hugging Face", "LangChain", "RAG", "Prompt Engineering", "Generative AI", "NLP"],
}

# Archetype-specific repos_analyzed distributions
ARCHETYPE_REPOS = {
    "senior": (15, 4),
    "mid": (8, 3),
    "junior": (3, 2),
    "career_switcher": (6, 3),
    "github_heavy": (18, 4),
    "resume_heavy": (2, 1),
}


# ---------------------------------------------------------------------------
#  HuggingFace Download (best-effort)
# ---------------------------------------------------------------------------
def try_load_hf_datasets() -> list:
    """Try to load real HF datasets. Returns empty list if unavailable."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.info("'datasets' package not installed -- using simulation only")
        return []

    records = []

    # Try Resume dataset
    try:
        ds = load_dataset("Sachinkelenjaguri/Resume_dataset", split="train", trust_remote_code=True)
        logger.info(f"Loaded {len(ds)} resumes from HuggingFace")
        skill_patterns = {}
        for skill in ALL_SKILLS:
            escaped = re.escape(skill.lower())
            if skill in ("C++", "C#"):
                skill_patterns[skill] = re.compile(escaped)
            elif skill in ("R", "C"):
                skill_patterns[skill] = re.compile(r"\b" + escaped + r"\b(?!\+|#)")
            else:
                skill_patterns[skill] = re.compile(r"\b" + escaped + r"\b")

        for row in ds:
            text = row.get("Resume_str", "") or row.get("resume", "") or ""
            if text and len(text) > 50:
                text_lower = text.lower()
                skills = [s for s, p in skill_patterns.items() if p.search(text_lower)]
                if skills:
                    records.append({"source": "resume", "skills": skills})
    except Exception as e:
        logger.warning(f"Could not load Resume_dataset: {e}")

    # Try Job descriptions
    try:
        ds2 = load_dataset("jacob-hugging-face/job-descriptions", split="train", trust_remote_code=True)
        logger.info(f"Loaded {len(ds2)} job descriptions from HuggingFace")
        for row in ds2:
            text = row.get("job_description", "") or row.get("description", "") or ""
            if text and len(text) > 30:
                text_lower = text.lower()
                skills = [s for s, p in skill_patterns.items() if p.search(text_lower)]
                if skills:
                    records.append({"source": "job_description", "skills": skills})
    except Exception as e:
        logger.warning(f"Could not load job-descriptions: {e}")

    return records


# ---------------------------------------------------------------------------
#  Compute continuous features for a profile
# ---------------------------------------------------------------------------
def compute_profile_features(
    resume_set: set,
    github_set: set,
    all_role_skills: list,
    repos_analyzed: int,
) -> dict:
    """Compute the 6 continuous features for a candidate profile."""
    n_role = max(len(all_role_skills), 1)

    resume_in_role = sum(1 for s in all_role_skills if s in resume_set)
    github_in_role = sum(1 for s in all_role_skills if s in github_set)

    resume_skill_ratio = resume_in_role / n_role
    github_skill_ratio = github_in_role / n_role

    agreements = sum(
        1 for s in all_role_skills
        if (s in resume_set) == (s in github_set)
    )
    skill_source_agreement = agreements / n_role

    total_candidate = max(len(resume_set) + len(github_set), 1)
    resume_claim_density = len(resume_set) / total_candidate

    github_evidence_strength = min(repos_analyzed / 20.0, 1.0)

    return {
        "resume_skill_ratio": round(resume_skill_ratio, 4),
        "github_skill_ratio": round(github_skill_ratio, 4),
        "skill_source_agreement": round(skill_source_agreement, 4),
        "resume_claim_density": round(resume_claim_density, 4),
        "github_evidence_strength": round(github_evidence_strength, 4),
    }


def compute_category_match_score(
    skill: str,
    all_candidate_skills: set,
) -> float:
    """Compute category match score for a single skill."""
    cat = SKILL_CATEGORY.get(skill)
    if not cat or not all_candidate_skills:
        return 0.0
    same_cat = sum(
        1 for s in all_candidate_skills
        if SKILL_CATEGORY.get(s) == cat and s != skill
    )
    return min(same_cat / max(len(all_candidate_skills), 1), 1.0)


def determine_label(
    in_resume: int,
    in_github: int,
    resume_claim_density: float,
    category_match_score: float,
    skill_source_agreement: float,
    rng: np.random.RandomState = None,
) -> int:
    """
    Feature-conditioned label assignment. Nearly deterministic when
    the continuous features carry enough signal.
    """
    r = rng.random() if rng else np.random.random()

    if in_resume and in_github:
        # Both sources confirm -- virtually certain
        return 1

    if in_resume and not in_github:
        # Resume-only: use features to decide
        if resume_claim_density > 0.85:
            return 0  # Likely padding
        if category_match_score > 0.5:
            return 1  # Coherent skill cluster
        if skill_source_agreement > 0.6:
            return 1  # Consistent candidate
        return 1 if r < 0.90 else 0  # Small residual noise

    if in_github and not in_resume:
        # GitHub evidence is strong
        return 1 if r < 0.98 else 0

    # Neither source
    return 0 if r < 0.99 else 1


# ---------------------------------------------------------------------------
#  Simulated Candidate Profiles
# ---------------------------------------------------------------------------
def generate_candidate_profiles(n_candidates: int = 800) -> list:
    """
    Generate realistic candidate profiles by simulating different archetypes.

    Each profile has:
      - A primary skill cluster (their strength area)
      - resume_skills: what they put on their resume
      - github_skills: what's visible on their GitHub
      - target_role: the job they're applying for
      - archetype: candidate type (senior, mid, junior, etc.)
    """
    np.random.seed(42)
    profiles = []
    role_names = list(JOB_ROLES.keys())
    cluster_names = list(SKILL_CLUSTERS.keys())

    for i in range(n_candidates):
        # Pick a candidate archetype
        archetype = np.random.choice(
            ["senior", "mid", "junior", "career_switcher", "github_heavy", "resume_heavy"],
            p=[0.15, 0.30, 0.25, 0.10, 0.10, 0.10]
        )

        # Pick a primary skill cluster
        primary_cluster = np.random.choice(cluster_names)
        cluster_skills = SKILL_CLUSTERS[primary_cluster]

        # Pick target role
        target_role = np.random.choice(role_names)
        role_data = JOB_ROLES[target_role]
        required = role_data["required_skills"]
        nice_to_have = role_data.get("nice_to_have", [])

        # Generate skills based on archetype
        resume_skills = set()
        github_skills = set()

        if archetype == "senior":
            for s in required:
                if np.random.random() < 0.85:
                    resume_skills.add(s)
                if np.random.random() < 0.70:
                    github_skills.add(s)
            for s in nice_to_have:
                if np.random.random() < 0.50:
                    resume_skills.add(s)
                if np.random.random() < 0.40:
                    github_skills.add(s)
            for s in cluster_skills:
                if np.random.random() < 0.80:
                    resume_skills.add(s)
                if np.random.random() < 0.65:
                    github_skills.add(s)

        elif archetype == "mid":
            for s in required:
                if np.random.random() < 0.65:
                    resume_skills.add(s)
                if np.random.random() < 0.45:
                    github_skills.add(s)
            for s in nice_to_have:
                if np.random.random() < 0.30:
                    resume_skills.add(s)
                if np.random.random() < 0.20:
                    github_skills.add(s)
            for s in cluster_skills:
                if np.random.random() < 0.60:
                    resume_skills.add(s)
                if np.random.random() < 0.40:
                    github_skills.add(s)

        elif archetype == "junior":
            for s in required:
                if np.random.random() < 0.35:
                    resume_skills.add(s)
                if np.random.random() < 0.20:
                    github_skills.add(s)
            for s in nice_to_have:
                if np.random.random() < 0.15:
                    resume_skills.add(s)
                if np.random.random() < 0.10:
                    github_skills.add(s)
            for s in cluster_skills[:3]:
                if np.random.random() < 0.50:
                    resume_skills.add(s)
                if np.random.random() < 0.30:
                    github_skills.add(s)

        elif archetype == "career_switcher":
            for s in cluster_skills:
                if np.random.random() < 0.75:
                    resume_skills.add(s)
                if np.random.random() < 0.55:
                    github_skills.add(s)
            for s in required:
                if np.random.random() < 0.25:
                    resume_skills.add(s)
                if np.random.random() < 0.15:
                    github_skills.add(s)

        elif archetype == "github_heavy":
            for s in required:
                if np.random.random() < 0.30:
                    resume_skills.add(s)
                if np.random.random() < 0.70:
                    github_skills.add(s)
            for s in cluster_skills:
                if np.random.random() < 0.25:
                    resume_skills.add(s)
                if np.random.random() < 0.75:
                    github_skills.add(s)

        elif archetype == "resume_heavy":
            for s in required:
                if np.random.random() < 0.75:
                    resume_skills.add(s)
                if np.random.random() < 0.15:
                    github_skills.add(s)
            for s in nice_to_have:
                if np.random.random() < 0.45:
                    resume_skills.add(s)
                if np.random.random() < 0.05:
                    github_skills.add(s)

        # Add some random extra skills (people have diverse backgrounds)
        extra_count = np.random.randint(0, 5)
        for _ in range(extra_count):
            s = np.random.choice(ALL_SKILLS)
            if np.random.random() < 0.6:
                resume_skills.add(s)
            if np.random.random() < 0.3:
                github_skills.add(s)

        # Simulate repos_analyzed for this archetype
        mean_repos, std_repos = ARCHETYPE_REPOS[archetype]
        repos_analyzed = int(np.clip(
            np.random.normal(mean_repos, std_repos), 0, 30
        ))

        profiles.append({
            "archetype": archetype,
            "target_role": target_role,
            "resume_skills": list(resume_skills),
            "github_skills": list(github_skills),
            "required": required,
            "nice_to_have": nice_to_have,
            "repos_analyzed": repos_analyzed,
        })

    return profiles


# ---------------------------------------------------------------------------
#  Convert Profiles to Training Rows (9-feature format)
# ---------------------------------------------------------------------------
def profiles_to_training_data(profiles: list) -> pd.DataFrame:
    """
    Convert candidate profiles into the 9-feature training format.

    For each profile, we create one row per skill in (required + nice_to_have).
    Labels are determined by feature-conditioned logic rather than flat
    probabilities, making them nearly deterministic and learnable.
    """
    rng = np.random.RandomState(42)
    rows = []

    for prof in profiles:
        resume_set = set(prof["resume_skills"])
        github_set = set(prof["github_skills"])
        all_role_skills = prof["required"] + prof["nice_to_have"]
        all_candidate_skills = resume_set | github_set

        # Compute profile-level features
        pf = compute_profile_features(
            resume_set, github_set, all_role_skills, prof["repos_analyzed"]
        )

        for skill in all_role_skills:
            in_resume = 1 if skill in resume_set else 0
            in_github = 1 if skill in github_set else 0
            is_req = 1 if skill in prof["required"] else 0

            cat_score = compute_category_match_score(skill, all_candidate_skills)

            label = determine_label(
                in_resume, in_github,
                pf["resume_claim_density"],
                cat_score,
                pf["skill_source_agreement"],
                rng,
            )

            rows.append({
                "in_resume": in_resume,
                "in_github": in_github,
                "is_required": is_req,
                "resume_skill_ratio": pf["resume_skill_ratio"],
                "github_skill_ratio": pf["github_skill_ratio"],
                "skill_source_agreement": pf["skill_source_agreement"],
                "resume_claim_density": pf["resume_claim_density"],
                "github_evidence_strength": pf["github_evidence_strength"],
                "category_match_score": round(cat_score, 4),
                "label": label,
            })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
#  Convert HF Records to Training Rows (if HF data available)
# ---------------------------------------------------------------------------
def hf_records_to_training_data(records: list) -> pd.DataFrame:
    """Convert HuggingFace extracted records to 9-feature training format."""
    rng = np.random.RandomState(42)
    rows = []

    skill_counts = {}
    for rec in records:
        for s in rec["skills"]:
            skill_counts[s] = skill_counts.get(s, 0) + 1
    total = max(len(records), 1)
    popularity = {s: min(c / total, 1.0) for s, c in skill_counts.items()}

    for rec in records:
        rec_skills = set(rec["skills"])
        # Simulate profile-level features from the record
        n_skills = len(rec_skills)
        resume_skill_ratio = min(n_skills / 15.0, 1.0)
        repos_analyzed = int(np.clip(rng.normal(8, 4), 0, 25))
        github_evidence_strength = min(repos_analyzed / 20.0, 1.0)

        for skill in rec["skills"]:
            is_req = 1 if skill in REQUIRED_SKILLS else 0
            pop = popularity.get(skill, 0.1)

            cat_score = compute_category_match_score(skill, rec_skills)

            if rec["source"] == "resume":
                github_prob = min(0.3 + pop * 0.5, 0.75)
                in_github = 1 if rng.random() < github_prob else 0
                in_resume = 1

                # Simulate agreement & density
                agreement = 0.5 + rng.random() * 0.3
                density = 0.4 + rng.random() * 0.3

                label = determine_label(
                    in_resume, in_github, density, cat_score, agreement, rng
                )

                rows.append({
                    "in_resume": 1, "in_github": in_github,
                    "is_required": is_req,
                    "resume_skill_ratio": round(resume_skill_ratio, 4),
                    "github_skill_ratio": round(resume_skill_ratio * (0.3 + pop * 0.4), 4),
                    "skill_source_agreement": round(agreement, 4),
                    "resume_claim_density": round(density, 4),
                    "github_evidence_strength": round(github_evidence_strength, 4),
                    "category_match_score": round(cat_score, 4),
                    "label": label,
                })
            else:
                # Job description record -- simulate candidate fit
                quality = rng.choice(
                    ["strong", "resume_only", "github_only", "missing"],
                    p=[0.25, 0.25, 0.15, 0.35]
                )
                if quality == "strong":
                    in_r, in_g = 1, 1
                elif quality == "resume_only":
                    in_r, in_g = 1, 0
                elif quality == "github_only":
                    in_r, in_g = 0, 1
                else:
                    in_r, in_g = 0, 0

                agreement = 0.4 + rng.random() * 0.4
                density = 0.3 + rng.random() * 0.4

                label = determine_label(
                    in_r, in_g, density, cat_score, agreement, rng
                )

                rows.append({
                    "in_resume": in_r, "in_github": in_g,
                    "is_required": 1,
                    "resume_skill_ratio": round(resume_skill_ratio * 0.8, 4),
                    "github_skill_ratio": round(resume_skill_ratio * 0.4, 4),
                    "skill_source_agreement": round(agreement, 4),
                    "resume_claim_density": round(density, 4),
                    "github_evidence_strength": round(github_evidence_strength, 4),
                    "category_match_score": round(cat_score, 4),
                    "label": label,
                })

        # Add some gap examples
        missing = [s for s in REQUIRED_SKILLS if s not in rec_skills]
        for skill in rng.choice(missing, min(3, len(missing)), replace=False) if missing else []:
            github_chance = rng.random() < 0.1
            cat_score = compute_category_match_score(skill, rec_skills)
            label = determine_label(
                0, 1 if github_chance else 0, 0.5, cat_score, 0.5, rng
            )
            rows.append({
                "in_resume": 0, "in_github": 1 if github_chance else 0,
                "is_required": 1,
                "resume_skill_ratio": round(resume_skill_ratio * 0.3, 4),
                "github_skill_ratio": round(resume_skill_ratio * 0.1, 4),
                "skill_source_agreement": round(0.5 + rng.random() * 0.3, 4),
                "resume_claim_density": round(0.4 + rng.random() * 0.3, 4),
                "github_evidence_strength": round(github_evidence_strength, 4),
                "category_match_score": round(cat_score, 4),
                "label": label,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("  BUILD-TIME DATASET PROCESSOR (9-feature format)")
    logger.info("=" * 60)

    frames = []

    # --- Try HuggingFace datasets first ---
    hf_records = try_load_hf_datasets()
    if hf_records:
        hf_df = hf_records_to_training_data(hf_records)
        frames.append(hf_df)
        logger.info(f"HuggingFace data: {len(hf_df)} training rows")

    # --- Generate simulated candidate profiles ---
    logger.info("Generating simulated candidate profiles...")
    profiles = generate_candidate_profiles(n_candidates=800)
    profile_df = profiles_to_training_data(profiles)
    frames.append(profile_df)
    logger.info(f"Simulated profile data: {len(profile_df)} training rows")

    # --- Combine all ---
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Log stats
    logger.info(f"Total training rows: {len(df)}")
    logger.info(f"  Positive rate: {df['label'].mean():.1%}")
    feature_cols = [
        "in_resume", "in_github", "is_required",
        "resume_skill_ratio", "github_skill_ratio",
        "skill_source_agreement", "resume_claim_density",
        "github_evidence_strength", "category_match_score",
    ]
    for col in feature_cols:
        logger.info(f"  {col}: mean={df[col].mean():.3f}")

    # Save to CSV
    output_path = OUTPUT_DIR / "hf_processed.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")

    # Save metadata
    archetype_counts = {}
    for p in profiles:
        a = p["archetype"]
        archetype_counts[a] = archetype_counts.get(a, 0) + 1

    meta = {
        "total_rows": len(df),
        "positive_rate": round(float(df["label"].mean()), 4),
        "hf_rows": len(frames[0]) if len(frames) > 1 else 0,
        "simulated_rows": len(profile_df),
        "candidate_profiles": len(profiles),
        "archetype_distribution": archetype_counts,
        "unique_roles_simulated": len(JOB_ROLES),
        "features": feature_cols,
    }
    with open(OUTPUT_DIR / "hf_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Archetype distribution: {archetype_counts}")
    logger.info("=" * 60)
    logger.info("  BUILD COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
