#!/usr/bin/env python3
"""
=============================================================================
 Build-Time Dataset Processor
=============================================================================
 Generates a large, realistic training dataset by:
   1. Trying to download real HuggingFace resume/job datasets (if network available)
   2. Falling back to comprehensive simulation using skills_master.json + job_roles.json
      to create realistic candidate profiles with varied skill distributions

 The simulation approach creates MUCH better training data than the old synthetic
 generator because it models actual candidate archetypes (junior, senior, career
 switcher, specialist, etc.) with realistic skill overlap patterns.

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


# ---------------------------------------------------------------------------
#  HuggingFace Download (best-effort)
# ---------------------------------------------------------------------------
def try_load_hf_datasets() -> list:
    """Try to load real HF datasets. Returns empty list if unavailable."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.info("'datasets' package not installed — using simulation only")
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

    This creates realistic patterns:
      - Strong candidates: many skills in both resume and GitHub
      - Resume-heavy: list skills but few GitHub projects
      - GitHub-heavy: active coder but weak resume
      - Career switchers: mismatch between skills and target role
      - Junior: few skills, some gaps
      - Senior: many skills, strong coverage
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
            # Senior: knows most required + cluster skills, strong GitHub
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
            # Mid-level: decent coverage, some gaps
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
            # Junior: limited skills, fewer GitHub projects
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
            # Juniors know basics from their cluster
            for s in cluster_skills[:3]:
                if np.random.random() < 0.50:
                    resume_skills.add(s)
                if np.random.random() < 0.30:
                    github_skills.add(s)

        elif archetype == "career_switcher":
            # Career switcher: strong in one area, weak in target role
            for s in cluster_skills:
                if np.random.random() < 0.75:
                    resume_skills.add(s)
                if np.random.random() < 0.55:
                    github_skills.add(s)
            # Weak in target role's specifics
            for s in required:
                if np.random.random() < 0.25:
                    resume_skills.add(s)
                if np.random.random() < 0.15:
                    github_skills.add(s)

        elif archetype == "github_heavy":
            # GitHub-heavy: active open source but sparse resume
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
            # Resume-heavy: well-written resume, limited GitHub
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

        profiles.append({
            "archetype": archetype,
            "target_role": target_role,
            "resume_skills": list(resume_skills),
            "github_skills": list(github_skills),
            "required": required,
            "nice_to_have": nice_to_have,
        })

    return profiles


# ---------------------------------------------------------------------------
#  Convert Profiles to Training Rows
# ---------------------------------------------------------------------------
def profiles_to_training_data(profiles: list) -> pd.DataFrame:
    """
    Convert candidate profiles into the 4-feature training format.

    For each profile, we create one row per skill in (required + nice_to_have).
    The label represents whether the candidate truly has that skill.

    Label logic:
      - both_confirmed=1 → label=1 (97% confidence, 3% noise)
      - resume only → label=1 with 75% probability (some resume padding)
      - github only → label=1 with 88% probability (code evidence is strong)
      - neither → label=0 with 96% probability (small false-positive rate)
    """
    np.random.seed(42)
    rows = []

    for prof in profiles:
        resume_set = set(prof["resume_skills"])
        github_set = set(prof["github_skills"])
        all_role_skills = prof["required"] + prof["nice_to_have"]

        for skill in all_role_skills:
            in_resume = 1 if skill in resume_set else 0
            in_github = 1 if skill in github_set else 0
            both = 1 if (in_resume and in_github) else 0
            is_req = 1 if skill in prof["required"] else 0

            # Determine label based on evidence strength
            if both:
                label = 1 if np.random.random() < 0.97 else 0
            elif in_resume and not in_github:
                # Resume claim without GitHub proof — less reliable
                label = 1 if np.random.random() < 0.75 else 0
            elif in_github and not in_resume:
                # GitHub evidence is strong (they actually coded it)
                label = 1 if np.random.random() < 0.88 else 0
            else:
                # No evidence at all — mostly a gap
                label = 0 if np.random.random() < 0.96 else 1

            rows.append({
                "in_resume": in_resume,
                "in_github": in_github,
                "both_confirmed": both,
                "is_required": is_req,
                "label": label,
            })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
#  Convert HF Records to Training Rows (if HF data available)
# ---------------------------------------------------------------------------
def hf_records_to_training_data(records: list) -> pd.DataFrame:
    """Convert HuggingFace extracted records to training format."""
    np.random.seed(42)
    rows = []

    skill_counts = {}
    for rec in records:
        for s in rec["skills"]:
            skill_counts[s] = skill_counts.get(s, 0) + 1
    total = max(len(records), 1)
    popularity = {s: min(c / total, 1.0) for s, c in skill_counts.items()}

    for rec in records:
        for skill in rec["skills"]:
            is_req = 1 if skill in REQUIRED_SKILLS else 0
            pop = popularity.get(skill, 0.1)

            if rec["source"] == "resume":
                github_prob = min(0.3 + pop * 0.5, 0.75)
                in_github = 1 if np.random.random() < github_prob else 0
                rows.append({
                    "in_resume": 1, "in_github": in_github,
                    "both_confirmed": 1 if in_github else 0,
                    "is_required": is_req,
                    "label": 1,
                })
            else:
                quality = np.random.choice(
                    ["strong", "resume_only", "github_only", "missing"],
                    p=[0.25, 0.25, 0.15, 0.35]
                )
                if quality == "strong":
                    rows.append({"in_resume": 1, "in_github": 1, "both_confirmed": 1, "is_required": 1, "label": 1})
                elif quality == "resume_only":
                    rows.append({"in_resume": 1, "in_github": 0, "both_confirmed": 0, "is_required": 1,
                                 "label": 1 if np.random.random() < 0.75 else 0})
                elif quality == "github_only":
                    rows.append({"in_resume": 0, "in_github": 1, "both_confirmed": 0, "is_required": 1,
                                 "label": 1 if np.random.random() < 0.85 else 0})
                else:
                    rows.append({"in_resume": 0, "in_github": 0, "both_confirmed": 0, "is_required": 1, "label": 0})

        # Add some gap examples
        missing = [s for s in REQUIRED_SKILLS if s not in rec["skills"]]
        for skill in np.random.choice(missing, min(3, len(missing)), replace=False) if missing else []:
            github_chance = np.random.random() < 0.1
            rows.append({
                "in_resume": 0, "in_github": 1 if github_chance else 0,
                "both_confirmed": 0, "is_required": 1,
                "label": 1 if github_chance else 0,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("  BUILD-TIME DATASET PROCESSOR")
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
    for col in ["in_resume", "in_github", "both_confirmed", "is_required"]:
        logger.info(f"  {col}: {df[col].mean():.1%} positive")

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
    }
    with open(OUTPUT_DIR / "hf_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Archetype distribution: {archetype_counts}")
    logger.info("=" * 60)
    logger.info("  BUILD COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
