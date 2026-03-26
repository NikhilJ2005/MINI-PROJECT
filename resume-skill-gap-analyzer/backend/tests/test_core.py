"""
Unit tests for core backend functionality:
  - Skill normalization
  - Feature engineering
  - ML model train/predict
  - Skill gap analyzer status classification
  - Report generator prerequisites & difficulty
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from modules.feature_engineering import (
    FeatureEngineer,
    _build_canonical_map,
    _normalize_skill,
)
from modules.ml_model import LR_WEIGHT, DT_WEIGHT, FEATURE_NAMES
from modules.report_generator import ReportGenerator


# ----------------------------------------------------------------
#  Fixtures
# ----------------------------------------------------------------
SKILLS_MASTER = {
    "Languages": ["Python", "JavaScript", "TypeScript", "Java", "Go"],
    "Frameworks": ["React", "Django", "Flask", "Next.js", "Spring Boot"],
    "Tools": ["Docker", "Kubernetes", "Git", "Terraform"],
    "Cloud": ["AWS", "Azure", "GCP"],
    "Concepts": ["Machine Learning", "Deep Learning", "REST API", "CI/CD"],
}


@pytest.fixture
def canonical_map():
    return _build_canonical_map(SKILLS_MASTER)


@pytest.fixture
def skill_aliases():
    aliases_path = BACKEND_DIR / "data" / "skill_aliases.json"
    with open(aliases_path) as f:
        return json.load(f)


# ================================================================
#  1. Skill Normalization Tests
# ================================================================
class TestSkillNormalization:
    def test_exact_match_preserved(self, canonical_map):
        assert _normalize_skill("Python", canonical_map) == "Python"
        assert _normalize_skill("React", canonical_map) == "React"

    def test_lowercase_normalizes(self, canonical_map):
        assert _normalize_skill("python", canonical_map) == "Python"
        assert _normalize_skill("javascript", canonical_map) == "JavaScript"

    def test_dash_underscore_normalizes(self, canonical_map):
        assert _normalize_skill("next.js", canonical_map) == "Next.js"
        assert _normalize_skill("spring boot", canonical_map) == "Spring Boot"

    def test_unknown_skill_preserved(self, canonical_map):
        assert _normalize_skill("SomeObscureSkill", canonical_map) == "SomeObscureSkill"

    def test_redux_alias_not_react(self, skill_aliases):
        """Verify redux maps to Redux, not React."""
        assert skill_aliases["redux"] == "Redux"
        assert skill_aliases["Redux"] == "Redux"

    def test_common_aliases(self, skill_aliases):
        assert skill_aliases["k8s"] == "Kubernetes"
        assert skill_aliases["JS"] == "JavaScript"
        assert skill_aliases["TS"] == "TypeScript"
        assert skill_aliases["tf"] == "TensorFlow"
        assert skill_aliases["dotnet"] == "C#"
        assert skill_aliases[".NET"] == "C#"


# ================================================================
#  2. Feature Engineering Tests
# ================================================================
class TestFeatureEngineering:
    def test_feature_count(self):
        """Verify 13 features are defined."""
        assert len(FEATURE_NAMES) == 13

    def test_skill_matrix_shape(self):
        fe = FeatureEngineer()
        df = fe.create_skill_matrix(
            claimed_skills=["Python", "React"],
            demonstrated_skills=["Python", "Docker"],
            required_skills=["Python", "React", "Docker", "AWS"],
            nice_to_have_skills=["TypeScript"],
            repos_analyzed=10,
            skills_master=SKILLS_MASTER,
        )
        # Should have one row per role skill (4 required + 1 nice-to-have)
        assert len(df) == 5
        # Should have all 13 feature columns plus 'skill' and 'is_required'
        for feat in FEATURE_NAMES:
            assert feat in df.columns

    def test_binary_features_correct(self):
        fe = FeatureEngineer()
        df = fe.create_skill_matrix(
            claimed_skills=["Python"],
            demonstrated_skills=["Docker"],
            required_skills=["Python", "Docker", "AWS"],
            nice_to_have_skills=[],
            repos_analyzed=5,
            skills_master=SKILLS_MASTER,
        )
        python_row = df[df["skill_name"] == "Python"].iloc[0]
        assert python_row["in_resume"] == 1
        assert python_row["in_github"] == 0

        docker_row = df[df["skill_name"] == "Docker"].iloc[0]
        assert docker_row["in_resume"] == 0
        assert docker_row["in_github"] == 1

        aws_row = df[df["skill_name"] == "AWS"].iloc[0]
        assert aws_row["in_resume"] == 0
        assert aws_row["in_github"] == 0

    def test_github_evidence_strength_capped(self):
        fe = FeatureEngineer()
        df = fe.create_skill_matrix(
            claimed_skills=["Python"],
            demonstrated_skills=[],
            required_skills=["Python"],
            nice_to_have_skills=[],
            repos_analyzed=100,
        )
        assert df.iloc[0]["github_evidence_strength"] <= 1.0


# ================================================================
#  3. ML Model Tests
# ================================================================
class TestMLModel:
    def test_ensemble_weights_sum_to_one(self):
        assert abs(LR_WEIGHT + DT_WEIGHT - 1.0) < 1e-9

    def test_dt_weight_higher(self):
        """DT should have higher weight due to better recall."""
        assert DT_WEIGHT > LR_WEIGHT

    def test_weights_are_updated(self):
        """Verify the new 35/65 split."""
        assert LR_WEIGHT == 0.35
        assert DT_WEIGHT == 0.65


# ================================================================
#  4. Report Generator Tests
# ================================================================
class TestReportGenerator:
    def test_prerequisites_exist(self):
        rg = ReportGenerator()
        assert "Kubernetes" in rg.SKILL_PREREQUISITES
        assert "Docker" in rg.SKILL_PREREQUISITES["Kubernetes"]

    def test_topological_sort_prereqs_first(self):
        skills = ["Kubernetes", "Docker", "Git"]
        sorted_skills = ReportGenerator._topological_sort_skills(skills)
        # Docker should come before Kubernetes
        assert sorted_skills.index("Docker") < sorted_skills.index("Kubernetes")

    def test_topological_sort_no_prereqs(self):
        skills = ["Git", "SQL", "HTML"]
        sorted_skills = ReportGenerator._topological_sort_skills(skills)
        assert set(sorted_skills) == {"Git", "SQL", "HTML"}

    def test_difficulty_info(self):
        info = ReportGenerator._get_difficulty_info("Git")
        assert info["level"] == 1
        assert info["label"] == "Beginner"

        info = ReportGenerator._get_difficulty_info("Kubernetes")
        assert info["level"] == 3
        assert info["label"] == "Advanced"

    def test_missing_prerequisites(self):
        all_missing = {"Docker", "Kubernetes", "Linux", "Git"}
        prereqs = ReportGenerator._get_missing_prerequisites("Kubernetes", all_missing)
        assert "Docker" in prereqs
        assert "Linux" in prereqs

    def test_resource_hint_returns_string(self):
        hint = ReportGenerator._get_resource_hint("Python")
        assert isinstance(hint, str)
        assert len(hint) > 0

    def test_score_labels(self):
        rg = ReportGenerator()
        assert rg._get_score_label(95) == "Excellent"
        assert rg._get_score_label(80) == "Strong"
        assert rg._get_score_label(65) == "Good"
        assert rg._get_score_label(45) == "Fair"
        assert rg._get_score_label(25) == "Weak"
        assert rg._get_score_label(10) == "Poor"

    def test_learning_path_has_difficulty(self):
        rg = ReportGenerator()
        path = rg.generate_learning_path(["Kubernetes", "Docker"], ["TypeScript"])
        for item in path:
            assert "difficulty" in item
            assert "estimated_time" in item
        # Docker should come before Kubernetes in path
        skills_order = [p["skill"] for p in path if p["priority"] == "Critical"]
        assert skills_order.index("Docker") < skills_order.index("Kubernetes")

    def test_topological_sort_deep_chain(self):
        """Deep Learning → Machine Learning → Python chain should be ordered."""
        skills = ["Deep Learning", "Machine Learning", "Python"]
        sorted_skills = ReportGenerator._topological_sort_skills(skills)
        py_idx = sorted_skills.index("Python")
        ml_idx = sorted_skills.index("Machine Learning")
        dl_idx = sorted_skills.index("Deep Learning")
        assert py_idx < ml_idx < dl_idx

    def test_empty_missing_skills(self):
        """No missing skills should produce an empty learning path."""
        rg = ReportGenerator()
        path = rg.generate_learning_path([], [])
        assert path == []

    def test_unknown_skill_gets_default_difficulty(self):
        """Skills not in SKILL_DIFFICULTY should default to level 2 (Intermediate)."""
        info = ReportGenerator._get_difficulty_info("SomeObscureFramework")
        assert info["level"] == 2
        assert info["label"] == "Intermediate"
        assert info["estimated_time"] == "2-4 weeks"

    def test_learning_path_item_fields(self):
        """Each learning path item should have all required fields."""
        rg = ReportGenerator()
        path = rg.generate_learning_path(["Docker"], ["TypeScript"])
        for item in path:
            assert "skill" in item
            assert "priority" in item
            assert "suggested_path" in item
            assert "difficulty" in item
            assert "estimated_time" in item
            assert item["priority"] in ("Critical", "Recommended")

    def test_missing_prerequisites_for_advanced_skill(self):
        """Advanced skills should report missing prereqs from the missing set."""
        all_missing = {"Deep Learning", "Machine Learning", "Python"}
        prereqs = ReportGenerator._get_missing_prerequisites("Deep Learning", all_missing)
        assert "Machine Learning" in prereqs


# ================================================================
#  5. Skill Gap Analyzer Tests
# ================================================================
class TestSkillGapAnalyzer:
    def test_status_classification(self):
        """Test the four possible skill statuses."""
        from modules.skill_gap_analyzer import SkillGapAnalyzer

        analyzer = SkillGapAnalyzer(skills_master=SKILLS_MASTER)
        # Build a minimal skill matrix for predictions
        fe = FeatureEngineer()
        df = fe.create_skill_matrix(
            claimed_skills=["Python", "React"],
            demonstrated_skills=["Python", "Docker"],
            required_skills=["Python", "React", "Docker", "AWS"],
            nice_to_have_skills=[],
            repos_analyzed=5,
            skills_master=SKILLS_MASTER,
        )

        # Mock ML predictions (all predict present for simplicity)
        n = len(df)
        ml_predictions = {
            "lr_predictions": [1] * n,
            "dt_predictions": [1] * n,
        }
        probabilities = [0.9] * n

        result = analyzer.analyze(
            claimed_skills=["Python", "React"],
            demonstrated_skills=["Python", "Docker"],
            target_role="Software Engineer",
            job_roles_data={
                "Software Engineer": {
                    "required_skills": ["Python", "React", "Docker", "AWS"],
                    "nice_to_have": [],
                }
            },
            ml_predictions=ml_predictions,
            probabilities=probabilities,
            skill_matrix=df,
        )

        # Check statuses
        statuses = {r["skill"]: r["status"] for r in result["required_analysis"]}
        assert statuses["Python"] == "strong"       # in both resume and GitHub
        assert statuses["React"] == "claimed_only"  # only in resume
        assert statuses["Docker"] == "unclaimed"    # only on GitHub (not claimed on resume)
        assert statuses["AWS"] == "missing"         # in neither source

        # Unclaimed required skills are tracked separately
        assert "Docker" in result["unclaimed_required"]

    def _run_analyzer(self, claimed, demonstrated, required):
        """Helper: run the analyzer with minimal scaffolding."""
        from modules.skill_gap_analyzer import SkillGapAnalyzer

        analyzer = SkillGapAnalyzer(skills_master=SKILLS_MASTER)
        fe = FeatureEngineer()
        df = fe.create_skill_matrix(
            claimed_skills=claimed,
            demonstrated_skills=demonstrated,
            required_skills=required,
            nice_to_have_skills=[],
            repos_analyzed=5,
            skills_master=SKILLS_MASTER,
        )
        n = len(df)
        result = analyzer.analyze(
            claimed_skills=claimed,
            demonstrated_skills=demonstrated,
            target_role="Test Role",
            job_roles_data={"Test Role": {"required_skills": required, "nice_to_have": []}},
            ml_predictions={"lr_predictions": [1] * n, "dt_predictions": [1] * n},
            probabilities=[0.9] * n,
            skill_matrix=df,
        )
        return result

    def test_match_score_cannot_be_100_with_missing_required(self):
        """If any required skill is missing, match_score must be < 100."""
        result = self._run_analyzer(
            claimed=["Python", "React"],
            demonstrated=["Python", "Docker"],
            required=["Python", "React", "Docker", "AWS"],
        )
        # AWS is missing → match_score must be < 100
        assert "AWS" in result["missing_required"]
        assert result["match_score"] < 100.0

    def test_match_score_equals_100_only_when_all_required_present(self):
        """match_score should be 100 only when all required skills are 'strong' (resume+GitHub)."""
        result = self._run_analyzer(
            claimed=["Python", "React", "Docker"],
            demonstrated=["Python", "React", "Docker"],
            required=["Python", "React", "Docker"],
        )
        assert result["missing_required"] == []
        assert result["unclaimed_required"] == []
        assert result["match_score"] == 100.0

    def test_match_score_consistency(self):
        """match_score uses weighted formula: strong=1.0, claimed_only=0.4, unclaimed/missing=0."""
        result = self._run_analyzer(
            claimed=["Python"],
            demonstrated=["Docker"],
            required=["Python", "Docker", "AWS", "TypeScript"],
        )
        # Python=claimed_only (0.4 credit), Docker=unclaimed (0 credit),
        # AWS=missing (0), TypeScript=missing (0)
        expected_score = round(100 * (1 * 0.4) / 4, 1)  # = 10.0
        assert result["match_score"] == expected_score
        assert "Docker" in result["unclaimed_required"]
        assert "AWS" in result["missing_required"]
        assert "TypeScript" in result["missing_required"]

    def test_match_score_zero_when_all_required_missing(self):
        """match_score should be 0 when none of the required skills are present."""
        result = self._run_analyzer(
            claimed=["Go", "Java"],  # many skills, but none required
            demonstrated=["Go"],
            required=["Python", "Docker"],
        )
        assert result["match_score"] == 0.0
        assert len(result["missing_required"]) == 2

    def test_match_score_no_negative(self):
        """match_score must never be negative."""
        result = self._run_analyzer(
            claimed=[],
            demonstrated=[],
            required=["Python", "Docker"],
        )
        assert result["match_score"] >= 0.0

    def test_match_score_no_required_skills(self):
        """When the role has no required skills, match_score should be 0."""
        result = self._run_analyzer(
            claimed=["Python"],
            demonstrated=[],
            required=[],
        )
        assert result["match_score"] == 0.0
