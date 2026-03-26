"""
=============================================================================
 Skill Gap Analyzer Module
=============================================================================
 Role in the pipeline:
   This is the FIFTH stage. It takes the parsed resume skills, GitHub
   demonstrated skills, ML predictions, and the target job role's
   requirements — then computes a comprehensive gap analysis.

 Key metrics computed:
   - match_score:   Recruiter-friendly Role Fit Score (0-100).
                    Strong (resume+GitHub) = 1.0 weight,
                    Claimed-only (resume only) = 0.4 weight,
                    Unclaimed (GitHub only) = 0 weight (visible but no credit),
                    Missing = 0 weight.
   - gap_score:     100 - match_score
   - proof_rate:    Of required skills the candidate claims, % also on GitHub
   - confidence:    ML model's average confidence in the candidate's skills
   - Per-skill status: strong / claimed_only / unclaimed / missing
                       (nice-to-have skills also use: demonstrated_only)
=============================================================================
"""

from typing import Dict, List

import pandas as pd
from loguru import logger


def _build_canonical_map(skills_master: Dict[str, List[str]]) -> Dict[str, str]:
    """Build lowercase -> canonical name mapping from skills_master."""
    canonical = {}
    for category, skills in skills_master.items():
        for skill in skills:
            canonical[skill.lower()] = skill
            cleaned = skill.lower().replace("-", " ").replace("_", " ")
            canonical[cleaned] = skill
    return canonical


def _normalize_skill(skill: str, canonical_map: Dict[str, str]) -> str:
    """Return the canonical form of a skill name, or the original if not found."""
    if skill in canonical_map.values():
        return skill
    key = skill.lower().replace("-", " ").replace("_", " ").strip()
    return canonical_map.get(skill.lower(), canonical_map.get(key, skill))


class SkillGapAnalyzer:
    """Computes skill gaps between a candidate's profile and a target job role."""

    def __init__(self, skills_master: Dict[str, List[str]] = None) -> None:
        """Initialize the skill gap analyzer."""
        self._canonical_map = _build_canonical_map(skills_master) if skills_master else {}
        logger.info("[SkillGapAnalyzer] Initialized.")

    # -----------------------------------------------------------------
    #  Main Analysis Method
    # -----------------------------------------------------------------
    def analyze(
        self,
        claimed_skills: List[str],
        demonstrated_skills: List[str],
        target_role: str,
        job_roles_data: Dict,
        ml_predictions: Dict[str, list],
        probabilities: List[float],
        skill_matrix: pd.DataFrame,
        code_quality_score: float = None,
    ) -> Dict:
        """
        Perform a full skill gap analysis for a candidate against a target role.

        Business logic for skill status classification:
          - "strong":      Skill found in BOTH resume AND GitHub
                           → Full credit (1.0) — claimed and demonstrated
          - "claimed_only": Skill found in resume but NOT on GitHub
                           → Partial credit (0.4) — claimed, not yet proven
          - "unclaimed":   Skill found on GitHub but NOT in resume (required skills only)
                           → Zero credit — not positioned by the candidate for this role
          - "missing":     Skill found in NEITHER source
                           → Zero credit — explicit gap
          For nice-to-have skills, github-only keeps status "demonstrated_only".

        Scoring (recruiter-friendly Role Fit):
          - match_score = 100 * (strong*1.0 + claimed_only*0.4) / total_required
          - gap_score   = 100 - match_score
          - proof_rate  = 100 * strong / max(strong + claimed_only, 1)
          - confidence  = average ML probability for present required skills

        Args:
            claimed_skills:      Skills extracted from the resume.
            demonstrated_skills: Skills found on GitHub.
            target_role:         The job role to compare against.
            job_roles_data:      The full job roles dictionary.
            ml_predictions:      Dict with lr_predictions, dt_predictions keys.
            probabilities:    List of LR probability scores per skill.
            skill_matrix:        The skill matrix DataFrame from feature engineering.

        Returns:
            A comprehensive analysis dict (see structure below).
        """
        logger.info(f"[SkillGapAnalyzer] Analyzing gaps for role: {target_role}")
        logger.debug(f"[SkillGapAnalyzer] Input: {len(claimed_skills)} claimed, {len(demonstrated_skills)} demonstrated")

        # --- Step 1: Get the role's requirements ---
        role_data = job_roles_data.get(target_role, {})
        required_skills = role_data.get("required_skills", [])
        nice_to_have = role_data.get("nice_to_have", [])

        # --- Step 1b: Normalize all skill lists to canonical names ---
        if self._canonical_map:
            claimed_skills = [_normalize_skill(s, self._canonical_map) for s in claimed_skills]
            demonstrated_skills = [_normalize_skill(s, self._canonical_map) for s in demonstrated_skills]
            required_skills = [_normalize_skill(s, self._canonical_map) for s in required_skills]
            nice_to_have = [_normalize_skill(s, self._canonical_map) for s in nice_to_have]

        # Deduplicate skill lists (normalization may create duplicates)
        required_skills = list(dict.fromkeys(required_skills))
        nice_to_have = [s for s in dict.fromkeys(nice_to_have) if s not in required_skills]

        logger.debug(f"[SkillGapAnalyzer] After normalization: {len(claimed_skills)} claimed={claimed_skills[:5]}, "
                     f"{len(required_skills)} required={required_skills[:5]}")

        # --- Step 2: Build the combined skill set ---
        # Use sets for O(1) lookup instead of O(n) list scan
        claimed_set = set(claimed_skills)
        demonstrated_set = set(demonstrated_skills)
        all_combined = claimed_set | demonstrated_set

        # Build case-insensitive lookup maps as fallback
        # (handles edge cases where skill casing differs between sources)
        claimed_lower = {s.lower(): s for s in claimed_skills}
        demonstrated_lower = {s.lower(): s for s in demonstrated_skills}

        # --- Step 3: Classify each required skill ---
        required_analysis = []
        missing_required = []
        strengths = []
        claims_not_proven = []
        hidden_strengths = []
        unclaimed_required = []

        for i, skill in enumerate(required_skills):
            # Primary: exact match. Fallback: case-insensitive match.
            in_resume = skill in claimed_set or skill.lower() in claimed_lower
            in_github = skill in demonstrated_set or skill.lower() in demonstrated_lower

            # Determine skill status based on evidence from both sources.
            # Recruiter-friendly rule: a required skill found only on GitHub
            # (not claimed on the resume) is labelled "unclaimed" and earns
            # zero score credit, because the candidate did not position
            # themselves as having that skill.
            if in_resume and in_github:
                status = "strong"
                strengths.append(skill)
            elif in_resume and not in_github:
                status = "claimed_only"
                claims_not_proven.append(skill)
            elif not in_resume and in_github:
                status = "unclaimed"
                unclaimed_required.append(skill)
            else:
                status = "missing"
                missing_required.append(skill)

            # Get ML prediction and probability for this skill (if available)
            # The skill_matrix rows align with required + nice_to_have skills
            ml_pred = ml_predictions.get("ensemble_predictions", ml_predictions.get("lr_predictions", [0] * len(required_skills)))
            ml_prob = probabilities

            skill_analysis = {
                "skill": skill,
                "status": status,
                "in_resume": in_resume,
                "in_github": in_github,
                "ml_prediction": ml_pred[i] if i < len(ml_pred) else 0,
                "probability": round(ml_prob[i], 4) if i < len(ml_prob) else 0.0,
            }
            required_analysis.append(skill_analysis)

        # --- Step 4: Classify each nice-to-have skill ---
        nice_to_have_analysis = []
        missing_nice_to_have = []

        for skill in nice_to_have:
            in_resume = skill in claimed_set or skill.lower() in claimed_lower
            in_github = skill in demonstrated_set or skill.lower() in demonstrated_lower

            if in_resume and in_github:
                status = "strong"
            elif in_resume:
                status = "claimed_only"
            elif in_github:
                status = "demonstrated_only"
            else:
                status = "missing"
                missing_nice_to_have.append(skill)

            nice_to_have_analysis.append({
                "skill": skill,
                "status": status,
                "in_resume": in_resume,
                "in_github": in_github,
            })

        # --- Step 5: Calculate overall scores ---

        # Recruiter-friendly Role Fit Score:
        #   strong      (resume ✅ + github ✅) → full credit   (1.0)
        #   claimed_only (resume ✅ + github ❌) → partial credit (0.4)
        #   unclaimed   (resume ❌ + github ✅) → no credit     (0.0)
        #   missing     (resume ❌ + github ❌) → no credit     (0.0)
        # The score reflects only what the candidate has positioned
        # themselves as having; unclaimed GitHub skills do not inflate it.
        total_required = len(required_skills)
        strong_count = len(strengths)
        claimed_only_count = len(claims_not_proven)

        if total_required > 0:
            role_fit_score = round(
                100 * (strong_count * 1.0 + claimed_only_count * 0.4) / total_required, 1
            )
        else:
            role_fit_score = 0.0

        match_score = role_fit_score  # match_score is the recruiter-friendly role fit score

        # Gap score: inverse of match score
        gap_score = round(100 - match_score, 1)

        # Proof rate: of the required skills the candidate claims,
        # what fraction are also evidenced on GitHub?
        proof_rate = round(
            100 * strong_count / max(strong_count + claimed_only_count, 1), 1
        )

        # present_required counts skills that earn any credit (strong or claimed_only)
        present_required = strong_count + claimed_only_count

        # Confidence score: average ML probability across ALL required skills
        # This gives an unbiased view of how confident the ML model is overall
        all_probs = [
            max(0.0, min(float(probabilities[i]), 1.0))
            for i in range(min(len(required_skills), len(probabilities)))
            if not (isinstance(probabilities[i], float) and (probabilities[i] != probabilities[i]))  # skip NaN
        ]

        if all_probs:
            confidence = round(sum(all_probs) / len(all_probs) * 100, 1)
        else:
            confidence = 0.0

        # --- Step 6: Calculate nice-to-have score ---
        total_nice = len(nice_to_have)
        present_nice = total_nice - len(missing_nice_to_have)
        nice_to_have_score = round(min(present_nice / total_nice, 1.0) * 100, 1) if total_nice > 0 else 0.0

        # --- Step 7: Calculate composite ranking score ---
        # Weighted formula combining multiple signals:
        #   45% required skill coverage (most important)
        #   15% nice-to-have coverage
        #   15% ML confidence
        #   10% GitHub evidence bonus (ratio of demonstrated skills)
        #   10% verification strength (ratio of "strong" skills)
        #    5% claim penalty (penalize heavy resume-only claims)
        # Note: unclaimed_required skills (github-only required) do not
        # contribute to github_bonus since the candidate didn't claim them.
        github_skill_count = len(strengths)  # only strong (resume+GitHub) skills count
        total_present = present_required + present_nice
        github_bonus = min(100.0, (github_skill_count / max(total_required, 1)) * 100)
        strong_ratio = (len(strengths) / max(total_present, 1)) * 100 if total_present > 0 else 0.0

        # Claim penalty: if most skills are resume-only, reduce score
        claim_penalty_score = 100.0
        if total_present > 0:
            unverified_ratio = len(claims_not_proven) / max(total_present, 1)
            claim_penalty_score = max(0.0, 100.0 - (unverified_ratio * 50.0))

        # Adjust weights when code quality score is available
        if code_quality_score is not None:
            # Normalize code quality from 1-10 scale to 0-100
            cq_normalized = min(100.0, max(0.0, code_quality_score * 10))
            composite_score = min(100.0, round(
                match_score * 0.35 +
                nice_to_have_score * 0.12 +
                confidence * 0.12 +
                github_bonus * 0.08 +
                strong_ratio * 0.08 +
                claim_penalty_score * 0.05 +
                cq_normalized * 0.20,
                1
            ))
        else:
            composite_score = min(100.0, round(
                match_score * 0.45 +
                nice_to_have_score * 0.15 +
                confidence * 0.15 +
                github_bonus * 0.10 +
                strong_ratio * 0.10 +
                claim_penalty_score * 0.05,
                1
            ))

        logger.info(f"[SkillGapAnalyzer] RoleFit: {match_score}% | Gap: {gap_score}% | Confidence: {confidence}% | Composite: {composite_score}% | ProofRate: {proof_rate}%")
        logger.debug(f"[SkillGapAnalyzer] Missing required: {missing_required} | Unclaimed required: {unclaimed_required}")

        # Consistency guard: match_score must be < 100 when any required skill is missing.
        if match_score == 100.0 and missing_required:
            logger.error(
                f"[SCORE_GUARD] match_score=100% but {len(missing_required)} required skill(s) are missing: "
                f"{missing_required}. This indicates a scoring bug."
            )
        # Debug logging for 0% scores to help diagnose issues
        if match_score == 0 and total_required > 0:
            logger.warning(
                f"[SCORE_DEBUG] match_score=0% but {total_required} skills required! "
                f"Required: {required_skills} | "
                f"Claimed ({len(claimed_skills)}): {claimed_skills[:10]} | "
                f"Demonstrated ({len(demonstrated_skills)}): {demonstrated_skills[:10]}"
            )

        # --- Step 8: Compile and return the full analysis ---
        return {
            "match_score": match_score,
            "gap_score": gap_score,
            "confidence": confidence,
            "composite_score": composite_score,
            "nice_to_have_score": nice_to_have_score,
            "proof_rate": proof_rate,
            "unclaimed_required": unclaimed_required,
            "required_analysis": required_analysis,
            "nice_to_have_analysis": nice_to_have_analysis,
            "missing_required": missing_required,
            "missing_nice_to_have": missing_nice_to_have,
            "strengths": strengths,
            "claims_not_proven": claims_not_proven,
            "hidden_strengths": hidden_strengths,
        }
