"""
=============================================================================
 ML Model — Logistic Regression + Decision Tree Ensemble
=============================================================================
 Two models working together for accuracy and explainability:
   Logistic Regression → calibrated probability scores (0-1)
   Decision Tree       → explainable IF/THEN rules

 Training pipeline:
   1. Split data 80/20 (stratified to preserve class balance)
   2. Train both models with class_weight='balanced'
   3. Evaluate on test set (accuracy, F1, precision, recall, AUC)
   4. Run 5-fold cross-validation for honest accuracy
   5. Save models to disk for fast startup next time
=============================================================================
"""

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------
# Ensemble weights: LR provides calibrated probabilities,
# DT adds explainability with tree-based splits
LR_WEIGHT = 0.35
DT_WEIGHT = 0.65

# Prediction threshold: above this = model says "candidate has skill"
PREDICTION_THRESHOLD = 0.5

# The 13 features our model uses for each skill (3 binary + 10 continuous)
FEATURE_NAMES = [
    'in_resume', 'in_github', 'is_required',
    'resume_skill_ratio', 'github_skill_ratio',
    'skill_source_agreement', 'resume_claim_density',
    'github_evidence_strength', 'category_match_score',
    'skill_rarity_score', 'profile_consistency_score',
    'both_sources', 'source_ratio_interaction',
]


class SkillGapMLModel:
    """Trains and runs ML models for skill presence prediction."""

    FEATURE_NAMES = FEATURE_NAMES

    def __init__(self) -> None:
        """
        Initialize both models with optimized hyperparameters.
        class_weight='balanced' handles imbalanced skill datasets.
        """
        # Logistic Regression with PolynomialFeatures for interaction learning
        self.lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ('classifier', LogisticRegression(
                C=1.5,
                max_iter=3000,
                random_state=42,
                class_weight='balanced',
                solver='lbfgs'
            ))
        ])

        # Decision Tree — tuned for 13 features
        self.dt_model = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )

        self.is_trained = False
        self.metrics: Dict = {}
        self.dataset_source = "not trained yet"
        self.model_save_path = Path("./models_saved")
        self.model_save_path.mkdir(exist_ok=True)

        # Accuracy fields used by report_generator
        self.lr_accuracy: float = 0.0
        self.dt_accuracy: float = 0.0

        logger.info("[MLModel] Initialized LR + DT ensemble.")

    # =================================================================
    #  TRAINING
    # =================================================================
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_source: str = "synthetic",
        use_cross_validation: bool = True,
    ) -> Dict:
        """
        Full training pipeline with evaluation.

        Steps:
          1. Split data 80/20 (stratified)
          2. Train both models
          3. Evaluate on test set
          4. Run 5-fold cross-validation
          5. Save models to disk
        """
        self.dataset_source = dataset_source
        logger.info(f"Training on {len(X)} samples from {dataset_source}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        # -- Train/Test Split (stratified to preserve class balance) --
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # -- Train Both Models --
        logger.info("Training Logistic Regression...")
        self.lr_pipeline.fit(X_train, y_train)

        logger.info("Training Decision Tree...")
        self.dt_model.fit(X_train, y_train)

        self.is_trained = True

        # -- Evaluate on Test Set --
        lr_metrics = self._evaluate(self.lr_pipeline, X_test, y_test, "LR")
        dt_metrics = self._evaluate(self.dt_model, X_test, y_test, "DT")

        self.lr_accuracy = round(lr_metrics['accuracy'] * 100, 2)
        self.dt_accuracy = round(dt_metrics['accuracy'] * 100, 2)

        # -- Cross Validation (5-fold for honest accuracy) --
        cv_scores = {}
        if use_cross_validation and len(X) >= 50:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            lr_cv = cross_val_score(self.lr_pipeline, X, y, cv=cv, scoring='f1')
            dt_cv = cross_val_score(self.dt_model, X, y, cv=cv, scoring='f1')
            cv_scores = {
                'lr_cv_f1_mean': round(float(lr_cv.mean()), 4),
                'lr_cv_f1_std': round(float(lr_cv.std()), 4),
                'dt_cv_f1_mean': round(float(dt_cv.mean()), 4),
                'dt_cv_f1_std': round(float(dt_cv.std()), 4),
            }
            logger.info(
                f"Cross-validation F1 | "
                f"LR: {lr_cv.mean():.3f} +/-{lr_cv.std():.3f} | "
                f"DT: {dt_cv.mean():.3f} +/-{dt_cv.std():.3f}"
            )

        # -- Store All Metrics --
        self.metrics = {
            'lr': lr_metrics,
            'dt': dt_metrics,
            'cross_validation': cv_scores,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'dataset_source': dataset_source,
            'features': self.FEATURE_NAMES,
        }

        self._log_training_results()
        self.save_models()

        return self.metrics

    def _evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series, label: str) -> dict:
        """Compute evaluation metrics for one model."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
        return metrics

    # =================================================================
    #  PREDICTION
    # =================================================================
    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Run both models and return ensemble predictions.

        Ensemble = LR_WEIGHT * LR + DT_WEIGHT * DT
        If ensemble probability > PREDICTION_THRESHOLD -> "has skill"
        """
        if not self.is_trained:
            logger.warning("[MLModel] Models not trained yet -- returning default predictions")
            n = len(X)
            return {
                "lr_predictions": [0] * n,
                "dt_predictions": [0] * n,
                "lr_probabilities": [0.5] * n,
                "dt_probabilities": [0.5] * n,
                "ensemble_predictions": [0] * n,
                "ensemble_probabilities": [0.5] * n,
                "warning": "model_not_trained",
            }

        # Ensure correct feature order
        for col in self.FEATURE_NAMES:
            if col not in X.columns:
                X[col] = 0
        X = X[self.FEATURE_NAMES]

        lr_proba = self.lr_pipeline.predict_proba(X)[:, 1]
        dt_proba = self.dt_model.predict_proba(X)[:, 1]

        # Ensemble: weighted average of both models
        ensemble_proba = (
            LR_WEIGHT * lr_proba +
            DT_WEIGHT * dt_proba
        )

        # Round probabilities for cleaner output
        lr_probs_rounded = [round(float(p), 4) for p in lr_proba]
        dt_probs_rounded = [round(float(p), 4) for p in dt_proba]
        ens_probs_rounded = [round(float(p), 4) for p in ensemble_proba]

        logger.debug(f"[MLModel] Generated predictions for {len(X)} skills.")

        return {
            'lr_predictions': (lr_proba >= PREDICTION_THRESHOLD).astype(int).tolist(),
            'lr_probabilities': lr_probs_rounded,
            'dt_predictions': (dt_proba >= PREDICTION_THRESHOLD).astype(int).tolist(),
            'dt_probabilities': dt_probs_rounded,
            'ensemble_predictions': (ensemble_proba >= PREDICTION_THRESHOLD).astype(int).tolist(),
            'ensemble_probabilities': ens_probs_rounded,
        }

    # =================================================================
    #  FEATURE IMPORTANCE
    # =================================================================
    def get_feature_importance(self) -> Dict:
        """
        Feature importance from both models.
        Shows which features matter most for predictions.
        """
        if not self.is_trained:
            return {}

        dt_imp = dict(zip(
            self.FEATURE_NAMES,
            [round(float(v), 4) for v in self.dt_model.feature_importances_]
        ))

        # LR coefficients are expanded by PolynomialFeatures — extract top ones
        try:
            poly = self.lr_pipeline.named_steps['poly']
            poly_names = poly.get_feature_names_out(self.FEATURE_NAMES)
            coefs = abs(self.lr_pipeline.named_steps['classifier'].coef_[0])
            # Return only the top 20 most important coefficients for readability
            coef_pairs = sorted(zip(poly_names, coefs), key=lambda x: -x[1])
            lr_coefs = {name: round(float(val), 4) for name, val in coef_pairs[:20]}
        except Exception:
            lr_coefs = {}

        return {
            'dt_importance': dt_imp,
            'lr_coefficients': lr_coefs,
            'explanation': (
                "DT importance: how often each feature splits the tree. "
                "LR coefficients: how much each feature (including interactions) moves the probability."
            )
        }

    # =================================================================
    #  MODEL SUMMARY
    # =================================================================
    def get_model_summary(self) -> Dict:
        """
        Human-readable summary of both models.
        Used by report_generator.py for the ML insights section.
        """
        return {
            "models_used": ["Logistic Regression", "Decision Tree"],
            "is_trained": self.is_trained,
            "lr_accuracy": self.lr_accuracy,
            "dt_accuracy": self.dt_accuracy,
            "feature_importance": self.get_feature_importance(),
            "lr_explanation": (
                "Logistic Regression calculates the probability "
                "a skill is genuinely present (0% to 100%) based "
                "on resume claims, GitHub evidence, and 8 continuous "
                "profile-level signals including skill rarity and "
                "profile consistency. Outputs a calibrated confidence "
                "score via StandardScaler normalization."
            ),
            "dt_explanation": (
                "Decision Tree builds IF/THEN rules from data using "
                "11 features. Example: IF skill in GitHub AND "
                "category_match_score > 0.5 AND profile_consistency > 0.4 "
                "THEN confidence=99%. Fully explainable -- you can "
                "trace exactly why each decision was made."
            ),
            "ensemble_explanation": (
                f"Final score = {int(LR_WEIGHT*100)}% LR + "
                f"{int(DT_WEIGHT*100)}% DT. "
                "LR provides calibrated probabilities while DT "
                "captures non-linear feature interactions. Together "
                "they reduce individual errors."
            ),
            "training_explanation": (
                "Models trained on real HuggingFace resume/job datasets + "
                "synthetic augmentation for comprehensive coverage. "
                f"Features used: {', '.join(self.FEATURE_NAMES)}. "
                "class_weight='balanced' handles imbalanced datasets. "
                "Cross-validation ensures honest accuracy reporting."
            ),
            "metrics": self.metrics,
            "feature_names": self.FEATURE_NAMES,
        }

    # =================================================================
    #  LOGGING
    # =================================================================
    def _log_training_results(self):
        """Log a clean training summary to console."""
        lr = self.metrics.get('lr', {})
        dt = self.metrics.get('dt', {})
        cv = self.metrics.get('cross_validation', {})

        logger.info("=" * 60)
        logger.info("          ML MODEL TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(
            f"  Logistic Regression | "
            f"Acc: {lr.get('accuracy', 0):.1%} | "
            f"F1: {lr.get('f1', 0):.3f} | "
            f"AUC: {lr.get('roc_auc', 0):.3f}"
        )
        logger.info(
            f"  Decision Tree       | "
            f"Acc: {dt.get('accuracy', 0):.1%} | "
            f"F1: {dt.get('f1', 0):.3f} | "
            f"AUC: {dt.get('roc_auc', 0):.3f}"
        )
        if cv:
            logger.info(
                f"  Cross-Val (5-fold)  | "
                f"LR F1: {cv.get('lr_cv_f1_mean', 0):.3f} "
                f"+/-{cv.get('lr_cv_f1_std', 0):.3f} | "
                f"DT F1: {cv.get('dt_cv_f1_mean', 0):.3f} "
                f"+/-{cv.get('dt_cv_f1_std', 0):.3f}"
            )
        logger.info(
            f"  Dataset: {self.dataset_source} | "
            f"Samples: {self.metrics.get('train_samples', 0)} train "
            f"/ {self.metrics.get('test_samples', 0)} test"
        )
        logger.info("=" * 60)

    # =================================================================
    #  SAVE / LOAD MODELS
    # =================================================================
    def save_models(self):
        """Save both models to disk with joblib."""
        try:
            joblib.dump(self.lr_pipeline,
                        self.model_save_path / "lr_model.pkl")
            joblib.dump(self.dt_model,
                        self.model_save_path / "dt_model.pkl")
            with open(self.model_save_path / "metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Models saved to {self.model_save_path}")
        except Exception as e:
            logger.warning(f"Could not save models: {e}")

    def load_models(self) -> bool:
        """
        Load previously saved models from disk.
        Faster than retraining on every startup.
        Returns True if loaded successfully.
        """
        try:
            lr_path = self.model_save_path / "lr_model.pkl"
            dt_path = self.model_save_path / "dt_model.pkl"
            metrics_path = self.model_save_path / "metrics.json"

            if not (lr_path.exists() and dt_path.exists()):
                logger.info("No saved models found -- will train fresh")
                return False

            self.lr_pipeline = joblib.load(lr_path)
            self.dt_model = joblib.load(dt_path)

            if metrics_path.exists():
                with open(metrics_path) as f:
                    self.metrics = json.load(f)
                self.dataset_source = self.metrics.get(
                    'dataset_source', 'loaded from disk'
                )

            self.lr_accuracy = round(
                self.metrics.get('lr', {}).get('accuracy', 0) * 100, 2
            )
            self.dt_accuracy = round(
                self.metrics.get('dt', {}).get('accuracy', 0) * 100, 2
            )

            self.is_trained = True
            logger.info(
                f"Loaded saved models | "
                f"LR acc: {self.lr_accuracy}% | "
                f"DT acc: {self.dt_accuracy}%"
            )
            return True

        except Exception as e:
            logger.warning(f"Could not load saved models: {e}")
            return False
