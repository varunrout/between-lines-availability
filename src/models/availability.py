"""
Phase 7 – ML Availability Classifier.

Predicts the probability that a build-up event contains a findable
between-lines option.

Two models are trained and compared:
  - Logistic Regression (interpretable baseline)
  - XGBoost Classifier (performance model)

Usage
-----
    from src.models.availability import AvailabilityModel

    model = AvailabilityModel()
    model.fit(feature_df, target_series)
    probs = model.predict_proba(feature_df)
    model.evaluate(feature_df, target_series)
    model.save("outputs/availability_model.pkl")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.config import CV_FOLDS, RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns used by the model
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # Ball location
    "x",
    "y",
    # Possession context
    "event_sequence_number",
    "under_pressure_flag",
    # Opponent shape
    "defensive_line_x",
    "midfield_line_x",
    "line_gap_depth",
    "defensive_compactness_y",
    "n_opponents_in_frame",
    # Team shape proxy
    "n_total_candidates",
    # Possession phase (one-hot encoded separately)
    "possession_phase_first_phase",
    "possession_phase_middle_buildup",
    "possession_phase_advanced_buildup",
    # Ball zone (one-hot encoded separately)
    "ball_zone_defensive_third",
    "ball_zone_middle_third",
]

TARGET_COLUMN = "findable_option_available"


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class AvailabilityModel:
    """Wraps Logistic Regression and XGBoost classifiers for availability."""

    def __init__(self):
        self.lr_pipeline: Optional[Pipeline] = None
        self.xgb_model = None
        self.feature_cols: list[str] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        model: str = "both",
        calibrate_lr: bool = True,
    ) -> "AvailabilityModel":
        """Fit model(s) on the provided feature DataFrame.

        Parameters
        ----------
        df:
            Must contain FEATURE_COLUMNS (or subset) and TARGET_COLUMN.
        model:
            ``"lr"``, ``"xgb"``, or ``"both"`` (default).
        calibrate_lr:
            Whether to apply Platt scaling calibration to LR.
        """
        X, y = self._prepare(df)
        self.feature_cols = list(X.columns)

        if model in ("lr", "both"):
            lr = LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
            )
            if calibrate_lr:
                lr = CalibratedClassifierCV(lr, cv=3, method="sigmoid")
            self.lr_pipeline = Pipeline(
                [("scaler", StandardScaler()), ("clf", lr)]
            )
            self.lr_pipeline.fit(X, y)
            logger.info("Logistic Regression fitted on %d samples", len(y))

        if model in ("xgb", "both") and XGBOOST_AVAILABLE:
            self.xgb_model = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            )
            self.xgb_model.fit(X, y)
            logger.info("XGBoost fitted on %d samples", len(y))

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict_proba(
        self,
        df: pd.DataFrame,
        model: str = "xgb",
    ) -> np.ndarray:
        """Return probability of class 1 (findable option available)."""
        X, _ = self._prepare(df, is_train=False)
        if model == "lr" or (model == "xgb" and self.xgb_model is None):
            return self.lr_pipeline.predict_proba(X)[:, 1]
        return self.xgb_model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    def evaluate(
        self,
        df: pd.DataFrame,
        plot: bool = True,
    ) -> dict[str, float]:
        """Cross-validated evaluation on the full dataset.

        Returns a dict with ``roc_auc_lr``, ``roc_auc_xgb``,
        ``avg_precision_lr``, ``avg_precision_xgb``.
        """
        X, y = self._prepare(df)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        metrics: dict[str, float] = {}

        if self.lr_pipeline is not None:
            auc_scores = cross_val_score(
                self.lr_pipeline, X, y, cv=cv, scoring="roc_auc"
            )
            metrics["roc_auc_lr_mean"] = float(auc_scores.mean())
            metrics["roc_auc_lr_std"] = float(auc_scores.std())
            logger.info(
                "LR ROC-AUC: %.3f ± %.3f", auc_scores.mean(), auc_scores.std()
            )

        if self.xgb_model is not None:
            auc_scores = cross_val_score(
                self.xgb_model, X, y, cv=cv, scoring="roc_auc"
            )
            metrics["roc_auc_xgb_mean"] = float(auc_scores.mean())
            metrics["roc_auc_xgb_std"] = float(auc_scores.std())
            logger.info(
                "XGB ROC-AUC: %.3f ± %.3f", auc_scores.mean(), auc_scores.std()
            )

        if plot:
            self._plot_evaluation(X, y)

        return metrics

    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances from XGBoost (if fitted)."""
        if self.xgb_model is None:
            raise RuntimeError("XGBoost model not fitted.")
        importances = self.xgb_model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist fitted models to disk using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "lr_pipeline": self.lr_pipeline,
                    "xgb_model": self.xgb_model,
                    "feature_cols": self.feature_cols,
                },
                fh,
            )
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "AvailabilityModel":
        """Load a previously saved model."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls()
        obj.lr_pipeline = state["lr_pipeline"]
        obj.xgb_model = state["xgb_model"]
        obj.feature_cols = state["feature_cols"]
        obj._is_fitted = True
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """One-hot encode categoricals and select/fill feature columns."""
        df = df.copy()

        # One-hot encode possession_phase and ball_zone
        if "possession_phase" in df.columns:
            dummies = pd.get_dummies(
                df["possession_phase"], prefix="possession_phase"
            )
            df = pd.concat([df, dummies], axis=1)

        if "ball_zone" in df.columns:
            dummies = pd.get_dummies(df["ball_zone"], prefix="ball_zone")
            df = pd.concat([df, dummies], axis=1)

        # Select features that exist in this DataFrame
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        X = df[available].copy().fillna(0).astype(float)

        y = df[TARGET_COLUMN].astype(int) if (is_train and TARGET_COLUMN in df.columns) else None
        return X, y

    def _plot_evaluation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Plot calibration curves for fitted models."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            for ax, (name, mdl) in zip(
                axes,
                [("Logistic Regression", self.lr_pipeline),
                 ("XGBoost", self.xgb_model)],
            ):
                if mdl is None:
                    continue
                CalibrationDisplay.from_estimator(
                    mdl, X, y, n_bins=10, ax=ax, name=name
                )
                ax.set_title(f"Calibration – {name}")

            plt.tight_layout()
            plt.savefig("outputs/calibration_curves.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Calibration curves saved to outputs/calibration_curves.png")
        except Exception as exc:
            logger.warning("Could not plot calibration curves: %s", exc)
