"""
src/agents/detection_agent.py
==============================
Binary fault-detection agent (0 = normal, 1 = fault).
Uses XGBoost binary classifier.
Independent and fully importable.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class DetectionAgent:
    """
    Binary fault-detection agent wrapping an XGBoost classifier.

    Parameters
    ----------
    config : dict
        Supported keys (all optional):
        - n_estimators    (int,   default 300)
        - max_depth       (int,   default 6)
        - learning_rate   (float, default 0.05)
        - subsample       (float, default 0.8)
        - colsample_bytree(float, default 0.8)
        - use_gpu         (bool,  default False)
        - random_state    (int,   default 42)
        - scale_pos_weight(float, default 1.0) — set > 1 for imbalanced data
        - early_stopping_rounds (int, default None)
        - eval_fraction   (float, default 0.1) — fraction of train used as eval set
                                                  when early stopping is enabled
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}

        self.config = config
        self._early_stopping_rounds = config.get("early_stopping_rounds", None)
        self._eval_fraction = config.get("eval_fraction", 0.1)
        self._random_state = config.get("random_state", 42)
        self.feature_names: list[str] = []
        self.is_trained: bool = False

        tree_method = "gpu_hist" if config.get("use_gpu", False) else "hist"

        self.model = XGBClassifier(
            n_estimators=config.get("n_estimators", 300),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.05),
            subsample=config.get("subsample", 0.8),
            colsample_bytree=config.get("colsample_bytree", 0.8),
            scale_pos_weight=config.get("scale_pos_weight", 1.0),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=tree_method,
            random_state=self._random_state,
            use_label_encoder=False,
            verbosity=0,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "DetectionAgent":
        """
        Fit the detection model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) — binary int labels
        feature_names : optional list of column names for logging

        Returns
        -------
        self
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        logger.info(
            "[DetectionAgent] Training on %d samples, %d features.", *X.shape
        )

        fit_kwargs: dict = {}
        if self._early_stopping_rounds is not None:
            from sklearn.model_selection import train_test_split as tts

            X_tr, X_val, y_tr, y_val = tts(
                X, y,
                test_size=self._eval_fraction,
                random_state=self._random_state,
                stratify=y,
            )
            fit_kwargs = dict(
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self._early_stopping_rounds,
                verbose=False,
            )
            self.model.fit(X_tr, y_tr, **fit_kwargs)
        else:
            self.model.fit(X, y)

        self.is_trained = True
        logger.info("[DetectionAgent] Training complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 or 1)."""
        self._check_trained()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities, shape (n_samples, 2).
        Column 0 = P(normal), Column 1 = P(fault).
        """
        self._check_trained()
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute detection metrics on a labelled set.

        Returns
        -------
        dict with accuracy, roc_auc, confusion_matrix, classification_report
        """
        self._check_trained()
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_prob),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(
                y, y_pred, target_names=["Normal (0)", "Fault (1)"]
            ),
        }

        logger.info(
            "[DetectionAgent] Accuracy=%.4f  ROC-AUC=%.4f",
            metrics["accuracy"],
            metrics["roc_auc"],
        )
        return metrics

    def feature_importance(self) -> dict:
        """Return feature importances as {feature_name: score}."""
        self._check_trained()
        scores = self.model.feature_importances_
        return dict(zip(self.feature_names, scores.tolist()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the agent (model + metadata) to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("[DetectionAgent] Saved to '%s'.", path)

    @classmethod
    def load(cls, path: str | Path) -> "DetectionAgent":
        """Load a previously saved DetectionAgent."""
        with open(path, "rb") as f:
            agent = pickle.load(f)
        logger.info("[DetectionAgent] Loaded from '%s'.", path)
        return agent

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "DetectionAgent has not been trained yet. Call .train() first."
            )

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"DetectionAgent(status={status}, model={self.model})"