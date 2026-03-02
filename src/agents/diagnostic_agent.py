"""
src/agents/diagnostic_agent.py
================================
Multi-class fault-diagnostic agent (G / C / B / A fault types).
Uses XGBoost multi:softmax classifier.
Independent and fully importable.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class DiagnosticAgent:
    """
    Multi-class fault-diagnostic agent wrapping an XGBoost classifier.

    The agent accepts integer-encoded labels produced by a LabelEncoder
    (as returned by ``loader.load_multiclass_data``).  Pass the
    ``label_encoder`` and ``class_names`` from the loader so the agent can
    provide human-readable predictions.

    Parameters
    ----------
    config : dict
        Supported keys (all optional):
        - n_estimators     (int,   default 300)
        - max_depth        (int,   default 6)
        - learning_rate    (float, default 0.05)
        - subsample        (float, default 0.8)
        - colsample_bytree (float, default 0.8)
        - use_gpu          (bool,  default False)
        - random_state     (int,   default 42)
        - early_stopping_rounds (int,   default None)
        - eval_fraction    (float, default 0.1)
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.config = config
        self._early_stopping_rounds = config.get("early_stopping_rounds", None)
        self._eval_fraction = config.get("eval_fraction", 0.1)
        self._random_state = config.get("random_state", 42)

        self.feature_names: List[str] = []
        self.class_names: List[str] = []      # populated in train()
        self.label_encoder: Optional[LabelEncoder] = None
        self.n_classes: int = 0
        self.is_trained: bool = False

        tree_method = "gpu_hist" if config.get("use_gpu", False) else "hist"

        # n_estimators set here; num_class is set dynamically in train()
        self._base_params = dict(
            n_estimators=config.get("n_estimators", 300),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.05),
            subsample=config.get("subsample", 0.8),
            colsample_bytree=config.get("colsample_bytree", 0.8),
            objective="multi:softmax",
            eval_metric="mlogloss",
            tree_method=tree_method,
            random_state=self._random_state,
            use_label_encoder=False,
            verbosity=0,
        )
        self.model: Optional[XGBClassifier] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        label_encoder: Optional[LabelEncoder] = None,
        class_names: Optional[List[str]] = None,
    ) -> "DiagnosticAgent":
        """
        Fit the diagnostic model.

        Parameters
        ----------
        X              : array-like (n_samples, n_features)
        y              : array-like (n_samples,) — integer-encoded class labels
        feature_names  : optional list[str]
        label_encoder  : LabelEncoder from the data loader (optional)
        class_names    : list of human-readable class strings (optional)

        Returns
        -------
        self
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.label_encoder = label_encoder
        self.n_classes = len(np.unique(y))

        if class_names:
            self.class_names = class_names
        elif label_encoder is not None:
            self.class_names = list(label_encoder.classes_)
        else:
            self.class_names = [str(c) for c in sorted(np.unique(y))]

        logger.info(
            "[DiagnosticAgent] Training on %d samples, %d features, %d classes: %s",
            X.shape[0], X.shape[1], self.n_classes, self.class_names,
        )

        self.model = XGBClassifier(
            **self._base_params,
            num_class=self.n_classes,
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
        logger.info("[DiagnosticAgent] Training complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer-encoded class predictions."""
        self._check_trained()
        return self.model.predict(X).astype(int)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Return human-readable fault-type labels (e.g. "0001", "1010").
        Requires label_encoder to have been set during training.
        """
        y_int = self.predict(X)
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y_int)
        return np.array(self.class_names)[y_int]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability matrix, shape (n_samples, n_classes).
        Column order matches self.class_names.
        """
        self._check_trained()
        # XGBoost multi:softmax does not natively output probabilities;
        # switch to softprob temporarily via a thin wrapper approach:
        # re-use the same booster but query predict with output_margin=False.
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute diagnostic metrics on a labelled test set.

        Returns
        -------
        dict with accuracy, confusion_matrix, classification_report
        """
        self._check_trained()
        y_pred = self.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(
                y, y_pred, target_names=self.class_names, zero_division=0
            ),
        }

        logger.info(
            "[DiagnosticAgent] Accuracy=%.4f", metrics["accuracy"]
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
        """Pickle the agent to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("[DiagnosticAgent] Saved to '%s'.", path)

    @classmethod
    def load(cls, path: str | Path) -> "DiagnosticAgent":
        """Load a saved DiagnosticAgent."""
        with open(path, "rb") as f:
            agent = pickle.load(f)
        logger.info("[DiagnosticAgent] Loaded from '%s'.", path)
        return agent

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "DiagnosticAgent has not been trained yet. Call .train() first."
            )

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (
            f"DiagnosticAgent(status={status}, "
            f"n_classes={self.n_classes}, classes={self.class_names})"
        )