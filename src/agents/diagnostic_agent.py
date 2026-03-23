"""
src/agents/diagnostic_agent.py
================================
Ensemble fault-diagnostic agent composed of four independent binary
XGBoost classifiers — one per fault bit:

    Position 0 → G fault
    Position 1 → C fault
    Position 2 → B fault
    Position 3 → A fault

Each sub-model outputs 0 (fault absent) or 1 (fault present).
The four binary predictions are concatenated to form a 4-character
fault code string, e.g. "1010" (G=1, C=0, B=1, A=0).

Training labels must be passed as integer-encoded fault-code strings
(as returned by ``loader.load_multiclass_data``).  The LabelEncoder
is used only for decoding/encoding those multi-class labels; each
sub-model is trained on its own extracted bit column.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Fault bit order: index → fault name
FAULT_BITS: List[str] = ["G", "C", "B", "A"]


class DiagnosticAgent:
    """
    Ensemble fault-diagnostic agent: four binary XGBoost sub-models.

    Each sub-model independently detects one fault bit (G / C / B / A).
    Predictions are combined into a 4-character fault code string such
    as "0110" (G=0, C=1, B=1, A=0).

    Parameters
    ----------
    config : dict
        Supported keys (all optional, applied identically to each sub-model):
        - n_estimators          (int,   default 300)
        - max_depth             (int,   default 6)
        - learning_rate         (float, default 0.05)
        - subsample             (float, default 0.8)
        - colsample_bytree      (float, default 0.8)
        - use_gpu               (bool,  default False)
        - random_state          (int,   default 42)
        - early_stopping_rounds (int,   default None)
        - eval_fraction         (float, default 0.1)
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.config = config
        self._early_stopping_rounds = config.get("early_stopping_rounds", None)
        self._eval_fraction = config.get("eval_fraction", 0.1)
        self._random_state = config.get("random_state", 42)

        self.feature_names: List[str] = []
        self.label_encoder: Optional[LabelEncoder] = None
        # Maps fault name → trained XGBClassifier
        self.models: Dict[str, XGBClassifier] = {}
        self.is_trained: bool = False

        tree_method = "gpu_hist" if config.get("use_gpu", False) else "hist"
        self._base_params = dict(
            n_estimators=config.get("n_estimators", 300),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.05),
            subsample=config.get("subsample", 0.8),
            colsample_bytree=config.get("colsample_bytree", 0.8),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=tree_method,
            random_state=self._random_state,
            use_label_encoder=False,
            verbosity=0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Convert integer-encoded class labels → 4-char fault code strings.

        If a LabelEncoder is available its ``inverse_transform`` is used;
        otherwise labels are assumed to already be strings or directly
        castable to 4-char codes.
        """
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y_encoded.astype(int))
        return np.array([str(c).zfill(4) for c in y_encoded])

    @staticmethod
    def _extract_bit(fault_codes: np.ndarray, bit_index: int) -> np.ndarray:
        """
        Extract one binary column from an array of 4-char fault code strings.

        Parameters
        ----------
        fault_codes : array of strings, e.g. ["0110", "1001", ...]
        bit_index   : 0=G, 1=C, 2=B, 3=A

        Returns
        -------
        Binary numpy array (0 or 1), shape (n_samples,).
        """
        return np.array([int(code[bit_index]) for code in fault_codes], dtype=int)

    def _make_sub_model(self) -> XGBClassifier:
        """Instantiate one binary XGBClassifier from base params."""
        return XGBClassifier(**self._base_params)

    def _fit_sub_model(
        self,
        model: XGBClassifier,
        X: np.ndarray,
        y_bit: np.ndarray,
    ) -> XGBClassifier:
        """Fit a single binary sub-model, with optional early stopping."""
        if self._early_stopping_rounds is not None:
            from sklearn.model_selection import train_test_split as tts

            X_tr, X_val, y_tr, y_val = tts(
                X, y_bit,
                test_size=self._eval_fraction,
                random_state=self._random_state,
                stratify=y_bit,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self._early_stopping_rounds,
                verbose=False,
            )
        else:
            model.fit(X, y_bit)
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        label_encoder: Optional[LabelEncoder] = None,
        class_names: Optional[List[str]] = None,   # kept for API compat, unused
    ) -> "DiagnosticAgent":
        """
        Fit all four binary sub-models.

        Parameters
        ----------
        X             : array-like (n_samples, n_features)
        y             : array-like (n_samples,) — integer-encoded fault codes
                        as produced by LabelEncoder on strings like "0110".
        feature_names : optional list[str]
        label_encoder : LabelEncoder from the data loader (recommended)
        class_names   : ignored; kept for backward compatibility

        Returns
        -------
        self
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.label_encoder = label_encoder

        # Decode integer labels → 4-char fault code strings
        fault_codes = self._decode_labels(np.asarray(y))

        logger.info(
            "[DiagnosticAgent] Training 4 binary sub-models on %d samples, "
            "%d features. Fault bits: %s",
            X.shape[0], X.shape[1], FAULT_BITS,
        )

        self.models = {}
        for bit_idx, fault_name in enumerate(FAULT_BITS):
            y_bit = self._extract_bit(fault_codes, bit_idx)
            positive_rate = y_bit.mean()
            logger.info(
                "  [%s] positive rate = %.3f  (%d / %d)",
                fault_name, positive_rate, y_bit.sum(), len(y_bit),
            )
            model = self._make_sub_model()
            self._fit_sub_model(model, X, y_bit)
            self.models[fault_name] = model
            logger.info("  [%s] sub-model trained.", fault_name)

        self.is_trained = True
        logger.info("[DiagnosticAgent] All sub-models trained.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_bits(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run each sub-model and return per-fault binary predictions.

        Returns
        -------
        dict  { "G": array([0,1,...]), "C": ..., "B": ..., "A": ... }
        """
        self._check_trained()
        return {
            fault: self.models[fault].predict(X).astype(int)
            for fault in FAULT_BITS
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return integer-encoded fault code predictions (via LabelEncoder).

        If no LabelEncoder was supplied, returns fault code *strings* instead
        (same shape, dtype object).  Use ``predict_labels`` for strings always.
        """
        labels = self.predict_labels(X)
        if self.label_encoder is not None:
            return self.label_encoder.transform(labels)
        return labels

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Return 4-character fault code strings, e.g. "0110".

        Each character corresponds to one fault bit (G / C / B / A).
        """
        bit_preds = self._predict_bits(X)
        n = len(next(iter(bit_preds.values())))
        codes = np.array(
            [
                "".join(str(bit_preds[fault][i]) for fault in FAULT_BITS)
                for i in range(n)
            ]
        )
        return codes

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Return per-fault probability arrays.

        Returns
        -------
        dict  { "G": array shape (n,2), "C": ..., "B": ..., "A": ... }
        Each array[:,1] is P(fault present).
        """
        self._check_trained()
        return {
            fault: self.models[fault].predict_proba(X)
            for fault in FAULT_BITS
        }

    def predict_fault_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return P(fault present) for each fault as a (n_samples, 4) matrix.

        Column order: G, C, B, A.
        """
        proba = self.predict_proba(X)
        return np.column_stack([proba[fault][:, 1] for fault in FAULT_BITS])

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the ensemble on a labelled test set.

        Returns
        -------
        dict with keys:
          - "overall_accuracy"  : fault-code exact-match accuracy
          - "overall_f1"        : weighted F1 across all fault code classes
          - "per_fault"         : dict keyed by fault name, each with
                                  accuracy and f1_score (weighted, per bit)
        """
        self._check_trained()
        from sklearn.metrics import f1_score

        fault_codes_true = self._decode_labels(np.asarray(y))
        fault_codes_pred = self.predict_labels(X)

        overall_acc = accuracy_score(fault_codes_true, fault_codes_pred)
        overall_f1  = f1_score(fault_codes_true, fault_codes_pred, average="weighted", zero_division=0)
        logger.info(
            "[DiagnosticAgent] Overall accuracy=%.4f  F1(weighted)=%.4f",
            overall_acc, overall_f1,
        )

        per_fault: dict = {}
        for bit_idx, fault_name in enumerate(FAULT_BITS):
            y_true_bit = self._extract_bit(fault_codes_true, bit_idx)
            y_pred_bit = self._extract_bit(fault_codes_pred, bit_idx)
            bit_acc = accuracy_score(y_true_bit, y_pred_bit)
            bit_f1  = f1_score(y_true_bit, y_pred_bit, average="weighted", zero_division=0)
            per_fault[fault_name] = {
                "accuracy": bit_acc,
                "f1_score": bit_f1,
            }
            logger.info(
                "  [%s] accuracy=%.4f  F1(weighted)=%.4f", fault_name, bit_acc, bit_f1
            )

        metrics = {
            "overall_accuracy": overall_acc,
            "overall_f1": overall_f1,
            "per_fault": per_fault,
        }
        return metrics


    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the agent to a directory using XGBoost native JSON format.

        Layout produced under ``path/``::

            <path>/
            ├── meta.json      # feature_names, label_encoder classes, config
            ├── model_G.json   # XGBoost native JSON for the G sub-model
            ├── model_C.json
            ├── model_B.json
            └── model_A.json

        Each ``.json`` is written by ``XGBClassifier.save_model()`` and is
        fully portable across XGBoost versions (no pickle fragility).

        Parameters
        ----------
        path : str or Path
            Directory to create (or overwrite).  Created if it does not exist;
            existing model files are overwritten silently.
        """
        import json

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- per-sub-model XGBoost native JSON ---
        for fault in FAULT_BITS:
            model_path = save_dir / f"model_{fault}.json"
            self.models[fault].save_model(str(model_path))
            logger.info(
                "[DiagnosticAgent] Sub-model '%s' saved to '%s'.", fault, model_path
            )

        # --- metadata sidecar ---
        meta = {
            "fault_bits": FAULT_BITS,
            "feature_names": self.feature_names,
            "config": self.config,
            # Persist LabelEncoder classes so we can reconstruct it on load
            "label_encoder_classes": (
                self.label_encoder.classes_.tolist()
                if self.label_encoder is not None
                else None
            ),
        }
        meta_path = save_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("[DiagnosticAgent] Metadata saved to '%s'.", meta_path)
        logger.info("[DiagnosticAgent] Agent saved to directory '%s'.", save_dir)

    @classmethod
    def load(cls, path: str | Path) -> "DiagnosticAgent":
        """
        Load a DiagnosticAgent saved with :meth:`save`.

        Parameters
        ----------
        path : str or Path
            The directory previously passed to :meth:`save`.

        Returns
        -------
        DiagnosticAgent (trained, ready for inference)
        """
        import json

        save_dir = Path(path)
        if not save_dir.is_dir():
            raise FileNotFoundError(
                f"Expected a directory at '{save_dir}', but it does not exist. "
                "Pass the directory path returned by save(), not a file path."
            )

        # --- metadata ---
        meta_path = save_dir / "meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        agent = cls(config=meta.get("config", {}))
        agent.feature_names = meta["feature_names"]

        # Reconstruct LabelEncoder if classes were persisted
        le_classes = meta.get("label_encoder_classes")
        if le_classes is not None:
            le = LabelEncoder()
            le.classes_ = np.array(le_classes)
            agent.label_encoder = le

        # --- per-sub-model XGBoost native JSON ---
        agent.models = {}
        for fault in FAULT_BITS:
            model_path = save_dir / f"model_{fault}.json"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Sub-model file missing: '{model_path}'. "
                    "The save directory may be incomplete or corrupted."
                )
            model = XGBClassifier()
            model.load_model(str(model_path))
            agent.models[fault] = model
            logger.info(
                "[DiagnosticAgent] Sub-model '%s' loaded from '%s'.", fault, model_path
            )

        agent.is_trained = True
        logger.info("[DiagnosticAgent] Agent loaded from directory '%s'.", save_dir)
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
        trained_bits = list(self.models.keys()) if self.models else []
        return (
            f"DiagnosticAgent(status={status}, "
            f"fault_bits={FAULT_BITS}, trained_sub_models={trained_bits})"
        )