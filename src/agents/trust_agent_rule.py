from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TrustAgentConfig:
    """
    Configuration for the rule-based trust agent.
    """

    # Final decision thresholds
    accept_threshold: float = 0.80
    reject_threshold: float = 0.50

    # Fast-path accept threshold for strong low-risk cases
    fast_accept_threshold: float = 0.65

    # Detection confidence thresholds
    high_detection_conf_threshold: float = 0.90
    medium_detection_conf_threshold: float = 0.60

    # Diagnostic probability thresholds
    high_diag_prob_threshold: float = 0.80
    medium_diag_prob_threshold: float = 0.60
    low_diag_prob_threshold: float = 0.20

    # Residual thresholds
    low_residual_threshold: float = 0.20
    medium_residual_threshold: float = 0.50

    # Disagreement thresholds
    low_disagreement_threshold: float = 0.20
    medium_disagreement_threshold: float = 0.50

    # Memory-related thresholds
    max_recent_escalations_for_stability: int = 1


class TrustAgent:
    """
    Rule-based baseline trust arbitration agent.

    Expected input schema:

    {
      "detection": {
        "fault_detected": bool,
        "p_fault": float,
        "p_normal": float,
        "confidence": "low|medium|high"   # optional
      },
      "diagnostic": {
        "fault_code": str,
        "active_labels": list[str],
        "inactive_labels": list[str],      # optional
        "unlikely_labels": list[str],      # optional
        "probabilities": {"G": float, ...}
      },
      "residual": {
        "value": float,
        "severity": "low|medium|high"      # optional
      },
      "disagreement": {
        "score": float,
        "severity": "low|medium|high"      # optional
      },
      "memory_summary": {
        "recent_confidence_stable": bool,  # optional
        "recent_fault_pattern": list[str], # optional
        "recent_escalations": int          # optional
      }
    }

    Output schema:

    {
      "decision": "ACCEPT" | "ESCALATE" | "REJECT",
      "trust_score": float,
      "structured_justification": { ... }
    }
    """

    def __init__(self, config: Optional[TrustAgentConfig] = None):
        self.config = config or TrustAgentConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, trust_payload: Dict[str, Any]) -> Dict[str, Any]:
        detection = trust_payload.get("detection", {})
        diagnostic = trust_payload.get("diagnostic", {})
        residual = trust_payload.get("residual", {})
        disagreement = trust_payload.get("disagreement", {})
        memory_summary = trust_payload.get("memory_summary", {})

        # -----------------------------
        # Extract core signals
        # -----------------------------
        fault_detected = bool(detection.get("fault_detected", False))
        p_fault = float(detection.get("p_fault", 0.0))
        p_normal = float(detection.get("p_normal", max(0.0, 1.0 - p_fault)))
        detection_confidence = detection.get(
            "confidence", self._confidence_bucket(p_fault)
        )

        fault_code = str(diagnostic.get("fault_code", "unknown"))
        active_labels = list(diagnostic.get("active_labels", []))
        inactive_labels = list(diagnostic.get("inactive_labels", []))
        unlikely_labels = list(diagnostic.get("unlikely_labels", []))
        diag_probs = diagnostic.get("probabilities", {})

        residual_value = float(residual.get("value", 0.0))
        disagreement_score = float(disagreement.get("score", 0.0))

        recent_confidence_stable = bool(
            memory_summary.get("recent_confidence_stable", True)
        )
        recent_fault_pattern = list(memory_summary.get("recent_fault_pattern", []))
        recent_escalations = int(memory_summary.get("recent_escalations", 0))

        # -----------------------------
        # Derived trust features
        # -----------------------------
        detection_margin = abs(p_fault - p_normal)
        max_diag_prob = max((float(v) for v in diag_probs.values()), default=0.0)
        strong_active_count = sum(
            1
            for label in active_labels
            if float(diag_probs.get(label, 0.0)) >= self.config.high_diag_prob_threshold
        )

        contradiction_flag = self._compute_contradiction_flag(
            fault_detected=fault_detected,
            p_fault=p_fault,
            active_labels=active_labels,
            diag_probs=diag_probs,
        )

        residual_severity = residual.get(
            "severity", self._residual_bucket(residual_value)
        )
        disagreement_severity = disagreement.get(
            "severity", self._disagreement_bucket(disagreement_score)
        )

        consistency_level = self._compute_consistency_level(
            contradiction_flag=contradiction_flag,
            residual_value=residual_value,
            disagreement_score=disagreement_score,
            fault_detected=fault_detected,
            strong_active_count=strong_active_count,
        )

        trust_score = self._compute_trust_score(
            p_fault=p_fault,
            detection_margin=detection_margin,
            max_diag_prob=max_diag_prob,
            contradiction_flag=contradiction_flag,
            residual_value=residual_value,
            disagreement_score=disagreement_score,
            recent_confidence_stable=recent_confidence_stable,
            recent_escalations=recent_escalations,
        )

        decision = self._decision_from_score_and_flags(
            trust_score=trust_score,
            contradiction_flag=contradiction_flag,
            residual_value=residual_value,
            disagreement_score=disagreement_score,
        )

        structured_justification = {
            "fault_detected": fault_detected,
            "detection_confidence": detection_confidence,
            "fault_code": fault_code,
            "active_labels": active_labels,
            "inactive_labels": inactive_labels,
            "unlikely_labels": unlikely_labels,
            "detection_margin": round(detection_margin, 6),
            "max_diagnostic_probability": round(max_diag_prob, 6),
            "strong_active_count": strong_active_count,
            "residual_value": round(residual_value, 6),
            "residual_severity": residual_severity,
            "disagreement_score": round(disagreement_score, 6),
            "disagreement_severity": disagreement_severity,
            "contradiction_flag": contradiction_flag,
            "consistency_level": consistency_level,
            "recent_confidence_stable": recent_confidence_stable,
            "recent_fault_pattern": recent_fault_pattern,
            "recent_escalations": recent_escalations,
            "decision_rationale": self._decision_rationale(
                decision=decision,
                trust_score=trust_score,
                contradiction_flag=contradiction_flag,
                residual_severity=residual_severity,
                disagreement_severity=disagreement_severity,
                detection_confidence=detection_confidence,
            ),
        }

        return {
            "decision": decision,
            "trust_score": round(trust_score, 6),
            "structured_justification": structured_justification,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _confidence_bucket(self, p_fault: float) -> str:
        if p_fault >= self.config.high_detection_conf_threshold:
            return "high"
        if p_fault >= self.config.medium_detection_conf_threshold:
            return "medium"
        return "low"

    def _residual_bucket(self, residual_value: float) -> str:
        if residual_value < self.config.low_residual_threshold:
            return "low"
        if residual_value < self.config.medium_residual_threshold:
            return "medium"
        return "high"

    def _disagreement_bucket(self, disagreement_score: float) -> str:
        if disagreement_score < self.config.low_disagreement_threshold:
            return "low"
        if disagreement_score < self.config.medium_disagreement_threshold:
            return "medium"
        return "high"

    def _compute_contradiction_flag(
        self,
        fault_detected: bool,
        p_fault: float,
        active_labels: List[str],
        diag_probs: Dict[str, float],
    ) -> bool:
        if not diag_probs:
            return False

        max_diag_prob = max(float(v) for v in diag_probs.values())

        # Detection strongly says fault, but diagnostic is weak everywhere
        if fault_detected and p_fault >= self.config.high_detection_conf_threshold:
            if max_diag_prob < self.config.medium_diag_prob_threshold:
                return True

        # Detection says no fault, but diagnostic strongly indicates one
        if (not fault_detected) and max_diag_prob >= self.config.high_diag_prob_threshold:
            return True

        # Detection says fault, but no diagnostic labels are active
        if fault_detected and len(active_labels) == 0:
            return True

        return False

    def _compute_consistency_level(
        self,
        contradiction_flag: bool,
        residual_value: float,
        disagreement_score: float,
        fault_detected: bool,
        strong_active_count: int,
    ) -> str:
        if contradiction_flag:
            return "low"

        if (
            residual_value < self.config.low_residual_threshold
            and disagreement_score < self.config.low_disagreement_threshold
            and (not fault_detected or strong_active_count >= 1)
        ):
            return "high"

        if (
            residual_value < self.config.medium_residual_threshold
            and disagreement_score < self.config.medium_disagreement_threshold
        ):
            return "medium"

        return "low"

    def _compute_trust_score(
        self,
        p_fault: float,
        detection_margin: float,
        max_diag_prob: float,
        contradiction_flag: bool,
        residual_value: float,
        disagreement_score: float,
        recent_confidence_stable: bool,
        recent_escalations: int,
    ) -> float:
        score = 0.0

        # Positive contributions
        score += 0.30 * p_fault
        score += 0.15 * detection_margin
        score += 0.25 * max_diag_prob

        # Penalties
        score -= 0.20 * min(max(residual_value, 0.0), 1.0)
        score -= 0.20 * min(max(disagreement_score, 0.0), 1.0)

        if contradiction_flag:
            score -= 0.25

        if not recent_confidence_stable:
            score -= 0.10

        if recent_escalations > self.config.max_recent_escalations_for_stability:
            score -= 0.10

        return max(0.0, min(1.0, score))

    def _decision_from_score_and_flags(
        self,
        trust_score: float,
        contradiction_flag: bool,
        residual_value: float,
        disagreement_score: float,
    ) -> str:
        # Hard reject conditions
        if contradiction_flag:
            return "REJECT"

        if (
            residual_value >= self.config.medium_residual_threshold
            and disagreement_score >= self.config.medium_disagreement_threshold
        ):
            return "REJECT"

        # Fast-path accept for clean, consistent, low-risk cases
        if (
            not contradiction_flag
            and residual_value < self.config.low_residual_threshold
            and disagreement_score < self.config.low_disagreement_threshold
            and trust_score >= self.config.fast_accept_threshold
        ):
            return "ACCEPT"

        # Threshold-based mapping
        if trust_score >= self.config.accept_threshold:
            return "ACCEPT"

        if trust_score < self.config.reject_threshold:
            return "REJECT"

        return "ESCALATE"

    def _decision_rationale(
        self,
        decision: str,
        trust_score: float,
        contradiction_flag: bool,
        residual_severity: str,
        disagreement_severity: str,
        detection_confidence: str,
    ) -> List[str]:
        reasons: List[str] = []

        reasons.append(f"trust_score={trust_score:.3f}")
        reasons.append(f"detection_confidence={detection_confidence}")
        reasons.append(f"residual_severity={residual_severity}")
        reasons.append(f"disagreement_severity={disagreement_severity}")

        if contradiction_flag:
            reasons.append("contradiction detected between detection and diagnostic signals")

        if decision == "ACCEPT":
            reasons.append("confidence and consistency are sufficient for acceptance")
        elif decision == "ESCALATE":
            reasons.append("uncertainty is moderate, so human review is recommended")
        else:
            reasons.append("evidence suggests low trust or strong inconsistency")

        return reasons

    def __repr__(self) -> str:
        return (
            "TrustAgent("
            f"accept_threshold={self.config.accept_threshold}, "
            f"reject_threshold={self.config.reject_threshold}, "
            f"fast_accept_threshold={self.config.fast_accept_threshold})"
        )