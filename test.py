import os
import json
import argparse
from datetime import datetime
import shutil

import numpy as np
import shap

from src.agents.detection_agent import DetectionAgent
from src.agents.diagnostic_agent import DiagnosticAgent, FAULT_BITS
from src.agents.explanation_agent import ExplanationAgent
from src.agents.trust_agent_rule import TrustAgent
from src.agents.trust_agent_llm import TrustAgentLLM


# -----------------------------
# CLI arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--print", action="store_true", help="Print JSON payloads to console")
parser.add_argument("--explain", action="store_true", help="Run explanation agent")
parser.add_argument("--trust", action="store_true", help="Run trust agent")
parser.add_argument("--save", action="store_true", help="Save outputs to disk")
parser.add_argument("--verbose", action="store_true", help="Print status messages")
parser.add_argument("--outdir", type=str, default="outputs", help="Base output directory")

# Input measurements
parser.add_argument("--Ia", type=float, default=10.5, help="Phase A current")
parser.add_argument("--Ib", type=float, default=-5.2, help="Phase B current")
parser.add_argument("--Ic", type=float, default=-4.8, help="Phase C current")
parser.add_argument("--Va", type=float, default=220.1, help="Phase A voltage")
parser.add_argument("--Vb", type=float, default=-110.0, help="Phase B voltage")
parser.add_argument("--Vc", type=float, default=-109.5, help="Phase C voltage")

# Explanation agent config
parser.add_argument(
    "--model",
    type=str,
    default="llama3.2:1b",
    help="Ollama model name for the explanation agent",
)
parser.add_argument(
    "--explanation-timeout",
    type=int,
    default=600,
    help="Timeout in seconds for the explanation agent request",
)

# Trust agent config
parser.add_argument(
    "--trust-mode",
    type=str,
    default="rule",
    choices=["rule", "llm"],
    help="Choose trust agent mode: rule or llm",
)
parser.add_argument(
    "--trust-model",
    type=str,
    default="llama3:8b",
    help="Ollama model name for the LLM trust agent",
)
parser.add_argument(
    "--trust-timeout",
    type=int,
    default=600,
    help="Timeout in seconds for the LLM trust agent request",
)

args = parser.parse_args()

# Default behavior: save only, silent
if not args.print and not args.save and not args.explain and not args.trust:
    args.save = True


# -----------------------------
# Directory handling
# -----------------------------
base_dir = args.outdir
latest_dir = os.path.join(base_dir, "latest")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
archive_dir = os.path.join(base_dir, timestamp)

if args.save or args.explain or args.trust:
    os.makedirs(base_dir, exist_ok=True)

    if os.path.exists(latest_dir):
        shutil.move(latest_dir, archive_dir)

    os.makedirs(latest_dir, exist_ok=True)

payload_file = os.path.join(latest_dir, "explanation_payload.json")
summary_file = os.path.join(latest_dir, "explanation_summary.json")
explanation_file = os.path.join(latest_dir, "explanation.txt")
trust_payload_file = os.path.join(latest_dir, "trust_payload.json")
trust_output_file = os.path.join(latest_dir, "trust_output.json")
memory_summary_file = os.path.join(latest_dir, "memory_summary.json")
memory_file = os.path.join(base_dir, "memory.json")


# -----------------------------
# Helper functions
# -----------------------------
def top_k_shap(feature_names, shap_values, k=3):
    items = list(zip(feature_names, shap_values))
    items = sorted(items, key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": name, "shap_value": float(val)} for name, val in items[:k]]


def positive_top_k_shap(feature_names, shap_values, k=3):
    items = [(name, float(val)) for name, val in zip(feature_names, shap_values) if float(val) > 0]
    items.sort(key=lambda x: x[1], reverse=True)
    return [{"feature": name, "shap_value": val} for name, val in items[:k]]


def negative_top_k_shap(feature_names, shap_values, k=3):
    items = [(name, float(val)) for name, val in zip(feature_names, shap_values) if float(val) < 0]
    items.sort(key=lambda x: x[1])  # most negative first
    return [{"feature": name, "shap_value": val} for name, val in items[:k]]


def confidence_bucket(p_fault: float) -> str:
    if p_fault >= 0.90:
        return "high"
    if p_fault >= 0.60:
        return "medium"
    return "low"


def severity_bucket(value: float, low_th: float = 0.20, med_th: float = 0.50) -> str:
    if value < low_th:
        return "low"
    if value < med_th:
        return "medium"
    return "high"


def compute_residual(p_fault, diagnostic_probabilities, active_labels):
    active_diag_mean = (
        sum(diagnostic_probabilities[label] for label in active_labels) / len(active_labels)
        if active_labels else 0.0
    )
    return abs(float(p_fault) - float(active_diag_mean))


def compute_disagreement(fault_detected, p_fault, diagnostic_probabilities, active_labels):
    max_diag_prob = max(diagnostic_probabilities.values()) if diagnostic_probabilities else 0.0
    active_diag_mean = (
        sum(diagnostic_probabilities[label] for label in active_labels) / len(active_labels)
        if active_labels else 0.0
    )

    components = [
        abs(float(p_fault) - float(active_diag_mean)),
        1.0 if fault_detected and len(active_labels) == 0 else 0.0,
        1.0 if (not fault_detected) and max_diag_prob >= 0.8 else 0.0,
    ]
    return sum(components) / len(components)


def load_memory(memory_file):
    if os.path.exists(memory_file):
        with open(memory_file, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "p_fault_history": [],
        "fault_code_history": [],
        "decision_history": [],
        "residual_history": [],
        "disagreement_history": [],
    }


def save_memory(memory_file, memory):
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


def update_memory(
    memory,
    p_fault,
    fault_code,
    residual_value,
    disagreement_score,
    decision=None,
    max_len=10,
):
    memory["p_fault_history"].append(float(p_fault))
    memory["fault_code_history"].append(str(fault_code))
    memory["residual_history"].append(float(residual_value))
    memory["disagreement_history"].append(float(disagreement_score))

    if decision is not None:
        memory["decision_history"].append(str(decision))

    for key in memory:
        memory[key] = memory[key][-max_len:]

    return memory


def build_memory_summary(memory):
    p_hist = memory.get("p_fault_history", [])
    fault_hist = memory.get("fault_code_history", [])
    decision_hist = memory.get("decision_history", [])
    residual_hist = memory.get("residual_history", [])
    disagreement_hist = memory.get("disagreement_history", [])

    recent_p_fault = p_hist[-5:]
    recent_fault_pattern = fault_hist[-5:]
    recent_decision_pattern = decision_hist[-5:]
    recent_residual = residual_hist[-5:]
    recent_disagreement = disagreement_hist[-5:]

    recent_confidence_stable = True
    if len(recent_p_fault) >= 2:
        recent_confidence_stable = (max(recent_p_fault) - min(recent_p_fault)) < 0.15

    recent_fault_pattern_stable = True
    if len(recent_fault_pattern) >= 2:
        recent_fault_pattern_stable = len(set(recent_fault_pattern)) == 1

    recent_decision_stable = True
    if len(recent_decision_pattern) >= 2:
        recent_decision_stable = len(set(recent_decision_pattern)) == 1

    recent_escalations = sum(1 for d in recent_decision_pattern if d == "ESCALATE")
    recent_rejects = sum(1 for d in recent_decision_pattern if d == "REJECT")
    recent_accepts = sum(1 for d in recent_decision_pattern if d == "ACCEPT")

    recent_residual_mean = (
        sum(recent_residual) / len(recent_residual) if recent_residual else 0.0
    )
    recent_disagreement_mean = (
        sum(recent_disagreement) / len(recent_disagreement) if recent_disagreement else 0.0
    )

    return {
        "recent_confidence_stable": recent_confidence_stable,
        "recent_fault_pattern": recent_fault_pattern,
        "recent_fault_pattern_stable": recent_fault_pattern_stable,
        "recent_decision_pattern": recent_decision_pattern,
        "recent_decision_stable": recent_decision_stable,
        "recent_escalations": recent_escalations,
        "recent_rejects": recent_rejects,
        "recent_accepts": recent_accepts,
        "recent_residual_mean": round(recent_residual_mean, 6),
        "recent_disagreement_mean": round(recent_disagreement_mean, 6),
    }


# -----------------------------
# Load models
# -----------------------------
det_agent = DetectionAgent.load("models/detection_agent")
diag_agent = DiagnosticAgent.load("models/diagnostic_agent")

# -----------------------------
# Input sample from CLI
# -----------------------------
new_data = np.array([[args.Ia, args.Ib, args.Ic, args.Va, args.Vb, args.Vc]])
feature_names = det_agent.feature_names or ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


# -----------------------------
# Detection
# -----------------------------
is_fault = det_agent.predict(new_data)
det_proba = det_agent.predict_proba(new_data)[0]
p_normal = float(det_proba[0])
p_fault = float(det_proba[1])

det_explainer = shap.TreeExplainer(det_agent.model)
det_shap_sample = det_explainer.shap_values(new_data)[0]
det_shap_dict = {name: float(val) for name, val in zip(feature_names, det_shap_sample)}

fault_detected = bool(int(is_fault[0]) == 1)
detection_confidence = confidence_bucket(p_fault)

det_top_abs = top_k_shap(feature_names, det_shap_sample, k=3)
det_top_supporting = positive_top_k_shap(feature_names, det_shap_sample, k=3)
det_top_opposing = negative_top_k_shap(feature_names, det_shap_sample, k=3)

# -----------------------------
# Diagnostic
# -----------------------------
fault_code = str(diag_agent.predict_labels(new_data)[0])
diag_proba = diag_agent.predict_fault_proba(new_data)[0]

predicted_bits = {fault: int(fault_code[i]) for i, fault in enumerate(FAULT_BITS)}
active_labels = [fault for fault in FAULT_BITS if predicted_bits[fault] == 1]
inactive_labels = [fault for fault in FAULT_BITS if predicted_bits[fault] == 0]
unlikely_labels = [fault for fault, prob in zip(FAULT_BITS, diag_proba) if float(prob) < 0.5]

diag_shap = {}
diag_top_abs = {}
diag_top_supporting = {}
diag_top_opposing = {}

for fault in FAULT_BITS:
    model = diag_agent.models[fault]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(new_data)[0]

    diag_shap[fault] = {name: float(val) for name, val in zip(feature_names, shap_vals)}
    diag_top_abs[fault] = top_k_shap(feature_names, shap_vals, k=3)
    diag_top_supporting[fault] = positive_top_k_shap(feature_names, shap_vals, k=3)
    diag_top_opposing[fault] = negative_top_k_shap(feature_names, shap_vals, k=3)

diagnostic_probabilities = {fault: float(prob) for fault, prob in zip(FAULT_BITS, diag_proba)}

# -----------------------------
# Full payload
# -----------------------------
payload = {
    "metadata": {
        "timestamp": timestamp,
        "archived_previous_latest_to": archive_dir if os.path.exists(archive_dir) else None,
    },
    "input_sample": {
        "Ia": float(args.Ia),
        "Ib": float(args.Ib),
        "Ic": float(args.Ic),
        "Va": float(args.Va),
        "Vb": float(args.Vb),
        "Vc": float(args.Vc),
    },
    "detection": {
        "predicted_label": int(is_fault[0]),
        "predicted_text": "fault" if fault_detected else "normal",
        "fault_detected": fault_detected,
        "detection_confidence": detection_confidence,
        "p_normal": p_normal,
        "p_fault": p_fault,
        "shap": det_shap_dict,
        "top_features": det_top_abs,
        "top_supporting_features": det_top_supporting,
        "top_opposing_features": det_top_opposing,
    },
    "diagnostic": {
        "fault_code": fault_code,
        "bit_order": FAULT_BITS,
        "predicted_bits": predicted_bits,
        "active_labels": active_labels,
        "inactive_labels": inactive_labels,
        "unlikely_labels": unlikely_labels,
        "probabilities": diagnostic_probabilities,
        "shap": diag_shap,
        "top_features": diag_top_abs,
        "top_supporting_features": diag_top_supporting,
        "top_opposing_features": diag_top_opposing,
    },
}

# -----------------------------
# Compact summary for explanation
# -----------------------------
summary_payload = {
    "fault_detected": fault_detected,
    "detection_confidence": detection_confidence,
    "fault_code": fault_code,
    "bit_order": FAULT_BITS,
    "predicted_bits": predicted_bits,
    "active_labels": active_labels,
    "inactive_labels": inactive_labels,
    "unlikely_labels": unlikely_labels,
    "detection_probability": {
        "p_fault": p_fault,
        "p_normal": p_normal,
    },
    "detection_top_supporting_features": det_top_supporting,
    "detection_top_opposing_features": det_top_opposing,
    "diagnostic_probabilities": diagnostic_probabilities,
    "diagnostic_top_supporting_features": {
        fault: diag_top_supporting[fault] for fault in FAULT_BITS
    },
    "diagnostic_top_opposing_features": {
        fault: diag_top_opposing[fault] for fault in FAULT_BITS
    },
}

# -----------------------------
# Compute trust signals
# -----------------------------
residual_value = compute_residual(p_fault, diagnostic_probabilities, active_labels)
disagreement_score = compute_disagreement(
    fault_detected,
    p_fault,
    diagnostic_probabilities,
    active_labels,
)

memory = load_memory(memory_file)
memory_summary = build_memory_summary(memory)

# -----------------------------
# Build trust payload
# -----------------------------
trust_payload = {
    "detection": {
        "fault_detected": fault_detected,
        "p_fault": p_fault,
        "p_normal": p_normal,
        "confidence": detection_confidence,
    },
    "diagnostic": {
        "fault_code": fault_code,
        "active_labels": active_labels,
        "inactive_labels": inactive_labels,
        "unlikely_labels": unlikely_labels,
        "probabilities": diagnostic_probabilities,
    },
    "residual": {
        "value": residual_value,
        "severity": severity_bucket(residual_value),
    },
    "disagreement": {
        "score": disagreement_score,
        "severity": severity_bucket(disagreement_score),
    },
    "memory_summary": memory_summary,
}

# -----------------------------
# Save JSON outputs
# -----------------------------
if args.save or args.explain or args.trust:
    with open(payload_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    with open(trust_payload_file, "w", encoding="utf-8") as f:
        json.dump(trust_payload, f, indent=2)

    with open(memory_summary_file, "w", encoding="utf-8") as f:
        json.dump(memory_summary, f, indent=2)

    if args.verbose:
        print(f"Saved full payload to {payload_file}")
        print(f"Saved summary payload to {summary_file}")
        print(f"Saved trust payload to {trust_payload_file}")
        print(f"Saved memory summary to {memory_summary_file}")

# -----------------------------
# Optional: print JSON outputs
# -----------------------------
if args.print:
    print("=== Full Payload ===")
    print(json.dumps(payload, indent=2))

    print("\n=== Explanation Summary Payload ===")
    print(json.dumps(summary_payload, indent=2))

    print("\n=== Trust Payload ===")
    print(json.dumps(trust_payload, indent=2))

    print("\n=== Memory Summary ===")
    print(json.dumps(memory_summary, indent=2))

# -----------------------------
# Optional: run explanation agent
# -----------------------------
if args.explain:
    explainer = ExplanationAgent(
        model_name=args.model,
        timeout=args.explanation_timeout,
    )

    try:
        explanation = explainer.explain(summary_payload)

        with open(explanation_file, "w", encoding="utf-8") as f:
            f.write(explanation)

        print("\n=== Explanation ===\n")
        print(explanation)

        if args.verbose:
            print(f"\nSaved explanation to {explanation_file}")

    except Exception as e:
        print(f"\n[WARNING] Explanation agent failed: {e}")

        if args.verbose:
            print("Continuing to trust agent without explanation output.")

# -----------------------------
# Optional: run trust agent
# -----------------------------
if args.trust:
    if args.trust_mode == "llm":
        trust_agent = TrustAgentLLM(
            model_name=args.trust_model,
            timeout=args.trust_timeout,
        )
    else:
        trust_agent = TrustAgent()

    trust_output = trust_agent.evaluate(trust_payload)

    with open(trust_output_file, "w", encoding="utf-8") as f:
        json.dump(trust_output, f, indent=2)

    memory = update_memory(
        memory,
        p_fault=p_fault,
        fault_code=fault_code,
        residual_value=residual_value,
        disagreement_score=disagreement_score,
        decision=trust_output["decision"],
    )
    save_memory(memory_file, memory)

    print("\n=== Trust Output ===\n")
    print(json.dumps(trust_output, indent=2))

    if args.verbose:
        print(f"\nSaved trust output to {trust_output_file}")
        print(f"Updated memory file at {memory_file}")