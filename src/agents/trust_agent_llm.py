from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from src.agents.trust_agent import TrustAgent


class TrustAgentLLM:
    """
    LLM-based trust arbitration agent with rule-based fallback.

    Input:
        trust_payload: structured trust features

    Output schema:
    {
      "decision": "ACCEPT" | "ESCALATE" | "REJECT",
      "trust_score": float in [0,1],
      "structured_justification": {
        "summary": str,
        "key_factors": list[str],
        "risk_flags": list[str],
        "recommended_action": str
      }
    }
    """

    VALID_DECISIONS = {"ACCEPT", "ESCALATE", "REJECT"}

    def __init__(
        self,
        model_name: str = "llama3:8b",
        ollama_url: str = "http://localhost:11434/api/generate",
        timeout: int = 600,
        fallback_agent: Optional[TrustAgent] = None,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.fallback_agent = fallback_agent or TrustAgent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, trust_payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(trust_payload)

        try:
            raw_text = self._query_ollama(prompt)
            parsed = self._extract_json(raw_text)
            validated = self._validate_output(parsed)
            return validated
        except Exception as e:
            fallback = self.fallback_agent.evaluate(trust_payload)
            fallback.setdefault("metadata", {})
            fallback["metadata"]["fallback_used"] = True
            fallback["metadata"]["fallback_reason"] = str(e)
            fallback["metadata"]["source"] = "rule_based_fallback"
            return fallback

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    def _build_prompt(self, trust_payload: Dict[str, Any]) -> str:
        return f"""
You are a trust arbitration agent for an electrical fault diagnosis system.

Your task is to decide whether the model prediction should be:
- ACCEPT
- ESCALATE
- REJECT

You must use ONLY the provided JSON input.
Do NOT invent physical causes, components, or interpretations not present in the input.
Do NOT output markdown.
Do NOT output any text before or after the JSON.

Decision guidance:
- Prefer ACCEPT when confidence is high, contradiction is absent, and residual/disagreement are low.
- Prefer ESCALATE when evidence is mixed or uncertainty is moderate.
- Prefer REJECT when contradiction is present or inconsistency is severe.

Return exactly one valid JSON object in this schema:
{{
  "decision": "ACCEPT | ESCALATE | REJECT",
  "trust_score": 0.0,
  "structured_justification": {{
    "summary": "short summary",
    "key_factors": ["factor 1", "factor 2"],
    "risk_flags": ["flag 1", "flag 2"],
    "recommended_action": "short action"
  }}
}}

Input JSON:
{json.dumps(trust_payload, indent=2)}
""".strip()

    # ------------------------------------------------------------------
    # Ollama call
    # ------------------------------------------------------------------

    def _query_ollama(self, prompt: str) -> str:
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise ValueError("Ollama response missing 'response' field.")
        return data["response"]

    # ------------------------------------------------------------------
    # Parsing / validation
    # ------------------------------------------------------------------

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        raw_text = raw_text.strip()

        # First try direct parse
        try:
            obj = json.loads(raw_text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Fallback: extract first JSON object from text
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM output.")

        candidate = raw_text[start:end + 1]
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            raise ValueError("Parsed JSON is not an object.")
        return obj

    def _validate_output(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        if "decision" not in obj:
            raise ValueError("LLM output missing 'decision'.")
        if "trust_score" not in obj:
            raise ValueError("LLM output missing 'trust_score'.")
        if "structured_justification" not in obj:
            raise ValueError("LLM output missing 'structured_justification'.")

        decision = str(obj["decision"]).strip().upper()
        if decision not in self.VALID_DECISIONS:
            raise ValueError(f"Invalid decision: {decision}")

        try:
            trust_score = float(obj["trust_score"])
        except Exception as e:
            raise ValueError(f"Invalid trust_score: {obj['trust_score']}") from e

        trust_score = max(0.0, min(1.0, trust_score))

        sj = obj["structured_justification"]
        if not isinstance(sj, dict):
            raise ValueError("'structured_justification' must be an object.")

        summary = str(sj.get("summary", "")).strip()
        key_factors = sj.get("key_factors", [])
        risk_flags = sj.get("risk_flags", [])
        recommended_action = str(sj.get("recommended_action", "")).strip()

        if not isinstance(key_factors, list):
            key_factors = [str(key_factors)]
        if not isinstance(risk_flags, list):
            risk_flags = [str(risk_flags)]

        key_factors = [str(x) for x in key_factors]
        risk_flags = [str(x) for x in risk_flags]

        return {
            "decision": decision,
            "trust_score": round(trust_score, 6),
            "structured_justification": {
                "summary": summary,
                "key_factors": key_factors,
                "risk_flags": risk_flags,
                "recommended_action": recommended_action,
            },
            "metadata": {
                "fallback_used": False,
                "source": "llm",
                "model_name": self.model_name,
            },
        }

    def __repr__(self) -> str:
        return (
            f"TrustAgentLLM(model_name={self.model_name}, "
            f"timeout={self.timeout})"
        )