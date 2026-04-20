import json
import requests


class ExplanationAgent:
    """
    Explanation Agent

    Converts structured model outputs (detection + diagnostic + SHAP)
    into human-readable explanations using a local LLM (Ollama).
    """

    def __init__(
        self,
        model_name: str = "llama3:8b",
        ollama_url: str = "http://localhost:11434/api/generate",
        timeout: int = 600,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.timeout = timeout

    def explain(self, payload: dict) -> str:
        prompt = self._build_prompt(payload)
        return self._query_ollama(prompt)

    def explain_from_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return self.explain(payload)

    def _build_prompt(self, payload: dict) -> str:
        return f"""
You are an electrical fault explanation agent.

Your task is to summarize the provided JSON only.
You must not infer, expand, interpret, or guess anything beyond the JSON.

Critical grounding rules:
1. Use ONLY facts explicitly present in the JSON.
2. Do NOT invent physical causes, component names, circuit behavior, or system mechanisms.
3. Do NOT expand label names. Labels such as G, C, B, and A must be written exactly as given.
4. Do NOT assign meanings to G, C, B, or A unless the JSON explicitly defines them.
5. Do NOT infer meanings for Ia, Ib, Ic, Va, Vb, or Vc beyond them being feature names.
6. Do NOT say a label is active unless it appears in active_labels or predicted_bits with value 1.
7. Do NOT say a label is unlikely unless it appears in unlikely_labels or has explicitly low probability in the JSON.
8. Do NOT confuse feature names with fault labels.
9. If information is missing, omit it. Do not guess.
10. If the JSON and any inference conflict, use the JSON only.

How to interpret fields:
- fault labels come from: active_labels, inactive_labels, unlikely_labels, predicted_bits, bit_order
- detection confidence comes from: detection_confidence
- fault code comes from: fault_code
- supporting evidence comes from:
  - detection_top_supporting_features
  - diagnostic_top_supporting_features
- opposing evidence comes from:
  - detection_top_opposing_features
  - diagnostic_top_opposing_features

Allowed content:
- whether fault is detected
- detection confidence
- fault code
- active labels
- unlikely labels
- feature names that support the prediction
- feature names that oppose the prediction

Forbidden content:
- any guessed expansion of labels
- any guessed physical interpretation
- any guessed cause of the fault
- any guessed circuit or power-system explanation

Return output in exactly this format and nothing else:

Explanation:
<A short factual summary using only the JSON. Maximum 2 sentences.>

Key Evidence:
- <fact from JSON>
- <fact from JSON>
- <fact from JSON>

Diagnostic Summary:
- Fault detected: <yes/no>
- Detection confidence: <low/medium/high>
- Fault code: <code>
- Active labels: <comma-separated labels or "none">
- Unlikely labels: <comma-separated labels or "none">

Now summarize this JSON:
{json.dumps(payload, indent=2)}
""".strip()

    def _query_ollama(self, prompt: str) -> str:
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"]

    def __repr__(self) -> str:
        return f"ExplanationAgent(model={self.model_name})"