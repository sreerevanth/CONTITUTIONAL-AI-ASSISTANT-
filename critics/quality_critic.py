"""
critics/quality_critic.py — Quality critic agent.

Evaluates: helpfulness, clarity, accuracy, appropriate length,
coherence, and relevance to the user's intent.

Constitutional category: QUALITY (weight=0.25)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from critics.base_critic import BaseCritic, CriticResult

# Minimum word threshold for a substantive response
MIN_RESPONSE_WORDS = 10
# Penalty trigger: responses shorter than this fraction of the prompt length
BREVITY_RATIO_THRESHOLD = 0.3


class QualityCritic(BaseCritic):
    """
    Quality Critic Agent.

    Scores helpfulness, coherence, and relevance.
    Does NOT penalise responses for being safe refusals —
    that is the domain of safety/ethics critics.
    """

    category = "QUALITY"
    critic_name = "quality_critic"

    def _build_prompt(self, user_prompt: str, response: str) -> str:
        rules_text = self._format_rules()
        return (
            f"You are a quality assurance auditor evaluating AI response quality "
            f"according to these standards:\n"
            f"{rules_text}\n\n"
            f"=== USER PROMPT ===\n{user_prompt}\n\n"
            f"=== AI RESPONSE ===\n{response}\n\n"
            f"Evaluate: Does the response clearly address the user's intent? "
            f"Is it coherent, appropriately detailed, and free of padding?\n"
            f"Respond ONLY with valid JSON:\n"
            f'{{"verdict": "pass" or "fail", "reason": "brief explanation", "score": 0.0-1.0}}\n'
            f"score=1.0 means excellent quality, 0.0 means useless.\n"
            f"JSON:"
        )

    def _heuristic_score(
        self, raw: str, user_prompt: str, response: str
    ) -> CriticResult:
        """
        Quality heuristics:
          - Length check (too short → low score)
          - Repetition detection (repeated 5-gram → penalty)
          - Basic coherence: response contains words from the prompt
        """
        response_words = response.split()
        prompt_words = set(user_prompt.lower().split())
        response_words_lower = set(response.lower().split())

        score = 1.0
        issues = []

        # Length penalty
        if len(response_words) < MIN_RESPONSE_WORDS:
            score -= 0.4
            issues.append("response too short")

        # Repetition detection: find repeated 5-grams
        ngrams = [
            " ".join(response_words[i : i + 5])
            for i in range(len(response_words) - 4)
        ]
        if len(ngrams) > 0:
            unique_ratio = len(set(ngrams)) / len(ngrams)
            if unique_ratio < 0.7:
                score -= 0.3
                issues.append(f"repetitive content (unique n-gram ratio={unique_ratio:.2f})")

        # Relevance: overlap of meaningful words
        overlap = len(prompt_words & response_words_lower) / max(len(prompt_words), 1)
        if overlap < 0.1 and len(prompt_words) > 3:
            score -= 0.2
            issues.append(f"low lexical overlap with prompt ({overlap:.2f})")

        score = round(max(0.0, min(1.0, score)), 3)
        verdict = "pass" if score >= 0.5 else "fail"
        reason = (
            "Quality heuristic issues: " + "; ".join(issues) + "."
            if issues
            else f"Response appears adequate (word count={len(response_words)})."
        )

        return CriticResult(
            critic_name=self.critic_name,
            verdict=verdict,
            reason=reason,
            score=score,
            rules_referenced=self.constitution_rules,
        )
