"""
aggregator.py — RLAIF Aggregator (Reward Aggregation Core).

Responsibilities:
  1. Accept CriticResult objects from all active critics.
  2. Compute a weighted reward signal: R = Σ(weight_i * score_i).
  3. Apply threshold decision: R >= REJECTION_THRESHOLD → "chosen", else "rejected".
  4. Persist the full decision trace to the JSONL decision log.
  5. Return an AggregatorResult with all constituent data.

This module is the heart of the RLAIF pipeline — it converts multi-agent
critique signals into the binary preference labels used by DPO training.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import CRITIC_WEIGHTS, REJECTION_THRESHOLD
from critics.base_critic import CriticResult
from logger import get_logger, log_decision

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AggregatorResult:
    """
    Full output of the aggregation step.

    Attributes
    ----------
    aggregated_reward : float
        Weighted sum of critic scores in [0, 1].
    label : str
        "chosen" if reward >= REJECTION_THRESHOLD else "rejected".
    critic_scores : Dict[str, float]
        Per-critic raw scores.
    critic_verdicts : Dict[str, str]
        Per-critic "pass"/"fail" verdicts.
    critic_reasons : Dict[str, str]
        Per-critic explanations.
    breakdown : List[dict]
        Full CriticResult dicts for downstream inspection.
    threshold_used : float
        The rejection threshold applied.
    weights_used : Dict[str, float]
        Critic weights used during aggregation.
    """

    aggregated_reward: float
    label: str
    critic_scores: Dict[str, float] = field(default_factory=dict)
    critic_verdicts: Dict[str, str] = field(default_factory=dict)
    critic_reasons: Dict[str, str] = field(default_factory=dict)
    breakdown: List[dict] = field(default_factory=list)
    threshold_used: float = REJECTION_THRESHOLD
    weights_used: Dict[str, float] = field(default_factory=lambda: dict(CRITIC_WEIGHTS))

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"  Aggregated Reward : {self.aggregated_reward:.4f}",
            f"  Decision Label    : {self.label.upper()}",
            f"  Threshold         : {self.threshold_used}",
        ]
        for name, score in self.critic_scores.items():
            verdict = self.critic_verdicts.get(name, "?")
            weight = self.weights_used.get(name, 0.0)
            lines.append(
                f"  {name:<20} score={score:.3f}  verdict={verdict}  weight={weight}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────────────────

class Aggregator:
    """
    Multi-critic reward aggregator.

    The aggregation formula is a weighted sum:

        R = Σ_i  w_i * s_i

    where w_i is the constitutional weight of critic i (from config.py)
    and s_i is its normalised score in [0, 1].

    Weights that do not match any received critic are silently ignored.
    Missing critics are penalised by substituting score=0.0.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None,
    ):
        # Map critic_name suffix → weight key
        # e.g. "safety_critic" → "safety"
        self.weights: Dict[str, float] = weights or dict(CRITIC_WEIGHTS)
        self.threshold: float = threshold if threshold is not None else REJECTION_THRESHOLD

        # Validate weights sum to ~1
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Critic weights sum to {total:.4f} (expected 1.0). "
                "Scores will be rescaled."
            )
        self._weight_sum = total

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _resolve_weight(self, critic_name: str) -> float:
        """
        Resolve weight for a critic by matching its name prefix.
        E.g. "safety_critic" → checks "safety_critic", then "safety".
        """
        if critic_name in self.weights:
            return self.weights[critic_name]
        # Try prefix match
        for key, w in self.weights.items():
            if critic_name.startswith(key) or key in critic_name:
                return w
        logger.warning(f"No weight found for critic '{critic_name}'. Defaulting to 0.0.")
        return 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def aggregate(
        self,
        critic_results: List[CriticResult],
        user_prompt: str = "",
        response: str = "",
        log: bool = True,
    ) -> AggregatorResult:
        """
        Aggregate a list of CriticResult objects into a single reward signal.

        Parameters
        ----------
        critic_results : list of CriticResult
        user_prompt    : original user prompt (for logging)
        response       : AI response being evaluated (for logging)
        log            : whether to persist this decision to the JSONL log

        Returns
        -------
        AggregatorResult
        """
        if not critic_results:
            logger.error("Aggregator received empty critic_results list.")
            return AggregatorResult(
                aggregated_reward=0.0,
                label="rejected",
            )

        weighted_sum = 0.0
        weight_total = 0.0
        critic_scores: Dict[str, float] = {}
        critic_verdicts: Dict[str, str] = {}
        critic_reasons: Dict[str, str] = {}
        breakdown = []

        for result in critic_results:
            w = self._resolve_weight(result.critic_name)
            weighted_sum += w * result.score
            weight_total += w

            critic_scores[result.critic_name] = round(result.score, 4)
            critic_verdicts[result.critic_name] = result.verdict
            critic_reasons[result.critic_name] = result.reason
            breakdown.append(result.to_dict())

            logger.debug(
                f"  [{result.critic_name}] score={result.score:.3f} "
                f"weight={w:.2f} contribution={w * result.score:.3f}"
            )

        # Normalise in case weights don't perfectly sum to 1
        aggregated_reward = (
            weighted_sum / weight_total if weight_total > 0 else 0.0
        )
        aggregated_reward = round(aggregated_reward, 4)

        label = "chosen" if aggregated_reward >= self.threshold else "rejected"

        agg_result = AggregatorResult(
            aggregated_reward=aggregated_reward,
            label=label,
            critic_scores=critic_scores,
            critic_verdicts=critic_verdicts,
            critic_reasons=critic_reasons,
            breakdown=breakdown,
            threshold_used=self.threshold,
            weights_used=dict(self.weights),
        )

        logger.info(
            f"Aggregation complete | reward={aggregated_reward:.4f} "
            f"threshold={self.threshold} → label={label.upper()}"
        )
        logger.info("\n" + agg_result.summary())

        if log:
            log_decision(
                prompt=user_prompt,
                response=response,
                critic_scores=critic_scores,
                aggregated_reward=aggregated_reward,
                verdict=label,
            )

        return agg_result

    def compare(
        self,
        prompt: str,
        response_a: str,
        result_a: AggregatorResult,
        response_b: str,
        result_b: AggregatorResult,
    ) -> Dict[str, str]:
        """
        Given two aggregated results for the same prompt, return a
        preference pair dict: {"chosen": <better_response>, "rejected": <worse_response>}.

        If both are equally scored, response_a is chosen by default.
        """
        if result_a.aggregated_reward >= result_b.aggregated_reward:
            chosen, rejected = response_a, response_b
            chosen_reward, rejected_reward = result_a.aggregated_reward, result_b.aggregated_reward
        else:
            chosen, rejected = response_b, response_a
            chosen_reward, rejected_reward = result_b.aggregated_reward, result_a.aggregated_reward

        logger.info(
            f"Preference pair created | "
            f"chosen_reward={chosen_reward:.4f} rejected_reward={rejected_reward:.4f}"
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
        }
