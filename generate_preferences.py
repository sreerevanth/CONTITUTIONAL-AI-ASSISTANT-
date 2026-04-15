"""
generate_preferences.py — Synthetic Preference Dataset Generator.

Pipeline:
  1. Load base generation model (OPT-125M or configured model).
  2. For each seed prompt → generate NUM_CANDIDATES responses.
  3. Run each (prompt, response) pair through all critics.
  4. Aggregator scores and labels each response.
  5. Select best (chosen) and worst (rejected) responses per prompt.
  6. Save structured preference dataset to data/preferences.json.

The resulting dataset is compatible with TRL DPOTrainer's expected format.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BASE_MODEL_ID,
    GENERATION_CONFIG,
    NUM_CANDIDATES,
    SEED_PROMPTS,
    ADVERSARIAL_PROMPTS,
    PREFERENCES_PATH,
    SAFE_FALLBACK,
)
from critics.safety_critic import SafetyCritic
from critics.ethics_critic import EthicsCritic
from critics.quality_critic import QualityCritic
from aggregator import Aggregator
from logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class PreferenceGenerator:
    """
    Orchestrates the full synthetic preference generation pipeline.
    Designed to be memory-efficient: loads models once, reuses across prompts.
    """

    def __init__(self):
        logger.info("Initialising PreferenceGenerator...")
        self._load_base_model()
        self._load_critics()
        self.aggregator = Aggregator()
        logger.info("PreferenceGenerator ready.")

    def _load_base_model(self):
        logger.info(f"Loading base generation model: {BASE_MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        logger.info("Base model loaded.")

    def _load_critics(self):
        logger.info("Loading critic agents...")
        self.critics = [
            SafetyCritic(),
            EthicsCritic(),
            QualityCritic(),
        ]
        logger.info(f"Loaded {len(self.critics)} critic agents.")

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate_response(self, prompt: str, temperature: float = None) -> str:
        """Generate a single response from the base model."""
        cfg = dict(GENERATION_CONFIG)
        if temperature is not None:
            cfg["temperature"] = temperature

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=cfg["max_new_tokens"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                do_sample=cfg["do_sample"],
                repetition_penalty=cfg["repetition_penalty"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _generate_candidates(self, prompt: str) -> List[str]:
        """Generate NUM_CANDIDATES diverse responses using temperature variation."""
        temperatures = [0.7 + (i * 0.3) for i in range(NUM_CANDIDATES)]
        candidates = []
        for i, temp in enumerate(temperatures):
            logger.info(f"  Generating candidate {i + 1}/{NUM_CANDIDATES} (T={temp:.1f})")
            response = self._generate_response(prompt, temperature=temp)
            if not response:
                response = "I can provide information on this topic."
            candidates.append(response)
        return candidates

    # ── Critique ──────────────────────────────────────────────────────────────

    def _critique_response(self, prompt: str, response: str) -> "AggregatorResult":
        """Run all critics + aggregator on a single (prompt, response) pair."""
        critic_results = [critic.evaluate(prompt, response) for critic in self.critics]
        return self.aggregator.aggregate(
            critic_results,
            user_prompt=prompt,
            response=response,
            log=True,
        )

    # ── Preference pair construction ─────────────────────────────────────────

    def _build_preference_pair(
        self,
        prompt: str,
        candidates: List[str],
    ) -> Dict[str, Any]:
        """
        Score all candidates, rank by reward, return chosen/rejected pair.
        Handles edge cases: single candidate, tied scores.
        """
        scored = []
        for candidate in candidates:
            agg_result = self._critique_response(prompt, candidate)
            scored.append((candidate, agg_result))

        # Sort descending by reward
        scored.sort(key=lambda x: x[1].aggregated_reward, reverse=True)

        chosen_response, chosen_agg = scored[0]
        rejected_response, rejected_agg = scored[-1]  # Worst if multiple

        # If only one candidate, synthesise a safe fallback as rejected
        if len(scored) == 1:
            rejected_response = SAFE_FALLBACK
            rejected_agg.aggregated_reward = 0.0

        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "chosen_reward": round(chosen_agg.aggregated_reward, 4),
            "rejected_reward": round(rejected_agg.aggregated_reward, 4),
            "chosen_critics": chosen_agg.critic_scores,
            "rejected_critics": rejected_agg.critic_scores,
        }

    # ── Main pipeline ────────────────────────────────────────────────────────

    def generate(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Run the full preference generation pipeline over a list of prompts.
        Returns a list of preference pair dicts.
        """
        dataset = []
        total = len(prompts)

        for idx, prompt in enumerate(prompts):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing prompt {idx + 1}/{total}: {prompt[:80]!r}")
            logger.info(f"{'='*60}")

            try:
                candidates = self._generate_candidates(prompt)
                pair = self._build_preference_pair(prompt, candidates)
                dataset.append(pair)

                logger.info(
                    f"Preference pair | "
                    f"chosen_reward={pair['chosen_reward']:.4f} "
                    f"rejected_reward={pair['rejected_reward']:.4f}"
                )
            except Exception as exc:
                logger.error(f"Failed on prompt {idx + 1}: {exc}", exc_info=True)
                continue

        return dataset

    def save(self, dataset: List[Dict[str, Any]]) -> None:
        """Persist dataset to PREFERENCES_PATH as JSON."""
        PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PREFERENCES_PATH, "w", encoding="utf-8") as fh:
            json.dump(dataset, fh, indent=2, ensure_ascii=False)
        logger.info(f"Dataset saved: {PREFERENCES_PATH} ({len(dataset)} pairs)")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_prompts = SEED_PROMPTS + ADVERSARIAL_PROMPTS

    logger.info(f"Starting preference generation | total prompts: {len(all_prompts)}")
    t0 = time.time()

    generator = PreferenceGenerator()
    dataset = generator.generate(all_prompts)
    generator.save(dataset)

    elapsed = time.time() - t0
    logger.info(f"\nPreference generation complete.")
    logger.info(f"  Total pairs   : {len(dataset)}")
    logger.info(f"  Elapsed time  : {elapsed:.1f}s")
    logger.info(f"  Output path   : {PREFERENCES_PATH}")

    # Print summary statistics
    if dataset:
        chosen_rewards = [d["chosen_reward"] for d in dataset]
        rejected_rewards = [d["rejected_reward"] for d in dataset]
        avg_margin = sum(c - r for c, r in zip(chosen_rewards, rejected_rewards)) / len(dataset)
        logger.info(f"  Avg reward margin (chosen - rejected): {avg_margin:.4f}")


if __name__ == "__main__":
    main()
