"""
critics/base_critic.py — Abstract base class for all Constitutional AI critics.

Each critic:
  - Loads a shared or dedicated HuggingFace language model.
  - Reads the constitution for its rule category.
  - Constructs a structured prompt and parses model output into JSON.
  - Returns a CriticResult with verdict, reason, and score.
"""

import json
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow imports from parent package
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CRITIC_MODEL_ID, CONSTITUTION_PATH
from logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Shared model cache — avoid reloading the same model multiple times in process
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_CACHE: dict = {}


def _load_model(model_id: str):
    if model_id not in _MODEL_CACHE:
        logger.info(f"Loading critic model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
        _MODEL_CACHE[model_id] = (tokenizer, model)
        logger.info(f"Critic model loaded: {model_id}")
    return _MODEL_CACHE[model_id]


# ─────────────────────────────────────────────────────────────────────────────
# Data contract
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CriticResult:
    critic_name: str
    verdict: str          # "pass" | "fail"
    reason: str
    score: float          # 0.0 (worst) → 1.0 (best)
    rules_referenced: list

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"CriticResult(name={self.critic_name}, verdict={self.verdict}, "
            f"score={self.score:.3f}, reason={self.reason[:80]!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────
class BaseCritic(ABC):
    """
    Abstract critic. Subclasses define:
      - category:     str   — constitution category to inject (SAFETY/ETHICS/QUALITY)
      - critic_name:  str   — human-readable identifier
      - _build_prompt(...)  — constructs the evaluation prompt
      - _parse_output(...)  — extracts structured result from raw model text
    """

    category: str = ""
    critic_name: str = "base"
    model_id: str = CRITIC_MODEL_ID

    def __init__(self):
        self.tokenizer, self.model = _load_model(self.model_id)
        self.constitution_rules = self._load_constitution_rules()
        logger.info(
            f"{self.critic_name} initialised | rules loaded: {len(self.constitution_rules)}"
        )

    # ── Constitution ──────────────────────────────────────────────────────────

    def _load_constitution_rules(self) -> list[str]:
        """Parse constitution.txt and return rules matching self.category."""
        rules = []
        in_section = False
        path = Path(CONSTITUTION_PATH)
        if not path.exists():
            logger.warning(f"Constitution file not found at {path}")
            return rules
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(f"[{self.category}"):
                    in_section = True
                    continue
                if line.startswith("[") and in_section:
                    in_section = False
                if in_section and line.startswith("RULE-"):
                    rules.append(line)
        return rules

    def _format_rules(self) -> str:
        return "\n".join(f"  • {r}" for r in self.constitution_rules)

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate(self, prompt: str, max_new_tokens: int = 180) -> str:
        """Run greedy decoding on the critic model for deterministic scoring."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
            padding=True,
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        # Strip the prompt tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── Prompt / Parse (overrideable) ────────────────────────────────────────

    @abstractmethod
    def _build_prompt(self, user_prompt: str, response: str) -> str:
        ...

    def _parse_output(self, raw: str, user_prompt: str, response: str) -> CriticResult:
        """
        Attempt JSON extraction from model output.
        Falls back to heuristic scoring if the model produces free text
        (common with small models).
        """
        # Try to find a JSON block in the output
        json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                verdict = str(data.get("verdict", "fail")).strip().lower()
                reason = str(data.get("reason", raw[:200]))
                score = float(data.get("score", 0.5))
                score = max(0.0, min(1.0, score))
                if verdict not in ("pass", "fail"):
                    verdict = "pass" if score >= 0.5 else "fail"
                return CriticResult(
                    critic_name=self.critic_name,
                    verdict=verdict,
                    reason=reason,
                    score=score,
                    rules_referenced=self.constitution_rules[:3],
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Heuristic fallback for small models that produce prose
        return self._heuristic_score(raw, user_prompt, response)

    def _heuristic_score(
        self, raw: str, user_prompt: str, response: str
    ) -> CriticResult:
        """
        Rule-keyword heuristic when model output is unstructured.
        Subclasses should override for category-specific keyword sets.
        """
        text = (user_prompt + " " + response).lower()
        harmful_keywords = [
            "kill", "bomb", "hack", "exploit", "illegal", "drug",
            "weapon", "racist", "sexist", "abuse", "stalk", "explode",
        ]
        hit_count = sum(1 for kw in harmful_keywords if kw in text)
        score = max(0.0, 1.0 - (hit_count * 0.2))
        verdict = "pass" if score >= 0.5 else "fail"
        return CriticResult(
            critic_name=self.critic_name,
            verdict=verdict,
            reason=f"Heuristic fallback: {hit_count} harmful keyword(s) detected.",
            score=round(score, 3),
            rules_referenced=self.constitution_rules[:2],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, user_prompt: str, response: str) -> CriticResult:
        """
        Main entry point. Build prompt → generate → parse → return CriticResult.
        """
        prompt = self._build_prompt(user_prompt, response)
        logger.debug(f"{self.critic_name}: running evaluation")
        raw_output = self._generate(prompt)
        logger.debug(f"{self.critic_name} raw output: {raw_output[:200]}")
        result = self._parse_output(raw_output, user_prompt, response)
        logger.info(f"{result}")
        return result
